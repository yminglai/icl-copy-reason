"""LoRA recipe transfer on released 7-8B models.

Arms: T = synthetic definition-use documents whose queried cells are unshown, L = twins whose queried cells are attested, Z = the base.
Inputs : a data directory (build_data), a base model id, a packed row file (pack_rows), a learning rate (calibrate_rate), seeds.
Outputs: RUN/config.json, RUN/train.jsonl, RUN/merged/ (adapter merged into the base weights, so every reading runs unmodified).

Recipe of record: LoRA r 16, alpha 32, dropout 0 on every attention and MLP projection (q, k, v, o, gate, up, down; the GPT-NeoX family names
them query_key_value / dense / dense_h_to_4h / dense_4h_to_h); AdamW betas (.9, .95), weight decay 0, clip 1; 30-step linear warm-up then cosine
to 10 %; global batch 32 rows of 2,048 tokens (micro-batch x accumulation); bf16 base + fp32 adapters, gradient checkpointing; the merged weights
must reproduce the adapter within .01 nats on a probe batch. Rate rule: SAFE rates have finite loss, clip fraction <= 20 % and general-text
validation loss within +.02 nats of step 0; SELECTED = the safe rate with the lowest held-out synthetic validation loss (ties -> lower; none -> 5e-5 + flag).
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from typing import Any

import numpy as np

from icl_copy_reason.io import read_jsonl, sha256_file
from icl_copy_reason.training.pack import SEQ, lora_rows, rows_from_text_stream

GLOBAL_BATCH = 32
MERGE_TOL = 0.01
GEN_VER = 3  # synthetic-document generator version: n_q in 2..4, twins differ only in the queried cells
REPLAY_CALIB = 3000  # replay documents reserved for the calibration stream
REPLAY_VAL = 400  # replay documents reserved for the calibration validation loss
CALIB_VAL_ROWS = 64  # held-out synthetic rows (tail of the calibration pack) for the synthetic validation loss
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TARGETS_BY_TYPE = {"gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]}
BASES = {"qwen": "Qwen/Qwen2.5-7B", "olmo2": "allenai/OLMo-2-1124-7B", "llama31": "meta-llama/Llama-3.1-8B", "comma": "common-pile/comma-v0.1-1t",
         "redpajama": "togethercomputer/RedPajama-INCITE-7B-Base"}


def device() -> str:
    return os.environ.get("ICR_DEVICE", "cuda")


def targets_for(config) -> list[str]:
    return TARGETS_BY_TYPE.get(getattr(config, "model_type", ""), TARGETS)


def model_sha(repo: str) -> str | None:
    """Hub commit of the base weights; None for a local path or offline."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo).sha if not os.path.exists(repo) else None
    except Exception:  # noqa: BLE001 - offline is not an error here
        return None


# ------------------------------------------------------------------------------------------------------------------------ data
def build_data(out: str, n_docs: int = 200000, seed: int = 0, calib_seed: int = 9, n_calib: int = 8000, n_replay: int = 60000,
               smollm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M") -> dict[str, Any]:
    """Write synth_T / synth_L twins (document k of both from Random(seed*10**6 + k)), calibration twins, the replay stream, the two held-out
    text sets (the same documents as the admixture runs' validation token files, stored as text so every base re-tokenises them) and data.json."""
    from icl_copy_reason.data_construction.synthetic_docs import make_doc

    os.makedirs(out, exist_ok=True)
    meta: dict[str, Any] = {"generator_version": GEN_VER, "n_docs": n_docs, "seed": seed, "calib_seed": calib_seed, "n_calib": n_calib, "n_replay": n_replay}
    stats: dict[str, list] = {"T": [], "L": []}
    sem = {"n_q_reduced": 0, "T_shown_query": 0, "L_unshown_query": 0}
    with open(f"{out}/synth_T.jsonl", "w") as ft, open(f"{out}/synth_L.jsonl", "w") as fl:
        for k in range(n_docs):
            dt, mt = make_doc(random.Random(seed * 10**6 + k), "compose", GEN_VER)
            dl, ml = make_doc(random.Random(seed * 10**6 + k), "attested", GEN_VER)
            ft.write(json.dumps({"k": k, "text": dt, **mt}) + "\n")
            fl.write(json.dumps({"k": k, "text": dl, **ml}) + "\n")
            stats["T"].append((mt["n_feat"], mt["n_shown"], mt["n_queries"], len(dt)))
            stats["L"].append((ml["n_feat"], ml["n_shown"], ml["n_queries"], len(dl)))
            sem["n_q_reduced"] += int(mt["n_q_draw"] > mt["n_queries"])
            sem["T_shown_query"] += mt["n_queries"] - mt["n_q_unshown"]
            sem["L_unshown_query"] += ml["n_q_unshown"]
    same = sum(1 for x, y in zip(stats["T"], stats["L"]) if x[:3] == y[:3])
    meta["twin_check"] = {"same_nfeat_nshown_nqueries": same, "of": n_docs, "mean_chars_T": float(np.mean([x[3] for x in stats["T"]])), "mean_chars_L": float(np.mean([x[3] for x in stats["L"]])), **sem}
    assert same == n_docs and sem["T_shown_query"] == 0 and sem["L_unshown_query"] == 0, f"twin check failed: {meta['twin_check']}"
    with open(f"{out}/calib_T.jsonl", "w") as fc, open(f"{out}/calib_L.jsonl", "w") as fl:
        for k in range(n_calib):
            d, m = make_doc(random.Random(calib_seed * 10**6 + k), "compose", GEN_VER)
            fc.write(json.dumps({"k": k, "text": d, **m}) + "\n")
            d, m = make_doc(random.Random(calib_seed * 10**6 + k), "attested", GEN_VER)
            fl.write(json.dumps({"k": k, "text": d, **m}) + "\n")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    with open(f"{out}/replay.jsonl", "w") as fr:
        for i, r in enumerate(ds):
            if i >= n_replay:
                break
            fr.write(json.dumps({"i": i, "id": r.get("id"), "text": r["text"]}) + "\n")
    tok = AutoTokenizer.from_pretrained(smollm_tokenizer)
    for which, n_tok in (("fwe", 2_000_000), ("pile", 1_000_000)):
        it = load_dataset("HuggingFaceFW/fineweb-edu", "CC-MAIN-2025-26", split="train", streaming=True) if which == "fwe" else load_dataset("json", data_files="hf://datasets/monology/pile-uncopyrighted/val.jsonl.zst", split="train", streaming=True)
        n = 0
        with open(f"{out}/val_{which}.jsonl", "w") as fv:
            for r in it:
                n += len(tok(r["text"], add_special_tokens=False)["input_ids"]) + 1
                fv.write(json.dumps({"text": r["text"]}) + "\n")
                if n >= n_tok:
                    break
        meta[f"val_{which}_tokens"] = n
    meta["sha256"] = {f: sha256_file(f"{out}/{f}") for f in ["synth_T.jsonl", "synth_L.jsonl", "calib_T.jsonl", "calib_L.jsonl", "replay.jsonl", "val_fwe.jsonl", "val_pile.jsonl"]}
    meta["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(meta, open(f"{out}/data.json", "w"), indent=1)
    return meta


def pack_rows(base: str, data: str, arm: str, out: str, syn_tokens: int = 0, replay_tokens: int = 0) -> dict[str, Any]:
    """Tokenise with the base's tokenizer; rows alternate synthetic / replay until each budget is met (calib: 48 steps' worth + the held-out tail)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    assert tok.eos_token_id is not None, "tokenizer without eos"
    if arm == "calib":
        syn_docs = (r["text"] for pair in zip(read_jsonl(f"{data}/calib_T.jsonl"), read_jsonl(f"{data}/calib_L.jsonl")) for r in pair)
        rep_docs = (r["text"] for i, r in enumerate(read_jsonl(f"{data}/replay.jsonl")) if i < REPLAY_CALIB)
        syn_tokens = syn_tokens or (48 * GLOBAL_BATCH // 2 + CALIB_VAL_ROWS) * SEQ
        rep_tokens = replay_tokens or syn_tokens
    else:
        syn_docs = (r["text"] for r in read_jsonl(f"{data}/synth_{arm}.jsonl"))
        rep_docs = (r["text"] for i, r in enumerate(read_jsonl(f"{data}/replay.jsonl")) if i >= REPLAY_CALIB + REPLAY_VAL)
        syn_tokens = syn_tokens or 20_000_000
        rep_tokens = replay_tokens or 20_000_000
    rows, info = lora_rows(tok, syn_docs, rep_docs, syn_tokens, rep_tokens, GLOBAL_BATCH, trim_to_batch=(arm != "calib"))
    np.save(out, rows)
    info.update({"base": base, "arm": arm, "sha256": sha256_file(out)})
    if arm in ("T", "L"):  # twin audit under THIS tokenizer: token counts of the first 3,000 pairs; tokens per arm equal by construction
        nt = [len(tok(r["text"], add_special_tokens=False)["input_ids"]) for i, r in enumerate(read_jsonl(f"{data}/synth_T.jsonl")) if i < 3000]
        nl = [len(tok(r["text"], add_special_tokens=False)["input_ids"]) for i, r in enumerate(read_jsonl(f"{data}/synth_L.jsonl")) if i < 3000]
        d = np.array(nt) - np.array(nl)
        info["twin_token_audit"] = {"pairs": len(d), "mean_tokens_T": float(np.mean(nt)), "mean_tokens_L": float(np.mean(nl)), "mean_diff": float(d.mean()),
                                    "mean_abs_diff": float(np.abs(d).mean()), "frac_equal": float((d == 0).mean()), "max_abs_diff": int(np.abs(d).max())}
        other = "L" if arm == "T" else "T"
        sib = out.replace(f"_{arm}.npy", f"_{other}.json")
        if os.path.exists(sib):
            o = json.load(open(sib))
            imb = abs(info["syn_docs_used"] - o["syn_docs_used"]) / max(1, o["syn_docs_used"] if arm == "L" else info["syn_docs_used"])
            info["twin_balance"] = {"tokens_equal": info["tokens"] == o["tokens"], "docs_imbalance": imb}
            assert info["tokens"] == o["tokens"] and imb <= 0.01, f"twin balance violated: {info['twin_balance']}"
    json.dump(info, open(out.replace(".npy", "") + ".json", "w"), indent=1)
    return info


# ----------------------------------------------------------------------------------------------------------------------- train
def lora_model(base: str, r: int, alpha: int, seed: int):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").to(device())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = targets_for(model.config)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, bias="none", task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_layers = model.config.num_hidden_layers
    n_lora = sum(1 for _, m in pm.named_modules() if isinstance(getattr(m, "lora_A", None), torch.nn.ModuleDict) and len(m.lora_A))
    assert n_lora == n_layers * len(targets), f"LoRA module count {n_lora} != {n_layers} layers x {len(targets)} targets"
    pm.lora_audit = {"n_lora_modules": n_lora, "n_layers": n_layers, "targets": targets, "model_type": model.config.model_type,
                     "trainable_params": sum(p.numel() for p in pm.parameters() if p.requires_grad)}
    return pm


def row_losses(pm, xb):
    import torch

    logits = pm(input_ids=xb).logits[:, :-1].float()
    return torch.nn.functional.cross_entropy(logits.transpose(1, 2), xb[:, 1:], reduction="none").mean(1)


def mean_loss(pm, rows, micro: int) -> float:
    import torch

    pm.eval()
    tot = 0.0
    with torch.no_grad(), torch.autocast(device(), dtype=torch.bfloat16):
        for b in range(0, len(rows), micro):
            xb = torch.from_numpy(rows[b : b + micro].astype(np.int64)).to(device())
            tot += float(row_losses(pm, xb).sum())
    pm.train()
    return tot / len(rows)


def train_loop(pm, rows, lr: float, steps: int, micro: int, log, val_rows=None, syn_val_rows=None, warmup: int = 30) -> dict[str, Any]:
    import torch

    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, steps - warmup)))))
    acc = GLOBAL_BATCH // micro
    n_clip = 0
    finite = True

    def val_loss():
        return None if val_rows is None else mean_loss(pm, val_rows, micro)

    def syn_val_loss():
        return None if syn_val_rows is None else mean_loss(pm, syn_val_rows, micro)

    v0, sv0 = val_loss(), syn_val_loss()
    log.write(json.dumps({"step": 0, "val": v0, "syn_val": sv0}) + "\n")
    log.flush()
    pm.train()
    t0 = time.time()
    rec: dict[str, Any] = {}
    s = 0
    for s in range(steps):
        tot = 0.0
        syn_l, rep_l = [], []
        for m in range(acc):
            xb = torch.from_numpy(rows[(s * acc + m) * micro : (s * acc + m + 1) * micro].astype(np.int64)).to(device())
            with torch.autocast(device(), dtype=torch.bfloat16):
                rl = row_losses(pm, xb)
                loss = rl.mean() / acc
            loss.backward()
            tot += float(loss)
            r0 = ((s * acc + m) * micro) % 2  # rows alternate synthetic (even index) / replay (odd index)
            syn_l += [float(x) for i, x in enumerate(rl.detach()) if (i + r0) % 2 == 0]
            rep_l += [float(x) for i, x in enumerate(rl.detach()) if (i + r0) % 2 == 1]
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        n_clip += int(gn > 1.0)
        if not math.isfinite(tot) or not math.isfinite(gn):
            finite = False
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec = {"step": s + 1, "lm_loss": tot, "syn_loss": float(np.mean(syn_l)) if syn_l else None, "replay_loss": float(np.mean(rep_l)) if rep_l else None,
               "grad_norm": gn, "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
        if (s + 1) % 16 == 0 or s + 1 == steps:  # adapter L2 norm as the dose record
            rec["adapter_norm"] = math.sqrt(sum(float((p.detach().float() ** 2).sum()) for p in params))
        if s + 1 == steps:
            rec["val"] = val_loss()
            rec["syn_val"] = syn_val_loss()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if (s + 1) % 10 == 0 or s + 1 == steps:
            print(json.dumps(rec), flush=True)
        if not finite:
            break
    return {"finite": finite, "n_clip": n_clip, "steps_done": s + 1, "val0": v0, "val_last": rec.get("val"), "syn_val0": sv0, "syn_val_last": rec.get("syn_val")}


def calib_val_rows(base: str, data: str) -> np.ndarray:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    docs = (r["text"] for i, r in enumerate(read_jsonl(f"{data}/replay.jsonl")) if REPLAY_CALIB <= i < REPLAY_CALIB + REPLAY_VAL)
    return np.stack([x[0] for x in rows_from_text_stream(tok, docs, 64 * SEQ, tok.eos_token_id)])


def calibrate_rate(base: str, data: str, packs: str, out: str, rates: list[float] | None = None, steps: int = 48, micro: int = 4, r: int = 16, alpha: int = 32, seed: int = 0) -> dict[str, Any]:
    import torch

    rates = rates or [5e-5, 1e-4, 2e-4, 5e-4]
    os.makedirs(out, exist_ok=True)
    allrows = np.load(packs, mmap_mode="r")
    val = calib_val_rows(base, data)
    assert len(allrows) >= steps * GLOBAL_BATCH + 2 * CALIB_VAL_ROWS, f"calibration packs hold {len(allrows)} rows < {steps * GLOBAL_BATCH + 2 * CALIB_VAL_ROWS}"
    rows = allrows[: steps * GLOBAL_BATCH]
    syn_val = np.array(allrows[steps * GLOBAL_BATCH :][0::2][:CALIB_VAL_ROWS])
    res = {}
    for lr in rates:
        pm = lora_model(base, r, alpha, seed)
        with open(f"{out}/calib_lr{lr:g}.jsonl", "w") as log:
            rr = train_loop(pm, rows, lr, steps, micro, log, val_rows=val, syn_val_rows=syn_val, warmup=min(30, steps // 2))
        rr["clip_frac"] = rr["n_clip"] / max(1, rr["steps_done"])
        rr["ok"] = bool(rr["finite"] and rr["clip_frac"] <= 0.20 and rr["val_last"] is not None and rr["val_last"] <= rr["val0"] + 0.02)
        res[f"{lr:g}"] = rr
        print("CALIB", lr, json.dumps(rr), flush=True)
        del pm
        torch.cuda.empty_cache()
    ok = [float(k) for k, v in res.items() if v["ok"]]
    best = min(ok, key=lambda lr: (res[f"{lr:g}"]["syn_val_last"], lr)) if ok else 5e-5
    sel = {"selected_lr": best, "flag": None if ok else "no rate qualified; 5e-5 by rule", "safe_rates": ok,
           "rule": "safe = finite & clip_frac<=.20 & val_last<=val0+.02; selected = argmin syn_val_last over safe rates; ties->lower; none->5e-5+flag", "per_lr": res}
    json.dump(sel, open(f"{out}/lr_sel.json", "w"), indent=1)
    return sel


def train(base: str, packs: str, lr: float, seed: int, out: str, r: int = 16, alpha: int = 32, steps: int = 0, micro: int = 4, calib_packs: str | None = None) -> int:
    """Train one arm and merge the adapter; returns the exit code (0 ok, 6 non-finite, 7 error, 8 merge check failed)."""
    import torch
    from transformers import AutoTokenizer

    os.makedirs(out, exist_ok=True)
    rows = np.load(packs, mmap_mode="r")
    steps = steps or len(rows) // GLOBAL_BATCH
    cfg: dict[str, Any] = {"base": base, "packs": packs, "packs_sha256": sha256_file(packs), "lr": lr, "seed": seed, "r": r, "alpha": alpha, "steps": steps, "global_batch": GLOBAL_BATCH, "seq": SEQ,
                           "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base_revision": model_sha(base)}
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
    pm = lora_model(base, r, alpha, seed)
    cfg["lora_audit"] = pm.lora_audit
    cfg["targets"] = pm.lora_audit["targets"]
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
    syn_val = None
    if calib_packs:
        cal = np.load(calib_packs, mmap_mode="r")
        syn_val = np.array(cal[48 * GLOBAL_BATCH :][0::2][:CALIB_VAL_ROWS])
    code = 0
    try:
        with open(f"{out}/train.jsonl", "w") as log:
            rr = train_loop(pm, rows, lr, steps, micro, log, syn_val_rows=syn_val, warmup=30)
        cfg.update(rr)
        json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
        if not rr["finite"]:
            print("non-finite loss; not merging")
            code = 6
        else:
            probe = rows[:micro]
            l_adapter = mean_loss(pm, probe, micro)
            merged = pm.merge_and_unload().to(torch.bfloat16)
            l_merged = mean_loss(merged, probe, micro)
            cfg["merge_check"] = {"loss_adapter": l_adapter, "loss_merged": l_merged, "abs_diff": abs(l_adapter - l_merged)}
            json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
            print("MERGE_CHECK", json.dumps(cfg["merge_check"]), flush=True)
            if cfg["merge_check"]["abs_diff"] > MERGE_TOL:
                print("MERGE_CHECK_FAILED")
                code = 8
            else:
                d = f"{out}/merged"
                merged.save_pretrained(d, safe_serialization=True)
                AutoTokenizer.from_pretrained(base).save_pretrained(d)
                open(f"{d}/.done", "w").write("ok")
                print(f"saved {d}", flush=True)
    except Exception as e:  # noqa: BLE001 - the exit code is the record
        print("TRAIN_ERROR", repr(e)[:500])
        code = 7
    print(f"TRAIN_EXIT={code}", flush=True)
    return code


def eval_loss(model_path: str, data: str, tag: str, out_dir: str) -> dict[str, Any]:
    """Mean token loss on the two held-out text sets re-tokenised with the model's tokenizer; per-row losses kept for a block bootstrap."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device()).eval()
    eos = tok.eos_token_id
    res = {}
    for k in ("fwe", "pile"):
        rows = np.stack([x[0] for x in rows_from_text_stream(tok, (r["text"] for r in read_jsonl(f"{data}/val_{k}.jsonl")), 10**9, eos)])
        per_row: list[float] = []
        with torch.no_grad():
            for b in range(0, len(rows), 8):
                xb = torch.from_numpy(rows[b : b + 8].astype(np.int64)).to(device())
                per_row += row_losses(model, xb).tolist()
        res[k] = {"mean_loss": float(np.mean(per_row)), "n_tokens": int(len(rows) * SEQ), "n_rows": int(len(rows)), "row_losses": [round(x, 5) for x in per_row]}
    os.makedirs(out_dir, exist_ok=True)
    mt = model_path.rstrip("/").replace("/", "_")
    json.dump(res, open(f"{out_dir}/loss{tag}_{mt}.json", "w"))
    return res


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("build-data")
    g.add_argument("--out", required=True)
    g.add_argument("--n-docs", type=int, default=200000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--calib-seed", type=int, default=9)
    g.add_argument("--n-calib", type=int, default=8000)
    g.add_argument("--n-replay", type=int, default=60000)
    p = sub.add_parser("pack")
    p.add_argument("--base", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--arm", required=True, choices=["T", "L", "calib"])
    p.add_argument("--out", required=True)
    p.add_argument("--syn-tokens", type=int, default=0)
    p.add_argument("--replay-tokens", type=int, default=0)
    c = sub.add_parser("calibrate")
    c.add_argument("--base", required=True)
    c.add_argument("--data", required=True)
    c.add_argument("--packs", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--rates", default="5e-5,1e-4,2e-4,5e-4")
    c.add_argument("--steps", type=int, default=48)
    c.add_argument("--micro", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--base", required=True)
    t.add_argument("--packs", required=True)
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=int, default=32)
    t.add_argument("--steps", type=int, default=0)
    t.add_argument("--micro", type=int, default=4)
    t.add_argument("--calib-packs", default=None)
    e = sub.add_parser("eval-loss")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--tag", default="_lora")
    e.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build-data":
        build_data(a.out, a.n_docs, a.seed, a.calib_seed, a.n_calib, a.n_replay)
    elif a.cmd == "pack":
        pack_rows(a.base, a.data, a.arm, a.out, a.syn_tokens, a.replay_tokens)
    elif a.cmd == "calibrate":
        calibrate_rate(a.base, a.data, a.packs, a.out, [float(x) for x in a.rates.split(",")], a.steps, a.micro)
    elif a.cmd == "train":
        return train(a.base, a.packs, a.lr, a.seed, a.out, a.r, a.alpha, a.steps, a.micro, a.calib_packs)
    else:
        eval_loss(a.model, a.data, a.tag, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
