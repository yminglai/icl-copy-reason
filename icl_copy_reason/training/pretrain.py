"""One continued-pretraining trainer for every admixture and fine-tune arm.

Inputs : a TrainSpec (base model + revision, data directory with a corpus manifest, or a row-map manifest, arm, learning rate, schedule,
         steps, save cadence, micro-batch, optional attention-mass regulariser heads) — normally resolved from configs/<experiment>.yaml.
Outputs: <out>/config.json, <out>/train.jsonl (one record per step), <out>/step<k>/ HF checkpoints, <out>/resume_latest.pt (exact resume state).

Two data layouts:
  replacement arms  the corpus pack packs_s<seed>.u16 with every 33rd row replaced by a row of the arm's synthetic / mined / rewritten pack
                    (synth_<file>_s<seed>.u16); the arm name selects the file (SYNTH_FILE); arm A = replay only.
  row-map arms      an explicit per-row source map (manifests.build_row_map): packs_row[j] / synth_row[j]; shapes and sha256 asserted at start.
Recipe of record: micro-batch x accumulation = rows_per_step rows of 2,048 tokens per step; AdamW betas (.9, .95) eps 1e-8 weight decay .01;
linear warm-up then constant lr (or warm-up then cosine to zero at --steps); fp32 master weights, bf16 autocast, clip 1.0; sdpa attention
for the LM forward and eager attention for the 8-prompt probe graph that every regularised arm runs (coefficient 0 in the plain arms).
Also here: head selection for the regulariser (select_heads), its coefficient calibration (calibrate_lambda), held-out loss (eval_loss) and
the engagement measurement (engagement).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from icl_copy_reason.io import sha256_file
from icl_copy_reason.training.pack import SEQ

ROWS_PER_STEP = {"smollm": 768, "pile": 1024, "smollm17": 1024}
SYNTH_FILE = {"B": "compose", "B_control": "attested", "D": "defuse", "D2": "enriched", "D3": "rendered", "E_M1": "eM1", "E_M2": "eM2", "E_src": "esrc"}
REPLACEMENT_ARMS = ("A", "B", "B_control", "C", "C_rand", "D", "D2", "D3", "E_M1", "E_M2", "E_src")
ROW_MAP_ARMS = ("F1", "F1_ctrl", "F2", "F2_ctrl", "calib", "calib_ctrl")
ARM_WORDS = {"A": "plain data (replay only)", "B": "synthetic definition-use documents", "B_control": "every-combination-shown twins", "C": "attention-mass regulariser (selected heads)",
             "C_rand": "attention-mass regulariser (random heads)", "D": "mined real definition-use documents", "D2": "mined documents, enriched", "D3": "mined documents, rendered",
             "E_M1": "rewritten real documents, mixture 1", "E_M2": "rewritten real documents, mixture 2", "E_src": "raw source documents of the rewritten set",
             "F1": "exercises as an admixture", "F1_ctrl": "admixture slots filled with replay", "F2": "exercise fine-tune with replay", "F2_ctrl": "matched replay fine-tune",
             "calib": "rate calibration slice (exercises + replay)", "calib_ctrl": "rate calibration slice (replay only)"}


def device() -> str:
    return os.environ.get("ICR_DEVICE", "cuda")


@dataclass
class TrainSpec:
    arm: str
    seed: int
    base: str
    revision: str | None
    lr: float
    out: str
    data: str | None = None  # directory with manifest_s<seed>.json, packs_s<seed>.u16, synth_*.u16, val_*.u16 (replacement arms)
    manifest: str | None = None  # row-map manifest (row-map arms and variant arms)
    steps: int = 636
    save_every: int = 32
    micro_bs: int = 8
    wd: float = 0.01
    warmup: int = 50
    sched: str = "constant"  # constant | cosine
    probe_graph: bool = True
    heads: str | None = None  # heads json (select_heads output) when probe_graph is on
    lam: float = 0.0
    calib: bool = False
    resume: bool = False
    val_batch: int = 16
    extra_row_map_arms: tuple[str, ...] = field(default_factory=tuple)

    @property
    def row_map(self) -> bool:
        return self.arm in ROW_MAP_ARMS or self.arm in self.extra_row_map_arms


def load_train_model(base: str, revision: str | None):
    """fp32 master weights, sdpa attention, every parameter trainable."""
    import torch

    from icl_copy_reason.readings.lm import LM

    lm = LM(base, device=device(), dtype=torch.float32, revision=revision)
    model = lm.model
    model.config._attn_implementation = "sdpa"
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    return lm, model


def grad_norm(params) -> float:
    import torch

    return float(torch.norm(torch.stack([p.grad.detach().norm() for p in params if p.grad is not None])))


# ---------------------------------------------------------------------------------------------------------------- probe graph
def token_span_cols(offsets, spans) -> list[int]:
    return [t for t in range(len(offsets)) if int(offsets[t][1]) > 0 and any(int(offsets[t][0]) < e and int(offsets[t][1]) > s for s, e in spans)]


def blank_and_cols(tok, items, enc, offs) -> tuple[list[int], list[list[int]]]:
    """The blank is the row before the first token at which the correct and the copied form diverge; cols = tokens of the copied form's
    second-marker spans in the demonstrations."""
    from icl_copy_reason.readings.ablation import line_sets
    from icl_copy_reason.tasks.prompts import four_forms, slots_of

    n_pad = (enc["attention_mask"] == 0).sum(1)
    blank, cols = [], []
    for i, it in enumerate(items):
        spans, _ = line_sets(it)
        cols.append(token_span_cols(offs[i], spans["competitor_marker"]))
        forms = dict(four_forms(it))
        first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
        comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
        ic = enc["input_ids"][i][int(n_pad[i]) :].tolist()
        ik = tok(it.prompt + " " + forms[comp])["input_ids"]
        d = next((j for j in range(min(len(ic), len(ik))) if ic[j] != ik[j]), min(len(ic), len(ik)) - 1)
        blank.append(int(n_pad[i]) + d - 1)
    return blank, cols


def probe_batch(tok, dev: str, seed: int, n: int = 8, template: str = "alt"):
    import torch  # noqa: F401 - tensors come from the tokenizer

    from icl_copy_reason.tasks.prompts import build, set_template

    set_template(template)
    items = build("compose_h", n, seed=seed)
    texts = [it.prompt + " " + it.answer for it in items]
    enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    blank, cols = blank_and_cols(tok, items, enc, offs)
    set_template("default")
    return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), blank, cols


def attention_mass(model, ids, am, blank, cols, heads):
    """Mean over heads and items of the attention probability from the blank row to the copied form's marker columns (eager attention)."""
    import torch

    model.config._attn_implementation = "eager"
    out = model(input_ids=ids, attention_mask=am, output_attentions=True)
    model.config._attn_implementation = "sdpa"
    rows = torch.arange(len(blank), device=ids.device)
    b = torch.tensor(blank, device=ids.device)
    masses = []
    for layer, h in heads:
        att = out.attentions[layer]
        if att is None:
            raise RuntimeError("attention weights not returned")
        row = att[rows, h, b]
        masses.append(torch.stack([row[i, cols[i]].sum() if cols[i] else row[i, :1].sum() * 0 for i in range(len(blank))]))
    return torch.stack(masses).mean()


def select_heads(dla_path: str, out: str, k: int = 10) -> dict[str, Any]:
    """Top-k positive heads (attribution towards the copied form) of the base checkpoint plus a layer-matched random set."""
    dla = json.load(open(dla_path))
    m = np.array(dla["head_contrib"]).mean(0)
    order = np.argsort(-m, kind="stable")
    heads = [tuple(h) for h in dla["heads"]]
    n_heads = dla["n_heads"]
    sel = [heads[i] for i in order[:k] if m[i] > 0]
    if len(sel) < k:
        raise SystemExit(f"fewer than {k} positive heads ({len(sel)}): base ineligible")
    rng = np.random.default_rng(2000 + 3)
    taken = set(sel)
    rand = []
    for layer, _ in sel:
        c = [hh for hh in range(n_heads) if (layer, hh) not in taken]
        pick = (layer, int(rng.choice(c)))
        rand.append(pick)
        taken.add(pick)
    rec = {"selected": [list(x) for x in sel], "random": [list(x) for x in rand], "dla": dla_path, "contrib": [float(m[i]) for i in order[:k]]}
    json.dump(rec, open(out, "w"), indent=1)
    return rec


# ------------------------------------------------------------------------------------------------------------------- training
def make_scheduler(opt, sched: str, warmup: int, steps: int):
    import torch

    if sched == "cosine":  # warm-up over `warmup` steps, then cosine from the peak to ZERO at `steps`
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s + 1 <= warmup else 0.0 if s + 1 >= steps else 0.5 * (1.0 + math.cos(math.pi * (s + 1 - warmup) / max(1, steps - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))


def _resolve(path: str, root: str | None) -> str:
    return path if os.path.isabs(path) or root is None else os.path.join(root, path)


def train(spec: TrainSpec, root: str | None = None) -> dict[str, Any]:
    """Run one arm; returns the final config record. `root` resolves relative file names inside a row-map manifest."""
    import torch

    dev = device()
    row_map = spec.row_map
    synth = None
    repl_pos: dict[int, int] = {}
    packs_row = synth_row = None
    if row_map:
        assert spec.manifest, "row-map arms need a manifest"
        man = json.load(open(spec.manifest))
        rps = man["rows_per_step"]
        assert man["arm"] == spec.arm and man["seed"] == spec.seed, "manifest belongs to another arm/seed"
        packs = np.load(_resolve(man["packs"]["file"], root), mmap_mode="r")
        assert packs.shape == (man["packs"]["rows"], SEQ), f"packs shape {packs.shape} != manifest"
        assert sha256_file(_resolve(man["packs"]["file"], root)) == man["packs"]["sha256"], "packs sha256 != manifest"
        if man["synth"]:
            assert man["synth"]["sha256"], "manifest synth sha256 not filled (manifests.fill_synth_sha)"
            synth = np.load(_resolve(man["synth"]["file"], root), mmap_mode="r")
            assert synth.shape == (man["synth"]["rows_expected"], SEQ), f"synth shape {synth.shape} != manifest"
            assert sha256_file(_resolve(man["synth"]["file"], root)) == man["synth"]["sha256"], "synth sha256 != manifest"
        packs_row = np.asarray(man["packs_row"], dtype=np.int64)
        synth_row = np.asarray(man["synth_row"], dtype=np.int64)
        assert len(packs_row) == len(synth_row) == man["n_rows"] == man["steps"] * rps and ((packs_row >= 0) ^ (synth_row >= 0)).all(), "row map malformed"
        assert spec.steps <= man["steps"], f"steps {spec.steps} beyond the manifest's {man['steps']}"
        if (synth_row >= 0).any():
            assert synth is not None and synth_row.max() < synth.shape[0], "row map references exercise rows beyond the file"
        assert packs_row.max() < packs.shape[0]
        val_path = {k: _resolve(man["val"][k]["file"], root) for k in ("fwe", "pile")}
        for k in val_path:
            assert sha256_file(val_path[k]) == man["val"][k]["sha256"], f"val_{k} sha256 != manifest"
    else:
        assert spec.data, "replacement arms need a data directory"
        man = json.load(open(f"{spec.data}/manifest_s{spec.seed}.json"))
        rps = man["rows_per_step"]
        packs = np.load(f"{spec.data}/packs_s{spec.seed}.u16", mmap_mode="r")
        val_path = {k: f"{spec.data}/val_{k}.u16" for k in ("fwe", "pile")}
        if spec.arm in SYNTH_FILE:
            synth = np.load(f"{spec.data}/synth_{SYNTH_FILE[spec.arm]}_s{spec.seed}.u16", mmap_mode="r")
            assert synth.shape[0] == len(man["replacement_rows"]), f"synthetic file has {synth.shape[0]} rows, manifest {len(man['replacement_rows'])}"
            repl_pos = {r: k for k, r in enumerate(man["replacement_rows"])}
    if spec.probe_graph:
        heads = json.load(open(spec.heads))
        hset = [tuple(x) for x in (heads["random"] if spec.arm == "C_rand" else heads["selected"])]
    else:
        assert spec.arm not in ("C", "C_rand"), "head arms need the probe graph"
        hset = []
    lam = float(spec.lam) if spec.arm in ("C", "C_rand") else 0.0
    torch.manual_seed(spec.seed)
    random.seed(spec.seed)
    lm, model = load_train_model(spec.base, spec.revision)
    tok = lm.tok
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=spec.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=spec.wd, fused=(dev == "cuda"))
    sched = make_scheduler(opt, spec.sched, spec.warmup, spec.steps)
    os.makedirs(spec.out, exist_ok=True)
    cfg: dict[str, Any] = {**asdict(spec), "lambda": lam, "heads_used": hset, "rows_per_step": rps, "manifest_sha": man["sha256"], "synth_file": SYNTH_FILE.get(spec.arm),
                           "arm_words": ARM_WORDS.get(spec.arm, spec.arm), "resumes": 0, "resume_events": []}
    if row_map:
        cfg.update({"manifest_arm": man["arm"], "synth_file": man["synth"] and man["synth"]["file"], "packs_file": man["packs"]["file"], "n_rows": man["n_rows"]})
    # exact resume: a rolling state (weights, AdamW moments, schedule, step, RNG) is written at every checkpoint step; the data position is a
    # function of the step, so a resumed run sees the same rows in the same order under the same schedule as an uninterrupted one.
    resume_path = f"{spec.out}/resume_latest.pt"
    start_step = 0
    n_clip = 0
    resave_at = None
    if not spec.calib and os.path.exists(f"{spec.out}/config.json"):
        old = json.load(open(f"{spec.out}/config.json"))
        if old.get("manifest_sha") == cfg["manifest_sha"] and old.get("arm") == spec.arm and old.get("seed") == spec.seed:
            cfg["resumes"] = old.get("resumes", 0)
            cfg["resume_events"] = old.get("resume_events", [])
    if not spec.calib and spec.resume and os.path.exists(resume_path):
        st = torch.load(resume_path, map_location=dev, weights_only=False)
        assert st["arm"] == spec.arm and st["seed"] == spec.seed and st["manifest_sha"] == man["sha256"], "resume state belongs to a different run"
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"]["torch"].cpu())
        random.setstate(st["rng"]["python"])
        np.random.set_state(st["rng"]["numpy"])
        if dev == "cuda" and st["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(st["rng"]["cuda"].cpu())
        start_step = int(st["step"])
        n_clip = int(st["n_clip"])
        cfg["resumes"] += 1
        cfg["resume_events"].append({"step": start_step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if os.path.exists(f"{spec.out}/train.jsonl"):
            keep = [ln for ln in open(f"{spec.out}/train.jsonl") if ln.strip() and json.loads(ln).get("step", 0) <= start_step]
            open(f"{spec.out}/train.jsonl", "w").writelines(keep)
        print(f"RESUMED from step {start_step} ({cfg['resumes']} resume(s) so far)", flush=True)
        del st
        resave_at = start_step if not os.path.exists(f"{spec.out}/step{start_step}/.done") else None
    json.dump(cfg, open(f"{spec.out}/config.json", "w"), indent=1)
    log = open(f"{spec.out}/train.jsonl", "a")
    val = {k: np.load(val_path[k], mmap_mode="r") for k in ("fwe", "pile")}

    def save(step: int) -> None:
        d = f"{spec.out}/step{step}"
        model.save_pretrained(d, safe_serialization=True)
        tok.save_pretrained(d)
        open(f"{d}/.done", "w").write("ok")
        print(f"saved {d}", flush=True)

    def save_resume(step: int, n_clip: int) -> None:
        if spec.calib or not spec.resume:
            return
        st = {"arm": spec.arm, "seed": spec.seed, "manifest_sha": man["sha256"], "step": step, "n_clip": n_clip, "model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "rng": {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state(),
                                                   "cuda": torch.cuda.get_rng_state() if dev == "cuda" else None}}
        torch.save(st, resume_path + ".tmp")
        os.replace(resume_path + ".tmp", resume_path)
        del st
        json.dump({"step": step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(f"{spec.out}/resume_latest.json", "w"))

    def val_loss(arr, n_seq: int = 256) -> float:
        vb = spec.val_batch  # equal-length rows, no padding: the token mean is the same for any micro-batch up to float rounding
        n_seq = min(n_seq, len(arr) // SEQ)
        model.eval()
        tot = 0.0
        cnt = 0
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            x = torch.from_numpy(np.array(arr[: n_seq * SEQ], dtype=np.int64)).view(n_seq, SEQ).to(dev)
            for b in range(0, n_seq, vb):
                xb = x[b : b + vb]
                out = model(input_ids=xb, labels=xb)
                tot += float(out.loss) * xb.shape[0]
                cnt += xb.shape[0]
        model.train()
        return tot / cnt

    row0 = (man["steps"] * rps if spec.calib else 0) if not row_map else 0  # the calibration slice of a corpus manifest; row-map calibration arms are their own index space
    steps = spec.steps
    if not spec.calib and start_step == 0:
        save(0)
    if resave_at is not None:
        save(resave_at)
    v0 = {k: val_loss(v) for k, v in val.items()} if spec.calib else None
    if spec.calib:
        log.write(json.dumps({"step": 0, "val": v0}) + "\n")
        log.flush()
    accum = rps // spec.micro_bs
    t0 = time.time()
    rec: dict[str, Any] = {}
    for step in range(start_step + 1, steps + 1):
        base = row0 + (step - 1) * rps
        lm_loss_acc = 0.0
        typ_sum = {"synth": 0.0, "packs": 0.0}
        typ_n = {"synth": 0, "packs": 0}
        for mb in range(accum):
            rows = np.arange(base + mb * spec.micro_bs, base + (mb + 1) * spec.micro_bs)
            if row_map:
                x = np.stack([np.array(packs[packs_row[r]] if packs_row[r] >= 0 else synth[synth_row[r]], dtype=np.int64) for r in rows])
            else:
                x = np.array(packs[rows[0] : rows[-1] + 1], dtype=np.int64)
                if synth is not None:
                    for j, r in enumerate(rows):
                        if int(r) in repl_pos:
                            x[j] = synth[repl_pos[int(r)]]
            x = torch.from_numpy(x).to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                if row_map and spec.calib:  # per-row token-mean CE split by row source (the calibration sanity check)
                    logits = model(input_ids=x).logits[:, :-1].float()
                    per_row = torch.nn.functional.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none").mean(1)
                    loss = per_row.mean() / accum
                    is_syn = torch.tensor([packs_row[r] < 0 for r in rows], device=per_row.device)
                    for name, msk in (("synth", is_syn), ("packs", ~is_syn)):
                        typ_sum[name] += float(per_row.detach()[msk].sum())
                        typ_n[name] += int(msk.sum())
                    del logits
                else:
                    loss = model(input_ids=x, labels=x).loss / accum
            loss.backward()
            lm_loss_acc += float(loss)
        g_lm = grad_norm(params) if (step == 1 or step % spec.save_every == 0) else None
        if spec.probe_graph:  # matched auxiliary graph (every regularised-recipe arm): eager probe forward, penalty = lambda * mass (lambda 0 in the plain arms)
            ids, am, blank, cols = probe_batch(tok, dev, seed=500000 + spec.seed * 10000 + step)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                mass = attention_mass(model, ids, am, blank, cols, hset)
            (lam * mass).backward()
        else:
            mass = torch.zeros(())
        total = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        clipped = total > 1.0
        n_clip += int(clipped)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step, "lm_loss": lm_loss_acc, "mass": float(mass.detach()), "penalty": lam * float(mass.detach()), "grad_norm": total, "clipped": clipped,
               "lr": sched.get_last_lr()[0], "tok_s": (step - start_step) * rps * SEQ / (time.time() - t0)}
        if g_lm is not None:
            rec["grad_norm_lm"] = g_lm
        if not math.isfinite(lm_loss_acc):
            rec["nonfinite"] = True
            log.write(json.dumps(rec) + "\n")
            log.flush()
            raise SystemExit("non-finite loss")
        if row_map and spec.calib:
            rec["loss_synth"] = typ_sum["synth"] / typ_n["synth"] if typ_n["synth"] else None
            rec["loss_packs"] = typ_sum["packs"] / typ_n["packs"] if typ_n["packs"] else None
        if spec.calib and step == steps:
            rec["val"] = {k: val_loss(v) for k, v in val.items()}
            rec["val0"] = v0
            rec["clip_fraction"] = n_clip / steps
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 8 == 0 or step == 1:
            print(json.dumps(rec), flush=True)
        if not spec.calib and (step % spec.save_every == 0 or step == steps):
            save(step)
            save_resume(step, n_clip)
    print(f"done {steps} steps in {(time.time() - t0) / 60:.1f} min ({start_step} resumed), clip fraction {n_clip / max(1, steps):.3f}", flush=True)
    cfg["final"] = {"steps_done": steps, "clip_fraction": n_clip / max(1, steps), "last": rec}
    json.dump(cfg, open(f"{spec.out}/config.json", "w"), indent=1)
    return cfg


def calibrate_lambda(base: str, revision: str | None, data: str, seed: int, heads_path: str, which: str, out: str, micro_bs: int = 8) -> dict[str, Any]:
    """lambda = 0.10 * median ||grad LM|| / median ||grad mass|| over 8 pairs of steps on the calibration slice; must land in [1e-3, 10]."""
    import torch

    dev = device()
    man = json.load(open(f"{data}/manifest_s{seed}.json"))
    rps = man["rows_per_step"]
    packs = np.load(f"{data}/packs_s{seed}.u16", mmap_mode="r")
    heads = json.load(open(heads_path))
    hset = [tuple(x) for x in (heads["random"] if which == "rand" else heads["selected"])]
    lm, model = load_train_model(base, revision)
    tok = lm.tok
    params = [p for p in model.parameters() if p.requires_grad]
    g_lm, g_att = [], []
    row0 = man["steps"] * rps
    accum = rps // micro_bs
    for k in range(8):
        base_row = row0 + k * rps
        for mb in range(accum):
            x = torch.from_numpy(np.array(packs[base_row + mb * micro_bs : base_row + (mb + 1) * micro_bs], dtype=np.int64)).to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                (model(input_ids=x, labels=x).loss / accum).backward()
        g_lm.append(grad_norm(params))
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        ids, am, blank, cols = probe_batch(tok, dev, seed=500000 + seed * 10000 + 1 + k)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            mass = attention_mass(model, ids, am, blank, cols, hset)
        mass.backward()
        g_att.append(grad_norm(params))
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        print(f"pair {k}: g_lm {g_lm[-1]:.4f} g_att {g_att[-1]:.6f} mass {float(mass):.4f}", flush=True)
    med_lm, med_att = float(np.median(g_lm)), float(np.median(g_att))
    lam = 0.10 * med_lm / med_att if med_att > 0 else None
    ok = lam is not None and 1e-3 <= lam <= 10
    res = {"which": which, "heads": hset, "g_lm": g_lm, "g_att": g_att, "median_g_lm": med_lm, "median_g_att": med_att, "lambda": lam, "in_range": ok}
    json.dump(res, open(out, "w"), indent=1)
    if not ok:
        raise SystemExit("lambda calibration failed: do not run the regularised arm")
    return res


def eval_loss(model_path: str, data: str, tag: str, out_dir: str, batch: int = 16) -> dict[str, Any]:
    """Mean token loss on the two held-out token files -> <out_dir>/loss<tag>_<model>.json."""
    import torch

    from icl_copy_reason.readings.lm import LM

    dev = device()
    lm = LM(model_path, device=dev, dtype=torch.bfloat16)
    model = lm.model.eval()
    res = {}
    for k in ("fwe", "pile"):
        arr = np.load(f"{data}/val_{k}.u16", mmap_mode="r")
        n_seq = len(arr) // SEQ
        x = torch.from_numpy(np.array(arr[: n_seq * SEQ], dtype=np.int64)).view(n_seq, SEQ)
        tot = 0.0
        with torch.no_grad():
            for b in range(0, n_seq, batch):
                xb = x[b : b + batch].to(dev)
                tot += float(model(input_ids=xb, labels=xb).loss) * xb.shape[0]
        res[k] = {"mean_loss": tot / n_seq, "n_tokens": int(n_seq * SEQ)}
    os.makedirs(out_dir, exist_ok=True)
    mt = model_path.rstrip("/").replace("/", "_")
    json.dump(res, open(f"{out_dir}/loss{tag}_{mt}.json", "w"))
    return res


def engagement(model_path: str, heads_path: str, tag: str, out_dir: str, n: int = 160, seed: int = 2) -> dict[str, Any]:
    """Selected / random heads' attention mass from the blank to the copied form's markers on the standard-template panel (item seed + 11)."""
    import torch

    from icl_copy_reason.readings.lm import LM
    from icl_copy_reason.tasks.prompts import build

    dev = device()
    lm = LM(model_path, device=dev, dtype=torch.bfloat16)
    model = lm.model.eval()
    tok = lm.tok
    heads = json.load(open(heads_path))
    res = {}
    items = build("compose_h", n, seed=seed + 11)
    for name in ("selected", "random"):
        hset = [tuple(x) for x in heads[name]]
        masses = []
        for b in range(0, n, 8):
            bp = items[b : b + 8]
            texts = [it.prompt + " " + it.answer for it in bp]
            enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            blank, cols = blank_and_cols(tok, bp, enc, offs)
            with torch.no_grad():
                model.config._attn_implementation = "eager"
                out = model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev), output_attentions=True)
                bt = torch.tensor(blank, device=dev)
                for i in range(len(bp)):
                    masses.append(float(np.mean([float(out.attentions[layer][i, h, bt[i], cols[i]].sum()) if cols[i] else 0.0 for (layer, h) in hset])))
        res[name] = {"mean_mass": float(np.mean(masses)), "items": masses}
    os.makedirs(out_dir, exist_ok=True)
    mt = model_path.rstrip("/").replace("/", "_")
    json.dump(res, open(f"{out_dir}/engage{tag}_{mt}.json", "w"))
    return res


def pick_rate_corpus(run_dir: str, rates: list[str], tol: float = 0.02, min_steps: int = 64) -> dict[str, Any]:
    """Rate rule for the admixture runs: the largest rate whose calibration run (arm A, 64 steps) has finite loss throughout, clip fraction <= .20
    and held-out FineWeb-Edu loss at the end <= start + tol; ties -> lower."""
    rows = {}
    for r in rates:
        f = f"{run_dir}/lrcalib_{r}/train.jsonl"
        if not os.path.exists(f):
            rows[r] = {"missing": True}
            continue
        recs = [json.loads(ln) for ln in open(f)]
        last = recs[-1]
        steps = [x for x in recs if "lm_loss" in x]
        finite = all(np.isfinite(x["lm_loss"]) for x in steps) and not any(x.get("nonfinite") for x in steps) and len(steps) >= min_steps
        clipf = last.get("clip_fraction", sum(int(x["clipped"]) for x in steps) / max(1, len(steps)))
        v0 = (recs[0].get("val") or {}).get("fwe")
        v1 = (last.get("val") or {}).get("fwe")
        ok = finite and clipf <= 0.20 and v0 is not None and v1 is not None and v1 <= v0 + tol
        rows[r] = {"finite": finite, "clip_fraction": clipf, "val_fwe_start": v0, "val_fwe_end": v1, "ok": ok}
    ok_rates = [r for r in rates if rows[r].get("ok")]
    return {"rates": rows, "selected_lr": max(ok_rates, key=float) if ok_rates else None, "rule": f"largest rate: finite, clip <= .20, FWE end <= start + {tol}; ties -> lower"}
