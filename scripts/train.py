#!/usr/bin/env python3
"""Train one arm of a training experiment, or run one of its preparation steps, from configs/<experiment>.yaml.

usage:
  python scripts/train.py --config synthetic-docs-360m --arm B --seed 0                 train (continued pretraining)
  python scripts/train.py --config synthetic-docs-360m --mode corpus-manifest --seed 0  pack the replay corpus + synthetic packs + validation files
  python scripts/train.py --config synthetic-docs-360m --mode calibrate --lr 1e-3        64-step rate calibration run (arm A, calibration slice)
  python scripts/train.py --config synthetic-docs-360m --mode select-heads --dla F       heads for the regularised arm
  python scripts/train.py --config synthetic-docs-360m --mode calibrate-lambda --which sel
  python scripts/train.py --config exercises-finetune --mode row-maps                    write the per-row source maps of every index space
  python scripts/train.py --config exercises-finetune --arm F2 --seed 0                  train a row-map arm (manifest from the config)
  python scripts/train.py --config lora-transfer --base qwen --arm T --seed 0            LoRA transfer (pack + train; calibration via --mode calibrate)
  python scripts/train.py --config heads-induce-train --arm B --seed 0 --select-seed 3   guided direct fine-tune
  python scripts/train.py --config <any> --dry-run                                       print the resolved plan without loading anything
Paths: ICR_DATA_ROOT (released data), ICR_RESULTS_ROOT (new runs; default <data>/../runs), ICR_HF_HOME (weights), ICR_DEVICE (cuda | cpu).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl_copy_reason.config import Config, add_config_args, config_from_args, data_root, results_root  # noqa: E402


def resolve_plan(cfg: Config, args: argparse.Namespace) -> dict:
    """Everything a run needs, resolved from the config + CLI overrides (no model, no data loaded)."""
    trainer = cfg.get("trainer")
    model = cfg.get("model", {})
    tr = cfg.get("training", {})
    out_root = results_root() / cfg.get("results_dir", cfg.name)
    if cfg.get("runs_subdir"):
        out_root = out_root / str(cfg.get("runs_subdir"))
    plan = {"experiment": cfg.name, "trainer": trainer, "mode": args.mode, "arm": args.arm, "seed": args.seed, "data_root": str(data_root()), "runs_root": str(out_root)}
    if trainer == "pretrain":
        arms = cfg.get("arms", {})
        arm_cfg = arms.get(args.arm or "", {})
        base = model.get("base")
        rev = model.get("revision")
        if args.block:  # replication blocks override base / seed / data directory
            blk = cfg.get("blocks", {})[args.block]
            base, rev = blk.get("base", base), blk.get("revision", rev)
            plan["block"] = blk
        lr = args.lr if args.lr is not None else arm_cfg.get("lr", tr.get("lr_selected"))
        micro_bs = tr.get("micro_bs")
        val_batch = tr.get("val_batch", 16)
        run_root = out_root
        if args.block:  # a block's own rate, micro-batch, validation batch and data directory
            blk = cfg.get("blocks", {})[args.block]
            picked = (blk.get("calibration") or {}).get("picked") if isinstance(blk.get("calibration"), dict) else None
            blk_lr = blk.get("lr") if isinstance(blk.get("lr"), dict) else picked
            if args.lr is None and isinstance(blk_lr, dict) and args.arm in blk_lr:
                lr = blk_lr[args.arm]
            micro_bs = blk.get("micro_bs", micro_bs)
            val_batch = blk.get("val_batch", val_batch)
            run_root = out_root / args.block
        plan.update({"base": base, "revision": rev, "arm_words": arm_cfg.get("words"), "lr": lr, "steps": arm_cfg.get("steps", tr.get("steps")),
                     "save_every": arm_cfg.get("save_every", tr.get("save_every")), "micro_bs": micro_bs, "val_batch": val_batch, "wd": tr.get("wd", 0.01),
                     "warmup": arm_cfg.get("warmup", tr.get("warmup", 50)), "sched": arm_cfg.get("sched", tr.get("sched", "constant")),
                     "probe_graph": bool(tr.get("probe_graph", False)), "rows_per_step": arm_cfg.get("rows_per_step", cfg.get("data.rows_per_step")),
                     "data_dir": str(run_root / "data"), "manifest": (str(run_root / "data" / arm_cfg["manifest"].format(seed=args.seed)) if arm_cfg.get("manifest") else None),
                     "out": str(run_root / f"{args.arm}_s{args.seed}") if args.arm else None, "lambda": arm_cfg.get("lambda", 0.0),
                     "eval": cfg.get("evaluation", {})})
    elif trainer == "lora":
        bases = cfg.get("bases", {})
        b = bases.get(args.base or "", {})
        plan.update({"base_key": args.base, "base": b.get("repo"), "base_note": b.get("note"), "arm": args.arm, "lr": args.lr if args.lr is not None else b.get("lr_selected", tr.get("lr_default")),
                     "r": tr.get("r"), "alpha": tr.get("alpha"), "micro_bs": tr.get("micro_bs"), "steps": tr.get("steps"), "rates": tr.get("lr_candidates"),
                     "data_dir": str(out_root / "data"), "out": str(out_root / (args.base or "base") / f"{args.arm}_s{args.seed}") if args.arm else None})
    elif trainer == "direct_finetune":
        arms = cfg.get("arms", {})
        arm_cfg = arms.get(args.arm or "", {})
        plan.update({"base": model.get("base"), "revision": model.get("revision"), "arm_words": arm_cfg.get("words"), "steps": arm_cfg.get("steps", tr.get("steps")),
                     "bs": tr.get("bs"), "lr": tr.get("lr"), "warmup": tr.get("warmup"), "sched_steps": arm_cfg.get("sched_steps", tr.get("sched_steps")),
                     "save_steps": arm_cfg.get("save_steps"), "eval_every": arm_cfg.get("eval_every"), "select_seed": args.select_seed, "accum": args.accum,
                     "out": str(out_root)})
    elif trainer == "evaluation":
        plan.update({"tasks": cfg.get("tasks"), "checkpoints": cfg.get("checkpoints"), "launcher": cfg.get("launcher")})
    return plan


def morph_dir(cfg: Config) -> Path:
    """<ICR_RESULTS_ROOT>/<experiment>/morph — where the per-checkpoint readings, panels and losses of a run live (what the report builders read)."""
    d = results_root() / cfg.get("results_dir", cfg.name) / "morph"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tokenizer(base: str, revision: str | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base, revision=revision)


def _exercise_records(cfg: Config, types: list[str], evalh=None) -> tuple[list[dict], dict, str]:
    """Verified exercise records of the shipped pool (pass == true, requested types, de-duplicated), with the packed-text format."""
    from icl_copy_reason.data_construction import exercises as X
    from icl_copy_reason.data_construction import verification as V

    fmt = X.load_prompts().get("packed_text_format", "{context}\n{query}\nAnswer: {answer}")
    pool = cfg.get("data.exercise_pool", "data/exercises/exercises.jsonl.gz")
    pool_path = Path(pool) if os.path.isabs(pool) else data_root().parent / pool
    recs, info = V.select_passing([pool_path], types, fmt, evalh)
    return recs, info, fmt


def _source_texts(corpus_dir: str) -> dict[str, str]:
    """sha256 -> source document text when docs.jsonl (not shipped; see data/README.md) sits next to the corpus; else empty (no residual fill)."""
    from icl_copy_reason.io import read_jsonl

    p = Path(corpus_dir) / "docs.jsonl"
    if not p.exists():
        return {}
    return {d.get("sha256") or d.get("doc_sha256"): d["text"] for d in read_jsonl(p)}


def run_pack(cfg: Config, args: argparse.Namespace, plan: dict) -> int:
    """Pack exercise rows (or rewritten documents) into the synthetic-row file an index space expects (synth_<stem>_s<seed>.u16 + layout + summary)."""
    import numpy as np

    from icl_copy_reason.data_construction import verification as V
    from icl_copy_reason.training import manifests as M
    from icl_copy_reason.training import pack as PK

    data_dir = args.out or plan["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    corpus_dir = args.corpus_dir or str(results_root() / cfg.get("data.corpus_experiment", cfg.name) / "data")
    tok = _tokenizer(plan["base"], plan["revision"])
    eos = PK.eos_id(tok)
    if args.mixture:  # rewritten real documents: rows of rewritten text replace corpus rows (replacement_rows of the scale)
        from icl_copy_reason.data_construction import exercises as X
        from icl_copy_reason.io import read_jsonl

        types = cfg.get("data.mixtures", {})[args.mixture]
        src = Path(cfg.get("data.rewritten_records") or results_root() / "rewritten-docs-data" / "verified.jsonl")
        fmt = X.load_prompts(data_root() / "exercises" / "rewrite_prompts.json").get("packed_text_format", "{context}\n{query}\nAnswer: {answer}")
        docs = [V.packed_text(r["exercise"], fmt) for r in read_jsonl(src) if r.get("pass") and r.get("type") in types]
        n_rows = int(cfg.get("data.replacement_rows"))
        rows = [r for r, _ in PK.rows_from_text_stream(tok, docs, n_rows * PK.SEQ, eos)][:n_rows]
        arr = np.stack(rows).astype(np.uint16)
        out = Path(corpus_dir) / f"synth_e{args.mixture}_s{args.seed}.u16"
        with open(out, "wb") as fh:
            np.save(fh, arr)
        print(json.dumps({"out": str(out), "rows": int(arr.shape[0]), "documents": len(docs), "types": types}))
        return 0
    if args.variant:
        v = cfg.get("variants", {})[args.variant]
        space = {"kind": v.get("kind", "admixture"), "rps": v["rps"], "steps": v["steps"], "residues": v["residues"], "mod": v["mod"]}
        types = v.get("types") or cfg.get("data.mixtures", {}).get(v.get("mixture", "F1"), [])
        synth = v["synth"]
    else:
        space = cfg.get("index_spaces", {})[args.arm]
        types = cfg.get("data.mixtures", {})[args.arm]
        synth = space["synth"]
    pr, sr, meta, _ = M.rule(space["kind"], space)
    positions = [int(j) for j in np.flatnonzero(sr >= 0)[np.argsort(sr[sr >= 0])]]  # slot rows ordered by their exercise-row index
    assert len(positions) == meta["n_synth_rows_expected"]
    windows_dir = cfg.get("data.windows_dir")
    evalh = V.load_eval_windows(windows_dir, required=False)[0] if windows_dir else None
    recs, info, fmt = _exercise_records(cfg, types, evalh)
    packs = np.load(f"{corpus_dir}/packs_s0.u16", mmap_mode="r")
    pk = cfg.get("data.packing", {})
    arr, layout, summary = PK.build_exercise_pack(
        tok, recs, _source_texts(corpus_dir), positions, render=lambda ex: V.packed_text(ex, fmt), once_only_types=tuple(pk.get("once_only_types", ["a"])),
        max_exposure=float(pk.get("max_mean_exposure", 3.0)), pack_seed=int(pk.get("pack_seed", 10**6 + 13)), interleave_seed=int(pk.get("interleave_seed", 13)),
        seed_offset=int(args.seed), revert_row=lambda pos: packs[pos], residual_cap=int(pk.get("residual_cap", PK.RESIDUAL_CAP)))
    out = Path(data_dir) / synth
    with open(out, "wb") as fh:
        np.save(fh, arr.astype(np.uint16))
    stem = out.name.replace("synth_", "").split(".")[0]
    json.dump(layout, open(Path(data_dir) / f"pack_layout_{stem}.json", "w"), separators=(",", ":"))
    json.dump({**summary, "selection": info, "types": types, "rows": int(arr.shape[0])}, open(Path(data_dir) / f"pack_summary_{stem}.json", "w"), indent=1)
    print(json.dumps({"out": str(out), "rows": int(arr.shape[0]), "exercises": len(recs), "reverted": summary["rows_reverted_to_replay"]}))
    return 0


def run_overlap_check(cfg: Config, args: argparse.Namespace, plan: dict) -> int:
    """Remove manifest rows whose 50-character windows hit the evaluation-window store (check 4 at manifest time)."""
    import numpy as np

    from icl_copy_reason.data_construction import verification as V
    from icl_copy_reason.training import manifests as M

    windows_dir = cfg.get("data.windows_dir") or str(results_root() / "exercises-data")
    store, _ = V.load_eval_windows(windows_dir)
    tok = _tokenizer(plan["base"], plan["revision"])
    corpus_dir = args.corpus_dir or str(results_root() / cfg.get("data.corpus_experiment", cfg.name) / "data")
    packs = np.load(f"{corpus_dir}/packs_s0.u16", mmap_mode="r")
    for mp in (args.manifests or "").split(","):
        if not mp:
            continue
        man = json.load(open(mp))
        synth = np.load(os.path.join(os.path.dirname(mp), os.path.basename(man["synth"]["file"])), mmap_mode="r") if man.get("synth") else None
        print(mp, json.dumps(M.overlap_check(mp, store, tok, packs, synth)))
    return 0




def run_freeze(cfg: Config, args: argparse.Namespace) -> int:
    """Write the record fixed before a run (revisions, hashes, rules) to <ICR_RESULTS_ROOT>/fixed_records/<experiment>[_<block>].json."""
    from icl_copy_reason.training import freeze as F

    out = results_root() / "fixed_records" / f"{cfg.name}{'_' + args.block if args.block else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    F.write_freeze(cfg, str(out), block=args.block)
    print("wrote", out)
    return 0


def run_pretrain(cfg: Config, args: argparse.Namespace, plan: dict) -> int:
    from icl_copy_reason.training import manifests as M
    from icl_copy_reason.training import pretrain as P

    data_dir = plan["data_dir"]
    if args.mode == "corpus-manifest":
        d = cfg.get("data", {})
        M.build_corpus_manifest(data_dir, args.seed, plan["base"], plan["revision"], d.get("corpus", "smollm"), d.get("steps", plan["steps"]), d.get("calib_steps", 64),
                                d.get("doc_ver", 1), d.get("token_dtype", "uint16"), tuple(d.get("synthetic_modes", ["compose", "attested"])))
        return 0
    if args.mode == "row-maps":
        spaces = M.index_spaces_from_config(cfg.get("index_spaces", {}), args.seed, {"replay0": args.replay0, "ctrl0": args.ctrl0})
        corpus_dir = args.corpus_dir or str(results_root() / cfg.get("data.corpus_experiment", cfg.name) / "data")
        packs = f"{corpus_dir}/packs_s0.u16"
        val = {"fwe": f"{corpus_dir}/val_fwe.u16", "pile": f"{corpus_dir}/val_pile.u16"}
        os.makedirs(data_dir, exist_ok=True)
        packs_sha = None
        for arm, p in spaces.items():
            if args.arm and arm != args.arm:
                continue
            synth = f"{data_dir}/{p['synth']}" if p.get("synth") else None
            out = f"{data_dir}/manifest_{p['file_stem']}_s{args.seed}.json"
            man = M.build_row_map(arm, p["kind"], p, args.seed, packs, val, synth, out, packs_sha=packs_sha)
            packs_sha = man["packs"]["sha256"]
            print(arm, out, man["meta"])
        return 0
    if args.mode == "select-heads":
        P.select_heads(args.dla, f"{data_dir}/heads.json")
        return 0
    if args.mode == "calibrate-lambda":
        P.calibrate_lambda(plan["base"], plan["revision"], data_dir, args.seed, f"{data_dir}/heads.json", args.which, f"{data_dir}/lambda_{args.which}.json", plan["micro_bs"])
        return 0
    if args.mode == "pick-rate":
        if cfg.get("calibration.fwe_tolerances") is not None:  # exercise fine-tunes: the held-out-loss rule
            print("rate pick for this experiment: apply the held-out-loss rule in the config (calibration.*) to the calibration runs under",
                  results_root() / cfg.get("results_dir", cfg.name) / "calibration")
            return 0
        rates = [str(r) for r in cfg.get("training.lr_candidates", [])]
        res = P.pick_rate_corpus(str(results_root() / cfg.get("results_dir", cfg.name) / "calibration"), rates)
        print(json.dumps(res, indent=1))
        return 0
    if args.mode == "eval-loss":
        P.eval_loss(args.model, data_dir, args.tag or "", str(morph_dir(cfg)))
        return 0
    if args.mode == "engagement":
        P.engagement(args.model, f"{data_dir}/heads.json", args.tag or "", str(morph_dir(cfg)))
        return 0
    calib = args.mode == "calibrate"
    arm = args.arm or ("A" if calib and not plan.get("manifest") else args.arm)
    assert arm, "--arm is required"
    out = plan["out"] if not calib else str(results_root() / cfg.name / "calibration" / f"lrcalib_{plan['lr']:g}")
    spec = P.TrainSpec(arm=arm, seed=args.seed, base=plan["base"], revision=plan["revision"], lr=float(plan["lr"]), out=out, data=data_dir, manifest=plan.get("manifest"),
                       steps=int(args.steps or plan["steps"]) if not calib else int(cfg.get("calibration.steps", 64)), save_every=int(plan["save_every"]), micro_bs=int(plan["micro_bs"]),
                       wd=float(plan["wd"]), warmup=int(plan["warmup"]) if not calib else int(cfg.get("calibration.warmup", plan["warmup"])), sched=plan["sched"] if not calib else cfg.get("calibration.sched", "constant"),
                       probe_graph=plan["probe_graph"], heads=f"{data_dir}/heads.json" if plan["probe_graph"] else None, lam=float(plan.get("lambda") or 0.0), calib=calib,
                       resume=not calib, val_batch=int(plan.get("val_batch", 16)), extra_row_map_arms=tuple(cfg.get("row_map_arms", [])))
    P.train(spec, root=str(data_root().parent))
    return 0


def run_lora(cfg: Config, args: argparse.Namespace, plan: dict) -> int:
    from icl_copy_reason.training import lora as L

    data_dir = plan["data_dir"]
    base = plan["base"]
    if args.mode == "build-data":
        g = cfg.get("data", {})
        L.build_data(data_dir, g.get("n_docs", 200000), g.get("seed", 0), g.get("calib_seed", 9), g.get("n_calib", 8000), g.get("n_replay", 60000))
        return 0
    work = results_root() / cfg.name / args.base
    os.makedirs(work, exist_ok=True)
    calib_packs = str(work / "packs_calib.npy")
    if not os.path.exists(calib_packs):
        L.pack_rows(base, data_dir, "calib", calib_packs)
    if args.mode == "calibrate":
        L.calibrate_rate(base, data_dir, calib_packs, str(work), [float(x) for x in plan["rates"]], int(cfg.get("calibration.steps", 48)), int(plan["micro_bs"]))
        return 0
    if args.mode == "eval-loss":
        L.eval_loss(args.model, data_dir, args.tag or "", str(results_root() / cfg.name / "readings"))
        return 0
    assert args.arm in ("T", "L"), "--arm T|L"
    out = plan["out"]
    os.makedirs(out, exist_ok=True)
    packs = f"{out}/packs_{args.arm}.npy"
    if not os.path.exists(packs):
        L.pack_rows(base, data_dir, args.arm, packs)
    return L.train(base, packs, float(plan["lr"]), args.seed, out, int(plan["r"]), int(plan["alpha"]), int(args.steps or 0), int(plan["micro_bs"]), calib_packs)


def run_direct(cfg: Config, args: argparse.Namespace, plan: dict) -> int:
    from icl_copy_reason.training import direct_finetune as D

    D.train(args.arm, args.seed, plan["out"], plan["base"], plan["revision"], int(args.steps or plan["steps"]), int(plan.get("eval_every") or 250), int(plan["bs"]), float(plan["lr"]),
            int(plan["warmup"]), plan.get("save_steps"), args.out_suffix or "", plan.get("sched_steps"), args.select_seed, args.dla, args.accum)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--mode", default="train", choices=["train", "calibrate", "corpus-manifest", "row-maps", "select-heads", "calibrate-lambda", "pick-rate", "eval-loss", "engagement",
                                                         "build-data", "pack", "overlap-check", "freeze"])
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", help="replication block name (configs with a `blocks` section)")
    ap.add_argument("--base", help="LoRA base key (configs with a `bases` section)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--select-seed", type=int, default=3)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--dla", help="attribution file for head selection / guided arms")
    ap.add_argument("--which", default="sel", choices=["sel", "rand"])
    ap.add_argument("--model", help="checkpoint path for eval-loss / engagement")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--replay0", type=int)
    ap.add_argument("--ctrl0", type=int)
    ap.add_argument("--corpus-dir", help="directory holding packs_s0.u16 / val_*.u16 for row-map manifests")
    ap.add_argument("--variant", help="pack: variant name (configs with a `variants` section)")
    ap.add_argument("--mixture", help="pack: rewritten-document mixture (M1 | M2)")
    ap.add_argument("--manifests", help="overlap-check: comma-separated manifest paths")
    ap.add_argument("--out", help="pack: output directory (default the experiment's data directory)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    if cfg is None:
        ap.error("--config is required")
    plan = resolve_plan(cfg, args)
    if args.mode == "pack":
        return run_pack(cfg, args, plan)
    if args.mode == "overlap-check":
        return run_overlap_check(cfg, args, plan)
    if args.mode == "freeze":
        return run_freeze(cfg, args)
    if args.dry_run:
        print(json.dumps(plan, indent=1, default=str))
        return 0
    if "HF_HOME" not in os.environ and os.environ.get("ICR_HF_HOME"):
        os.environ["HF_HOME"] = os.environ["ICR_HF_HOME"]
    trainer = cfg.get("trainer")
    if trainer == "pretrain":
        return run_pretrain(cfg, args, plan)
    if trainer == "lora":
        return run_lora(cfg, args, plan)
    if trainer == "direct_finetune":
        return run_direct(cfg, args, plan)
    print(f"config {cfg.name} is an evaluation experiment: use its launcher {cfg.get('launcher')}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
