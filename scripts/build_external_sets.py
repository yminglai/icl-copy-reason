#!/usr/bin/env python
"""Symbol-label / exchanged-label evaluation sets, the composition panels and the far assays: build and score.

usage:
  python scripts/build_external_sets.py label-maps [--out data/external/label_maps.json]
  python scripts/build_external_sets.py build-labels --sets trec6,dbpedia7 --n 1000 --out <dir>      # items jsonl per set (needs the Hub)
  python scripts/build_external_sets.py build-panels --config external-sets [--panels-dir D]          # SCAN / ELEMENTAL / MiniSCAN / arith / chain-map
  python scripts/build_external_sets.py fetch-bigbench [--dest data/external/bigbench]               # BIG-bench task JSON for the far assays
  python scripts/build_external_sets.py score-labels --model M [--revision R] --tag T --sets trec6,dbpedia7,yahoo5,ng6 [--n 1000]
  python scripts/build_external_sets.py score-near   --model M --tag T --sets sst2,agnews,subj
  python scripts/build_external_sets.py score-panels --model M --tag T --tasks scan,elemental,miniscan,arith,chainmap
  python scripts/build_external_sets.py score-far    --model M --tag T --assays list_functions,unnatural_icl,babi_1
  python scripts/build_external_sets.py assays --config exercises-finetune --model M --tag T [--near 1] [--max-batch 16]   # a training run's OOD block
Outputs: <scoring.out_dir>/<tag>/... (default <ICR_RESULTS_ROOT>/external-sets/<tag>/).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl_copy_reason.config import add_config_args, config_from_args, results_root  # noqa: E402
from icl_copy_reason.data_construction import external_sets as X  # noqa: E402
from icl_copy_reason.io import write_json  # noqa: E402

BIGBENCH_TASKS = ["list_functions", "unnatural_in_context_learning"]


def _lm(a, dtype: str):
    from icl_copy_reason.readings.lm import load_model

    return load_model(a.model, a.revision, dtype)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["label-maps", "build-labels", "build-panels", "fetch-bigbench", "score-labels", "score-near", "score-panels", "score-far", "assays"])
    add_config_args(ap)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--assays", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--panels-dir", default=None)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--near", type=int, default=0, help="assays: 1 adds the near block (symbol / exchanged-label in-context sets)")
    ap.add_argument("--scale", default=None, help="assays: 360M | 1p7B (selects the suites of a training config)")
    ap.add_argument("--max-batch", type=int, default=None, help="assays: generation batch size")
    a = ap.parse_args(argv)
    cfg = config_from_args(a) or config_from_args(SimpleNamespace(config="external-sets", set=[]))
    root = Path(__file__).resolve().parents[1]
    out_root = Path(a.out) if a.out and a.cmd != "score-panels" else Path(cfg.get("scoring.out_dir") or results_root() / "external-sets")
    ext = cfg if cfg.name == "external-sets" else config_from_args(SimpleNamespace(config="external-sets", set=[]))
    n = a.n or int(cfg.get("n_items", 1000))
    if a.cmd == "label-maps":
        p = Path(a.out or root / "data" / "external" / "label_maps.json")
        write_json(X.label_maps(), p)
        print("wrote", p)
        return 0
    if a.cmd == "build-labels":
        out = Path(a.out or out_root / "label_sets")
        out.mkdir(parents=True, exist_ok=True)
        for ds in (a.sets or ",".join(cfg.get("label_sets", []))).split(","):
            items, names, src = X.build_label_set(ds, n)
            with open(out / f"{ds}.jsonl", "w") as f:
                for it in items:
                    f.write(json.dumps({**it, "names": names}) + "\n")
            print(ds, len(items), src)
        return 0
    if a.cmd == "build-panels":
        d = Path(a.panels_dir or cfg.get("panels.dir") or out_root / "panels")
        rep = X.build_panels(d, (a.tasks or ",".join(cfg.get("panels.tasks", []))).split(","), int(cfg.get("panels.n_scan", 200)), int(cfg.get("panels.per_cat", 5)), int(cfg.get("panels.n_miniscan", 200)))
        print(json.dumps(rep, indent=1))
        return 0
    if a.cmd == "fetch-bigbench":
        import urllib.request

        dest = Path(a.dest or root / cfg.get("bigbench_dir", "data/external/bigbench"))
        base = "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/"
        for task in BIGBENCH_TASKS:
            if task == "list_functions":
                for fi in range(1, 251):
                    fid = f"c{fi:03d}"
                    (dest / "list_functions" / fid).mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(base + f"list_functions/{fid}/task.json", dest / "list_functions" / fid / "task.json")
            else:
                for sub in X.UNNATURAL:
                    (dest / task / sub).mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(base + f"{task}/{sub}/task.json", dest / task / sub / "task.json")
            print("fetched", task)
        return 0
    if not a.model or not a.tag:
        raise SystemExit("--model and --tag are required for scoring")
    lm = _lm(a, cfg.get("scoring.dtype", "bfloat16"))
    if a.cmd == "score-labels":
        X.score_label_sets(lm, out_root, a.tag, (a.sets or ",".join(ext.get("label_sets", []))).split(","), n)
    elif a.cmd == "score-near":
        X.score_label_sets(lm, out_root, a.tag, (a.sets or ",".join(ext.get("near_sets", []))).split(","), n)
    elif a.cmd == "score-panels":
        d = Path(a.panels_dir or ext.get("panels.dir") or out_root / "panels")
        out_file = Path(a.out) if a.out else out_root / a.tag / "panels.json"
        X.score_panels(lm, d, out_file, (a.tasks or ",".join(ext.get("panels.tasks", []))).split(","), a.batch_size or 8)
    elif a.cmd == "assays":
        # the OOD block of a training experiment at a stage boundary: far assays + the composition panels (+ the near block at the end).
        # Written where the report builders read them: far -> <root>/rewritten-docs-admixture/far/<tag>/, panels -> <root>/<experiment>/morph/icl_<tag>.json,
        # unseen symbol-label sets -> <root>/<experiment>/icl/<tag>/, near in-context sets -> <root>/external-assays/full/<tag>/.
        exp_dir = results_root() / cfg.get("results_dir", cfg.name)
        bs = a.max_batch or a.batch_size or int(ext.get("scoring.batch_size", 16))
        far = cfg.get("evaluation.far_assays") or ext.get("far_assays", [])
        X.SRC = str(root / ext.get("bigbench_dir", "data/external/bigbench"))
        X.score_far_assays(lm, results_root() / "rewritten-docs-admixture" / "far", a.tag, list(far), n, bs)
        d = Path(a.panels_dir or ext.get("panels.dir") or results_root() / "external-sets" / "panels")
        X.score_panels(lm, d, exp_dir / "morph" / f"icl_{a.tag}.json", list(ext.get("panels.tasks", [])), min(bs, 8))
        if a.near:
            unseen = cfg.get("evaluation.unseen_family.datasets") or ext.get("label_sets", [])
            X.score_label_sets(lm, exp_dir / "icl", a.tag, list(unseen), int(cfg.get("evaluation.unseen_family.n", n)))
            X.score_label_sets(lm, results_root() / "external-assays" / "full", a.tag, list(ext.get("near_sets", [])), n)
    else:
        X.SRC = str(root / ext.get("bigbench_dir", "data/external/bigbench"))
        X.score_far_assays(lm, out_root, a.tag, (a.assays or ",".join(ext.get("far_assays", []))).split(","), n, a.batch_size or int(ext.get("scoring.batch_size", 16)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
