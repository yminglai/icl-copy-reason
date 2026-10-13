#!/usr/bin/env python
"""Synthetic definition-use documents, their every-combination-shown control, the preparation checks, and the mined real documents.

usage:
  python scripts/build_synthetic_docs.py generate --mode compose --seed 0 --n 5 [--version 1|3]          # print documents
  python scripts/build_synthetic_docs.py sample --config synthetic-docs-data --out data/synthetic       # 1,000-document samples per mode
  python scripts/build_synthetic_docs.py disjoint                                                        # generator pools vs the diagnostic's pools
  python scripts/build_synthetic_docs.py eligibility --results-dir <dir> --model-tag <tag> --tag _screen  # base-checkpoint screen before an admixture run
  python scripts/build_synthetic_docs.py learning-rate --calib-dir <dir> --rates 3e-4,1e-3,3e-3           # calibration pick
  python scripts/build_synthetic_docs.py mined <step> --config mined-docs-data --mine-dir M --data-dir D [--pool P1] [--workers N]
      arm D  steps: pool candidates label fit select audit overlap record
      arm D2 steps: scan verify pack render d2-audit surprisal proxies d2-record
      extras: refvocab (--random-docs F) survey (--word-terms) window-audit reread
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl_copy_reason.config import add_config_args, config_from_args, results_root  # noqa: E402
from icl_copy_reason.data_construction import mining, synthetic_docs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["generate", "sample", "disjoint", "eligibility", "learning-rate", "mined"])
    ap.add_argument("step", nargs="?", default=None, help="mined: pipeline step")
    add_config_args(ap)
    ap.add_argument("--mode", default="compose", choices=list(synthetic_docs.MODES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--tag", default="_screen")
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--rates", default="3e-4,1e-3,3e-3")
    ap.add_argument("--mine-dir", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--critique-dir", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--panels", default=None, help="mined overlap: directory with the composition panels json")
    ap.add_argument("--random-docs", default=None, help="refvocab / survey: random.jsonl of the arm-D pool step")
    ap.add_argument("--word-terms", action="store_true")
    ap.add_argument("--max-files", type=int, default=None, help="mined scan: smoke tests only")
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    if a.cmd == "generate":
        import random

        rng = random.Random(a.seed)
        for _ in range(a.n):
            d, meta = synthetic_docs.make_doc(rng, a.mode, a.version)
            print(d)
            print(meta, "\n" + "-" * 60)
        return 0
    if a.cmd == "sample":
        c = cfg or config_from_args(SimpleNamespace(config="synthetic-docs-data", set=[]))
        out = Path(a.out or Path(__file__).resolve().parents[1] / "data" / "synthetic")
        n = a.n if a.n != 3 else int(c.get("sample.n", 1000))
        for ver in sorted({int(c.get("generator.version_360m", 1)), int(c.get("generator.version_1p7b", 3))}):
            for mode in c.get("generator.modes", list(synthetic_docs.MODES)):
                p = out / f"sample_{mode}_v{ver}.jsonl.gz"
                synthetic_docs.write_sample(mode, int(c.get("generator.seed", 0)), n, ver, p)
                print("wrote", p)
        return 0
    if a.cmd == "disjoint":
        w, m = synthetic_docs.diagnostic_pools()
        ov = synthetic_docs.pools_disjoint_from(w, m)
        print(json.dumps({"overlap": ov, "disjoint": not ov["words"] and not ov["markers"]}))
        return 0 if not ov["words"] and not ov["markers"] else 1
    if a.cmd == "eligibility":
        print(json.dumps(synthetic_docs.base_eligibility(a.results_dir, a.model_tag, a.tag), indent=1))
        return 0
    if a.cmd == "learning-rate":
        print(json.dumps(synthetic_docs.learning_rate_pick(a.calib_dir, a.rates.split(",")), indent=1))
        return 0
    # mined documents
    c = cfg or config_from_args(SimpleNamespace(config="mined-docs-data", set=[]))
    mine = a.mine_dir or c.get("mine_dir") or results_root() / "mined-docs-data" / "mine"
    data = a.data_dir or c.get("data_dir")
    if not data:
        raise SystemExit("--data-dir (admixture data directory with manifest_s0.json) is required for the mined-documents steps")
    mining.configure(mine, data, a.critique_dir or c.get("critique_dir"), max_files=a.max_files)
    ad = c.get("arm_d", {})
    steps = {
        "pool": lambda: mining.arm_d_pool(SimpleNamespace(data=data, out=mine, skip_docs=None, target=ad.get("pool_target", 300000), random_every=ad.get("random_every", 20))),
        "candidates": lambda: mining.arm_d_candidates(SimpleNamespace(mine=mine, n=ad.get("candidates", 900), n_random=ad.get("candidates_random", 300), seed=ad.get("seed", 0))),
        "label": lambda: mining.arm_d_label(SimpleNamespace(inp=f"{mine}/cands.jsonl", out=f"{mine}/labels.jsonl", models=",".join(mining.LABEL_MODELS), workers=a.workers)),
        "fit": lambda: mining.arm_d_fit(SimpleNamespace(mine=mine, seed=ad.get("seed", 0))),
        "select": lambda: mining.arm_d_select(SimpleNamespace(mine=mine, data=data, seed=ad.get("seed", 0))),
        "audit": lambda: mining.arm_d_audit(SimpleNamespace(mine=mine, n=ad.get("audit_n", 150), seed=ad.get("audit_seed", 1), workers=a.workers)),
        "overlap": lambda: mining.arm_d_overlap(SimpleNamespace(mine=mine, panels=a.panels)),
        "record": lambda: mining.arm_d_record(SimpleNamespace(mine=mine, data=data, panels=a.panels)),
        "scan": lambda: mining.d2_scan(SimpleNamespace(pool=a.pool, workers=a.workers)),
        "verify": lambda: mining.d2_verify(SimpleNamespace(pool=a.pool, workers=a.workers)),
        "pack": lambda: mining.d2_pack(SimpleNamespace()),
        "render": lambda: mining.d3_render(SimpleNamespace()),
        "d2-audit": lambda: mining.d2_audit(SimpleNamespace(workers=min(a.workers, 12))),
        "surprisal": lambda: mining.d2_surprisal(SimpleNamespace()),
        "proxies": lambda: mining.d2_proxies(SimpleNamespace()),
        "d2-record": lambda: mining.d2_record(SimpleNamespace()),
        "refvocab": lambda: mining.build_refvocab(a.random_docs or f"{mine}/random.jsonl", workers=a.workers),
        "survey": lambda: mining.pool_survey(a.random_docs or f"{mine}/random.jsonl", int(c.get("survey.n_per_pool_word_terms" if a.word_terms else "survey.n_per_pool", 1500)), word_terms=a.word_terms),
        "window-audit": lambda: mining.window_audit(),
        "reread": lambda: mining.window_reread(workers=min(a.workers, 12)),
    }
    if a.step not in steps:
        raise SystemExit(f"mined: step must be one of {sorted(steps)}")
    steps[a.step]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
