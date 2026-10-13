#!/usr/bin/env python3
"""Build the diagnostic prompts (made-up endings, one combination withheld) and write the item sets of record.

usage: python scripts/make_prompts.py [--config diagnostic] [--out data/diagnostic] [--seeds 0,1,2,3] [--n 160] [--show]
Writes items_<task>_n<n>_seed<seed>.jsonl.gz for every task and seed in the config, the ablation panels (seed + 11), the code-scheme and
alternate-template controls, plus templates.json and pools.json. --show prints one prompt per task instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl_copy_reason.config import add_config_args, config_from_args, data_root  # noqa: E402
from icl_copy_reason.tasks import prompts as P  # noqa: E402
from icl_copy_reason.tasks.items import ABLATION_SEED_OFFSET, item_file, items_of_record, write_items  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--out", default=None, help="output directory (default <ICR_DATA_ROOT>/diagnostic)")
    ap.add_argument("--seeds", default=None, help="comma-separated item seeds (default from the config)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--show", action="store_true", help="print one prompt per task and exit")
    a = ap.parse_args(argv)
    cfg = config_from_args(a, "diagnostic")
    tasks = cfg.get("tasks", ["bind_a", "bind_b", "compose_s", "compose_h"])
    n = a.n or cfg.get("n", 160)
    seeds = [int(s) for s in (a.seeds.split(",") if a.seeds else cfg.get("seeds", [0, 1, 2, 3]))]
    if a.show:
        for t in tasks:
            it = P.build(t, 1, seed=seeds[0])[0]
            print(f"--- {t} ---\n{it.prompt} {it.answer}\n")
        return 0
    out = Path(a.out) if a.out else data_root() / "diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in seeds:
        for t in tasks:
            written.append(write_items(items_of_record(t, n, seed), item_file(t, n, seed, root=out)))
        written.append(write_items(items_of_record("compose_h", n, seed, ABLATION_SEED_OFFSET), item_file("compose_h", n, seed, "ablation_panel", root=out)))
    for ctrl, kw in cfg.get("controls", {}).items():
        for seed in seeds[:2]:
            written.append(write_items(items_of_record("compose_h", n, seed, **kw), item_file("compose_h", n, seed, ctrl, root=out)))
    P.set_template("default")
    json.dump({"templates": P.TEMPLATES, "header_default": "In this language:\n", "code_header": P.CODE_HEADER, "code_sep": P.CODE_SEP},
              open(out / "templates.json", "w"), indent=1)
    json.dump({"affix_pool": P.AFFIX_POOL, "words_a": P.WORDS_A, "words_b": P.WORDS_B, "onsets": P.ONSETS, "vowels": P.VOWELS, "codas": P.CODAS,
               "code_pool": P.CODE_POOL, "code_words_a": P.CODE_WORDS_A, "code_words_b": P.CODE_WORDS_B}, open(out / "pools.json", "w"), indent=1)
    print(f"wrote {len(written)} item files + templates.json + pools.json under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
