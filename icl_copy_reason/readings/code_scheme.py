"""The code-scheme diagnostic (attribute words -> a two-symbol code, no stem) scored under two generation rules.

  prefix  the first line's symbols, truncated to the target length when they begin with the target (the earlier, looser rule);
  strict  the first line's symbols must equal the target sequence exactly (the rule of record).
Items = build("compose_h", n, seed, n_single=2, n_combo=2, scheme="code"), greedy decoding, 8 new tokens. The copy-probability share
and attention ablation of the code scheme are the standard readings with scheme="code" (copy_share.run / ablation.run).

Inputs : model, revisions, seed. Outputs: schema of the released codestrict_<model>_s<seed>.json (rates + raw generations), via `run`.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

from ..io import read_json, write_json
from ..tasks.prompts import build, parse_form, slots_of
from .copy_share import generate, load_model, revision_key


def score_strict(items, gens) -> tuple[int, int, list[dict]]:
    strict = prefix = 0
    recs = []
    for it, g in zip(items, gens):
        _, affs = parse_form(g, "code")
        tgt = [x for _, x in slots_of(it.lang, it.ia, it.ib)]
        s = affs == tgt
        p = s or (len(affs) > len(tgt) and affs[:len(tgt)] == tgt)
        strict += s
        prefix += p
        recs.append({"gen": g, "target": tgt, "strict": bool(s), "prefix": bool(p)})
    return strict, prefix, recs


def run(model: str, steps: str = "main", seed: int = 0, n: int = 160, bs: int = 16, out_dir: str | Path = ".", device: str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mtag = model.split("/")[-1]
    outp = out_dir / f"codestrict_{mtag}_s{seed}.json"
    res = read_json(outp) if outp.exists() else {}
    for step in steps.split(","):
        rev = revision_key(step)
        key = rev or "main"
        if key in res:
            continue
        t0 = time.time()
        lm = load_model(model, revision=rev, device=device)
        items = build("compose_h", n, seed=seed, n_single=2, n_combo=2, scheme="code")
        gens = []
        for b in range(0, len(items), bs):
            gens += generate(lm, [it.prompt for it in items[b:b + bs]])
        strict, prefix, recs = score_strict(items, gens)
        res[key] = {"n": len(items), "exact_strict": round(strict / len(items), 4), "exact_prefix": round(prefix / len(items), 4), "fingerprint": lm.fingerprint(), "items": recs}
        print(f"{mtag} {key}: strict {strict / len(items):.3f} prefix {prefix / len(items):.3f} [{time.time() - t0:.0f}s]", flush=True)
        write_json(res, outp, indent=0)
        lm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outp
