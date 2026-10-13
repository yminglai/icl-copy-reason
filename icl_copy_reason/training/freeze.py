"""Records fixed before a run: file hashes, rules, families, manifests, environment.

Every training experiment that compares arms wrote such a record before anything trained, and its launcher refuses to run without it.
Inputs : a config (configs/<experiment>.yaml) whose `freeze` section lists the files to hash and the manifests to summarise, plus the rules
         and families that the analysis reads back (minimum effect, loss tolerances, assay families, comparators, labels, predictions).
Outputs: <data_root>/results/fixed_records/<experiment>[_<block>].json (never overwritten without force=True).
Also here: the unseen-family qualification rule (first 200 items, natural-label Wilson-95 lower bound above chance and natural accuracy <= .95;
the first three qualifying datasets in the fixed order) used by the exercise fine-tune blocks.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

from icl_copy_reason.config import Config, data_root
from icl_copy_reason.io import sha256_file, write_json
from icl_copy_reason.training.manifests import manifest_summary

DATASET_ORDER = [("trec6", "TREC-6 (coarse question type), k = 12"), ("dbpedia7", "DBpedia-14 restricted to the first seven classes by label index, k = 14"),
                 ("yahoo5", "Yahoo Answers Topics restricted to the first five classes, k = 10"), ("20ng6", "20 Newsgroups, first six groups, k = 12 (fallback)")]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head(cwd: str | None = None) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=False).stdout.strip() or None
    except OSError:
        return None


def env_versions() -> dict[str, str]:
    out = {}
    for mod in ("torch", "transformers", "lm_eval", "peft", "datasets"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - optional dependency
            out[mod] = "absent"
    return out


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (c - r) / d


def qualify_unseen(screen_summary: dict[str, dict[str, Any]], order: list[tuple[str, str]] | None = None, max_pick: int = 3, ceiling: float = 0.95) -> dict[str, Any]:
    """The base-only qualification of the unseen-symbol family: Wilson-95 lower bound of natural-label accuracy above chance and natural <= ceiling."""
    order = order or DATASET_ORDER
    alias = {"20ng6": "ng6"}
    q = []
    for ds, _ in order:
        s = screen_summary.get(alias.get(ds, ds))
        if not s:
            q.append({"ds": ds, "screened": False})
            continue
        lb = wilson_lower(round(s["natural_acc"] * s["n"]), s["n"])
        ok = lb > s["chance"] and s["natural_acc"] <= ceiling
        q.append({"ds": alias.get(ds, ds), "n": s["n"], "natural": s["natural_acc"], "wilson_lb": round(lb, 4), "chance": s["chance"], "qualifies": bool(ok)})
    chosen = [x["ds"] for x in q if x.get("qualifies")][:max_pick]
    assays = []
    for ds in chosen:
        assays.append({"id": f"{ds}_sul", "kind": "unseen", "ds": ds, "metric": "sul_correct", "label": f"{ds} symbol-label accuracy (unseen dataset)"})
        assays.append({"id": f"{ds}_flip", "kind": "unseen", "ds": ds, "metric": "flipped_follows_context", "label": f"{ds} exchanged-label following (unseen dataset)"})
    return {"qualification": q, "datasets": chosen, "assays": assays, "rule": f"first {max_pick} in order with Wilson-95 LB(natural) > chance and natural <= {ceiling} on the first 200 items"}


def freeze_record(cfg: Config, block: str | None = None, root: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the record from the config's `freeze` section: hashes of the listed files, summaries of the listed manifests, the copied rules."""
    fz = cfg.get("freeze", {}) or {}
    root = root or str(data_root().parent)
    rec: dict[str, Any] = {"experiment": cfg.name, "block": block, "frozen_utc": utc(), "git_head": git_head(root), "env": env_versions(),
                           "config_sha256": sha256_file(cfg.path) if cfg.path else None, "config": cfg.as_dict()}
    files = {}
    for f in fz.get("files", []):
        p = f if os.path.isabs(f) else os.path.join(root, f)
        files[f] = sha256_file(p) if os.path.exists(p) else None
    rec["files_sha256"] = files
    mans = {}
    for name, f in (fz.get("manifests") or {}).items():
        p = f if os.path.isabs(f) else os.path.join(root, f)
        if not os.path.exists(p):
            mans[name] = {"file": f, "status": "absent"}
            continue
        m = json.load(open(p))
        mans[name] = {"file": f, "sha256": sha256_file(p), **manifest_summary(m)}
    rec["manifests"] = mans
    for key in ("rules", "families", "comparators", "labels", "damage", "predictions", "priors", "calibration", "stop_rule", "min_effect_pp", "loss_tol_nats"):
        if key in fz:
            rec[key] = fz[key]
    if extra:
        rec.update(extra)
    return rec


def write_freeze(cfg: Config, out: str, block: str | None = None, root: str | None = None, extra: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    if os.path.exists(out) and not force:
        raise SystemExit(f"REFUSE: {out} exists (fixed before the run); pass force=True to overwrite")
    rec = freeze_record(cfg, block, root, extra)
    write_json(rec, out)
    return rec
