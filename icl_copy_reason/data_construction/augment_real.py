"""The first set of LLM-rewritten real documents (types a-g) and the pipeline specification shared by every exercise build.

Inputs : a config (configs/rewritten-docs-data.yaml for the first set, configs/exercises-data.yaml for the topped-up set):
         prompt table path, output directory, sampling spec (quotas, residue rule, type rule), API model names.
Outputs: the sampling spec consumed by `documents.sample_documents`, and `run_pipeline`, which chains
         windows -> sample -> generate -> verify -> (retry -> verify) -> audit with the functions of `exercises`.

First set: 60,000 target documents cut to 40,000 (then 28,000 when a stage is short), quotas per pool at the 60,000
target scaled proportionally, eligible-index % 25 == 7, type by a hash of the document text over the six types
{a, b, c, e, f, g}; mixtures M1 = all six and M2 = {a, e, f, g}; audit gate >= .85 per type and >= .90 pooled.
Topped-up set: fresh documents (eligible-index % 25 == 8, quotas = first-set quotas x 40/60), type per document drawn
with Random(13) over {h: 27,000, b: 5,000, c: 3,000, e: 3,000, g: 2,000}; mixtures F1 = {a (first set, once), b, c, e, g, h}
and F2 = {b, c, e, g, h}; audit gate >= .85 per type.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ApiSettings, Config, data_root, results_root
from . import exercises as E
from . import verification as V
from .documents import P4_EXERCISE_FILES, sample_documents


def sampling_spec(P: dict, cfg: Config | None, target: int | None = None) -> dict[str, Any]:
    """The sampler spec from the prompt table's `sampling` block plus the config's exclusion / record paths."""
    s = dict(P["sampling"])
    get = (lambda k, d=None: cfg.get(k, d)) if cfg else (lambda k, d=None: d)
    spec = {
        "quotas": s["quotas"], "target_docs": int(s["target_docs"]), "target": int(target or get("sampling.target", s["target_docs"])), "mod": int(s["mod"]), "residue": int(s["residue"]),
        "min_chars": int(P["document_chars"]["min"]), "max_chars": int(P["document_chars"]["max"]), "forbidden_regex": P["forbidden_regex"],
        "type_rule": "draw" if "type_weights" in s else "hash", "types": sorted(P["types"]), "type_weights": s.get("type_weights"), "type_seed": s.get("type_seed"),
        "p4_files": P4_EXERCISE_FILES, "mining_dir": get("exclusions.mining_dir"), "selected_docs": get("exclusions.selected_docs"), "files_records": get("exclusions.mining_dir"),
        "pilot_type": "h" if "type_weights" in s else None, "pilot_per_pool": 5 if "type_weights" in s else 24,
    }
    return spec


def paths(cfg: Config | None, name: str = "rewritten-docs-data") -> dict[str, Path]:
    base = Path(cfg.get("output_dir")) if cfg and cfg.get("output_dir") else results_root() / name
    return {"base": base, "docs": base / "docs.jsonl", "gen": base / "gen.jsonl", "gen_retry": base / "gen_retry.jsonl", "verified": base / "verified.jsonl",
            "windows": Path(cfg.get("windows_dir")) if cfg and cfg.get("windows_dir") else base / "eval_windows"}


def prompt_table(cfg: Config | None, default: str = "rewrite_prompts.json") -> dict:
    p = cfg.get("prompts") if cfg else None
    return E.load_prompts(p or data_root() / "exercises" / default)


def api_settings(cfg: Config | None) -> ApiSettings:
    return cfg.api if cfg else ApiSettings.from_env()


def run_pipeline(cfg: Config | None, stage: str, name: str, default_prompts: str, workers: int = 32, pilot: int = 0, extend_windows_with_datasets: bool = True, target: int | None = None) -> Any:
    """One stage of the exercise pipeline: windows | sample | generate | verify | retry | verify-retry | audit | all."""
    P = prompt_table(cfg, default_prompts)
    pth = paths(cfg, name)
    pth["base"].mkdir(parents=True, exist_ok=True)
    settings = api_settings(cfg)
    stages = ["windows", "sample", "generate", "verify", "retry", "verify-retry", "audit"] if stage == "all" else [stage]
    out = None
    for st in stages:
        if st == "windows":
            out = V.build_eval_windows(pth["windows"], datasets=extend_windows_with_datasets)[1]
        elif st == "sample":
            out = sample_documents(sampling_spec(P, cfg, target), pth["base"], workers=workers, pilot=pilot)
        elif st == "generate":
            out = E.generate(P, pth["docs"], pth["gen"], workers=workers, settings=settings)
        elif st == "verify":
            out = E.verify(P, pth["gen"], pth["docs"], pth["verified"], pth["windows"], workers=workers, resume=True, settings=settings)
        elif st == "retry":
            if not P.get("retry_type"):
                continue
            out = E.generate(P, pth["docs"], pth["gen_retry"], workers=workers, retry_from=pth["verified"], settings=settings)
        elif st == "verify-retry":
            if not P.get("retry_type"):
                continue
            out = E.verify(P, pth["gen_retry"], pth["docs"], pth["verified"], pth["windows"], workers=workers, resume=True, settings=settings)
        elif st == "audit":
            out = E.audit(P, pth["verified"], pth["docs"], per_type=50, workers=min(workers, 16), settings=settings, seed=13 if P.get("retry_type") else 12)
        else:
            raise SystemExit(f"unknown stage {st}")
    return out
