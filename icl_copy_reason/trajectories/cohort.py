"""Systematic survey of released checkpoint families: final-checkpoint screen (stage A) and trajectory dispositions (stage B).

Inputs : the family registry (data/manifests/revisions/*.json with `survey_status`), per-family ablation and copy-share files at
         the final checkpoint (stage A) and over the selected grid (stage B), a declared-failures map (family -> error text).
Outputs: stage A rows (pooled four-way correct count, exact match, copied-form rate, null adherence, the eight ablation effects
         with item-bootstrap intervals, D, class = composes / partial / copies), the two contrast families, stage B
         summaries with entry state, onset censoring class and disposition, and the cohort flow with its verdict.
Fail-closed: a family without result files and without a declared failure raises.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..io import read_json, write_json
from . import crossings
from .registry import Family, load_family, result_key

CONDS = ["competitor", "single2", "sham", "otherbundles", "wordB", "competitor_marker", "wordA", "single2_marker"]
SEPARATE_REPO_FAMILIES = {"intellect-1-10b", "openllama-7b-v2", "bloom-7b1", "tinyllama-1.1b"}
COMPOSER_MIN_CORRECT = 160


def model_tag(model_id: str) -> str:
    return model_id.split("/")[-1]


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - adj) / den


def boot_mean(x: Sequence[float], rng: np.random.Generator, B: int = 2000) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def classify(k_correct: int, n: int) -> str:
    """composes: at least half of the pooled withheld items correct; partial: Wilson lower bound above chance; else copies."""
    if k_correct >= COMPOSER_MIN_CORRECT * n / 320:
        return "composes"
    return "partial" if wilson_lower(k_correct, n) > crossings.CHANCE_FOUR_WAY else "copies"


def stage_a(results_dir: str | Path, families: Sequence[str], seeds: Sequence[int] = (2, 3), n: int = 160, tag: str = "_survey",
            failures: dict[str, str] | None = None) -> dict[str, Any]:
    """Final-checkpoint screen of every family: class and the two readings' statistics; then the two contrast families
    (traceable copiess ranked by pooled accuracy, then number of available checkpoints, then model id)."""
    failures = failures or {}
    rows: dict[str, Any] = {}
    fams = {f: load_family(f) for f in families}
    for fam, F in fams.items():
        t = model_tag(F.final_model)
        files = {(kind, s): crossings.result_file(results_dir, kind, t, s, tag) for kind in ("knockout", "triage") for s in seeds}
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            if fam in failures:
                rows[fam] = {"model": F.final_model, "class": "technical_failure", "error": failures[fam]}
                continue
            raise crossings.TrajectoryError(f"{fam}: missing {missing} and no declared failure")
        ko = [crossings.rec(read_json(files[("knockout", s)])["main"]) for s in seeds]
        tr = [crossings.rec(read_json(files[("triage", s)])["main"])["compose_h"] for s in seeds]
        for r in tr:
            if len(r["four_way_items"]) != n:
                raise crossings.TrajectoryError(f"{fam}: copy-share file has {len(r['four_way_items'])} items, expected {n}")
        acc_items = np.array([int(i["acc"]) for r in tr for i in r["four_way_items"]])
        k = int(acc_items.sum())
        nn = int(len(acc_items))
        rng = np.random.default_rng(0)
        base = np.concatenate([[it["margin"] for it in r["none"]["items"]] for r in ko])
        anull = np.concatenate([[it["a_null"] for it in r["none"]["items"]] for r in ko])
        eff = {c: boot_mean(np.concatenate([np.array([it["margin"] for it in r[c]["items"]]) - np.array([it["margin"] for it in r["none"]["items"]]) for r in ko]), rng) for c in CONDS}
        D = np.concatenate([np.array([it["margin"] for it in r["competitor_marker"]["items"]]) + np.array([it["margin"] for it in r["wordB"]["items"]])
                            - 2 * np.array([it["margin"] for it in r["none"]["items"]]) for r in ko])
        rows[fam] = {"model": F.final_model, "class": classify(k, nn), "four_way_correct": k, "n": nn, "four_way_acc": k / nn, "wilson_lower": wilson_lower(k, nn),
                     "exact": float(np.mean([r["exact"] for r in tr])), "competitor_rate_4way": float(np.mean([r["four_way_competitor"] for r in tr])),
                     "a_null": boot_mean(anull, rng), "margin": boot_mean(base, rng), "effects": eff, "D": boot_mean(D, rng),
                     "series_status": F.survey_status or "n/a"}
    ready = [f for f, r in rows.items() if r["class"] == "copies" and fams[f].survey_status == "B-ready"]
    ready.sort(key=lambda f: (-rows[f]["four_way_acc"], -(fams[f].n_available or 0), fams[f].final_model))
    contrasts = ready[:2]
    composing_ready = [f for f, r in rows.items() if r["class"] == "composes" and fams[f].survey_status == "B-ready"]
    composing_unavailable = [f for f, r in rows.items() if r["class"] == "composes" and fams[f].survey_status != "B-ready"]
    return {"rows": rows, "contrasts": contrasts, "contrast_shortfall": max(0, 2 - len(contrasts)), "stage_b_composing": composing_ready,
            "models_that_compose_series_unavailable": composing_unavailable}


def merge_separate_repo_series(results_dir: str | Path, F: Family, grid: str | None, seeds: Sequence[int], tag: str) -> tuple[str, list[str], dict[str, float]]:
    """Families whose checkpoints are separate repos or tags: merge the per-checkpoint files into one file keyed `pos<position>`
    under the tag `<family>-merged`. Returns (model tag, keys, position table)."""
    entries = F.revisions(grid)
    mt = f"{F.family}-merged"
    for kind in ("knockout", "triage"):
        for sd in seeds:
            merged = {}
            for e in entries:
                ref = e["ref"]
                if "@" in ref:
                    repo, rev = ref.split("@", 1)
                    src = crossings.result_file(results_dir, kind, model_tag(repo), sd, tag)
                    key = rev
                else:
                    src = crossings.result_file(results_dir, kind, model_tag(ref), sd, tag)
                    key = "main"
                if not src.exists():
                    raise crossings.TrajectoryError(f"{F.family}: missing {src}")
                d = read_json(src)
                if key not in d:
                    raise crossings.TrajectoryError(f"{F.family}: {src} lacks key {key}")
                merged[f"pos{int(e['position'])}"] = d[key]
            write_json(merged, crossings.result_file(results_dir, kind, mt, sd, tag))
    keys = [f"pos{int(e['position'])}" for e in entries]
    return mt, keys, {k: float(e["position"]) for k, e in zip(keys, entries)}


def restrict_to_grid(results_dir: str | Path, mt: str, keys: Sequence[str], seeds: Sequence[int], tag: str, grid_name: str) -> str:
    """Copy the files of a run keeping only the grid's keys, under the tag `<model tag>-<grid>`; returns the new model tag."""
    gt = f"{mt}-{grid_name}"
    for kind in ("knockout", "triage"):
        for sd in seeds:
            src = crossings.result_file(results_dir, kind, mt, sd, tag)
            if not src.exists():
                raise crossings.TrajectoryError(f"missing {src}")
            d = read_json(src)
            miss = [k for k in keys if k not in d]
            if miss:
                raise crossings.TrajectoryError(f"{kind} s{sd} lacks checkpoints {miss}")
            write_json({k: d[k] for k in keys}, crossings.result_file(results_dir, kind, gt, sd, tag))
    return gt


def stage_b(results_dir: str | Path, a: dict[str, Any], grid: str = "grid_20", seeds: Sequence[int] = (2, 3), n: int = 160, tag: str = "_survey",
            out_dir: str | Path | None = None) -> dict[str, Any]:
    """Trajectory summaries for the stage-B composing families and the contrasts, with entry state, onset class and disposition."""
    out: dict[str, Any] = {}
    for fam in a["stage_b_composing"] + a["contrasts"]:
        F = load_family(fam)
        if fam in SEPARATE_REPO_FAMILIES:
            mt, keys, table = merge_separate_repo_series(results_dir, F, grid, seeds, tag)
        else:
            mt = model_tag(F.repo)
            keys = [result_key(r) for r in F.refs(grid)]
            table = F.position_table(grid)
        gt = restrict_to_grid(results_dir, mt, keys, seeds, tag, grid)

        def pos(k, table=table, F=F):
            return table.get(k, F.position_of(k))

        r = crossings.summarize(results_dir, gt, seeds, tag, n, pos, descriptive=True)
        r.update({"family": fam, "grid": grid, "position_unit": F.position_unit})
        if out_dir is not None:
            write_json(r, Path(out_dir) / f"trajectory_{fam}_s{'-'.join(map(str, seeds))}.json")
        out[fam] = disposition(results_dir, gt, keys, seeds, tag, r, role="composes" if fam in a["stage_b_composing"] else "contrast", position_unit=F.position_unit)
    return out


def disposition(results_dir, gt: str, keys: Sequence[str], seeds: Sequence[int], tag: str, r: dict[str, Any], role: str, position_unit: str | None) -> dict[str, Any]:
    """Entry state from D's first interval, onset censoring class, strict-pass / failure / non-evaluable / counter-example."""
    x = np.array(r["stage1_checkpoints_B"])
    lo = np.array(r["D_ci_lo"])
    hi = np.array(r["D_ci_hi"])
    m = np.array(r["D_mean_series"])
    entry = "copy-dependent" if lo[0] > 0 else ("word-dependent" if hi[0] < 0 else "indeterminate")
    on = r.get("four_way_onset", {}).get("x")
    fw0 = np.mean([np.mean([int(i["acc"]) for i in crossings.rec(read_json(crossings.result_file(results_dir, "triage", gt, sd, tag))[keys[0]])["compose_h"]["four_way_items"]]) for sd in seeds])
    if on is None:
        onset = "left-censored" if fw0 > crossings.CHANCE_FOUR_WAY else "right-censored"
    else:
        onset = "left-censored" if on < x[0] else ("right-censored" if on > x[-1] else "interior")
    T = r["T"]
    tvals = [T.get("T1_marker_dependence_falls"), T.get("T2_never_returns"), T.get("T5_indifference_not_before_crossover"), T.get("T6_onset_after_crossover")]
    one_change = r["D_sign_changes_of_mean"] == 1
    never_negative = bool(np.all(m >= 0))
    first_neg = r["D_first_negative_mean"]
    if all(v is True for v in tvals) and one_change:
        disp = "strict_pass"
    elif any(v is False for v in tvals) or (all(v is not None for v in tvals) and not one_change):
        disp = "failure"
    else:
        disp = "non_evaluable"
    counter = None
    if onset == "interior" and never_negative:
        counter = "composition without an observed shift"
        disp = "counter_example"
    elif onset == "interior" and first_neg is not None and on is not None and on <= first_neg:
        counter = "timing counter-example (onset at or before D's first negative checkpoint)"
        disp = "counter_example"
    return {"model_tag": gt, "n_checkpoints": len(keys), "position_unit": position_unit, "role": role, "entry_state": entry, "onset_class": onset,
            "onset": r.get("four_way_onset"), "four_way_first_checkpoint": float(fw0), "T": T, "D_sign_changes": r["D_sign_changes_of_mean"],
            "D_first_negative": first_neg, "D_last_positive": r["D_last_positive_mean"], "disposition": disp, "counter_example": counter, "traj": r}


def cohort_flow(a: dict[str, Any], b: dict[str, Any] | None, reference_families: Sequence[str] = ("olmo3-7b",)) -> dict[str, Any]:
    """Which families were attempted, failed, composed, were traced; the verdict on whether the ordering recurs outside the
    reference families (the families whose earlier trajectories motivated the test)."""
    rows = a["rows"]

    def names(pred):
        return sorted(f for f, r in rows.items() if pred(r))

    flow: dict[str, Any] = {"attempted": sorted(rows), "technical_failure": names(lambda r: r["class"] == "technical_failure"),
                            "composes": names(lambda r: r["class"] == "composes"), "partial": names(lambda r: r["class"] == "partial"),
                            "models_that_copy": names(lambda r: r["class"] == "copies"), "stage_b_composing": a["stage_b_composing"],
                            "models_that_compose_series_unavailable": a["models_that_compose_series_unavailable"], "contrasts": a["contrasts"], "contrast_shortfall": a["contrast_shortfall"]}
    if b is not None:
        for k in ("entry_state", "onset_class", "disposition"):
            flow[k] = {fam: v[k] for fam, v in b.items()}
        new = [f for f, v in b.items() if v["role"] == "composes" and f not in reference_families]
        evaluable = [f for f in new if b[f]["disposition"] in ("strict_pass", "failure", "counter_example")]
        passes = [f for f in evaluable if b[f]["disposition"] == "strict_pass"]
        counters = [f for f, v in b.items() if v["disposition"] == "counter_example"]
        if counters:
            verdict = "counter-example: the universal ordering is falsified"
        elif not new:
            verdict = "no new models that compose: the screen does not broaden the trajectory evidence" if not a["models_that_compose_series_unavailable"] else "composes with an untraceable series: the screen is incomplete"
        elif not evaluable:
            verdict = "no independent test (no evaluable new composes)"
        elif len(passes) == len(evaluable):
            verdict = "supported: the ordering recurs in the named families (not universality)"
        else:
            verdict = "mixed: route-dependent ordering"
        flow["verdict"] = {"new_models_that_compose": new, "evaluable": evaluable, "strict_passes": passes, "counter_examples": counters, "verdict": verdict}
    return flow
