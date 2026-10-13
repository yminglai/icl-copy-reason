"""Per-checkpoint statistics of a traced run and the sign-change interval of the two input dependences.

Inputs : attention-ablation result files (one per item seed; keys = revisions; each revision holds the conditions `none`,
         `competitor_marker`, `wordB`, `wordA`, `single2_marker`, `sham`, `competitor`, ... with per-item `margin` and `a_null`)
         and, optionally, copy-share result files (`compose_h.four_way_items` with per-item `acc` / `competitor`).
Outputs: the trajectory summary (schema of the released `trajectory_*.json` summaries): per-checkpoint means with stratified
         item-bootstrap intervals, the D = Δ(competitor marker) + Δ(queried word) series, the last positive / first negative
         checkpoint of its mean, the null-adherence release, the fitted four-way onset, the descriptors T1-T6, a joint bootstrap
         of the interval; the compound-event tests (H4 shift interval, H5 two-step release) for the one run they were fixed for;
         the first-crossing table (`first_crossings`) and the margin-conditioned D (`d_conditioned`).
Definitions. Per-item paired Δmargin = margin(ablation) - margin(none) on the same item (items aligned by position; the generator
is deterministic per seed). D = Δ(competitor_marker) + Δ(wordB). Intervals: item bootstrap stratified within seed, numpy
default_rng(0), 2000 resamples, percentile 95 %. Only token-labelled checkpoints enter the trajectory statistics; the released
model is reported separately. Fail-closed: a missing seed, checkpoint, condition or a wrong item count raises instead of
producing a summary.
"""
from __future__ import annotations

import collections
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..io import read_json
from .onsets import fit_onset, fit_onset_positions

BOOTSTRAP = 2000
ONSET_BOOTSTRAP = 300
REQUIRED_CONDS = ["competitor_marker", "wordB", "wordA", "single2_marker", "sham", "competitor"]
RELEASE_HI, RELEASE_LO = 0.80, 0.55
CHANCE_FOUR_WAY = 0.25

# compound-event thresholds fixed before the OLMo-2-7B run of record (kept for the released summary's H4 / H5 fields)
H4_POSITIVE_UPTO = 336
H4_NEGATIVE_FROM = 630
H4_ANULL_INDIFF_MAX_TOKENS = 1049
H4_JOINT_PROB = 0.90
H5_OTHER_SHARE_MIN = 0.35
H5_OTHER_SHARE_LATE_MAX = 0.30
H5_DROP_MIN = 0.15
H5_ONSET_RATIO_MIN = 2.0
H5_SUSTAINED = 0.35


class TrajectoryError(RuntimeError):
    """Raised instead of producing a summary when an input is missing or malformed."""


def rec(d: dict) -> dict:
    return d["main"] if isinstance(d, dict) and "main" in d else d


class Panel:
    """Per-item arrays for one (run, checkpoint), items stacked by seed with stratum labels."""

    def __init__(self, recs: Sequence[dict], seeds: Sequence[int], n: int):
        self.seeds = list(seeds)
        self.n = n
        for s, r in zip(seeds, recs):
            for c in ["none"] + REQUIRED_CONDS:
                if c not in r or "items" not in r[c]:
                    raise TrajectoryError(f"condition {c} missing for seed {s}")
                if len(r[c]["items"]) != n:
                    raise TrajectoryError(f"seed {s} condition {c} has {len(r[c]['items'])} items, expected {n}")
        self.base = np.concatenate([[float(it["margin"]) for it in r["none"]["items"]] for r in recs])
        self.anull = np.concatenate([[float(it["a_null"]) for it in r["none"]["items"]] for r in recs])
        self.delta = {c: np.concatenate([np.array([float(it["margin"]) for it in r[c]["items"]]) - np.array([float(it["margin"]) for it in r["none"]["items"]]) for r in recs])
                      for c in REQUIRED_CONDS}
        self.D = self.delta["competitor_marker"] + self.delta["wordB"]
        self.strata = np.concatenate([[i] * n for i in range(len(recs))])


def strat_idx(rng: np.random.Generator, n_seeds: int, n: int) -> np.ndarray:
    """One stratified resample: indices into the stacked arrays, each seed's block resampled within itself."""
    return np.concatenate([i * n + rng.integers(0, n, n) for i in range(n_seeds)])


def mean_ci(x: Sequence[float], n_seeds: int, n: int, B: int = BOOTSTRAP) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    x = np.asarray(x, float)
    bs = np.array([x[strat_idx(rng, n_seeds, n)].mean() for _ in range(B)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def panel_stats(p: Panel) -> dict[str, Any]:
    k = (len(p.seeds), p.n)
    st: dict[str, Any] = {"n": int(len(p.base)), "a_null": mean_ci(p.anull, *k), "base_margin": mean_ci(p.base, *k), "D": mean_ci(p.D, *k),
                          "per_seed": {str(s): {"a_null": float(p.anull[p.strata == i].mean()), "D": float(p.D[p.strata == i].mean())} for i, s in enumerate(p.seeds)}}
    for c in REQUIRED_CONDS:
        st[c] = mean_ci(p.delta[c], *k)
    st["wordA_minus_wordB"] = mean_ci(p.delta["wordA"] - p.delta["wordB"], *k)
    return st


def crossing_events(x: np.ndarray, Dmat: np.ndarray):
    """x: checkpoint positions (sorted); Dmat: items x checkpoints. On the mean series: last positive, first negative after it,
    whether it never returns to positive, and the number of sign changes."""
    m = Dmat.mean(0)
    pos = x[m > 0]
    neg = x[m < 0]
    last_pos = float(pos.max()) if len(pos) else None
    cand = [v for v in neg if last_pos is None or v > last_pos]
    first_neg = float(min(cand)) if cand else None
    no_return = bool(first_neg is not None and np.all(m[x >= first_neg] < 0))
    return last_pos, first_neg, no_return, int(np.sum(np.sign(m[1:]) != np.sign(m[:-1])))


def result_file(results_dir: str | Path, kind: str, model_tag: str, seed: int, tag: str) -> Path:
    """The per-run result file; a gzipped copy (`.json.gz`, as the released readings ship) is accepted when the plain file is absent."""
    p = Path(results_dir) / f"{kind}_{model_tag}_s{seed}{tag}.json"
    gz = p.with_suffix(".json.gz")
    return gz if (not p.exists() and gz.exists()) else p


def load_seed_files(results_dir: str | Path, kind: str, model_tag: str, seeds: Sequence[int], tag: str) -> list[dict]:
    out = []
    for s in seeds:
        f = result_file(results_dir, kind, model_tag, s, tag)
        if not f.exists():
            raise TrajectoryError(f"missing {f}")
        out.append(read_json(f))
    return out


def summarize(results_dir: str | Path, model_tag: str, seeds: Sequence[int], tag: str, n: int, position_of: Callable[[str], float | None],
              descriptive: bool = True, copy_share_tag: str | None = None, B: int = BOOTSTRAP, onset_B: int = ONSET_BOOTSTRAP) -> dict[str, Any]:
    """The trajectory summary of one run from its ablation (+ copy-share) files. `position_of` maps a revision key to its training
    position (None = released model). descriptive=True writes the descriptors T1-T6 and the joint bootstrap of the interval;
    descriptive=False evaluates the compound events H4 / H5 fixed for the OLMo-2-7B run."""
    ko = load_seed_files(results_dir, "knockout", model_tag, seeds, tag)
    ttag = tag if copy_share_tag is None else copy_share_tag
    have_triage = all(result_file(results_dir, "triage", model_tag, s, ttag).exists() for s in seeds)
    if not descriptive and not have_triage:
        raise TrajectoryError("the compound events need the copy-share files")
    tr = load_seed_files(results_dir, "triage", model_tag, seeds, ttag) if have_triage else []
    keys = [k for k in ko[0] if all(k in d for d in ko)]
    stage = sorted([k for k in keys if position_of(k) is not None], key=position_of)
    if not stage:
        raise TrajectoryError("no position-labelled checkpoints")
    x = np.array([position_of(k) for k in stage])
    panels = {k: Panel([rec(d[k]) for d in ko], seeds, n) for k in keys}
    rows = []
    for k in keys:
        st = panel_stats(panels[k])
        st["key"] = k
        st["x"] = position_of(k)
        st["released_model"] = position_of(k) is None
        items4 = [[(int(i["acc"]), int(i["competitor"])) for i in rec(t[k])["compose_h"]["four_way_items"]] for t in tr if k in t]
        if items4:
            if len(items4) != len(seeds) or any(len(v) != n for v in items4):
                raise TrajectoryError(f"copy-share {k} must have {len(seeds)} seeds x {n} items")
            a = np.array(items4).reshape(-1, 2)
            st["four_way_acc"] = float(a[:, 0].mean())
            st["other_marker_share"] = float((1 - a[:, 0] - a[:, 1]).mean())
            st["_acc_items"] = a[:, 0]
            st["_other_items"] = 1 - a[:, 0] - a[:, 1]
        rows.append(st)
    byk = {r["key"]: r for r in rows}
    Dmat = np.stack([panels[k].D for k in stage], 1)
    An = np.stack([panels[k].anull for k in stage], 1)
    last_pos, first_neg, no_return, nchg = crossing_events(x, Dmat)
    ci_lo = np.array([byk[k]["D"][1] for k in stage])
    ci_hi = np.array([byk[k]["D"][2] for k in stage])
    an_mean = An.mean(0)
    first_null = next((float(v) for v, a in zip(x, an_mean) if a >= RELEASE_HI), None)
    release = None if first_null is None else next((float(v) for v, a in zip(x, an_mean) if a <= RELEASE_LO and v > first_null), None)
    out: dict[str, Any] = {"model": model_tag, "tag": tag, "seeds": list(seeds), "n_per_seed": n, "stage1_checkpoints_B": x.tolist(),
                           "released_model_reported_separately": {k: {"D": byk[k]["D"], "a_null": byk[k]["a_null"]} for k in keys if position_of(k) is None},
                           "D_mean_series": Dmat.mean(0).tolist(), "D_ci_lo": ci_lo.tolist(), "D_ci_hi": ci_hi.tolist(),
                           "D_last_positive_mean": last_pos, "D_first_negative_mean": first_neg, "D_no_return": no_return, "D_sign_changes_of_mean": nchg,
                           "anull": {"first_ge_.80": first_null, "release_le_.55_after_null": release}}
    have4 = [k for k in stage if "_acc_items" in byk[k]]
    if have4:
        out["four_way_onset"] = fit_onset_positions(np.array([position_of(k) for k in have4]), [byk[k]["_acc_items"] for k in have4], CHANCE_FOUR_WAY, B=onset_B)
    rng = np.random.default_rng(0)
    n_seeds = len(seeds)
    if descriptive:
        mk = np.array([byk[k]["competitor_marker"][0] for k in stage])
        wb = np.abs(np.array([byk[k]["wordB"][0] for k in stage]))
        peak = float(mk.max())
        after = mk[x >= first_neg] if first_neg is not None else None
        onset_x = out.get("four_way_onset", {}).get("x")
        out["T"] = {"T1_marker_dependence_falls": bool(mk[-1] < 0.5 * peak), "T1_ratio_last_over_peak": float(mk[-1] / peak) if peak > 0 else None,
                    "T2_never_returns": None if after is None else bool(np.all(after < 0.8 * peak)),
                    "T3_word_dependence_grows": bool(wb[-1] > 2 * wb.min()), "T3_ratio_last_over_min": float(wb[-1] / wb.min()) if wb.min() > 0 else None,
                    "T4_single_sign_change": bool(nchg <= 1),
                    "T5_indifference_not_before_crossover": None if (release is None or last_pos is None) else bool(release >= last_pos),
                    "T6_onset_after_crossover": None if (first_neg is None or onset_x is None) else bool(onset_x > first_neg)}
        ints = []
        for _ in range(B):
            idx = strat_idx(rng, n_seeds, n)
            lp, fn, nr, _ = crossing_events(x, Dmat[idx])
            ints.append((lp, fn, nr))
        cnt = collections.Counter((lp, fn) for lp, fn, nr in ints)
        out["joint_bootstrap"] = {"P_no_sign_change": float(np.mean([lp is None or fn is None for lp, fn, nr in ints])),
                                  "interval_distribution": {f"{lp}-{fn}": c / B for (lp, fn), c in sorted(cnt.items(), key=lambda t: -t[1])[:8]},
                                  "P_no_return": float(np.mean([nr for _, _, nr in ints]))}
    else:
        out.update(_compound_events(x, Dmat, An, ci_lo, ci_hi, release, last_pos, first_neg, byk, have4, position_of, out, rng, n_seeds, n, B, onset_B))
    for r in rows:
        r.pop("_acc_items", None)
        r.pop("_other_items", None)
    out["rows"] = rows
    return out


def _compound_events(x, Dmat, An, ci_lo, ci_hi, release, last_pos, first_neg, byk, have4, position_of, out, rng, n_seeds, n, B, onset_B) -> dict[str, Any]:
    early = x <= H4_POSITIVE_UPTO
    late = x >= H4_NEGATIVE_FROM
    h4_point = bool(early.any() and late.any() and np.all(ci_lo[early] > 0) and np.all(ci_hi[late] < 0) and release is not None and release <= H4_ANULL_INDIFF_MAX_TOKENS)
    ev = []
    for _ in range(B):
        idx = strat_idx(rng, n_seeds, n)
        m = Dmat[idx].mean(0)
        a = An[idx].mean(0)
        fnull = next((v for v, aa in zip(x, a) if aa >= RELEASE_HI), None)
        rel = None if fnull is None else next((v for v, aa in zip(x, a) if aa <= RELEASE_LO and v > fnull), None)
        ev.append(bool(np.all(m[early] > 0) and np.all(m[late] < 0) and rel is not None and rel <= H4_ANULL_INDIFF_MAX_TOKENS))
    p4 = float(np.mean(ev))
    res = {"H4_crossover": {"pass": bool(h4_point and p4 >= H4_JOINT_PROB), "pointwise": h4_point, "joint_probability": p4, "interval": [last_pos, first_neg], "release_tokens": release}}
    mid = [k for k in have4 if 630 <= position_of(k) <= 1700]
    late4 = [k for k in have4 if position_of(k) >= 2500]
    last3 = have4[-3:]
    if not (mid and late4 and len(last3) == 3 and "four_way_onset" in out):
        raise TrajectoryError("H5 needs four-way records in the 630B-1.7T window, at >= 2.5T and at the last three checkpoints")

    def share(ks):
        return float(np.mean([byk[k]["other_marker_share"] for k in ks]))

    h5a = share(mid) >= H5_OTHER_SHARE_MIN
    h5b = share(late4) <= H5_OTHER_SHARE_LATE_MAX and share(mid) - share(late4) >= H5_DROP_MIN
    on = out["four_way_onset"]["x"]
    h5c = bool(on is not None and on / H4_NEGATIVE_FROM >= H5_ONSET_RATIO_MIN)
    h5d = all(byk[k]["four_way_acc"] >= H5_SUSTAINED for k in last3)
    acc = np.stack([byk[k]["_acc_items"] for k in have4], 1)
    oth = np.stack([byk[k]["_other_items"] for k in have4], 1)
    x4 = np.array([position_of(k) for k in have4])
    midm = np.array([k in mid for k in have4])
    latem = np.array([k in late4 for k in have4])
    l3m = np.array([k in last3 for k in have4])
    ev5 = []
    for _ in range(onset_B):
        idx = strat_idx(rng, n_seeds, n)
        o = oth[idx].mean(0)
        a = acc[idx].mean(0)
        xc = fit_onset(np.log10(x4), [acc[idx][:, j] for j in range(len(have4))], CHANCE_FOUR_WAY, B=1)[0]
        onb = 10**xc if np.isfinite(xc) else None
        ev5.append(bool(o[midm].mean() >= H5_OTHER_SHARE_MIN and o[latem].mean() <= H5_OTHER_SHARE_LATE_MAX and o[midm].mean() - o[latem].mean() >= H5_DROP_MIN
                        and onb is not None and onb / H4_NEGATIVE_FROM >= H5_ONSET_RATIO_MIN and np.all(a[l3m] >= H5_SUSTAINED)))
    res["H5_two_step_release"] = {"pass": bool(h5a and h5b and h5c and h5d), "mid_other_share": share(mid), "late_other_share": share(late4),
                                  "h5a_mid_ge_.35": bool(h5a), "h5b_late_le_.30_drop_ge_.15": bool(h5b), "h5c_onset_ratio_ge_2": h5c,
                                  "h5d_sustained_last3_ge_.35": bool(h5d), "joint_probability": float(np.mean(ev5))}
    return res


def summary_markdown(obj: dict[str, Any], title: str) -> str:
    import json

    lines = [f"# {title}", ""]
    for k, v in obj.items():
        if k in ("rows", "stats", "D_mean_series", "D_ci_lo", "D_ci_hi"):
            continue
        lines.append(f"- **{k}**: `{json.dumps(v, default=float)[:800]}`")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- derived tables
def first_crossings(summary: dict[str, Any], label: str, min_position: float = 5.0) -> dict[str, Any]:
    """First sampled checkpoint (at or after `min_position`) at which null adherence falls below .8 and below .5, D first turns
    negative and four-way accuracy first exceeds chance, plus the fitted onset; released models excluded."""
    def val(v):
        return v[0] if isinstance(v, (list, tuple)) else v

    rows = sorted([r for r in summary["rows"] if not r.get("released_model") and r.get("x")], key=lambda r: r["x"])
    an = [(r["x"], val(r["a_null"])) for r in rows]
    fw = [(r["x"], val(r.get("four_way_acc"))) for r in rows]
    D = [(r["x"], val(r["D"])) for r in rows]
    x = [r["x"] for r in rows]

    def first(seq, pred):
        return next((xx for xx, v in seq if v is not None and pred(v) and xx >= min_position), None)

    onset = summary.get("four_way_onset", {}).get("x") if isinstance(summary.get("four_way_onset"), dict) else None
    cens = summary.get("four_way_onset", {}).get("censored_fraction") if isinstance(summary.get("four_way_onset"), dict) else None
    return dict(label=label, n=len(x), span=[x[0], x[-1]], a08=first(an, lambda a: a < 0.8), a05=first(an, lambda a: a < 0.5), dneg=first(D, lambda v: v < 0),
                fw25=first(fw, lambda v: v > CHANCE_FOUR_WAY), fitted_onset_B=onset, onset_censored_fraction=cens,
                min_a_null=min(a for xx, a in an if xx >= min_position), last_four_way=fw[-1][1], position_unit=summary.get("position_unit", "tokens_B"))


def d_conditioned(variant_files: Sequence[str | Path], band: float = 1.0, min_band: int = 10) -> dict[str, Any]:
    """D per checkpoint pooled over the item seeds of float32 re-scoring files (per-item `sum` log-probabilities): over all items,
    restricted to items whose baseline margin is within `band` nats of zero, and as the intercept of a least-squares fit of the
    per-item d on the per-item margin (D at m = 0). Answers whether D's sign change merely tracks the margin's own sign change."""
    recs = [read_json(f) for f in variant_files]
    keys = sorted([k for k in recs[0] if "tokens" in k and all(k in r for r in recs)], key=lambda k: int(re.search(r"tokens(\d+)B", k).group(1)))
    rows = []
    for k in keys:
        m = np.concatenate([[it["sum"] for it in r[k]["none"]["items"]] for r in recs])
        R = np.concatenate([[a["sum"] - b["sum"] for a, b in zip(r[k]["competitor_marker"]["items"], r[k]["none"]["items"])] for r in recs])
        W = np.concatenate([[a["sum"] - b["sum"] for a, b in zip(r[k]["wordB"]["items"], r[k]["none"]["items"])] for r in recs])
        D = R + W
        inband = np.abs(m) < band
        A = np.vstack([np.ones_like(m), m]).T
        coef = np.linalg.lstsq(A, D, rcond=None)[0]
        rows.append(dict(tokens_B=int(re.search(r"tokens(\d+)B", k).group(1)), base_mean=float(m.mean()), D_all=float(D.mean()), n_band=int(inband.sum()),
                         D_band=float(D[inband].mean()) if inband.sum() >= min_band else None, D_at_m0=float(coef[0]), slope=float(coef[1])))

    def first(key):
        return next((r["tokens_B"] for r in rows if r[key] is not None and r[key] < 0), None)

    return {"rows": rows, "summary": {"first_negative_all_B": first("D_all"), "first_negative_band_B": first("D_band"), "first_negative_at_m0_B": first("D_at_m0")}}


def seed_of(path: str | Path) -> str | None:
    m = re.search(r"_s(\d)[_.]", os.path.basename(str(path)))
    return m.group(1) if m else None
