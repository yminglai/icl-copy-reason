"""The real-language check: the same withheld-combination task in Southern Quechua noun morphology (UniMorph `que`).

Paradigm: possessor x case, both realised by fixed suffix strings in the fixed order stem-POSS-CASE
every cell used equals the exact
concatenation lemma + POSS + CASE in the UniMorph rows (`verify_unimorph`). Only vowel-final lemmas
GEN, COM/DISTR/PRIV and
prefix-confusable marker pairs are excluded so that the four cells of a lemma are exactly stem + markerA + markerB.
  feature A (first suffix)  possessor  y 'my' | yki 'your' | n 'his' | nchik 'our' | nku 'their'
  feature B (second suffix) case       man 'to' | manta 'from' | pi 'in' | wan 'with' | kama 'until' | paq 'for' | rayku 'because'
Presentation is identical to the made-up-ending task (tasks.prompts): header, "<wordA> <wordB> <stem> = <form>" lines, two lines per
single-feature value and per shown combination, one combination withheld, the query on a new lemma
the copied candidate is the shown
combination sharing the query's possessor. Forms are real orthography (hyphen=True gives the segmented variant stem-A-B).

  gate     contamination check: zero-shot (named header, no demonstrations) and format-only demonstrations with disjoint markers;
           four-way likelihood accuracy must be at chance (.25) — schema of the released gate_<model>_s<seed>_real[h].json.
  measure  generation taxonomy + four-way likelihood and the eight attention-ablation sets — schema of triage_/knockout_<model>_s<seed>_real[h].json.
  report tables (`report_*`) reproduce the appendix tables of this check from those files.

Inputs : model, revision, seed, hyphen flag
the UniMorph `que` file for `verify_unimorph` (path from the config / data card).
Outputs: JSON files under out_dir with the same layout as the standard readings.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..io import read_json, write_json
from ..readings.ablation import scored_logprobs
from ..readings.bootstrap import wilson as _wilson
from ..readings.copy_share import generate, load_model, seq_logprob
from .prompts import Lang, MItem

HEADER = "In this language:\n"
HEADER_NAMED = "In Quechua (Runasimi):\n"
SEP = " = "
SETS = ["competitor", "single2", "sham", "otherbundles", "wordB", "competitor_marker", "wordA", "single2_marker"]


POSS = {"y": ("my", "PSS1S"), "yki": ("your", "PSS2S"), "n": ("his", "PSS3S"), "nchik": ("our", "PSS1PI"), "nku": ("their", "PSS3P")}
CASE = {"man": ("to", "DAT"), "manta": ("from", "ABL"), "pi": ("in", "ESS"), "wan": ("with", "INS"), "kama": ("until", "TERM"),
        "paq": ("for", "BEN"), "rayku": ("because", "CAUSV")}


def _pairs(pool: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    """unordered marker pairs where neither marker is a prefix of the other"""
    ks = list(pool)
    return [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:] if not (a.startswith(b) or b.startswith(a))]


POSS_PAIRS = _pairs(POSS)      # 7 pairs
CASE_PAIRS = _pairs(CASE)      # 20 pairs

# vowel-final lower-case noun lemmas of UniMorph que (256) minus function words / obvious loans / a lemma identical to a suffix
_EXCLUDE = {"hina", "ima", "ama", "ari", "grasya", "karru", "liwru", "ripuwlika", "simana", "yupanki", "haku", "pisi", "sapa", "kuska"}
LEMMAS = [w for w in """achachi achiwa aklla allpa allpaqa allqu allqucha alqa ama amu anka anku antanka anti apanqura api aqha ari arma
atillcha atipa awana awki awqa aycha ayllu ayni ayqi ayrampu chaki chala challa challwa chanka chaski chawpi chikchi chincha chinka chiqa
chuku chumpi chunta chuqi churi grasya haku hanka harawi hina hucha huqari ichhuna ikma illa illapa ima inti ispa kachi kallpa kanka karru
khachu khallki khata khuchi khuya kicha killa kina kinwa kuka kunka kunti kuraka kusi kuska kuti kututu layqa liwru llaki llalli llama
llanthu llaqta llasa lliklla lliphlli llulla lluqsi machu maki malli mallki mama manqu masi mikhuna milla mitma muya muyu nina pacha palla
pampa pani panta para paska patacha paylla phallcha phaqsa phuru phuti phuyu pikchu pirqa pirwa pisi pisqu pitu puchka pukara puna punku
puqu puti qalla qallu qaqa qara qata qawa qhari qhichwa qhincha qillqa qillqana qisa qulla qullpa qullqi quri qusi quwi quya raki rankhi
raphi raphra raqacha raymi ripuwlika riqsi rirpu rumi runa rupha ruru rutu saksa salla sallqa sama sansa sapa saphi simana simi sinchi
siqsi sisa suka sukha suni sunqu suti suyru suyu taka taki taku tampu tamya tanta taski tayta thanta thatki thiti tika tiksi tipi tiyana
tupa tupu tuqtu turi tuta uhu ukhu ullanta ullpu uma umalli unu upa upalla urma urpi urqu uruya usa ususi uturunku uya uywa wakcha waki
wallpa wamphu wampu wamra wanaku wantu wanu waqta waqya wasi waska wathiya watu watya wawa wawqi wayllu wayna wayra wayrachina wiksa wiqi
yapa yaqulla yaya yunka yupa yupana yupanki yupi yura yuri yuya""".split() if w not in _EXCLUDE]



# ----------------------------------------------------------------------------------------------------------------------- UniMorph check
def unimorph_tag(poss: str | None, case: str | None) -> str:
    p = POSS[poss][1] if poss else None
    c = CASE[case][1] if case else "NOM"
    parts = ["N"] + ([p] if p else []) + [c] + ([p] if (p and c == "BEN") else []) + ["SG"]   # UniMorph que repeats PSS after BEN/GEN
    return ";".join(parts)


def verify_unimorph(path: str | Path) -> dict:
    """check every (lemma, poss, case) cell used here against the UniMorph rows; returns counts (+ the rows for one example lemma)"""
    rows = {}
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) == 3:
            rows[(f[0], f[2])] = f[1]
    n_ok = n_bad = 0
    bad = []
    for lem in LEMMAS:
        for p in list(POSS) + [None]:
            for c in list(CASE) + [None]:
                if p is None and c is None:
                    continue
                tag = unimorph_tag(p, c)
                want = lem + (p or "") + (c or "")
                got = rows.get((lem, tag))
                if got == want:
                    n_ok += 1
                else:
                    n_bad += 1
                    bad.append((lem, tag, got, want))
    ex = sorted((tag, form) for (lem, tag), form in rows.items() if lem == "wasi" and tag in {unimorph_tag(p, c) for p in list(POSS) + [None] for c in list(CASE) + [None] if p or c})
    return {"n_lemmas": len(LEMMAS), "cells_checked": n_ok + n_bad, "cells_ok": n_ok, "cells_mismatch": n_bad, "mismatches": bad[:10], "example_rows_wasi": ex}


# ----------------------------------------------------------------------------------------------------------------------- items
class RealLang(Lang):
    """Lang with real-orthography concatenation (no hyphens unless hyphen=True)"""
    hyphen: bool = False

    def form(self, stem, ia, ib):
        affs = [aff for _, aff in self.slots(ia, ib)]
        return "-".join([stem] + affs) if self.hyphen else stem + "".join(affs)


def real_lang(lang: dict, hyphen: bool) -> RealLang:
    L = RealLang(**{k: v for k, v in lang.items() if k in Lang.__dataclass_fields__})
    L.hyphen = hyphen
    return L


def make_real_item(task: str, rng: random.Random, n_single=2, n_combo=2, hyphen=False, header=HEADER) -> MItem:
    """same interface as morph_tasks.make_item (compose_h / compose_s); lines_meta = [(stem, ia, ib)] in prompt order"""
    pa = rng.choice(POSS_PAIRS)
    pb = rng.choice(CASE_PAIRS)
    if rng.random() < .5:
        pa = pa[::-1]
    if rng.random() < .5:
        pb = pb[::-1]
    held = (rng.randrange(2), rng.randrange(2))
    lang = RealLang((POSS[pa[0]][0], POSS[pa[1]][0]), (CASE[pb[0]][0], CASE[pb[1]][0]), pa, pb, held, "ab", "real")
    lang.hyphen = hyphen
    stems = rng.sample(LEMMAS, 4 * n_single + 3 * n_combo + 1)
    cells = []
    for ia in range(2):
        cells += [(ia, None)] * n_single
    for ib in range(2):
        cells += [(None, ib)] * n_single
    for ia in range(2):
        for ib in range(2):
            if (ia, ib) != held:
                cells += [(ia, ib)] * n_combo
    order = list(range(len(cells)))
    rng.shuffle(order)
    meta = [(stems[i], cells[i][0], cells[i][1]) for i in order]
    lines = [f"{lang.phrase(s, ia, ib)}{SEP}{lang.form(s, ia, ib)}" for s, ia, ib in meta]
    stem = stems[-1]
    if task == "compose_h":
        ia, ib = held
    elif task == "compose_s":
        ia, ib = rng.choice([(a, b) for a in range(2) for b in range(2) if (a, b) != held])
    else:
        raise ValueError(task)
    ctx = header + "\n".join(lines) + "\n"
    prompt = ctx + lang.phrase(stem, ia, ib) + SEP.rstrip()
    it = MItem(task, prompt, lang.form(stem, ia, ib), stem, ia, ib, asdict(lang), [m[0] for m in meta])
    it.lines_meta = meta
    it.hyphen = hyphen
    it.header = header
    it.lang["hyphen"] = hyphen
    return it


def build_real(task: str, n: int, seed=0, **kw) -> list[MItem]:
    rng = random.Random(f"real|{task}|{seed}")
    return [make_real_item(task, rng, **kw) for _ in range(n)]


def four_forms_real(it: MItem):
    L = real_lang(it.lang, it.lang.get("hyphen", False))
    return [((a, b), L.form(it.stem, a, b)) for a in range(2) for b in range(2)]


def competitor_of(it: MItem) -> tuple[int, int]:
    return (it.ia, 1 - it.ib)          # first marker = possessor (feature a) always, order stem-POSS-CASE


def gate_prompts(it: MItem, rng: random.Random, kind: str, n_single=2, n_combo=2) -> str:
    """zs: named header + query only.  fmt: named header + demonstrations for OTHER lemmas with DISJOINT marker pairs (all their cells)."""
    lang = real_lang(it.lang, it.lang.get("hyphen", False))
    query = lang.phrase(it.stem, it.ia, it.ib) + SEP.rstrip()
    if kind == "zs":
        return HEADER_NAMED + query
    qa, qb = set(it.lang["affA"]), set(it.lang["affB"])
    pa = rng.choice([p for p in POSS_PAIRS if not (set(p) & qa)])
    pb = rng.choice([p for p in CASE_PAIRS if not (set(p) & qb)])
    dl = RealLang((POSS[pa[0]][0], POSS[pa[1]][0]), (CASE[pb[0]][0], CASE[pb[1]][0]), pa, pb, (-1, -1), "ab", "real")
    dl.hyphen = lang.hyphen
    stems = rng.sample([w for w in LEMMAS if w != it.stem], 4 * n_single + 4 * n_combo)
    cells = [(ia, None) for ia in range(2) for _ in range(n_single)] + [(None, ib) for ib in range(2) for _ in range(n_single)] + \
            [(ia, ib) for ia in range(2) for ib in range(2) for _ in range(n_combo)]
    rng.shuffle(cells)
    lines = [f"{dl.phrase(s, ia, ib)}{SEP}{dl.form(s, ia, ib)}" for s, (ia, ib) in zip(stems, cells)]
    return HEADER_NAMED + "\n".join(lines) + "\n" + query


# ----------------------------------------------------------------------------------------------------------------------- spans
def real_line_sets(it: MItem):
    """char spans in it.prompt for each knockout set (same semantics as morph_knockout.line_sets) + query start"""
    L = real_lang(it.lang, it.lang.get("hyphen", False))
    lines = it.prompt.split("\n")
    header, demos, query = lines[0], lines[1:-1], lines[-1]
    assert len(demos) == len(it.lines_meta)
    comp = competitor_of(it)
    spans = {k: [] for k in SETS}
    pos = len(header) + 1
    for d, (s, ia, ib) in zip(demos, it.lines_meta):
        span = (pos, pos + len(d))
        rhs_start = pos + len(L.phrase(s, ia, ib)) + len(SEP)
        rhs = d[len(L.phrase(s, ia, ib)) + len(SEP):]
        assert rhs == L.form(s, ia, ib)
        if ia is not None and ib is not None:
            if (ia, ib) == comp:
                spans["competitor"].append(span)
                m = it.lang["affB"][ib]
                off = len(rhs) - len(m)
                spans["competitor_marker"].append((rhs_start + off, rhs_start + off + len(m)))
            else:
                spans["otherbundles"].append(span)
        else:
            if ib is not None and ib == it.ib:            # single line carrying the correct SECOND marker
                spans["single2"].append(span)
                m = it.lang["affB"][ib]
                off = len(rhs) - len(m)
                spans["single2_marker"].append((rhs_start + off, rhs_start + off + len(m)))
            if ia is not None and ia == 1 - it.ia:        # sham: first feature's OTHER value
                spans["sham"].append(span)
        pos += len(d) + 1
    qstart = pos
    words = query.split(" ")
    spans["wordA"].append((qstart, qstart + len(words[0])))
    w1 = qstart + len(words[0]) + 1
    spans["wordB"].append((w1, w1 + len(words[1])))
    return spans, qstart


# ----------------------------------------------------------------------------------------------------------------------- scoring
def categorise_real(it: MItem, gen: str) -> str:
    """correct | competitor | other_attested | single_affix | stem_error | malformed (first whitespace word of the continuation)"""
    line = gen.strip().split("\n")[0].strip() if gen.strip() else ""
    w = line.split(" ")[0] if line else ""
    w = re.sub(r"[^\w\-]+$", "", w)
    hy = it.lang.get("hyphen", False)
    if hy:
        parts = w.split("-")
        stem, rest = parts[0], "".join(parts[1:])
        if stem != it.stem:
            return "stem_error"
        affs = parts[1:]
    else:
        if not w.startswith(it.stem):
            return "stem_error"
        rest = w[len(it.stem):]
        affs = None
    A, B = it.lang["affA"], it.lang["affB"]
    forms = {(a, b): A[a] + B[b] for a in range(2) for b in range(2)}
    if affs is not None and len(affs) == 1 and affs[0] in set(A) | set(B):
        return "single_affix"
    if affs is None and rest in set(A) | set(B):
        return "single_affix"
    key = [k for k, f in forms.items() if f == rest] if affs is None else [k for k, f in forms.items() if [A[k[0]], B[k[1]]] == affs]
    if not key:
        return "malformed"
    k = key[0]
    if k == (it.ia, it.ib):
        return "correct"
    return "competitor" if k == competitor_of(it) else "other_attested"


def wilson(k: int, n: int, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    lo, hi = _wilson(k, n, z)
    return (round(lo, 3), round(hi, 3))


def token_stats(lm, items) -> dict:
    """token counts of the four forms (in context, leading space) and marker/token boundary alignment of the competitor second marker"""
    nt, aligned, n_mark_tok = [], [], []
    for it in items[:64]:
        for _, f in four_forms_real(it):
            nt.append(len(lm.tok.encode(" " + f, add_special_tokens=False)))
        spans, _ = real_line_sets(it)
        enc = lm.tok(it.prompt, return_offsets_mapping=True, add_special_tokens=False)
        for s, e in spans["competitor_marker"]:
            toks = [(a, b) for a, b in enc["offset_mapping"] if a < e and b > s]
            n_mark_tok.append(len(toks))
            aligned.append(int(all(a >= s and b <= e for a, b in toks)))
    return {"form_tokens_mean": round(float(np.mean(nt)), 2), "form_tokens_max": int(max(nt)), "competitor_marker_tokens_mean": round(float(np.mean(n_mark_tok)), 2),
            "competitor_marker_token_aligned": round(float(np.mean(aligned)), 3)}


def four_way_lik(lm, items, prompts_fn, bs):
    """four-form full-likelihood scores; returns per-item dicts (acc, competitor, margin, a_null, best)"""
    per = []
    for b in range(0, len(items), bs):
        bp = items[b:b + bs]
        prompts, conts, tags = [], [], []
        for it in bp:
            p = prompts_fn(it)
            for bundle, form in four_forms_real(it):
                prompts.append(p)
                conts.append(form)
                tags.append(bundle)
        lps = seq_logprob(lm, prompts, conts)
        for j, it in enumerate(bp):
            cand = dict(zip(tags[4 * j:4 * j + 4], lps[4 * j:4 * j + 4]))
            tgt, comp = (it.ia, it.ib), competitor_of(it)
            best = max(cand, key=cand.get)
            margin = cand[tgt] - cand[comp]
            per.append({"acc": int(best == tgt), "competitor": int(best == comp), "margin": round(margin, 4), "a_null": round(float(1 / (1 + np.exp(margin))), 4),
                        "lp": {f"{a}{b}": round(v, 3) for (a, b), v in cand.items()}})
    return per


def summarise(per: list[dict]) -> dict:
    k = sum(x["acc"] for x in per)
    n = len(per)
    return {"four_way_acc": round(k / n, 3), "four_way_acc_ci": wilson(k, n), "competitor": round(float(np.mean([x["competitor"] for x in per])), 3),
            "margin": round(float(np.mean([x["margin"] for x in per])), 3), "a_null": round(float(np.mean([x["a_null"] for x in per])), 3), "n": n}


def load(model, rev, device=None):
    lm = load_model(model, revision=rev, device=device)
    lm.model.config._attn_implementation = "eager"
    lm.text_cfg._attn_implementation = "eager"
    return lm


def zs_generation(lm, items, bs) -> dict:
    """stricter recall check: zero-shot greedy generation (named header, no demonstrations); exact match of the held-out form, and whether the
    generated word carries the correct possessor / case suffix (stem + correct POSS + anything / stem + anything + correct CASE)."""
    prompts = [gate_prompts(it, random.Random(0), "zs") for it in items]
    gens = []
    for b in range(0, len(items), bs):
        gens += generate(lm, prompts[b:b + bs], max_new=10)
    exact = poss = case = stem = 0
    ex = []
    for it, g in zip(items, gens):
        w = re.sub(r"[^\w\-]+$", "", (g.strip().split("\n")[0].strip().split(" ") or [""])[0]).replace("-", "")
        A, B = it.lang["affA"][it.ia], it.lang["affB"][it.ib]
        exact += w == it.answer.replace("-", "")
        stem += w.startswith(it.stem)
        rest = w[len(it.stem):] if w.startswith(it.stem) else ""
        poss += rest.startswith(A)
        case += rest.endswith(B) and len(rest) > 0
        if len(ex) < 8:
            ex.append((it.answer, w))
    n = len(items)
    return {"exact": round(exact / n, 3), "exact_ci": wilson(exact, n), "stem_copied": round(stem / n, 3), "correct_poss_suffix": round(poss / n, 3),
            "correct_case_suffix": round(case / n, 3), "examples": ex}


def run_gate(lm, items, seed, bs, rec: dict | None = None) -> dict:
    rec = rec if rec is not None else {}
    rec.setdefault("tokens", token_stats(lm, items))
    rng = random.Random(f"gate|{seed}")
    for kind in ("zs", "fmt"):
        cache = {id(it): gate_prompts(it, rng, kind) for it in items}     # rng consumed in the same order as the original run
        if kind in rec:
            continue
        per = four_way_lik(lm, items, lambda it, cache=cache: cache[id(it)], bs)
        rec[kind] = summarise(per)
        rec[kind]["items"] = per
        rec[kind]["example_prompt"] = cache[id(items[0])]
    if "zs_gen" not in rec:
        rec["zs_gen"] = zs_generation(lm, items, bs)
    return rec


def run_main(lm, items, bs, ko_bs) -> tuple[dict, dict]:
    # --- triage-style: greedy generation taxonomy + four-way likelihood
    gens = []
    for b in range(0, len(items), bs):
        gens += generate(lm, [it.prompt for it in items[b:b + bs]], max_new=10)
    cats = [categorise_real(it, g) for it, g in zip(items, gens)]
    per4 = four_way_lik(lm, items, lambda it: it.prompt, bs)
    tri = {"n": len(items), "taxonomy": {c: round(cats.count(c) / len(cats), 3) for c in sorted(set(cats))},
           "copy": round(float(np.mean([c != "stem_error" for c in cats])), 3), "exact": round(float(np.mean([c == "correct" for c in cats])), 3),
           "items": [{"category": c, "gen": g.strip().split("\n")[0][:40]} for c, g in zip(cats, gens)],
           "four_way_acc": summarise(per4)["four_way_acc"], "four_way_competitor": summarise(per4)["competitor"], "four_way_margin": summarise(per4)["margin"],
           "a_null": summarise(per4)["a_null"], "four_way_items": per4, "examples": [(it.answer, g.strip().split("\n")[0][:40]) for it, g in list(zip(items, gens))[:5]],
           "tokens": token_stats(lm, items)}
    # --- knockouts
    conds = ["none"] + SETS
    per = {c: [] for c in conds}
    for b in range(0, len(items), ko_bs):
        bp = items[b:b + ko_bs]
        spans_all = [real_line_sets(it) for it in bp]
        for c in conds:
            prompts, conts, tags, spans_l, qs = [], [], [], [], []
            for it, (spans, qstart) in zip(bp, spans_all):
                for bundle, form in four_forms_real(it):
                    prompts.append(it.prompt)
                    conts.append(form)
                    tags.append(bundle)
                    spans_l.append(None if c == "none" else spans[c])
                    qs.append(qstart)
            lps = scored_logprobs(lm, prompts, conts, spans_l, qs)
            for j, it in enumerate(bp):
                cand = dict(zip(tags[4 * j:4 * j + 4], lps[4 * j:4 * j + 4]))
                tgt, comp = (it.ia, it.ib), competitor_of(it)
                best = max(cand, key=cand.get)
                margin = cand[tgt] - cand[comp]
                per[c].append({"acc": int(best == tgt), "competitor": int(best == comp), "margin": round(margin, 4), "a_null": round(float(1 / (1 + np.exp(margin))), 4)})
    ko = {"n": len(items), "scheme": "real", "sets": SETS}
    for c in conds:
        ko[c] = {"four_way_acc": round(float(np.mean([x["acc"] for x in per[c]])), 3), "competitor": round(float(np.mean([x["competitor"] for x in per[c]])), 3),
                 "margin": round(float(np.mean([x["margin"] for x in per[c]])), 3), "a_null": round(float(np.mean([x["a_null"] for x in per[c]])), 3), "items": per[c]}
    for c in SETS:
        d = np.array([x["margin"] for x in per[c]]) - np.array([x["margin"] for x in per["none"]])
        ko[c]["delta_margin"] = round(float(d.mean()), 3)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(d), (2000, len(d)))
        bs_ = d[idx].mean(1)
        ko[c]["delta_margin_ci"] = [round(float(np.percentile(bs_, 2.5)), 3), round(float(np.percentile(bs_, 97.5)), 3)]
    return tri, ko


def gate(model: str, steps: str = "main", n: int = 160, seed: int = 0, bs: int = 16, hyphen: bool = False, tag: str = "_real", out_dir: str | Path = ".",
         device: str | None = None) -> Path:
    """Contamination gate for `model`; resumable. Returns <out_dir>/gate_<model>_s<seed><tag>[h].json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = build_real("compose_h", n, seed=seed, hyphen=hyphen)
    mtag = model.split("/")[-1]
    hy = "h" if hyphen else ""
    rev = None if steps in ("main", "") else (steps if steps.startswith("stage") else f"step{steps}")
    key = rev or "main"
    t0 = time.time()
    outp = out_dir / f"gate_{mtag}_s{seed}{tag}{hy}.json"
    res = read_json(outp) if outp.exists() else {}
    if key in res and "zs_gen" in res[key]:
        return outp
    lm = load(model, rev, device)
    rec = run_gate(lm, items, seed, bs, res.get(key))
    rec["fingerprint"] = lm.fingerprint()
    rec["hyphen"] = hyphen
    rec["seconds"] = rec.get("seconds", 0) + round(time.time() - t0)
    res[key] = rec
    write_json(res, outp)
    print(f"GATE {mtag} {key}: zs 4-way {rec['zs']['four_way_acc']:.3f} {rec['zs']['four_way_acc_ci']} | fmt 4-way {rec['fmt']['four_way_acc']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    return outp


def measure(model: str, steps: str = "main", n: int = 160, seed: int = 0, bs: int = 16, ko_bs: int = 8, hyphen: bool = False, tag: str = "_real",
            out_dir: str | Path = ".", device: str | None = None) -> tuple[Path, Path]:
    """Generation taxonomy, four-way likelihood and the attention ablations; resumable. Returns (triage path, knockout path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = build_real("compose_h", n, seed=seed, hyphen=hyphen)
    mtag = model.split("/")[-1]
    hy = "h" if hyphen else ""
    rev = None if steps in ("main", "") else (steps if steps.startswith("stage") else f"step{steps}")
    key = rev or "main"
    t0 = time.time()
    outt = out_dir / f"triage_{mtag}_s{seed}{tag}{hy}.json"
    outk = out_dir / f"knockout_{mtag}_s{seed}{tag}{hy}.json"
    rt = read_json(outt) if outt.exists() else {}
    rk = read_json(outk) if outk.exists() else {}
    if key in rt and key in rk:
        return outt, outk
    lm = load(model, rev, device)
    tri, ko = run_main(lm, items, bs, ko_bs)
    fp = lm.fingerprint()
    tri.update({"fingerprint": fp, "hyphen": hyphen, "scheme": "real"})
    ko.update({"fingerprint": fp, "hyphen": hyphen, "seconds": round(time.time() - t0)})
    rt[key] = {"compose_h": tri, "fingerprint": fp}
    rk[key] = ko
    write_json(rt, outt)
    write_json(rk, outk)
    print(f"RUN {mtag} {key}: exact {tri['exact']:.2f} 4way {tri['four_way_acc']:.2f} A_null {tri['a_null']:.2f} [{time.time() - t0:.0f}s]", flush=True)
    return outt, outk


# ---------------------------------------------------------------- report tables of the real-language check
MODELS = [("pythia-1.4b", "Pythia-1.4B"), ("pythia-2.8b", "Pythia-2.8B"), ("pythia-12b", "Pythia-12B"), ("OLMo-2-1124-7B", "OLMo-2-7B"),
          ("OLMo-2-1124-13B", "OLMo-2-13B"), ("Qwen2.5-7B", "Qwen2.5-7B"), ("Qwen3-1.7B-Base", "Qwen3-1.7B-Base"), ("Qwen3-4B-Base", "Qwen3-4B-Base"),
          ("Qwen3-8B-Base", "Qwen3-8B-Base"), ("gemma-3-4b-pt", "Gemma-3-4B-pt")]
REPORT_SETS = ["competitor", "competitor_marker", "single2", "single2_marker", "sham", "otherbundles", "wordB", "wordA"]


REPORT_DIR: Path | None = None  # set by report() before the table functions run


def load_result(kind, model, seed, hy):
    p = Path(REPORT_DIR) / f"{kind}_{model}_s{seed}_real{hy}.json"
    if not p.exists():
        return None
    d = read_json(p)
    return d[list(d)[0]]


def summary_both(seeds):
    """compact side-by-side: real orthography vs hyphenated, pooled seeds"""
    print(f"### Summary: real orthography vs hyphenated (segmented) variant, {len(seeds)} seeds pooled (n = {160*len(seeds)}); made-up-ending reference values from the main fingerprint table\n")
    print("| model | nonce A_null | real: A_null | 4-way | exact gen | KO comp. marker | KO word B | hyphenated: A_null | 4-way | exact gen | KO comp. marker | KO word B |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    nonce = {"Pythia-1.4B": ".81", "Pythia-2.8B": ".79", "Pythia-12B": ".84", "OLMo-2-7B": ".24", "OLMo-2-13B": ".17", "Qwen2.5-7B": ".02", "Qwen3-1.7B-Base": ".11", "Qwen3-4B-Base": ".01", "Qwen3-8B-Base": ".00", "Gemma-3-4B-pt": ".11"}
    for m, name in MODELS:
        row = [name, nonce.get(name, "")]
        for hy in ("", "h"):
            tr = [load_result("triage", m, s, hy) for s in seeds]
            tr = [r["compose_h"] for r in tr if r]
            ko = [load_result("knockout", m, s, hy) for s in seeds]
            ko = [r for r in ko if r]
            if not tr or not ko:
                row += ["–"] * 5
                continue
            fw = sum((r["four_way_items"] for r in tr), [])
            cats = sum((r["items"] for r in tr), [])
            base = np.array(sum(([x["margin"] for x in r["none"]["items"]] for r in ko), []))
            d1 = np.array(sum(([x["margin"] for x in r["competitor_marker"]["items"]] for r in ko), [])) - base
            d2 = np.array(sum(([x["margin"] for x in r["wordB"]["items"]] for r in ko), [])) - base
            row += [f"{np.mean([x['a_null'] for x in fw]):.2f}", f"{np.mean([x['acc'] for x in fw]):.2f}", f"{np.mean([x['category']=='correct' for x in cats]):.2f}", f"{d1.mean():+.1f}", f"{d2.mean():+.1f}"]
        print("| " + " | ".join(row) + " |")


HF = {"pythia-1.4b": "EleutherAI/pythia-1.4b", "pythia-2.8b": "EleutherAI/pythia-2.8b", "pythia-12b": "EleutherAI/pythia-12b", "OLMo-2-1124-7B": "allenai/OLMo-2-1124-7B",
      "OLMo-2-1124-13B": "allenai/OLMo-2-1124-13B", "Qwen2.5-7B": "Qwen/Qwen2.5-7B", "Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B-Base", "Qwen3-4B-Base": "Qwen/Qwen3-4B-Base",
      "Qwen3-8B-Base": "Qwen/Qwen3-8B-Base", "gemma-3-4b-pt": "unsloth/gemma-3-4b-pt"}
_TOK = {}


def lennorm_acc(m, seeds, hyphen):
    """four-way accuracy with each form's summed log-prob divided by its token count (tokenised as ' '+form)"""
    from transformers import AutoTokenizer
    if m not in _TOK:
        _TOK[m] = AutoTokenizer.from_pretrained(HF[m])
    t = _TOK[m]
    acc = []
    for s in seeds:
        r = load_result("triage", m, s, "h" if hyphen else "")
        if r is None:
            continue
        items = build_real("compose_h", 160, seed=s, hyphen=hyphen)
        for it, x in zip(items, r["compose_h"]["four_way_items"]):
            ff = dict(four_forms_real(it))
            sc = {k: v / len(t.encode(" " + ff[(int(k[0]), int(k[1]))], add_special_tokens=False)) for k, v in x["lp"].items()}
            best = max(sc, key=sc.get)
            acc.append((int(best[0]), int(best[1])) == (it.ia, it.ib))
    return float(np.mean(acc)) if acc else float("nan")


def report(results_dir: str | Path, hyphen: bool = False, seeds=(0, 1), summary: bool = False) -> None:
    """Print the markdown tables of the real-language check (gate, pooled gate, fingerprint, four-way winner, attention ablations)."""
    global REPORT_DIR
    REPORT_DIR = Path(results_dir)
    a = type("A", (), {"hyphen": hyphen})()
    hy = "h" if hyphen else ""
    seeds = list(seeds)
    if summary:
        summary_both(seeds)
        return None
    if a.summary:
        summary_both(seeds)
        return None
    print(f"### Contamination gate ({'hyphenated' if hy else 'real orthography'}; seed 0, n = 160; four-way full-form likelihood; chance .25)\n")
    print("| model | zero-shot 4-way [Wilson 95%] | A_null | format-only demos (disjoint markers) 4-way [95%] | A_null | form tokens (mean/max) | competitor-marker tokens (mean) | marker/token aligned |")
    print("|---|---|---|---|---|---|---|---|")
    for m, name in MODELS:
        g = load_result("gate", m, 0, hy)
        if g is None:
            print(f"| {name} | – | | – | | | | |")
            continue
        t = g["tokens"]
        print(f"| {name} | {g['zs']['four_way_acc']:.3f} [{g['zs']['four_way_acc_ci'][0]:.2f}, {g['zs']['four_way_acc_ci'][1]:.2f}] | {g['zs']['a_null']:.2f} | "
              f"{g['fmt']['four_way_acc']:.3f} [{g['fmt']['four_way_acc_ci'][0]:.2f}, {g['fmt']['four_way_acc_ci'][1]:.2f}] | {g['fmt']['a_null']:.2f} | "
              f"{t['form_tokens_mean']:.1f}/{t['form_tokens_max']} | {t['competitor_marker_tokens_mean']:.2f} | {t['competitor_marker_token_aligned']:.2f} |")
    # pooled gate over seeds with per-slot decomposition (possessor slot / case slot; chance .5 each)
    print(f"\n### Contamination gate pooled over seeds {seeds} (n = {160*len(seeds)}): zero-shot four-way, per-slot accuracy (chance .5) and the model's marker preferences\n")
    print("| model | zs 4-way [95%] | zs possessor slot | zs case slot | fmt 4-way [95%] | fmt possessor slot | fmt case slot | most-chosen possessor (zs) | most-chosen case (zs) | zs free generation: exact / correct POSS suffix / correct CASE suffix |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m, name in MODELS:
        row = [name]
        pref = {}
        for kind in ("zs", "fmt"):
            acc, pa, pb, prefA, prefB = [], [], [], {}, {}
            for s in seeds:
                g = load_result("gate", m, s, hy)
                if g is None:
                    continue
                items = build_real("compose_h", 160, seed=s, hyphen=a.hyphen)
                for it, x in zip(items, g[kind]["items"]):
                    best = max(x["lp"], key=x["lp"].get)
                    ia, ib = int(best[0]), int(best[1])
                    acc.append(x["acc"])
                    pa.append(ia == it.ia)
                    pb.append(ib == it.ib)
                    prefA[it.lang["affA"][ia]] = prefA.get(it.lang["affA"][ia], 0) + 1
                    prefB[it.lang["affB"][ib]] = prefB.get(it.lang["affB"][ib], 0) + 1
            if not acc:
                row += ["–", "", ""]
                continue
            lo, hi = _wilson(sum(acc), len(acc))
            row += [f"{np.mean(acc):.3f} [{lo:.2f}, {hi:.2f}]", f"{np.mean(pa):.2f}", f"{np.mean(pb):.2f}"]
            pref[kind] = (max(prefA, key=prefA.get) + f" {max(prefA.values())/len(acc):.2f}", max(prefB, key=prefB.get) + f" {max(prefB.values())/len(acc):.2f}")
        row += list(pref.get("zs", ("", "")))
        zg = [load_result("gate", m, s, hy) for s in seeds]
        zg = [g["zs_gen"] for g in zg if g and "zs_gen" in g]
        row.append((f"{np.mean([g['exact'] for g in zg]):.3f} / {np.mean([g['correct_poss_suffix'] for g in zg]):.2f} / {np.mean([g['correct_case_suffix'] for g in zg]):.2f} ({len(zg)} seeds)") if zg else "–")
        print("| " + " | ".join(row) + " |")
    print(f"\n### Fingerprint ({len(seeds)} seeds pooled, n = {160*len(seeds)} held-out items; generation = greedy, 10 new tokens)\n")
    print("| model | exact (gen) | 4-way acc [95%] | 4-way acc, length-normalised | competitor rate (4-way) | A_null | taxonomy: correct / competitor / other attested / single affix / stem error / malformed |")
    print("|---|---|---|---|---|---|---|")
    for m, name in MODELS:
        recs = [load_result("triage", m, s, hy) for s in seeds]
        recs = [r["compose_h"] for r in recs if r]
        if not recs:
            print(f"| {name} | – | | | | |")
            continue
        fw = sum((r["four_way_items"] for r in recs), [])
        cats = sum((r["items"] for r in recs), [])
        n = len(fw)
        k = sum(x["acc"] for x in fw)
        lo, hi = _wilson(k, n)
        tax = {c: np.mean([x["category"] == c for x in cats]) for c in ["correct", "competitor", "other_attested", "single_affix", "stem_error", "malformed"]}
        ln = lennorm_acc(m, seeds, a.hyphen)
        print(f"| {name} | {tax['correct']:.2f} | {k/n:.2f} [{lo:.2f}, {hi:.2f}] | {ln:.2f} | {np.mean([x['competitor'] for x in fw]):.2f} | {np.mean([x['a_null'] for x in fw]):.2f} | "
              + " / ".join(f"{tax[c]:.2f}" for c in ["correct", "competitor", "other_attested", "single_affix", "stem_error", "malformed"]) + " |")
    print(f"\n### Four-way winner ({len(seeds)} seeds pooled): which of the four forms has the highest full-form likelihood\n")
    print("| model | correct (held-out) | competitor = attested, shares FIRST marker (possessor) | attested, shares SECOND marker (case) | attested, shares NEITHER marker | A_null (correct vs competitor) | A_null2 (correct vs case-sharing) |")
    print("|---|---|---|---|---|---|---|")
    for m, name in MODELS:
        recs = [(s, load_result("triage", m, s, hy)) for s in seeds]
        recs = [(s, r["compose_h"]) for s, r in recs if r]
        if not recs:
            print(f"| {name} | – | | | | | |")
            continue
        w = {"correct": 0, "comp": 0, "case": 0, "neither": 0}
        an2 = []
        n = 0
        for s, r in recs:
            items = build_real("compose_h", 160, seed=s, hyphen=a.hyphen)
            for it, x in zip(items, r["four_way_items"]):
                best = max(x["lp"], key=x["lp"].get)
                ia, ib = int(best[0]), int(best[1])
                k = "correct" if (ia, ib) == (it.ia, it.ib) else "comp" if ia == it.ia else "case" if ib == it.ib else "neither"
                w[k] += 1
                n += 1
                m2 = x["lp"][f"{it.ia}{it.ib}"] - x["lp"][f"{1-it.ia}{it.ib}"]
                an2.append(1 / (1 + np.exp(m2)))
        an = np.mean(sum(([x["a_null"] for x in r["four_way_items"]] for _, r in recs), []))
        print(f"| {name} | {w['correct']/n:.2f} | {w['comp']/n:.2f} | {w['case']/n:.2f} | {w['neither']/n:.2f} | {an:.2f} | {np.mean(an2):.2f} |")
    print(f"\n### Attention knockouts: Δmargin = change in log-odds(correct vs competitor) vs no knockout, {len(seeds)} seeds pooled, item bootstrap 95% CI\n")
    print("| model | A_null (base) | " + " | ".join(f"KO {s}" for s in REPORT_SETS) + " | GPU min |")
    print("|---|---|" + "---|" * len(REPORT_SETS) + "---|")
    for m, name in MODELS:
        recs = [load_result("knockout", m, s, hy) for s in seeds]
        recs = [r for r in recs if r]
        if not recs:
            print(f"| {name} | – |" + " |" * len(REPORT_SETS) + " |")
            continue
        base = np.array(sum(([x["margin"] for x in r["none"]["items"]] for r in recs), []))
        cells = []
        for s in REPORT_SETS:
            d = np.array(sum(([x["margin"] for x in r[s]["items"]] for r in recs), [])) - base
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(d), (2000, len(d)))
            bs = d[idx].mean(1)
            cells.append(f"{d.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
        an = np.mean(sum(([x["a_null"] for x in r["none"]["items"]] for r in recs), []))
        print(f"| {name} | {an:.2f} | " + " | ".join(cells) + f" | {sum(r.get('seconds', 0) for r in recs)/60:.1f} |")
