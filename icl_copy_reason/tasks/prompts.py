"""Diagnostic prompt generator: an in-context word-formation task with made-up endings and one withheld combination.

Each item is its own small language (endings re-sampled per item so nothing can be memorised): two binary features
  A (size)   words wA = (a0, a1)  e.g. ("small", "big")   -> endings affA[0], affA[1]
  B (number) words wB = (b0, b1)  e.g. ("one", "many")    -> endings affB[0], affB[1]
Surface form: "<wordA> <wordB> <stem> = <stem>-<affA>-<affB>" (fixed order A then B)
single-feature lines use one
feature ("<wordA> <stem> = <stem>-<affA>"). The demonstrations cover all four single-feature forms and THREE of the four
combinations
the withheld combination never appears in the demonstrations. Stems are pronounceable made-up strings,
never repeated within a prompt.

Query types (always on a new stem):
  bind_a / bind_b  "<wordA> <stem> ="         -> stem-affA            (one feature)
  compose_s        "<wordA> <wordB> <stem> =" -> stem-affA-affB       a combination SHOWN in the demonstrations (other stems)
  compose_h        "<wordA> <wordB> <stem> =" -> stem-affA-affB       the WITHHELD combination
The copied candidate of a withheld query is the shown combination that shares the first ending (categorise -> "competitor").

Schemes: "morph" (suffixes on a stem) and "code" (attribute words -> a two-symbol code, no stem; one line per form).
Templates: "default" and "alt" (other header, separator, word pairs and ending pool), switched with set_template().
Controls: order="ba" realises the endings B then A
n_single / n_combo set the demonstration counts.

Inputs : task name, number of items, seed and the controls above. Outputs: MItem records (prompt, answer, stem, feature
values, language, demonstration stems, and the donor / counterfactual fields used by activation patching).
Scoring helpers: parse_form, score (copy / exact / per-ending), categorise (taxonomy), score_cf (patching), four_forms.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

AFFIX_POOL = ["il", "nak", "or", "um", "esh", "oth", "ka", "tu", "ri", "mo", "lin", "dor", "sa", "ne", "po", "ta", "ku", "zi",
              "fen", "gor", "lu", "wa", "ex", "an", "ir", "on", "ul", "ek", "ip", "ab"]
WORDS_A = [("small", "big"), ("old", "new"), ("red", "blue"), ("hot", "cold")]
WORDS_B = [("one", "many"), ("here", "there"), ("mine", "yours")]
ONSETS = ["bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "kl", "kr", "pl", "pr", "sk", "sl", "sm", "sn", "sp", "st", "sw",
          "tr", "tw", "d", "f", "g", "k", "m", "n", "p", "r", "t", "v", "w", "z", "b", "l"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ee", "oo", "ou"]
CODAS = ["b", "ck", "d", "f", "g", "k", "l", "m", "n", "p", "rn", "sh", "t", "th", "v", "x", "z", "lt", "mp", "nd", "rb", "sk"]


def nonce_stem(rng: random.Random) -> str:
    return rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS)


@dataclass
class Lang:
    wA: tuple[str, str]
    wB: tuple[str, str]
    affA: tuple[str, str]
    affB: tuple[str, str]
    held: tuple[int, int]          # held-out bundle (ia, ib)
    order: str = "ab"              # realisation order of the suffixes: "ab" = stem-A-B, "ba" = stem-B-A (word order stays A B)
    scheme: str = "morph"          # "morph": stem-affA-affB suffixing; "code": attribute words → symbol code "symA symB" (no stem)

    def slots(self, ia: int | None, ib: int | None) -> list[tuple[str, str]]:
        """realised affix slots in surface order: [(feature, affix), ...]"""
        parts = []
        if ia is not None:
            parts.append(("a", self.affA[ia]))
        if ib is not None:
            parts.append(("b", self.affB[ib]))
        if self.order == "ba":
            parts = parts[::-1]
        return parts

    def form(self, stem: str, ia: int | None, ib: int | None) -> str:
        if self.scheme == "code":
            return " ".join(aff for _, aff in self.slots(ia, ib))
        return "-".join([stem] + [aff for _, aff in self.slots(ia, ib)])

    def phrase(self, stem: str, ia: int | None, ib: int | None) -> str:
        words = ([self.wA[ia]] if ia is not None else []) + ([self.wB[ib]] if ib is not None else [])
        if self.scheme == "code":
            return " ".join(words)
        return " ".join(words + [stem])


@dataclass
class MItem:
    task: str                      # bind_a | bind_b | compose_s | compose_h
    prompt: str                    # ends with " =" ; answer follows as " <form>"
    answer: str                    # e.g. "blick-il-nak"
    stem: str
    ia: int | None
    ib: int | None
    lang: dict
    demo_stems: list[str]
    # causal donor (differs in ONE feature value and in the stem)
    donor_prompt: str | None = None
    donor_answer: str | None = None
    donor_stem: str | None = None
    donor_feature: str | None = None      # "a" or "b": the intervened feature
    donor_ia: int | None = None
    donor_ib: int | None = None
    cf_answer: str | None = None          # base stem with the donor's value of the intervened feature (selective-write target)
    sv_prompt: str | None = None          # same-value donor: same feature values as the base, different stem (control)
    sv_stem: str | None = None

    def to_json(self):
        return asdict(self)


CODE_POOL = ["Q7", "K2", "Z9", "M4", "R1", "T8", "V3", "H6", "P5", "W0", "J2", "L7", "N4", "G9", "B3", "D8", "F1", "X6", "C5", "Y0"]
CODE_WORDS_A = [("red", "blue"), ("big", "small"), ("hot", "cold"), ("old", "new")]
CODE_WORDS_B = [("circle", "square"), ("north", "south"), ("left", "right")]
CODE_HEADER = "Codebook:\n"
CODE_SEP = " => "


def make_lang(rng: random.Random, order: str = "ab", scheme: str = "morph") -> Lang:
    if scheme == "code":
        syms = rng.sample(CODE_POOL, 4)
        return Lang(rng.choice(CODE_WORDS_A), rng.choice(CODE_WORDS_B), (syms[0], syms[1]), (syms[2], syms[3]),
                    (rng.randrange(2), rng.randrange(2)), order, "code")
    affs = rng.sample(AFFIX_POOL, 4)
    return Lang(rng.choice(WORDS_A), rng.choice(WORDS_B), (affs[0], affs[1]), (affs[2], affs[3]),
                (rng.randrange(2), rng.randrange(2)), order)


def demos(lang: Lang, rng: random.Random, n_single=2, n_combo=2) -> tuple[list[str], list[str]]:
    """n_single lines per single-feature value (4 values) + n_combo lines per non-held-out bundle (3 bundles)."""
    lines, stems = [], []
    used = set()

    def new_stem():
        while True:
            s = nonce_stem(rng)
            if s not in used:
                used.add(s)
                return s
    for ia in range(2):
        for _ in range(n_single):
            s = new_stem()
            stems.append(s)
            lines.append(f"{lang.phrase(s, ia, None)}{SEP(lang)}{lang.form(s, ia, None)}")
    for ib in range(2):
        for _ in range(n_single):
            s = new_stem()
            stems.append(s)
            lines.append(f"{lang.phrase(s, None, ib)}{SEP(lang)}{lang.form(s, None, ib)}")
    for ia in range(2):
        for ib in range(2):
            if (ia, ib) == lang.held:
                continue
            for _ in range(n_combo):
                s = new_stem()
                stems.append(s)
                lines.append(f"{lang.phrase(s, ia, ib)}{SEP(lang)}{lang.form(s, ia, ib)}")
    order = list(range(len(lines)))
    rng.shuffle(order)
    return [lines[i] for i in order], [stems[i] for i in order]


HEADER = "In this language:\n"
TEMPLATES = {
    "default": {"header": "In this language:\n", "sep": " = ", "wA": None, "wB": None, "pool": None},
    "alt": {"header": "Word forms in a made-up language (stem, then suffixes):\n", "sep": " -> ",
            "wA": [("tiny", "huge"), ("young", "ancient"), ("green", "yellow"), ("wet", "dry")],
            "wB": [("single", "several"), ("near", "far"), ("ours", "theirs")],
            "pool": ["ash", "ent", "ilk", "oom", "ust", "eck", "arn", "ith", "ulp", "orm", "esk", "ind", "ath", "ock", "ump", "erd", "ipt", "oft", "ung", "ext"]},
}
ACTIVE_TEMPLATE = "default"
_DEFAULT_WORDS_A, _DEFAULT_WORDS_B, _DEFAULT_AFFIX_POOL = list(WORDS_A), list(WORDS_B), list(AFFIX_POOL)


def set_template(name: str) -> None:
    """Switch header / separator / word pairs / ending pool globally (module level, read by make_lang / make_item);
    set_template("default") restores the original pools."""
    global ACTIVE_TEMPLATE, HEADER
    t = TEMPLATES[name]
    ACTIVE_TEMPLATE = name
    HEADER = t["header"]
    WORDS_A[:] = t["wA"] or _DEFAULT_WORDS_A
    WORDS_B[:] = t["wB"] or _DEFAULT_WORDS_B
    AFFIX_POOL[:] = t["pool"] or _DEFAULT_AFFIX_POOL


def SEP(lang=None):
    if lang is not None and getattr(lang, "scheme", "morph") == "code":
        return CODE_SEP
    return TEMPLATES[ACTIVE_TEMPLATE]["sep"]


def header_for(lang):
    return CODE_HEADER if getattr(lang, "scheme", "morph") == "code" else HEADER


def make_item(task: str, rng: random.Random, n_single=2, n_combo=2, donor_feature: str | None = None, order: str = "ab", scheme: str = "morph") -> MItem:
    lang = make_lang(rng, order, scheme)
    if scheme == "code":
        n_single, n_combo = 1, 1          # no stems → every demonstration line is unique (4 singles + 3 attested bundles)
    lines, dstems = demos(lang, rng, n_single, n_combo)
    used = set(dstems)
    while True:
        stem = nonce_stem(rng) if scheme != "code" else ""
        if stem not in used or scheme == "code":
            used.add(stem)
            break
    if task == "bind_a":
        ia, ib = rng.randrange(2), None
    elif task == "bind_b":
        ia, ib = None, rng.randrange(2)
    elif task == "compose_h":
        ia, ib = lang.held
    elif task == "compose_s":
        seen = [(a, b) for a in range(2) for b in range(2) if (a, b) != lang.held]
        ia, ib = rng.choice(seen)
    else:
        raise ValueError(task)
    ctx = header_for(lang) + "\n".join(lines) + "\n"
    prompt = ctx + lang.phrase(stem, ia, ib) + SEP(lang).rstrip()
    it = MItem(task, prompt, lang.form(stem, ia, ib), stem, ia, ib, asdict(lang), dstems)
    # causal donor: flip ONE present feature, new stem
    feats = [f for f, v in (("a", ia), ("b", ib)) if v is not None]
    if donor_feature is None:
        donor_feature = rng.choice(feats)
    if donor_feature in feats:
        while True:
            dstem = nonce_stem(rng) if scheme != "code" else ""
            if dstem not in used or scheme == "code":
                break
        dia, dib = ia, ib
        if donor_feature == "a":
            dia = 1 - ia
        else:
            dib = 1 - ib
        it.donor_prompt = ctx + lang.phrase(dstem, dia, dib) + SEP(lang).rstrip()
        it.donor_answer = lang.form(dstem, dia, dib)
        it.donor_stem, it.donor_feature, it.donor_ia, it.donor_ib = dstem, donor_feature, dia, dib
        it.cf_answer = lang.form(stem, dia, dib)
        used.add(dstem)
        while True:
            svstem = nonce_stem(rng) if scheme != "code" else ""
            if svstem not in used or scheme == "code":
                break
        it.sv_prompt = ctx + lang.phrase(svstem, ia, ib) + SEP(lang).rstrip()
        it.sv_stem = svstem
    return it


def build(task: str, n: int, seed=0, **kw) -> list[MItem]:
    rng = random.Random(f"morph|{task}|{seed}")
    return [make_item(task, rng, **kw) for _ in range(n)]


def parse_form(text: str, scheme: str = "morph") -> tuple[str, list[str]]:
    """Parse the generated continuation: morph → first whitespace-delimited word split on '-' (stem, affixes); code → the symbols on
    the first line (no stem; returns ("", symbols))."""
    line = text.strip().split("\n")[0].strip() if text.strip() else ""
    if scheme == "code":
        return "", [w for w in line.split(" ") if w]
    word = line.split(" ")[0] if line else ""
    parts = word.split("-")
    return parts[0], parts[1:]


def slots_of(lang: dict, ia, ib) -> list[tuple[str, str]]:
    return Lang(**lang).slots(ia, ib)


def score(it: MItem, gen: str) -> dict[str, bool]:
    stem, affs = parse_form(gen, it.lang.get("scheme", "morph"))
    lang = it.lang
    tgt = slots_of(lang, it.ia, it.ib)
    # strict in every scheme: the parsed first-line symbols (code) or first word (morph) must equal the target exactly; an overlong code line that
    # merely begins with the target is NOT exact (an earlier release truncated it; effect ≤ 1 item in 320, code_strict.json)
    exact = (stem == it.stem and affs == [a for _, a in tgt])
    res = {"copy": stem == it.stem, "exact": exact, "n_affixes_ok": len(affs) == len(tgt), "aff_a": None, "aff_b": None}
    for k, (f, a) in enumerate(tgt):
        res["aff_" + f] = len(affs) > k and affs[k] == a
    res["demo_stem_reuse"] = stem in it.demo_stems
    res["category"] = categorise(it, stem, affs)
    return res


def categorise(it: MItem, stem: str, affs: list[str]) -> str:
    """generation taxonomy for compose queries: correct | competitor (attested bundle sharing the first realised affix) | other_attested |
    single_affix | stem_error | malformed"""
    lang = it.lang
    if stem != it.stem:
        return "stem_error"
    if it.ia is None or it.ib is None:
        tgt = [a for _, a in slots_of(lang, it.ia, it.ib)]
        return "correct" if affs == tgt else ("malformed" if len(affs) != 1 else "wrong_affix")
    forms = {tuple(a for _, a in slots_of(lang, a_, b_)): (a_, b_) for a_ in range(2) for b_ in range(2)}
    if len(affs) == 1:
        return "single_affix"
    key = tuple(affs)
    if key not in forms:
        return "malformed"
    bundle = forms[key]
    if bundle == (it.ia, it.ib):
        return "correct"
    first = slots_of(lang, it.ia, it.ib)[0][0]
    competitor = (it.ia, 1 - it.ib) if first == "a" else (1 - it.ia, it.ib)
    return "competitor" if bundle == competitor else "other_attested"


def score_cf(it: MItem, gen: str) -> dict[str, bool]:
    """Selective-write scoring for an intervened generation: base stem kept, intervened affix = donor value, other affix kept."""
    stem, affs = parse_form(gen, it.lang.get("scheme", "morph"))
    lang = it.lang
    tgt = slots_of(lang, it.donor_ia, it.donor_ib)
    base = slots_of(lang, it.ia, it.ib)
    slot = [k for k, (f, _) in enumerate(tgt) if f == it.donor_feature][0]
    others = [k for k in range(len(tgt)) if k != slot]
    flipped = len(affs) > slot and affs[slot] == tgt[slot][1]
    other_kept = all(len(affs) > k and affs[k] == base[k][1] for k in others)
    return {"stem_kept": stem == it.stem, "stem_donor": stem == it.donor_stem, "flipped": flipped, "other_kept": other_kept,
            "selective": (stem == it.stem) and flipped and other_kept, "exact_cf": (stem == it.stem and affs == [a for _, a in tgt]),
            "exact_base": (stem == it.stem and affs == [a for _, a in base])}


def four_forms(it: MItem) -> list[tuple[tuple[int, int], str]]:
    """the four bundle forms for a compose query (stem fixed), tagged by bundle"""
    return [((a, b), Lang(**it.lang).form(it.stem, a, b)) for a in range(2) for b in range(2)]
