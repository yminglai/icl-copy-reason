"""Registry of released pretraining checkpoints per model family, with their training positions.

Inputs : data/manifests/revisions/<family>.json (one file per family: Hub repo, final model, grids of revisions with positions
         in billions of tokens or optimizer steps, candidate token tables where the grid was selected from a longer list) and
         data/manifests/revisions/hub_candidate_tokens.json (every token-labelled checkpoint of the surveyed same-repo families).
Outputs: Family objects (revisions of a grid, position mapping for result keys), the log-spaced grid selection used to pick 20
         checkpoints per family, and the Hub survey helpers that found and qualified the families (metadata only, no weights).
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import data_root
from ..io import read_json

PYTHIA_TOKENS_PER_STEP = 2097152
AMBER_TOKENS_B = 1260.0
AMBER_CHUNKS = 359


def revisions_dir() -> Path:
    return data_root() / "manifests" / "revisions"


@dataclass
class Family:
    """One traced family: where its checkpoints live and how result keys map to training positions."""

    family: str
    repo: str
    final_model: str
    position_unit: str | None
    grids: dict[str, list[dict[str, Any]]]
    default_grid: str | None
    seeds: list[int]
    tag: str
    n: int
    note: str = ""
    final_revision: str | None = None
    survey_status: str | None = None
    n_available: int | None = None
    summary_in_release: bool = False
    candidates_tokens_B: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Family:
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})

    def revisions(self, grid: str | None = None) -> list[dict[str, Any]]:
        g = grid or self.default_grid
        if g is None:
            return []
        return list(self.grids[g])

    def refs(self, grid: str | None = None) -> list[str]:
        return [r["ref"] for r in self.revisions(grid)]

    def position_table(self, grid: str | None = None) -> dict[str, float]:
        """result key -> position for every revision of every grid (the grid argument restricts it)."""
        out: dict[str, float] = {}
        grids = [grid] if grid else list(self.grids)
        for g in grids:
            for r in self.grids[g]:
                if r.get("position") is not None:
                    out[r["ref"]] = float(r["position"])
                    out[result_key(r["ref"])] = float(r["position"])
        return out

    def position_of(self, key: str) -> float | None:
        """Training position (billions of tokens, or steps for step-labelled families) of a result key; None for the released model."""
        t = self.position_table()
        if key in t:
            return t[key]
        if key in self.candidates_tokens_B:
            return float(self.candidates_tokens_B[key])
        return position_from_key(key)


def result_key(ref: str) -> str:
    """The key under which a revision appears in a result file: `repo@rev` -> rev, a separate model id -> `main`, else the ref."""
    if "@" in ref:
        return ref.split("@", 1)[1]
    if "/" in ref:
        return "main"
    return ref


def position_from_key(key: str) -> float | None:
    """Position in billions of tokens parsed from a revision name (OLMo `tokens<N>B`, Baichuan `train_<N>B`, Amber `ckpt_<k>`,
    Pythia `step<N>`); None for `main` and anything else."""
    m = re.search(r"tokens(\d+)B", key) or re.search(r"train_0*(\d+)B", key)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"ckpt_(\d+)", key)
    if m:
        return float(int(m.group(1)) * AMBER_TOKENS_B / AMBER_CHUNKS)
    m = re.fullmatch(r"step(\d+)", key)
    if m:
        return float(int(m.group(1)) * PYTHIA_TOKENS_PER_STEP / 1e9)
    m = re.fullmatch(r"pos(\d+)", key)
    if m:
        return float(m.group(1))
    return None


def families() -> list[str]:
    return sorted(p.stem for p in revisions_dir().glob("*.json") if p.stem != "hub_candidate_tokens")


def load_family(name: str) -> Family:
    name = name.removeprefix("traj-")
    p = revisions_dir() / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"no revisions manifest for family {name!r} in {revisions_dir()}")
    return Family.from_dict(read_json(p))


def hub_candidate_tokens() -> dict[str, dict[str, float]]:
    p = revisions_dir() / "hub_candidate_tokens.json"
    return read_json(p) if p.exists() else {}


# ----------------------------------------------------------------------------- grid selection (fixed before the runs)
def select_log_spaced(cands: list[tuple[str, float]], k: int = 20) -> list[tuple[str, float]]:
    """k log-spaced targets between the first and the last checkpoint, each mapped to its nearest checkpoint in log position;
    duplicates removed; while fewer than k remain, the unused checkpoint nearest the log midpoint of the largest gap is added."""
    cands = sorted(cands, key=lambda t: t[1])
    if len(cands) <= k:
        return cands
    logs = [math.log(t) for _, t in cands]
    lo, hi = logs[0], logs[-1]
    chosen: set[int] = set()
    for i in range(k):
        target = lo + (hi - lo) * i / (k - 1)
        chosen.add(min(range(len(cands)), key=lambda j: abs(logs[j] - target)))
    while len(chosen) < k and len(chosen) < len(cands):
        idx = sorted(chosen)
        gaps = sorted(((logs[b] - logs[a], a, b) for a, b in zip(idx, idx[1:])), reverse=True)
        added = False
        for _, a, b in gaps:
            free = [j for j in range(a + 1, b) if j not in chosen]
            if free:
                mid = (logs[a] + logs[b]) / 2
                chosen.add(min(free, key=lambda j: abs(logs[j] - mid)))
                added = True
                break
        if not added:
            break
    return [cands[j] for j in sorted(chosen)]


# branch-name patterns of the surveyed same-repo families and the tokens-per-step or token labels from their model cards
BRANCH_PATTERNS: dict[str, tuple[str, str, Any]] = {
    "olmo3-7b": ("allenai/Olmo-3-1025-7B", r"^stage1-step(\d+)$", lambda m: int(m.group(1)) * 5930.0 / 1413814),
    "apertus-8b": ("swiss-ai/Apertus-8B-2509", r"^step\d+-tokens(\d+)([BT])$", lambda m: float(m.group(1)) * (1000.0 if m.group(2) == "T" else 1.0)),
    "smollm3-3b": ("HuggingFaceTB/SmolLM3-3B-checkpoints", r"^stage[123]-step-(\d+)$", lambda m: int(m.group(1)) * 2359296 / 1e9),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints", r"^step-(\d+)$", lambda m: int(m.group(1)) * 2097152 / 1e9),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints", r"^step-(\d+)$", lambda m: int(m.group(1)) * 1572864 / 1e9),
    "comma-7b": ("common-pile/comma-v0.1-1t", r"^stage[12]-step\d+-tokens(\d+)B$", lambda m: float(m.group(1))),
    "lucie-7b": ("OpenLLM-France/Lucie-7B", r"^step0*(\d+)$", lambda m: int(m.group(1)) * 3100.0 / 753851),
    "redpajama-7b": ("togethercomputer/RedPajama-INCITE-7B-Base", r"^(\d+)(b|t)_tokens$", lambda m: float(m.group(1)) * (1000.0 if m.group(2) == "t" else 1.0)),
    "viking-7b": ("LumiOpen/Viking-7B", r"^(\d+)B$", lambda m: float(m.group(1))),
    "fineweb-edu-1.8b": ("HuggingFaceFW/ablation-model-fineweb-edu", r"^step-\d+-(\d+)BT$", lambda m: float(m.group(1))),
    "tinyllama-1.1b": ("TinyLlama/tinyLlama-intermediate-checkpoints", r"^step-\d+k-token-(\d+)B$", lambda m: float(m.group(1))),
}
MARIN_TAGS_TOKENS_B = {"kestrel": 2700.0, "ocelot": 3780.0, "jellyfish": 4780.0, "phoenix": 11100.0, "starling": 12400.0, "deeper-starling": 12700.0}


def candidates_from_branches(family: str, branches: list[str]) -> list[tuple[str, float]]:
    """Token-labelled candidate checkpoints of a same-repo family from its Hub branch names."""
    _, pat, f = BRANCH_PATTERNS[family]
    out = []
    for b in branches:
        m = re.match(pat, b)
        if m:
            t = f(m)
            if t > 0:
                out.append((b, t))
    return out


# ----------------------------------------------------------------------------- Hub survey (metadata only)
def probe_repo(repo: str, api=None) -> dict[str, Any]:
    """Model-card and ref metadata of one Hub repository: licence, files, branches and tags (no weights downloaded)."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    api = api or HfApi()
    out: dict[str, Any] = {"repo": repo}
    try:
        info = api.model_info(repo, files_metadata=False)
        out["exists"] = True
        out["gated"] = info.gated
        out["private"] = info.private
        cd = info.card_data or {}
        out["license"] = cd.get("license") if hasattr(cd, "get") else getattr(cd, "license", None)
        out["pipeline_tag"] = info.pipeline_tag
        out["library_name"] = info.library_name
        out["last_modified"] = str(info.last_modified) if info.last_modified else None
        cfg = info.config or {}
        out["config_arch"] = cfg.get("architectures") if cfg else None
        out["model_type"] = cfg.get("model_type") if cfg else None
        siblings = [s.rfilename for s in (info.siblings or [])]
        out["n_files"] = len(siblings)
        out["has_config"] = "config.json" in siblings
        out["has_tokenizer"] = any(f in siblings for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"))
        out["top_dirs"] = sorted({f.split("/")[0] for f in siblings if "/" in f})[:40]
    except RepositoryNotFoundError:
        return {"repo": repo, "exists": False, "error": "not found"}
    except GatedRepoError:
        out["exists"] = True
        out["gated"] = "gated-no-access"
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", None)
        return {"repo": repo, "exists": False if code == 404 else None, "error": f"http {code}"}
    try:
        refs = api.list_repo_refs(repo)
        out["branches"] = [b.name for b in refs.branches]
        out["tags_list"] = [t.name for t in refs.tags]
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        out["refs_error"] = repr(e)[:200]
    return out


def probe_repos(repos: list[str], retries: int = 6, pause: float = 5.0) -> list[dict[str, Any]]:
    """probe_repo over a list with back-off on rate limits (serial, as the survey was run)."""
    from huggingface_hub import HfApi

    api = HfApi()
    out = []
    for repo in repos:
        r: dict[str, Any] = {}
        for i in range(retries):
            r = probe_repo(repo, api)
            err = (r.get("error") or "") + (r.get("refs_error") or "")
            if "429" not in err:
                break
            time.sleep(pause * (i + 1))
        out.append(r)
    return sorted(out, key=lambda r: r["repo"])


CHECKPOINT_NAME_PATTERN = re.compile(r"(intermediate|checkpoint|ckpt|step|phase|stage|-\d+[bBtT]\b|\d+bt|tokens|preview|pretrain|midtrain|base)", re.I)


def author_checkpoint_repos(authors: list[str], limit: int = 2000) -> dict[str, dict[str, Any]]:
    """For each Hub author, every model id and the ones whose name suggests intermediate checkpoints."""
    from huggingface_hub import HfApi

    api = HfApi()
    out = {}
    for a in authors:
        ids = sorted(m.id for m in api.list_models(author=a, limit=limit))
        hits = [i for i in ids if CHECKPOINT_NAME_PATTERN.search(i.split("/", 1)[1])]
        out[a] = {"n_models": len(ids), "hits": hits}
    return out


def resolve_ref_shas(repo: str) -> dict[str, str]:
    """ref name -> commit sha for every branch and tag of a repo (metadata only)."""
    from huggingface_hub import HfApi

    refs = HfApi().list_repo_refs(repo)
    out = {b.name: b.target_commit for b in refs.branches}
    out.update({t.name: t.target_commit for t in refs.tags})
    return out


def tokenizer_marker_check(repo: str, n_per_marker: int = 6, seed: int = 0) -> dict[str, Any]:
    """How a tokenizer treats the diagnostic's markers: fraction of markers that are one token after the hyphen (first and
    second position), whether the hyphen is its own token, and whether marker characters align with token boundaries
    (so an attention-ablation span can cover exactly the marker). Tokenizer only, no weights."""
    import random

    from transformers import AutoTokenizer

    from ..tasks.prompts import AFFIX_POOL, nonce_stem

    out: dict[str, Any] = {"repo": repo}
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:200]
        return out
    out["vocab"] = len(tok)
    rng = random.Random(seed)
    fast = bool(getattr(tok, "is_fast", False))
    single1 = single2 = align1 = align2 = n1 = n2 = hyphen_own = nh = 0
    for aff in AFFIX_POOL:
        for _ in range(n_per_marker):
            stem = nonce_stem(rng)
            aff2 = rng.choice([a for a in AFFIX_POOL if a != aff])
            t1 = f" {stem}-{aff}"
            t2 = f" {stem}-{aff2}-{aff}"
            ids_stem = tok(f" {stem}", add_special_tokens=False)["input_ids"]
            ids_h = tok(f" {stem}-", add_special_tokens=False)["input_ids"]
            ids1 = tok(t1, add_special_tokens=False)["input_ids"]
            n1 += 1
            single1 += int(ids1[: len(ids_h)] == ids_h and len(ids1) == len(ids_h) + 1)
            nh += 1
            hyphen_own += int(ids_h[: len(ids_stem)] == ids_stem and len(ids_h) == len(ids_stem) + 1)
            if fast:
                align1 += int(_span_aligned(tok, t1, len(t1) - len(aff), len(t1)))
                align2 += int(_span_aligned(tok, t2, len(t2) - len(aff), len(t2)))
            ids_h2 = tok(f" {stem}-{aff2}-", add_special_tokens=False)["input_ids"]
            ids2 = tok(t2, add_special_tokens=False)["input_ids"]
            n2 += 1
            single2 += int(ids2[: len(ids_h2)] == ids_h2 and len(ids2) == len(ids_h2) + 1)
    out.update({"fast": fast, "single_after_hyphen_pos1": round(single1 / n1, 3), "single_after_hyphen_pos2": round(single2 / n2, 3),
                "hyphen_own_token": round(hyphen_own / nh, 3), "marker_aligned_pos1": round(align1 / n1, 3) if fast else None,
                "marker_aligned_pos2": round(align2 / n2, 3) if fast else None,
                "example": tok.convert_ids_to_tokens(tok(" glev-ek-dor", add_special_tokens=False)["input_ids"])})
    return out


def _span_aligned(tok, text: str, start: int, end: int) -> bool:
    """True when the character span [start, end) of text is covered exactly by whole tokens (leading spaces ignored)."""
    offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    ov = [(s, e) for s, e in offs if e > start and s < end]
    if not ov:
        return False
    s0 = min(s for s, _ in ov)
    e1 = max(e for _, e in ov)
    while s0 < start and text[s0] == " ":
        s0 += 1
    return s0 == start and e1 == end
