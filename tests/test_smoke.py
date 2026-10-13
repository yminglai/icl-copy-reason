"""Smoke tests (CPU, no network): CLIs answer --help, prompts are deterministic and never show the withheld combination,
the readings run on a tiny model, the shipped exercises parse and pass the offline checks, every config dry-runs, launchers
name real sub-commands, the checkpoint registry loads, the trajectory summary works on synthetic data."""
from __future__ import annotations

import glob
import gzip
import json
import os
import random
import re
import subprocess
import sys

import numpy as np
import pytest

from icl_copy_reason.config import load_config  # noqa: E402
from icl_copy_reason.data_construction import exercises, synthetic_docs, variants, verification
from icl_copy_reason.data_construction import external_sets as X
from icl_copy_reason.data_construction.documents import POOLS, assign_type_by_hash, pool_table
from icl_copy_reason.readings import ablation, bootstrap, copy_share, deletion, patching  # noqa: E402
from icl_copy_reason.readings.lm import wrap_model  # noqa: E402
from icl_copy_reason.tasks import prompts as P  # noqa: E402
from icl_copy_reason.tasks.items import items_of_record, load_items, write_items  # noqa: E402
from icl_copy_reason.tasks.prompts import build  # noqa: E402
from icl_copy_reason.tasks.quechua import build_real, categorise_real, four_forms_real, real_line_sets  # noqa: E402
from icl_copy_reason.training import manifests as M  # noqa: E402
from icl_copy_reason.training import pack as PK  # noqa: E402
from icl_copy_reason.training.freeze import freeze_record, qualify_unseen, wilson_lower  # noqa: E402
from icl_copy_reason.trajectories import crossings, onsets
from icl_copy_reason.trajectories.registry import (
    families,
    load_family,
    position_from_key,
    result_key,
    select_log_spaced,
)

# ---- from test_cli_help ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_module_help():
    r = subprocess.run([sys.executable, "-m", "icl_copy_reason", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "icl_copy_reason" in r.stdout


def test_every_script_help():
    scripts = sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py")))
    assert scripts, "no CLIs under scripts/"
    for s in scripts:
        r = subprocess.run([sys.executable, s, "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"{os.path.basename(s)} --help failed:\n{r.stderr[-800:]}"


# ---- from test_tasks ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def _held_form_absent(it) -> bool:
    lang = P.Lang(**it.lang)
    demos = it.prompt.split("\n")[1:-1]
    held_markers = [a for _, a in lang.slots(*lang.held)]
    for line in demos:
        rhs = line.split(P.SEP(lang))[-1]
        markers = rhs.split(" ") if lang.scheme == "code" else rhs.split("-")[1:]
        if markers == held_markers:
            return False
    return True


def test_deterministic_per_seed():
    a = P.build("compose_h", 5, seed=3)
    b = P.build("compose_h", 5, seed=3)
    c = P.build("compose_h", 5, seed=4)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert [x.prompt for x in a] != [x.prompt for x in c]


def test_withheld_combination_never_in_demonstrations():
    for task in ("compose_h", "compose_s", "bind_a", "bind_b"):
        for it in P.build(task, 40, seed=0):
            assert _held_form_absent(it)
            assert it.stem not in it.demo_stems
            assert len(it.prompt.split("\n")) == 1 + 2 * 4 + 2 * 3 + 1  # header + 8 single + 6 combination lines + query
    for it in P.build("compose_h", 20, seed=1):
        assert tuple(it.lang["held"]) == (it.ia, it.ib)
        assert it.answer not in it.prompt


def test_code_scheme_and_controls():
    for it in P.build("compose_h", 30, seed=0, scheme="code"):
        assert _held_form_absent(it)
        assert it.prompt.startswith(P.CODE_HEADER) and "-" not in it.answer and len(it.answer.split(" ")) == 2
    ba = P.build("compose_h", 3, seed=0, order="ba")[0]
    assert ba.answer.split("-")[1] == ba.lang["affB"][ba.ib]
    small = P.build("compose_h", 2, seed=0, n_single=1, n_combo=1)[0]
    assert len(small.prompt.split("\n")) == 1 + 4 + 3 + 1
    P.set_template("alt")
    try:
        alt = P.build("compose_h", 2, seed=0)[0]
        assert alt.prompt.startswith(P.TEMPLATES["alt"]["header"]) and " -> " in alt.prompt
    finally:
        P.set_template("default")
    assert P.build("compose_h", 1, seed=0)[0].prompt.startswith("In this language:")


def test_scoring_rules():
    it = P.build("compose_h", 1, seed=2)[0]
    assert P.score(it, " " + it.answer)["exact"] and P.categorise(it, *P.parse_form(it.answer)) == "correct"
    lang = P.Lang(**it.lang)
    first = P.slots_of(it.lang, it.ia, it.ib)[0][0]
    comp = (it.ia, 1 - it.ib) if first == "a" else (1 - it.ia, it.ib)
    assert P.categorise(it, *P.parse_form(lang.form(it.stem, *comp))) == "competitor"
    assert P.categorise(it, *P.parse_form("zzz-il-um")) == "stem_error"
    assert P.score_cf(it, " " + it.cf_answer)["selective"]
    assert {b for b, _ in P.four_forms(it)} == {(a, b) for a in range(2) for b in range(2)}


def test_item_set_round_trip(tmp_path):
    items = items_of_record("compose_h", 12, 0, offset=11)
    p = write_items(items, tmp_path / "items.jsonl.gz")
    back = load_items(p)
    assert [x.prompt for x in back] == [x.prompt for x in items]
    assert json.loads(json.dumps(back[0].lang)) == json.loads(json.dumps(items[0].lang)) and back[0].answer == items[0].answer


def test_quechua_items():
    items = build_real("compose_h", 10, seed=0)
    for it in items:
        assert it.answer == it.stem + it.lang["affA"][it.ia] + it.lang["affB"][it.ib]
        assert it.answer not in it.prompt.split("\n")[:-1]
        spans, qstart = real_line_sets(it)
        assert len(spans["competitor"]) == 2 and len(spans["single2"]) == 2 and qstart == it.prompt.rfind("\n") + 1
        assert categorise_real(it, " " + it.answer) == "correct"
        comp_form = dict(four_forms_real(it))[(it.ia, 1 - it.ib)]
        assert categorise_real(it, " " + comp_form) == "competitor"
    seg = build_real("compose_h", 2, seed=0, hyphen=True)[0]
    assert seg.answer.count("-") == 2


# ---- from test_readings ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



@pytest.fixture(scope="module")
def lm(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    return wrap_model(model, tok, device="cpu")


def test_copy_share_schema(lm):
    rec = copy_share.score_revision(lm, n=2, seed=0, bs=4, tasks=["bind_a", "compose_h"])
    for task in ("bind_a", "compose_h"):
        for k in ("copy", "exact", "items", "taxonomy", "n"):
            assert k in rec[task]
        assert len(rec[task]["items"]) == 2
    assert {"four_way_acc", "four_way_competitor", "four_way_margin", "four_way_items"} <= set(rec["compose_h"])
    assert 0.0 <= rec["induction"] <= 1.0 and isinstance(rec["fingerprint"], float)


def test_ablation_schema_and_paired_effect(lm):
    items = build("compose_h", 2, seed=11)
    rec = ablation.ablate_revision(lm, items, ["competitor_marker", "wordB", "sham"], bs=2, n_boot=50)
    assert rec["n"] == 2 and set(rec["none"]) >= {"four_way_acc", "competitor", "margin", "a_null", "items"}
    for c in ("competitor_marker", "wordB", "sham"):
        assert len(rec[c]["items"]) == 2 and len(rec[c]["delta_margin_ci"]) == 2
        d = np.mean([a["margin"] - b["margin"] for a, b in zip(rec[c]["items"], rec["none"]["items"])])
        assert abs(d - rec[c]["delta_margin"]) < 2e-3
    for it in items:
        spans, qstart = ablation.line_sets(it)
        assert len(spans["competitor"]) == 2 and len(spans["competitor_marker"]) == 2 and len(spans["wordB"]) == 1
        assert all(e <= qstart for s, e in spans["competitor"])  # demonstration spans precede the query line


def test_deletion_variants(lm):
    items = build("compose_h", 2, seed=11)
    v = deletion.variants(items[0])
    assert set(v) == set(deletion.VARIANTS)
    assert len(v["drop_competitor"].split("\n")) == len(v["full"].split("\n")) - 2
    assert len(v["drop_all_bundles"].split("\n")) == len(v["full"].split("\n")) - 6
    rec = deletion.delete_revision(lm, items, bs=2, variant_names=["full", "drop_competitor"])
    assert {"four_way_acc", "gen_exact", "items", "n"} <= set(rec["full"]) and rec["full"]["n"] == 2


def test_one_layer_word_patch(lm, tmp_path):
    out = patching.layer_sweep("tiny", seed=0, n=2, bs=2, layers="0", out_dir=tmp_path, device="cpu", max_new=3, lm=lm)
    rec = json.load(open(out))
    assert rec["n"] == 2 and "0" in rec["layers"] and len(rec["layers"]["0"]["followed"]) == 2
    assert 0.0 <= rec["identity_check"]["share_unchanged"] <= 1.0
    it = build("compose_s", 1, seed=200, donor_feature="b")[0]
    assert patching.donor_prompt(it) != it.prompt and patching.target_of(it) in ("into_attested", "into_unattested")


def test_bootstrap_reproducible():
    x = np.arange(20.0)
    a = bootstrap.bootstrap_mean(x, n_boot=200)
    b = bootstrap.bootstrap_mean(x, n_boot=200)
    assert a == b and a[1] <= a[0] <= a[2]
    lo, hi = bootstrap.wilson(8, 16)
    assert lo < 0.5 < hi


# ---- from test_data_construction ----
ROOT_data_construction = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT_data_construction, "data")
REQUIRED = {"doc_sha256", "pool", "file", "row", "type", "prompt_sha256", "retry", "exercise", "answer_norm", "n_tokens", "v1", "v2", "v3", "mech1", "mech2", "mech3", "mech4", "mech5", "mech6", "pass", "reasons"}


def _records(n: int = 200) -> list[dict]:
    out = []
    with gzip.open(os.path.join(DATA, "exercises", "exercises.jsonl.gz"), "rt") as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def test_shipped_exercises_parse_and_pass():
    recs = _records(200)
    assert len(recs) == 200
    for r in recs:
        assert REQUIRED <= set(r), REQUIRED - set(r)
        assert r["pass"] is True and r["reasons"] == []
        assert r["type"] in {"b", "c", "e", "g", "h"}
        assert r["pool"] in POOLS and len(r["doc_sha256"]) == 64
        ex = r["exercise"]
        assert all(isinstance(ex.get(k), str) and ex[k].strip() for k in ("context", "query", "answer"))


def test_offline_checks_agree_with_shipped_verdicts():
    P = exercises.load_prompts(os.path.join(DATA, "exercises", "prompts.json"))
    recs = _records(200)
    out = exercises.run_offline_checks(P, recs, tok=None, evalh=None)
    for r, o in zip(recs, out):
        assert o["mech1"] == r["mech1"] and o["mech3"] == r["mech3"], r["doc_sha256"]
        assert o["mech6"]["pass"] == r["mech6"]["pass"], r["doc_sha256"]
        assert o["mechanical_reasons"] == []  # every shipped record passes the checks that need no tokenizer / window store


def test_prompt_table_is_consistent():
    P = exercises.load_prompts(os.path.join(DATA, "exercises", "prompts.json"))
    for k in ("gen_system", "gen_user", "gen_user_h", "gen_ban_topup", "v1_system", "v1_user", "v2_system", "v2_user", "v3_system", "v3_user", "audit_system", "audit_user", "audit_system_h", "audit_user_h"):
        assert isinstance(P[k], str) and P[k]
    assert set(P["topup_types"]) == {"b", "c", "e", "g", "h"} and P["retry_type"] == "h"
    assert set(P["mixtures"]["F2"]) == {"b", "c", "e", "g", "h"} and set(P["mixtures"]["F1"]) == {"a", "b", "c", "e", "g", "h"}
    assert verification.class_name_norms(P) >= {"abbreviation", "company", "society"}
    for p in ("rewrite_prompts.json", "variant_prompts.json", "banned_labels.json", "source_pools.json", "examples.json"):
        json.load(open(os.path.join(DATA, "exercises", p)))
    assert set(pool_table()) == set(POOLS)


def test_normalisation_and_parse_demos():
    assert verification.norm_answer(' "The Red-Fox." ') == "red-fox"
    demos, mode = verification.parse_demos("Input: a b → Output: X1\nInput: c d -> Output: X2\nquery")
    assert mode == r"output\s*:" and [d["output"] for d in demos] == ["X1", "X2"] and demos[0]["input"] == "a b"
    assert verification.check1_answer_tokens_in_context("x1", "input a b output x1")
    assert not verification.check2_answer_not_demo_tail("x1", "a: X1\nb: X2")
    assert verification.parse_json_obj('```json\n{"a": 1}\n```') == {"a": 1}


def test_type_h_check6_on_a_shipped_record():
    P = exercises.load_prompts(os.path.join(DATA, "exercises", "prompts.json"))
    h = next(r for r in _records(400) if r["type"] == "h")
    r = verification.check6(h["exercise"], "h", verification.class_name_norms(P))
    assert r["pass"] and len(r["enforced"]) == 6
    broken = dict(h["exercise"], copy_candidate=h["exercise"]["answer"])
    assert not verification.check6(broken, "h", set())["m6_1_candidates_are_demo_outputs_ne_answer"]


def test_variant_checks_on_a_synthetic_type_k_exercise():
    ex = {
        "context": "Notes on the mill.\nfeatures: number (one / many) and role (tool / product); form = term-m1-m2\none hammer → hammer-ka\nmany hammer → hammer-zu\ntool anvil → anvil-pel\nproduct anvil → anvil-rof\none tool chisel → chisel-ka-pel\nmany tool saw → saw-zu-pel\none product plank → plank-ka-rof\nThe mill also sells the nail.",
        "query": "many product nail →",
        "answer": "nail-zu-rof",
        "justification": "composed",
        "type": "k",
        "prefix": "Notes on the mill.",
        "features": json.dumps([{"name": "number", "values": ["one", "many"], "markers": ["ka", "zu"]}, {"name": "role", "values": ["tool", "product"], "markers": ["pel", "rof"]}]),
        "demos": json.dumps([{"term": "hammer", "values": ["one", None], "form": "hammer-ka"}, {"term": "hammer", "values": ["many", None], "form": "hammer-zu"}, {"term": "anvil", "values": [None, "tool"], "form": "anvil-pel"},
                             {"term": "anvil", "values": [None, "product"], "form": "anvil-rof"}, {"term": "chisel", "values": ["one", "tool"], "form": "chisel-ka-pel"}, {"term": "saw", "values": ["many", "tool"], "form": "saw-zu-pel"},
                             {"term": "plank", "values": ["one", "product"], "form": "plank-ka-rof"}]),
        "query_term": "nail",
        "query_values": json.dumps(["many", "product"]),
        "competitor": "saw-zu-pel",
    }
    mk = variants.checkk(ex)
    assert mk["k0_struct"] and mk["k2_cells"] and mk["k6_compose"] and mk["k7_demos_attested"] and mk["k8_competitor"], {k: mk[k] for k in mk["enforced"]}
    m1k = variants.check1k(ex, variants.decode(ex))
    assert m1k["pass"]
    h2 = {"context": "Intro.\nInput: a → Output: X\n" * 1 + "".join(f"Input: w{i} → Output: Y{i % 3}\n" for i in range(8)) + "Note: the output depends on the role, not on shared words.", "query": "Input: q → Output:", "answer": "Y1", "key_term": "q", "copy_candidate": "Y0"}
    assert variants.check7(h2)["pass"] and variants.check8(h2)["m8_prefix_le_400"]


def test_synthetic_generator_deterministic_and_disjoint():
    a = synthetic_docs.generate("compose", 7, 20, ver=1)
    b = synthetic_docs.generate("compose", 7, 20, ver=1)
    assert [d["text"] for d in a] == [d["text"] for d in b]
    for ver in (1, 3):
        for mode in synthetic_docs.MODES:
            for d in synthetic_docs.generate(mode, 3, 50, ver=ver):
                assert d["n_queries"] >= 1 and (d["n_q_unshown"] == d["n_queries"] if mode == "compose" else d["n_q_unshown"] == 0)
    words, markers = synthetic_docs.diagnostic_pools()
    ov = synthetic_docs.pools_disjoint_from(words, markers)
    assert ov == {"words": [], "markers": []}
    rng = random.Random(0)
    text, meta = synthetic_docs.make_doc(rng, "compose", 3)
    assert text.endswith("\n") and meta["n_held"] >= 2 and meta["n_shown"] >= 2
    cfg = json.load(open(os.path.join(DATA, "synthetic", "config.json")))
    assert cfg["disjointness"] == {"diagnostic_words_overlap": [], "diagnostic_markers_overlap": []}


def test_shipped_synthetic_samples_match_generator():
    with gzip.open(os.path.join(DATA, "synthetic", "sample_compose_v3.jsonl.gz"), "rt") as f:
        first = [json.loads(next(f)) for _ in range(5)]
    regen = synthetic_docs.generate("compose", 0, 5, ver=3)
    assert [d["text"] for d in first] == [d["text"] for d in regen]


def test_label_maps_consistent():
    maps = json.load(open(os.path.join(DATA, "external", "label_maps.json")))
    assert maps == X.label_maps()
    for ds, d in X.LABEL_DATASETS.items():
        C = len(d["keep"])
        assert len(d["names"]) == C and d["k"] % C == 0 and len(set(X.SUL[:C])) == C
        m = maps["sets"][ds]["exchange_map"]
        assert set(m) == set(m.values()) and all(k != v for k, v in m.items())
    assert X.derange(2) == [1, 0] and X.derange(4) == [1, 2, 3, 0]
    assert assign_type_by_hash("some document", ["a", "b", "c"]) in {"a", "b", "c"}


def test_scan_oracle_and_shipped_files():
    assert os.path.exists(os.path.join(DATA, "external", "scan", "tasks_test_addprim_jump.txt"))
    assert X.scan_interpret("jump twice") == ["I_JUMP", "I_JUMP"]
    assert X.scan_interpret("walk left") == ["I_TURN_LEFT", "I_WALK"]
    with pytest.raises(KeyError):
        X.LABEL_DATASETS["missing"]


# ---- from test_training ----
ROOT_training = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_training)


TRAINING_CONFIGS = {"synthetic-docs-360m": ["B", "0"], "synthetic-docs-1p7b": ["B", "0"], "mined-docs": ["D2", "0"], "rewritten-docs-admixture": ["E_M1", "0"],
                    "exercises-finetune": ["F2", "0"], "exercises-finetune-replication": ["F2", "0", "--block", "S3B_s0"], "exercises-variants": ["T1_dose30", "0"],
                    "lora-transfer": ["T", "0", "--base", "qwen"], "heads-induce-train": ["B", "0"], "benchmarks": ["x", "0"], "external-assays": ["x", "0"]}


class ToyTok:
    """A byte-level tokenizer stand-in: one token per character, eos 0."""

    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [min(255, ord(c)) + 1 for c in text]}


@pytest.mark.parametrize("name", sorted(TRAINING_CONFIGS))
def test_config_dry_run(name):
    arm, seed, *extra = TRAINING_CONFIGS[name]
    r = subprocess.run([sys.executable, os.path.join(ROOT_training, "scripts", "train.py"), "--config", name, "--arm", arm, "--seed", seed, *extra, "--dry-run"],
                       cwd=ROOT_training, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    plan = json.loads(r.stdout)
    assert plan["experiment"] == name and plan["trainer"] in ("pretrain", "lora", "direct_finetune", "evaluation")
    if plan["trainer"] == "pretrain" and name not in ("mined-docs",):
        assert plan["lr"] is not None and plan["steps"] and plan["micro_bs"]
    cfg = load_config(name)
    assert cfg.get("launcher", "").startswith("icl_copy_reason/training/launch/") and os.path.exists(os.path.join(ROOT_training, cfg.get("launcher")))


def test_pack_exercise_rows_cycles_and_gate():
    tok = ToyTok()
    items = [tok(f"exercise {i} " + "x" * (300 + 37 * i))["input_ids"] for i in range(12)]
    raws = [tok("raw source " + "y" * 900)["input_ids"] for _ in items]
    rows, meta, info = PK.pack_exercise_rows(items, raws, target_rows=10, eos=0, max_exposure=3.0, seed=5)
    assert all(r.shape == (PK.SEQ,) and r.dtype == np.uint16 for r in rows)
    assert info["rows"] == len(rows) == len(meta) and info["mean_exposure"] <= 3.0 + 1e-9
    assert sum(int(k) * v for k, v in info["exposure_histogram"].items()) == info["instances"]
    rows1, _, info1 = PK.pack_exercise_rows(items, raws, target_rows=3, eos=0, once_only=True, seed=5)
    assert info1["max_exposure"] == 1 and info1["instances"] <= info1["unique"] and len(rows1) <= 3
    _, _, info_all = PK.pack_exercise_rows(items, raws, target_rows=50, eos=0, once_only=True, seed=5)
    assert info_all["instances"] == info_all["unique"] and info_all["max_exposure"] == 1, "once-only packs every exercise exactly once"
    rows2, _, _ = PK.pack_exercise_rows(items, raws, target_rows=10, eos=0, max_exposure=3.0, seed=5)
    assert all(np.array_equal(a, b) for a, b in zip(rows, rows2)), "packing is deterministic for a seed"


def test_build_exercise_pack_and_reverts(tmp_path):
    tok = ToyTok()
    recs = []
    docs = {}
    for t, k in (("a", 3), ("b", 4), ("c", 4)):
        for i in range(k):
            sha = f"{t}{i}"
            recs.append({"type": t, "doc_sha256": sha, "pool": "P1", "file": 0, "row": i, "exercise": {"context": "c" * (400 + 50 * i), "query": "q", "answer": "a"}})
            docs[sha] = "d" * 500
    corpus = np.arange(40 * PK.SEQ, dtype=np.uint16).reshape(40, PK.SEQ) % 200
    positions = [i for i in range(40) if i % 5 in (1, 3)]
    arr, layout, summ = PK.build_exercise_pack(tok, recs, docs, positions, render=lambda ex: ex["context"] + "\n" + ex["query"] + "\nAnswer: " + ex["answer"],
                                              revert_row=lambda pos: corpus[pos], max_exposure=1.0)
    assert arr.shape == (len(positions), PK.SEQ) and len(layout) == len(positions)
    assert summ["rows_reverted_to_replay"] == sum(1 for x in layout if x["type"] == "replay")
    for k, x in enumerate(layout):
        if x["type"] == "replay":
            assert np.array_equal(arr[k], corpus[x["row"]])


def test_row_maps_rules():
    pr, sr, meta = M.map_admixture(8, 3, (6, 13, 19), 20)
    assert len(pr) == 24 and ((pr >= 0) ^ (sr >= 0)).all() and meta["n_replacement"] == int((sr >= 0).sum())
    pr, sr, meta = M.map_finetune(4, 3, replay0=100)
    assert list(pr[:4]) == [100, -1, 101, -1] and list(sr[:4]) == [-1, 0, -1, 1]
    pr, sr, meta = M.map_finetune(4, 3, replay0=100, ctrl0=500)
    assert list(pr[:4]) == [100, 500, 101, 501] and (sr < 0).all()
    pr, sr, meta = M.map_calibration(4, 2, calib0=50, n_synth=12, tail=4)
    assert list(sr[1::2]) == [11, 10, 9, 8] and list(pr[0::2]) == [50, 51, 52, 53]
    pr, sr, meta = M.map_admixture_control(8, 3, (6, 13, 19), 20, pool0=300, pool_size=2)
    assert set(pr[np.array([i for i in range(24) if i % 20 in (6, 13, 19)])].tolist()) == {300, 301}
    assert M.replacement_rows(2, 66) == [32, 65, 98, 131]


def test_build_row_map_manifest(tmp_path):
    packs = np.zeros((120, PK.SEQ), dtype=np.uint16)
    pp = tmp_path / "packs_s0.u16"
    np.save(pp, packs)
    os.replace(str(pp) + ".npy", pp)
    for k in ("fwe", "pile"):
        np.save(tmp_path / f"val_{k}.u16", np.zeros(4 * PK.SEQ, dtype=np.uint16))
        os.replace(str(tmp_path / f"val_{k}.u16") + ".npy", tmp_path / f"val_{k}.u16")
    synth = np.ones((12, PK.SEQ), dtype=np.uint16)
    sp = tmp_path / "synth_f2_s0.u16"
    np.save(sp, synth)
    os.replace(str(sp) + ".npy", sp)
    p = {"rps": 8, "steps": 3, "replay0": 24}
    out = tmp_path / "manifest_f2_s0.json"
    man = M.build_row_map("F2", "finetune", p, 0, str(pp), {"fwe": str(tmp_path / "val_fwe.u16"), "pile": str(tmp_path / "val_pile.u16")}, str(sp), str(out))
    assert man["n_rows"] == 24 and man["synth"]["sha256"] and man["packs"]["rows"] == 120
    chk = M.check_row_map(str(out), "finetune", p)
    assert chk["rule_match"] and chk["synth_rows"] == 12
    assert M.manifest_summary(man)["arm"] == "F2"
    slim = tmp_path / "slim.json"
    M.write_manifest_without_rows({"seed": 0, "docs": [["src", f"id{i}"] for i in range(2000)], "sha256": {}}, str(slim))
    s = json.load(open(slim))
    assert "docs" not in s and s["docs_list"]["n_docs"] == 2000 and len(s["docs_list"]["sample_first_1000"]) == 1000


def test_window_hashes_and_lora_rows():
    h = M.window_hashes("a" * 60)
    assert len(h) == 11 and (h == h[0]).all()
    assert len(M.window_hashes("short")) == 0
    tok = ToyTok()
    rows, info = PK.lora_rows(tok, ["s" * 3000] * 40, ["r" * 3000] * 40, syn_tokens=4 * PK.SEQ, replay_tokens=4 * PK.SEQ, global_batch=2)
    assert rows.shape == (8, PK.SEQ) and info["steps"] == 4 and rows.dtype == np.uint32
    assert (rows[0] == tok("s")["input_ids"][0]).all() and (rows[1] == tok("r")["input_ids"][0]).all()


def test_freeze_and_qualification(tmp_path):
    cfg = load_config("exercises-finetune")
    r1 = freeze_record(cfg, root=ROOT_training)
    r2 = freeze_record(cfg, root=ROOT_training)
    for k in ("experiment", "config_sha256", "files_sha256", "manifests"):
        assert r1[k] == r2[k]
    assert abs(wilson_lower(50, 100) - 0.4038) < 1e-3
    screen = {"trec6": {"n": 200, "natural_acc": 0.6, "chance": 1 / 6}, "dbpedia7": {"n": 200, "natural_acc": 0.97, "chance": 1 / 7}, "yahoo5": {"n": 200, "natural_acc": 0.5, "chance": 0.2}, "ng6": {"n": 200, "natural_acc": 0.3, "chance": 1 / 6}}
    q = qualify_unseen(screen)
    assert q["datasets"] == ["trec6", "yahoo5", "ng6"] and len(q["assays"]) == 6


def test_launchers_parse():
    d = os.path.join(ROOT_training, "icl_copy_reason", "training", "launch")
    for f in sorted(os.listdir(d)):
        if f.endswith(".sh"):
            r = subprocess.run(["bash", "-n", os.path.join(d, f)], capture_output=True, text=True)
            assert r.returncode == 0, f"{f}: {r.stderr}"


# ---- from test_trajectories ----

CONDS = ["none", "competitor_marker", "wordB", "wordA", "single2_marker", "sham", "competitor"]


def test_registry_every_family_sorted():
    fams = families()
    assert len(fams) >= 29
    for name in fams:
        F = load_family(name)
        assert F.repo and F.final_model
        for grid, revs in F.grids.items():
            pos = [r["position"] for r in revs if r["position"] is not None]
            assert pos == sorted(pos), (name, grid)
            assert all(F.position_of(result_key(r["ref"])) is not None for r in revs if r["position"] is not None), (name, grid)
        if F.survey_status == "B-ineligible":
            assert F.default_grid is None
        else:
            assert F.refs(), name


def test_position_parsing():
    assert position_from_key("stage1-step80000-tokens336B") == 336.0
    assert position_from_key("step143000") == pytest.approx(143000 * 2097152 / 1e9)
    assert position_from_key("ckpt_359") == pytest.approx(1260.0)
    assert position_from_key("train_00220B") == 220.0
    assert position_from_key("main") is None
    assert result_key("bigscience/bloom-7b1-intermediate@global_step1000") == "global_step1000"
    assert result_key("PrimeIntellect/INTELLECT-1-step-15500") == "main"


def test_log_spaced_selection():
    cands = [(f"c{i}", float(2**i)) for i in range(40)]
    sel = select_log_spaced(cands, 20)
    assert len(sel) == 20 and sel[0] == cands[0] and sel[-1] == cands[-1]
    assert [t for _, t in sel] == sorted(t for _, t in sel)
    assert select_log_spaced(cands[:7], 20) == cands[:7]


def _synthetic_files(tmp_path, n=24, seeds=(2, 3)):
    """A run whose D is positive up to 300B and negative from 600B, with four-way accuracy rising at ~2T."""
    rng = np.random.default_rng(1)
    xs = [5, 20, 80, 300, 600, 1200, 2400, 4800]
    for s in seeds:
        ko, tr = {}, {}
        for x in xs:
            d_cm = 1.5 if x <= 300 else -0.3
            d_wb = 0.5 if x <= 300 else -1.5
            base = rng.normal(-1, 0.3, n)
            rev = {"fingerprint": 0.0, "n": n, "scheme": "morph"}
            for c in CONDS:
                shift = {"none": 0.0, "competitor_marker": d_cm, "wordB": d_wb, "wordA": -0.1, "single2_marker": 0.05, "sham": 0.0, "competitor": d_cm}[c]
                items = [{"margin": float(b + shift + rng.normal(0, 0.05)), "a_null": float(0.9 if x < 300 else 0.3)} for b in base]
                rev[c] = {"items": items}
            ko[f"stage1-step{x}-tokens{x}B"] = rev
            p = 0.1 if x < 1200 else (0.3 if x < 2400 else 0.7)
            acc = rng.random(n) < p
            tr[f"stage1-step{x}-tokens{x}B"] = {"compose_h": {"four_way_items": [{"acc": int(a), "competitor": int((not a) and rng.random() < 0.5)} for a in acc]}}
        json.dump(ko, open(tmp_path / f"knockout_toy_s{s}_t.json", "w"))
        json.dump(tr, open(tmp_path / f"triage_toy_s{s}_t.json", "w"))
    return n, xs


def test_summary_interval_and_schema(tmp_path):
    n, xs = _synthetic_files(tmp_path)
    r = crossings.summarize(tmp_path, "toy", [2, 3], "_t", n, position_from_key, descriptive=True, B=50, onset_B=20)
    assert r["D_last_positive_mean"] == 300.0 and r["D_first_negative_mean"] == 600.0 and r["D_no_return"]
    assert r["D_sign_changes_of_mean"] == 1 and r["T"]["T4_single_sign_change"]
    assert r["anull"]["release_le_.55_after_null"] == 300.0
    for k in ("stage1_checkpoints_B", "D_mean_series", "D_ci_lo", "D_ci_hi", "four_way_onset", "T", "joint_bootstrap", "rows"):
        assert k in r
    assert set(r["rows"][0]) >= {"n", "a_null", "D", "key", "x", "released_model", "four_way_acc", "other_marker_share", "competitor_marker", "wordB"}
    on = r["four_way_onset"]["x"]
    assert on is not None and 1000 < on < 4800
    fc = crossings.first_crossings(r, "toy")
    assert fc["dneg"] == 600.0 and fc["a05"] == 300.0 and fc["fw25"] in (1200.0, 2400.0)


def test_fit_onset_recovers_threshold():
    x = np.log10([10, 30, 100, 300, 1000, 3000, 10000])
    per = [np.array([1.0] * int(60 * p) + [0.0] * (60 - int(60 * p))) for p in (0.05, 0.08, 0.15, 0.4, 0.75, 0.9, 0.95)]
    xc, lo, hi, cens = onsets.fit_onset(x, per, 0.25, B=30)
    assert 2.2 < xc < 2.7 and cens == 0.0 and lo <= xc <= hi


def test_summary_markdown_and_tmp_cleanup(tmp_path):
    n, _ = _synthetic_files(tmp_path)
    r = crossings.summarize(tmp_path, "toy", [2, 3], "_t", n, position_from_key, descriptive=True, B=10, onset_B=5)
    md = crossings.summary_markdown(r, "toy")
    assert md.startswith("# toy") and "D_first_negative_mean" in md
    assert os.path.exists(tmp_path / "knockout_toy_s2_t.json")


# ---- from test_integration ----
ROOT_integration = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = os.path.join(ROOT_integration, "icl_copy_reason", "training", "launch")


def _choices(script: str) -> set[str]:
    """Positional choices of a CLI, read from its --help (argparse prints {a,b,c})."""
    r = subprocess.run([sys.executable, os.path.join(ROOT_integration, "scripts", script), "--help"], cwd=ROOT_integration, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    out = set()
    for m in re.finditer(r"\{([a-z0-9,_-]+)\}", r.stdout):
        out |= set(m.group(1).split(","))
    return out


def _modes(script: str) -> set[str]:
    r = subprocess.run([sys.executable, os.path.join(ROOT_integration, "scripts", script), "--help"], cwd=ROOT_integration, capture_output=True, text=True, timeout=120)
    m = re.search(r"--mode\s*\{([a-z0-9,_-]+)\}", r.stdout.replace("\n", " ").replace("  ", " "))
    return set(m.group(1).split(",")) if m else set()


def _help_text(script: str) -> str:
    r = subprocess.run([sys.executable, os.path.join(ROOT_integration, "scripts", script), "--help"], cwd=ROOT_integration, capture_output=True, text=True, timeout=120)
    return r.stdout


@pytest.fixture(scope="module")
def launchers():
    files = sorted(glob.glob(os.path.join(LAUNCH, "*.sh")))
    assert files
    return {os.path.basename(f): open(f).read() for f in files}



def test_subcommands_exist(launchers):
    sub = {s: _choices(s) for s in ("score_model.py", "build_external_sets.py", "build_exercises.py", "build_synthetic_docs.py")}
    train_modes = _modes("train.py")
    assert {"pack", "overlap-check", "freeze", "eval-loss"} <= train_modes
    pat = re.compile(r'scripts/([a-z_]+\.py)"?\s+([a-z][a-z0-9-]*)')
    for name, text in launchers.items():
        for m in pat.finditer(text):
            script, first = m.group(1), m.group(2)
            if script in sub:
                assert first in sub[script], f"{name}: {script} {first} is not a sub-command ({sorted(sub[script])})"
        for m in re.finditer(r'scripts/train\.py"[^\n;|&]*?--mode\s+([a-z-]+)', text):
            assert m.group(1) in train_modes, f"{name}: train.py --mode {m.group(1)}"
        if "make_tables.py" in text:
            for m in re.finditer(r'make_tables\.py"?\s+(\S+)', text):
                assert m.group(1).startswith("--"), f"{name}: make_tables.py takes no sub-command ({m.group(1)})"
        if "trace_checkpoints.py" in text:
            h = _help_text("trace_checkpoints.py")
            for flag in re.findall(r'trace_checkpoints\.py"[^\n|]*?(--[a-z-]+)', text):
                assert flag in h, f"{name}: trace_checkpoints.py {flag}"


def test_configs_named_exist(launchers):
    configs = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ROOT_integration, "configs", "*.yaml"))}
    texts = dict(launchers)
    for p in glob.glob(os.path.join(ROOT_integration, "icl_copy_reason", "**", "*.py"), recursive=True) + glob.glob(os.path.join(ROOT_integration, "scripts", "*.py")):
        texts[p] = open(p).read()
    for name, text in texts.items():
        for m in re.finditer(r"--config\s+([A-Za-z0-9][A-Za-z0-9_-]+)\b", text):
            c = m.group(1)
            if c.startswith("$") or c in {"traj", "is", "CONFIG", "name"}:
                continue
            if c.endswith("-") or "<" in c:
                continue
            assert c in configs or c.startswith("traj-") and c[5:] in {x[5:] for x in configs if x.startswith("traj-")} or c == "traj-pythia-2", f"{name}: --config {c} has no configs/{c}.yaml"
