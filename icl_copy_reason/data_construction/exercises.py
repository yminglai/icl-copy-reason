"""Real documents -> definition-use exercises: generate, retry, verify, audit (one OpenAI-compatible client).

Inputs : docs.jsonl from `documents.sample_documents`, a prompt table (data/exercises/prompts.json for the topped-up set with
         type h; data/exercises/rewrite_prompts.json for the first set, see `augment_real`), the evaluation-window store,
         and the API settings (ICR_API_BASE / ICR_API_KEY; generator / verifier / auditor model names from the config).
Outputs: gen.jsonl (one generator record per document), verified.jsonl (+ verify_summary.json; one record per document and
         per retry, every check and verifier verdict kept), audit.jsonl / audit.json (seeded sample judged by a third model),
         and router_calls.json (token accounting per step). Record schema = the released data/exercises/exercises.jsonl.gz.

Pipeline (fixed before the run): one generator call per document (JSON reply; one re-ask when the reply does not parse);
verifier V1 answers the query from the exercise context only (must equal the stated answer), V2 from its own knowledge
(must differ), V3 from the source document (must differ); checks 1-6 of `verification`; type-h documents whose first draw
fails get ONE retry generation; the audit samples 50 verified exercises per type (seed 13) and judges four booleans
(faithful, novel_binding, context_only, answer_correct) plus rule_not_shortcut for type h; a type is retained when >= .85
of its verified sample is all-true.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

from ..api_client import ChatClient
from ..config import ApiSettings, data_root
from ..io import read_json, read_jsonl, sha256_file, sha256_text, write_json
from . import verification as V

EX_KEYS_DEFAULT = ("context", "query", "answer", "justification", "type")
EX_KEYS_BY_TYPE = {"h": ("context", "query", "answer", "justification", "type", "key_term", "copy_candidate", "prior_candidate")}
AUDIT_KEYS4 = ("faithful", "novel_binding", "context_only", "answer_correct")
AUDIT_KEYS5 = AUDIT_KEYS4 + ("rule_not_shortcut",)
_lock = threading.Lock()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_prompts(path: str | Path | None = None) -> dict:
    """The prompt table (default data/exercises/prompts.json). Adds `_sha256` and `_path`."""
    p = Path(path) if path else data_root() / "exercises" / "prompts.json"
    P = read_json(p)
    P["_sha256"] = sha256_file(p)
    P["_path"] = str(p)
    return P


class Stats:
    """Throughput counters for one pipeline step."""

    def __init__(self, name: str):
        self.name, self.n, self.lat, self.inp, self.out, self.err, self.t0 = name, 0, 0.0, 0, 0, 0, time.time()

    def add(self, r: dict) -> None:
        with _lock:
            self.n += 1
            self.lat += r["latency"]
            u = r.get("usage") or {}
            self.inp += u.get("prompt_tokens") or 0
            self.out += u.get("completion_tokens") or 0
            self.err += int(bool(r.get("error")))
            if self.n % 200 == 0:
                print(self.line(), flush=True)

    def line(self) -> str:
        el = max(1e-6, time.time() - self.t0)
        return f"[{self.name}] calls={self.n} calls/min={60 * self.n / el:.1f} mean_latency={self.lat / max(1, self.n):.1f}s tokens in/out={self.inp}/{self.out} errors={self.err} elapsed={el / 60:.1f}min"

    def dump(self) -> dict:
        el = time.time() - self.t0
        return {"calls": self.n, "calls_per_min": 60 * self.n / max(1e-6, el), "mean_latency_s": self.lat / max(1, self.n), "prompt_tokens": self.inp, "completion_tokens": self.out, "errors": self.err, "wall_s": el}


def log_router(outdir: str | Path, step: str, model: str, st: Stats, **extra: Any) -> None:
    """Append one record per pipeline step (calls, tokens, errors) to <outdir>/router_calls.json."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "router_calls.json"
    recs = read_json(path) if path.exists() else []
    recs.append({"utc": utc(), "step": step, "model": model, **st.dump(), **extra})
    write_json(recs, path)


def make_client(P: dict, settings: ApiSettings | None = None) -> ChatClient:
    s = settings or ApiSettings.from_env()
    return ChatClient(s)


def models(P: dict, settings: ApiSettings | None = None) -> dict[str, str]:
    """generator / verifier / auditor model names: the config/env settings override the prompt table's defaults."""
    s = settings or ApiSettings.from_env()
    return {"generator": s.generator_model or P["generator_model"], "verifier": s.verifier_model or P["verifier_model"], "auditor": s.auditor_model or P["audit_model"]}


# ---------------------------------------------------------------- generate / retry
def gen_prompts(P: dict, doc: dict) -> tuple[str, str]:
    t = doc["type"]
    spec = P["types"][t]
    tpl = P.get("gen_user_h") if t == "h" else P["gen_user"]
    extra = ("\n" + P["gen_ban_topup"]) if P.get("gen_ban_topup") and t in P.get("topup_types", []) else ""
    return P["gen_system"], V.fill(tpl, type=t, name=spec["name"], spec=spec["spec"] + extra, document=doc["text"][: P["document_chars"]["prompt_truncate"]])


def generate(P: dict, docs_path: str | Path, out: str | Path, workers: int = 64, types: list[str] | None = None, timeout: float = 600, retry_from: str | Path | None = None,
             redo_failed: bool = False, dry_run: bool = False, settings: ApiSettings | None = None) -> dict:
    """One generator call per document (or per retry candidate when retry_from is given); resumable; appends to `out`."""
    out = Path(out)
    docs = [d for d in read_jsonl(docs_path) if (not types or d["type"] in types)]
    out.parent.mkdir(parents=True, exist_ok=True)
    is_retry = retry_from is not None
    if is_retry:
        docs = V.retry_candidates(retry_from, docs, P.get("retry_type", "h"))
        print(f"retry ({P.get('retry_type', 'h')}, one call per failed document): {len(docs)} documents selected from {retry_from}", flush=True)
        if dry_run:
            return {"n_retry_docs": len(docs), "pools": dict(collections.Counter(d["pool"] for d in docs))}
    if redo_failed and out.exists():
        keep = [d for d in read_jsonl(out) if d.get("parsed")]
        with open(out, "w") as f:
            for d in keep:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    done = {d["doc_sha256"] for d in read_jsonl(out)} if out.exists() else set()
    todo = [d for d in docs if d["sha256"] not in done]
    print(f"generate: {len(docs)} docs, {len(done)} done, {len(todo)} to do, workers {workers}", flush=True)
    M = models(P, settings)
    client = make_client(P, settings)
    model = M["generator"]
    st = Stats("generate")
    max_out = P["max_output_tokens"]["generator"]
    gs = P.get("generator_settings", {})
    extra_body = {"thinking": {"type": "disabled"}} if gs.get("thinking", "disabled") == "disabled" else None
    temp = P["temperature"].get("generator")
    outf = open(out, "a")

    def one(d: dict) -> bool:
        system, user = gen_prompts(P, d)
        psha = sha256_text(system + "\n\x00\n" + user)
        keys = EX_KEYS_BY_TYPE.get(d["type"], EX_KEYS_DEFAULT)
        rec = {"doc_sha256": d["sha256"], "pool": d["pool"], "file": d["file"], "row": d["row"], "type": d["type"], "generator_model": model, "prompt_sha256": psha, "prompts_file_sha256": P["_sha256"], "retry": is_retry}
        r = client.complete(system, user, model, temp, max_out, timeout_s=timeout, extra_body=extra_body)
        obj = V.parse_json_obj(r["content"]) if not r["error"] else None
        retry = None
        if obj is None and not r["error"]:
            retry = client.complete(system, user, model, temp, max_out, timeout_s=timeout, extra_body=extra_body)
            obj = V.parse_json_obj(retry["content"]) if not retry["error"] else None
        ok = obj is not None and all(isinstance(obj.get(k), str) and obj.get(k).strip() for k in ("context", "query", "answer"))
        rec.update({"max_tokens": max_out, "raw": r["content"], "finish_reason": r["finish_reason"], "usage": r["usage"], "latency": r["latency"], "temperature_applied": r["temperature_applied"], "attempts": r["attempts"],
                    "served_model": r["served_model"], "error": r["error"],
                    "parse_retry": None if retry is None else {"raw": retry["content"], "usage": retry["usage"], "latency": retry["latency"], "error": retry["error"], "finish_reason": retry["finish_reason"]},
                    "parsed": ok, "parse_failed": (not ok) and not r["error"], "utc": utc(),
                    "exercise": {k: (obj.get(k) if isinstance(obj.get(k), str) else (None if obj.get(k) is None else json.dumps(obj.get(k)))) for k in keys} if ok else None})
        st.add(r)
        if retry:
            st.add(retry)
        with _lock:
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outf.flush()
        return ok

    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, todo))
    outf.close()
    print(st.line(), flush=True)
    run = {"utc": utc(), "model": model, "workers": workers, "max_tokens": max_out, "retry": is_retry, "retry_from": str(retry_from) if retry_from else None, "n_docs": len(docs), "n_new": len(todo),
           "n_parsed_new": sum(res), "temperature_configured": temp, "temperature_dropped_by_endpoint": model in client.no_temperature, "throughput": st.dump(), "prompts_sha256": P["_sha256"]}
    write_json(run, str(out).replace(".jsonl", "") + "_run.json")
    log_router(out.parent, "retry" if is_retry else "generate", model, st, n_docs=len(docs), n_new=len(todo))
    return run


# ---------------------------------------------------------------- verify
def verifier_call(client: ChatClient, P: dict, model: str, system: str, user: str, timeout: float) -> dict:
    """reasoning effort low, the verifier budget; an EMPTY answer is retried once at the larger budget; still empty -> 'unverifiable' (fails)"""
    b = P["max_output_tokens"]["verifier"]
    b2 = P["max_output_tokens"].get("verifier_retry", 4 * b)
    effort = P.get("verifier_settings", {}).get("reasoning_effort", "low")
    temp = P["temperature"].get("verifier")
    kw = {"reasoning_effort": effort} if effort else {}
    r = client.complete(system, user, model, temp, b, timeout_s=timeout, **kw)
    r["budget_retry"] = False
    r["unverifiable"] = False
    if not r["error"] and not r["content"].strip():
        r2 = client.complete(system, user, model, temp, b2, timeout_s=timeout, **kw)
        r2["budget_retry"] = True
        r2["first_usage"] = r["usage"]
        r2["first_finish_reason"] = r["finish_reason"]
        r2["unverifiable"] = (not r2["error"]) and not r2["content"].strip()
        return r2
    return r


def get_tokenizer(P: dict):
    from transformers import AutoTokenizer

    name, rev = P["tokenizer"]["name"], P["tokenizer"].get("revision")
    return AutoTokenizer.from_pretrained(name, revision=rev)


def verify(P: dict, gen_path: str | Path, docs_path: str | Path, out: str | Path, windows_dir: str | Path, workers: int = 64, timeout: float = 180, resume: bool = False,
           settings: ApiSettings | None = None, with_check6: bool | None = None) -> dict:
    """V1/V2/V3 verifier calls + the mechanical checks -> verified.jsonl (sorted by pool, file, row, retry) and verify_summary.json."""
    out = Path(out)
    tok = get_tokenizer(P)
    evalh, evinfo = V.load_eval_windows(windows_dir)
    docs = {d["sha256"]: d for d in read_jsonl(docs_path)}
    gens = sorted([g for g in read_jsonl(gen_path) if g.get("parsed")], key=lambda g: (g["pool"], g["file"], g["row"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(d["doc_sha256"], bool(d.get("retry"))) for d in read_jsonl(out)} if out.exists() and resume else set()
    todo = [g for g in gens if (g["doc_sha256"], bool(g.get("retry"))) not in done]
    print(f"verify: {len(gens)} parsed exercises, {len(done)} done, {len(todo)} to do; eval windows {evinfo['n_windows']} (built {evinfo['built_utc']})", flush=True)
    M = models(P, settings)
    client = make_client(P, settings)
    model = M["verifier"]
    st = Stats("verify")
    class_norms = V.class_name_norms(P)
    use6 = P.get("check6", True) if with_check6 is None else with_check6
    trunc = P["document_chars"]["prompt_truncate"]

    def vrec(v: dict, vn: str, p: bool) -> dict:
        return {"raw": v["content"], "norm": vn, "pass": p, "usage": v["usage"], "latency": v["latency"], "error": v["error"], "budget_retry": v["budget_retry"], "unverifiable": v["unverifiable"],
                "finish_reason": v["finish_reason"], "temperature_applied": v["temperature_applied"]}

    def one(g: dict) -> dict:
        ex, t = g["exercise"], g["type"]
        mech = V.mechanical_checks(ex, t, P, tok, evalh, class_norms, with_check6=use6)
        ans_n = mech["answer_norm"]
        reasons = list(mech.pop("mechanical_reasons"))
        doc = docs.get(g["doc_sha256"], {}).get("text", "")
        v1 = verifier_call(client, P, model, P["v1_system"], V.fill(P["v1_user"], context=ex["context"], query=ex["query"]), timeout)
        v2 = verifier_call(client, P, model, P["v2_system"], V.fill(P["v2_user"], query=ex["query"]), timeout)
        v3 = verifier_call(client, P, model, P["v3_system"], V.fill(P["v3_user"], document=doc[:trunc], query=ex["query"]), timeout)
        for v in (v1, v2, v3):
            st.add(v)
        v1n, v2n, v3n = V.norm_answer(v1["content"]), V.norm_answer(v2["content"]), V.norm_answer(v3["content"])
        p1 = (v1n == ans_n) and not v1["error"] and not v1["unverifiable"]
        p2 = (v2n != ans_n) and not v2["error"] and not v2["unverifiable"]
        p3 = (v3n != ans_n) and not v3["error"] and not v3["unverifiable"]
        vreasons = []
        if not p1:
            vreasons.append("V1_unverifiable" if v1["unverifiable"] else ("V1_mismatch" if not v1["error"] else "V1_error"))
        if not p2:
            vreasons.append("V2_unverifiable" if v2["unverifiable"] else ("V2_solved_without_context" if not v2["error"] else "V2_error"))
        if not p3:
            vreasons.append("V3_unverifiable" if v3["unverifiable"] else ("V3_solved_from_document" if not v3["error"] else "V3_error"))
        reasons = vreasons + reasons
        return {**{k: g[k] for k in ("doc_sha256", "pool", "file", "row", "type", "prompt_sha256")}, "generator_model": g.get("generator_model") or g.get("model"), "retry": bool(g.get("retry")), "exercise": ex,
                **mech, "v1": vrec(v1, v1n, p1), "v2": vrec(v2, v2n, p2), "v3": vrec(v3, v3n, p3), "pass": not reasons, "reasons": reasons, "verifier_model": model,
                "eval_windows_sha256": evinfo["sha256_npy"], "utc": utc()}

    with cf.ThreadPoolExecutor(workers) as ex:
        new = list(ex.map(one, todo))
    allrec = (list(read_jsonl(out)) if done else []) + new
    allrec.sort(key=lambda r: (r["pool"], r["file"], r["row"], bool(r.get("retry"))))
    with open(out, "w") as f:
        for r in allrec:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(st.line(), flush=True)
    summ = summarise_verified(P, allrec, model, evinfo, client, st)
    name = "verify_summary.json" if out.name == "verified.jsonl" else out.name.replace(".jsonl", "") + "_summary.json"
    write_json(summ, out.parent / name)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("by_type", "total")}, indent=1))
    log_router(out.parent, "verify", model, st, n_exercises=len(todo))
    return summ


def summarise_verified(P: dict, allrec: list[dict], model: str, evinfo: dict, client: ChatClient | None, st: Stats | None) -> dict:
    types = sorted(P["types"])
    by = {t: collections.Counter() for t in types}
    for r in allrec:
        c = by.setdefault(r["type"], collections.Counter())
        c["n"] += 1
        c["pass"] += r["pass"]
        for k in ("mech1", "mech2", "mech3", "mech4", "mech5"):
            c[k + "_ok"] += bool(r[k])
        if "mech6" in r:
            c["mech6_ok"] += bool(r["mech6"]["pass"])
            for k in r["mech6"]["enforced"]:
                c[k + "_ok"] += bool(r["mech6"][k])
        for k in ("v1", "v2", "v3"):
            c[k + "_pass"] += bool(r[k]["pass"])
            c[k + "_budget_retry"] += bool(r[k].get("budget_retry"))
            c[k + "_error"] += bool(r[k]["error"])
            c[k + "_unverifiable"] += bool(r[k].get("unverifiable"))
        for why in r["reasons"]:
            c["reason:" + why] += 1
    tot: collections.Counter = collections.Counter()
    for c in by.values():
        tot.update(c)
    rtype = P.get("retry_type", "h")
    rt = [r for r in allrec if r.get("retry")]
    first_fail = {r["doc_sha256"] for r in allrec if r["type"] == rtype and not r.get("retry") and not r["pass"]}
    retry_info = {"rule": P.get("retry_rule"), "n_first_pass_failed": len(first_fail), "n_retried": len(rt), "n_retry_pass": sum(r["pass"] for r in rt), "retry_pass_rate": sum(r["pass"] for r in rt) / max(1, len(rt)),
                  "n_docs_pass_after_retry": len({r["doc_sha256"] for r in allrec if r["type"] == rtype and r["pass"]}), "retry_reasons": dict(collections.Counter(x for r in rt for x in r["reasons"]))}
    return {"utc": utc(), "verifier_model": model, "retry": retry_info, "temperature_configured": P["temperature"].get("verifier"), "temperature_dropped_by_endpoint": bool(client and model in client.no_temperature),
            "reasoning_effort": P.get("verifier_settings", {}).get("reasoning_effort"), "budgets": [P["max_output_tokens"]["verifier"], P["max_output_tokens"].get("verifier_retry")],
            "n": len(allrec), "n_pass": tot["pass"], "pass_rate": tot["pass"] / max(1, len(allrec)),
            "rates": {k: tot[k] / max(1, len(allrec)) for k in ("v1_pass", "v2_pass", "v3_pass", "mech1_ok", "mech2_ok", "mech3_ok", "mech4_ok", "mech5_ok", "mech6_ok")},
            "by_type": {t: dict(c) for t, c in by.items() if c}, "total": dict(tot), "eval_windows": {k: evinfo.get(k) for k in ("built_utc", "n_windows", "sha256_npy")},
            "tokenizer": P["tokenizer"], "throughput": st.dump() if st else None, "prompts_sha256": P["_sha256"]}


# ---------------------------------------------------------------- audit
def audit(P: dict, verified_path: str | Path, docs_path: str | Path, per_type: int = 50, types: list[str] | None = None, workers: int = 16, timeout: float = 300,
          include_unverified: bool = False, tag: str = "", seed: int = 13, gate: float = 0.85, settings: ApiSettings | None = None) -> dict:
    """Seeded sample of verified exercises per type judged by the auditor model -> audit<tag>.jsonl / audit<tag>.json; a type is retained at >= gate all-true."""
    docs = {d["sha256"]: d for d in read_jsonl(docs_path)}
    recs = list(read_jsonl(verified_path))
    rng = random.Random(seed)
    types = types or P.get("topup_types") or sorted(P["types"])
    sample: list[dict] = []
    for t in types:
        cands = [r for r in recs if r["type"] == t and (r["pass"] or include_unverified)]
        sample += rng.sample(cands, min(per_type, len(cands)))
    M = models(P, settings)
    client = make_client(P, settings)
    model = M["auditor"]
    st = Stats("audit")
    print(f"audit: {len(sample)} exercises ({per_type} per type; types {types}; all={include_unverified}) with {model}", flush=True)
    trunc = P["document_chars"]["prompt_truncate"]
    temp = P["temperature"].get("audit", 0)
    budget = P["max_output_tokens"]["audit"]

    def one(r: dict) -> dict:
        ex = r["exercise"]
        h = r["type"] == "h" and "audit_system_h" in P
        sysm, usr, keys = (P["audit_system_h"], P["audit_user_h"], AUDIT_KEYS5) if h else (P["audit_system"], P["audit_user"], AUDIT_KEYS4)
        user = V.fill(usr, document=docs.get(r["doc_sha256"], {}).get("text", "")[:trunc], context=ex["context"], query=ex["query"], answer=ex["answer"])
        c = client.complete(sysm, user, model, temp, budget, timeout_s=timeout)
        obj = V.parse_json_obj(c["content"]) if not c["error"] else None
        if obj is None and not c["error"]:
            c = client.complete(sysm, user + "\nReturn ONLY the JSON object.", model, temp, 1500, timeout_s=timeout)
            obj = V.parse_json_obj(c["content"]) if not c["error"] else None
        st.add(c)
        vals = {k: (obj.get(k) is True) if obj else None for k in keys}
        return {"doc_sha256": r["doc_sha256"], "type": r["type"], "verified_pass": r["pass"], "audit": vals, "keys": list(keys), "all_true": all(vals[k] for k in keys) if obj else False,
                "note": (obj or {}).get("note"), "raw": c["content"], "usage": c["usage"], "latency": c["latency"], "error": c["error"], "model": model, "utc": utc()}

    with cf.ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, sample))
    d = Path(verified_path).parent
    with open(d / f"audit{tag}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    def share(rs: list[dict]) -> float:
        return sum(o["all_true"] for o in rs) / max(1, len(rs))

    ver = [o for o in out if o["verified_pass"]]
    per = {t: {"n": len([o for o in out if o["type"] == t]), "n_verified": len([o for o in ver if o["type"] == t]), "all_true_verified": share([o for o in ver if o["type"] == t]),
               "all_true": share([o for o in out if o["type"] == t]), **{k: sum(bool((o["audit"] or {}).get(k)) for o in out if o["type"] == t) for k in (AUDIT_KEYS5 if t == "h" else AUDIT_KEYS4)}} for t in types}
    retained = [t for t in types if per[t]["n_verified"] > 0 and per[t]["all_true_verified"] >= gate]
    pooled = share([o for o in ver if o["type"] in retained])
    summ = {"utc": utc(), "model": model, "seed": seed, "per_type_target": per_type, "n": len(out), "n_verified_in_sample": len(ver), "share_all_true": share(out), "share_all_true_verified_only": share(ver),
            "gate_per_type": gate, "per_type": per, "retained_types": retained, "dropped_types": [t for t in types if t not in retained], "pooled_share_retained": pooled, "rule": P.get("audit_rule"),
            "errors": sum(bool(o["error"]) for o in out), "unparsed": sum(o["audit"] is None or None in o["audit"].values() for o in out), "throughput": st.dump(), "prompts_sha256": P["_sha256"], "include_unverified": include_unverified}
    write_json(summ, d / f"audit{tag}.json")
    print(json.dumps(summ, indent=1))
    log_router(d, "audit", model, st, n_exercises=len(sample))
    return summ


def run_offline_checks(P: dict, records: list[dict], tok=None, evalh=None) -> list[dict]:
    """The mechanical checks on shipped records (no API, no tokenizer when tok is None) — used by the tests and by users auditing the release."""
    class_norms = V.class_name_norms(P)
    return [V.mechanical_checks(r["exercise"], r["type"], P, tok, evalh, class_norms) for r in records]


def default_dirs(cfg: dict) -> dict[str, Path]:
    """Where a build writes: <ICR_RESULTS_ROOT>/<experiment>/{documents,generated,verified}."""
    from ..config import results_root

    base = Path(cfg.get("output_dir") or results_root() / cfg.get("name", "exercises-data"))
    return {"base": base, "windows": Path(cfg.get("windows_dir") or base / "eval_windows")}


def _env_note() -> str:
    return "set ICR_API_BASE and ICR_API_KEY" if not os.environ.get("ICR_API_KEY") else "API configured"
