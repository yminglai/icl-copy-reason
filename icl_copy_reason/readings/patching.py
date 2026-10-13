"""Activation patching: word -> slot residual patching, the selective-write subspace (DAS), induction-head ablation, the layer sweep and
the layer x position patch map.

Three procedures share this module:
  run            per model x revision (schema of the released causal_<model>_s<seed><tag>.json):
                 (1) cue conflict at the second slot with the stem teacher-forced
                 (2) per-head induction scores, zero / mean / resample
                 ablation of the top-k induction heads vs matched random heads (forced choice and, optionally, free generation);
                 (3) at a pre-output position ("=" / last stem token / the feature word): the full residual patch from a donor prompt
                 (other value of ONE feature, other stem) and, unless no_das, a rank-k orthonormal subspace trained so that interchanging
                 the donor's residual flips the intervened ending; floors = random rotation and same-value donor; a logistic probe.
  layer_sweep    the word -> slot patch at EVERY layer on the shown-combination panel (donor = same prompt, other value of the second
                 feature word); redirection = the generation parses to the combination the donor word denotes; identity check.
                 Schema of the released patch_sweep_<model>_s<seed>_lsweep.json.
  patch_map      layer x position patching from the word-swapped prompt into the withheld-combination prompt, scored by the decision
                 logit difference at the blank with clean labels fixed. Schema of the released mech patch_<model>_s<seed>.json.

Inputs : model name/revision, seeds, panel sizes, layers. Outputs: JSON records with per-item fields, written under out_dir.
"""
from __future__ import annotations

import gc
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..io import read_json, write_json
from ..tasks.prompts import WORDS_A, WORDS_B, Lang, build, categorise, four_forms, parse_form, score, score_cf, slots_of
from .copy_share import fc_prefix, fc_prefix_pair, load_model


def fit_logistic(X, y, n_classes, l2=1e-3, l1=0.0, steps=300, lr=0.5, seed=0, standardize=True):
    """Multinomial logistic regression with optional L1 (proximal), full batch. Returns (W, b, mu, sd)."""
    if standardize:
        mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    else:
        mu, sd = torch.zeros(1, X.shape[1], device=X.device), torch.ones(1, X.shape[1], device=X.device)
    Xs = (X - mu) / sd
    W = torch.zeros(X.shape[1], n_classes, device=X.device, requires_grad=True)
    b = torch.zeros(n_classes, device=X.device, requires_grad=True)
    lossf = torch.nn.CrossEntropyLoss()
    if l1 == 0.0:
        opt = torch.optim.LBFGS([W, b], lr=lr, max_iter=steps, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = lossf(Xs @ W + b, y) + l2 * (W * W).sum()
            loss.backward()
            return loss

        opt.step(closure)
    else:
        opt = torch.optim.Adam([W, b], lr=0.05)
        for _ in range(steps * 3):
            opt.zero_grad()
            loss = lossf(Xs @ W + b, y) + l2 * (W * W).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():  # proximal L1 step
                W.copy_(torch.sign(W) * torch.clamp(W.abs() - 0.05 * l1, min=0))
    return W.detach(), b.detach(), mu, sd


def predict(W, b, mu, sd, X):
    return ((X - mu) / sd @ W + b).argmax(1)


def attn_module(lm, layer):
    blk = lm.layers[layer]
    return getattr(blk, "attention", None) or blk.self_attn


def out_proj(lm, layer):
    attn = attn_module(lm, layer)
    return getattr(attn, "dense", None) or attn.o_proj


def head_slices(lm, layer):
    attn = attn_module(lm, layer)
    head_size = getattr(attn, "head_size", None) or attn.head_dim
    return attn.config.num_attention_heads, head_size


def induction_heads(lm, n=32, length=40, seed=0):
    """Per-head induction score [L, H]: mean attention from position t (second copy) to t-length+1."""
    g = torch.Generator().manual_seed(seed)
    vocab = lm.tok.vocab_size
    ids = torch.randint(1000, vocab - 1000, (n, length), generator=g)
    seq = torch.cat([ids, ids], dim=1).to(lm.device)
    with torch.no_grad():
        out = lm.model(input_ids=seq, output_attentions=True)
    scores = []
    for A in out.attentions:                                   # [B, H, T, T]
        A = A.float()
        t = torch.arange(length, 2 * length - 1, device=A.device)
        src = t - length + 1
        scores.append(A[:, :, t, src].mean(dim=(0, 2)).cpu())  # [H]
    return torch.stack(scores)                                 # [L, H]


def ablate_hook_factory(heads_by_layer, n_heads, head_size, mode="zero"):
    """returns make(layer) -> (pos_mask -> pre_hook) for generation-time ablation"""
    def make(layer):
        def with_mask(pos_mask):
            return ablate_hook(heads_by_layer, n_heads, head_size, pos_mask, mode)(layer)
        return with_mask
    return make


def ablate_hook(heads_by_layer, n_heads, head_size, pos_mask, mode="zero"):
    """forward_pre_hook on attention.dense acting on the given heads at masked positions: zero | mean (batch mean at the position) |
    resample (the same head's output from another item of the batch, torch.roll along the batch)."""
    def make(layer):
        def pre(module, inputs):
            x = inputs[0].clone()
            m = pos_mask.to(x.dtype)[:, :, None]
            for h in heads_by_layer[layer]:
                sl = slice(h * head_size, (h + 1) * head_size)
                cur = x[:, :, sl]
                if mode == "zero":
                    rep = torch.zeros_like(cur)
                elif mode == "mean":
                    act = pos_mask.any(dim=1)
                    src = cur[act] if act.sum() > 0 else cur
                    rep = src.mean(dim=0, keepdim=True).expand_as(cur)
                else:
                    # resample among rows that are ACTIVE at this call (same phase); fall back to the active-row mean if < 2 active
                    act = pos_mask.any(dim=1)
                    idx = torch.nonzero(act).flatten()
                    rep = cur.clone()
                    if len(idx) >= 2:
                        rep[idx] = cur[idx.roll(1)]
                    elif len(idx) == 1:
                        rep[idx] = cur[idx].mean(dim=0, keepdim=True)
                x[:, :, sl] = cur * (1 - m) + rep * m
            return (x,)
        return pre
    return make


def prefix_for(it, feature, lm=None):
    """teacher-forced prefix up to the slot of `feature` (surface order) and the (correct, other) candidate strings; tokenizer-aware
    (merged "-aff" tokens for Qwen-style tokenizers)."""
    lang = it.lang
    v, affs = (it.ia, lang["affA"]) if feature == "a" else (it.ib, lang["affB"])
    base = fc_prefix(it, feature)
    if lm is None:
        return " " + base + "-", (affs[v], affs[1 - v])
    pre, pr = fc_prefix_pair(lm, base, affs[v], affs[1 - v])
    if pr is None:
        pre, pr = base + "-", (affs[v], affs[1 - v])
    return " " + pre, pr


def generate_with_hook(lm, prompts, layer, fn, max_new=8):
    """Greedy decode with an intervention hook active during the prefill only (its effect persists through the KV cache)."""
    enc = lm.tok(prompts, return_tensors="pt", padding=True).to(lm.device)
    hd = lm.layers[layer].register_forward_hook(lambda m, i, o: ((fn(o[0]),) + tuple(o[1:])) if isinstance(o, tuple) else fn(o)) if fn is not None else None
    with torch.no_grad():
        try:
            out = lm.model(**enc, use_cache=True)
        finally:
            if hd is not None:
                hd.remove()
        pkv = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        toks = [nxt]
        am = enc["attention_mask"]
        for _ in range(max_new - 1):
            am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
            out = lm.model(input_ids=nxt[:, None], past_key_values=pkv, attention_mask=am, use_cache=True)
            pkv = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            toks.append(nxt)
    toks = torch.stack(toks, 1)
    return [lm.tok.decode(t, skip_special_tokens=True) for t in toks]


def generate_with_prehooks(lm, prompts, pre_hooks, stems, max_new=8):
    """Greedy decode with attention-output pre-hooks active only in the AFFIX phase: for item i, at decode steps where the text
    generated so far already contains the whole stem (so stem copying itself is never ablated). Prefill is unhooked."""
    enc = lm.tok(prompts, return_tensors="pt", padding=True).to(lm.device)
    B, T = enc["input_ids"].shape
    with torch.no_grad():
        out = lm.model(**enc, use_cache=True)
        pkv = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        toks = [nxt]
        am = enc["attention_mask"]
        for _ in range(max_new - 1):
            so_far = [lm.tok.decode(t, skip_special_tokens=True).strip() for t in torch.stack(toks, 1)]
            pm = torch.tensor([[len(sf) >= len(st) and sf.startswith(st)] for sf, st in zip(so_far, stems)], device=lm.device)
            hds = [out_proj(lm, layer).register_forward_pre_hook(mk(pm)) for (layer, mk) in pre_hooks]
            try:
                am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
                out = lm.model(input_ids=nxt[:, None], past_key_values=pkv, attention_mask=am, use_cache=True)
            finally:
                for h in hds:
                    h.remove()
            pkv = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            toks.append(nxt)
    toks = torch.stack(toks, 1)
    return [lm.tok.decode(t, skip_special_tokens=True) for t in toks]


def run(model: str, steps: str = "main", tasks: str = "bind_a,bind_b,compose_s", layers: str = "", position: str = "eq", ranks: str = "8",
        n_train: int = 240, n_test: int = 160, steps_opt: int = 300, bs: int = 16, n_heads_ablate: int = 8, seed: int = 0, tag: str = "",
        ablate_scope: str = "last", ablation: str = "zero", order: str = "ab", gen_ablation: bool = False, no_das: bool = False,
        n_control_draws: int = 2, holdout_pair: int = 0, skip_heads: bool = False, skip_das: bool = False, out_dir: str | Path = ".",
        device: str | None = None) -> Path:
    """Cue conflict + induction-head ablation (unless skip_heads) and the selective-write / full residual patch block (unless skip_das)
    for `model` at every revision in `steps`. The paper's word -> slot residual patch is `no_das=True, position="word", layers="4",
    skip_heads=True` (tag _path2 / _path4)
    the DAS replication is `position="word", layers="4", ranks="8", skip_heads=True` (tag _das3).
    Returns <out_dir>/causal_<model>_s<seed><tag>.json (schema of the released files)."""
    args = SimpleNamespace(model=model, steps=steps, tasks=tasks, layers=layers, position=position, ranks=ranks, n_train=n_train, n_test=n_test,
                           steps_opt=steps_opt, bs=bs, n_heads_ablate=n_heads_ablate, seed=seed, tag=tag, ablate_scope=ablate_scope, ablation=ablation,
                           order=order, gen_ablation=gen_ablation, no_das=no_das, n_control_draws=n_control_draws, holdout_pair=holdout_pair,
                           skip_heads=skip_heads, skip_das=skip_das)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    outp = out_dir / f"causal_{tag}_s{args.seed}{args.tag}.json"
    res = read_json(outp) if outp.exists() else {}
    torch.manual_seed(args.seed)
    for step in args.steps.split(","):
        rev = None if step in ("main", "") else (step if step.startswith("stage") else f"step{step}")
        step_key = rev or "main"
        if step_key in res:
            continue
        t0 = time.time()
        lm = load_model(args.model, revision=rev, device=device)
        lm.model.config._attn_implementation = "eager"
        for _L in lm.layers:
            attn_module(lm, 0).config._attn_implementation = "eager"
        lm.text_cfg._attn_implementation = "eager"
        n_heads, head_size = head_slices(lm, 0)
        layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted(set([lm.n_layers // 4, lm.n_layers // 2, (3 * lm.n_layers) // 4]))
        rec = {"n_layers": lm.n_layers}

        def fc_batch(prompts, pairs, interventions=None, pre_hooks=None):
            hds = []
            for (layer, fn) in (pre_hooks or []):
                hds.append(out_proj(lm, layer).register_forward_pre_hook(fn))
            try:
                enc, _ = lm.encode(prompts, [[len(p)] for p in prompts])
                lg, _ = lm.forward(enc, interventions=interventions)
            finally:
                for h in hds:
                    h.remove()
            out = []
            for k, (c, w) in enumerate(pairs):
                out.append(float(lg[k, lm.first_token_id(c)] > lg[k, lm.first_token_id(w)]))
            return out

        # ---------- (1) cue conflict + (2) induction heads
        if not args.skip_heads:
            ind = induction_heads(lm)
            flat = [(float(ind[lyr, h]), lyr, h) for lyr in range(ind.shape[0]) for h in range(ind.shape[1])]
            flat.sort(reverse=True)
            top = flat[:args.n_heads_ablate]
            rng = random.Random(args.seed)
            excluded = set((lyr, h) for s_, lyr, h in flat[:2 * args.n_heads_ablate])
            def matched_draw():
                out = []
                for s_, lyr, _h in top:
                    pool = [(lyr, hh) for hh in range(ind.shape[1]) if (lyr, hh) not in excluded and (lyr, hh) not in out]
                    out.append(rng.choice(pool))
                return [(0.0, lyr, h) for lyr, h in out]
            rnd_draws = [matched_draw() for _ in range(args.n_control_draws)]
            rnd_draws[0]
            rec["induction_heads"] = [(lyr, h, round(s, 3)) for s, lyr, h in top]
            rec["induction_top_score"] = round(top[0][0], 3)
            rec["ablation"] = {}
            for task in ["compose_h", "compose_s", "bind_b", "bind_a"]:
                items = build(task, args.n_test, seed=args.seed + 7, order=args.order)
                feature = "a" if task == "bind_a" else ("b" if task == "bind_b" else ("a" if args.order == "ba" else "b"))
                prompts, pairs = [], []
                for it in items:
                    pre, pair = prefix_for(it, feature, lm)
                    prompts.append(it.prompt + pre)
                    pairs.append(pair)
                base = []
                abl = {}
                # text-level cue manipulations (compose tasks, slot B): (i) the teacher-forced A affix replaced by the OTHER A affix
                # (bigram cue now points elsewhere / is ambiguous; the word cue unchanged) — word-following rate; (ii) the feature-B
                # WORD in the query replaced by the other value (bundle changes seen↔held-out) — following rate w.r.t. the new word.
                swapA, swapW = [], []
                for b in range(0, len(prompts), args.bs):
                    P, PR = prompts[b:b + args.bs], pairs[b:b + args.bs]
                    base += fc_batch(P, PR)
                    if task.startswith("compose"):
                        bp = items[b:b + args.bs]
                        PA, PRA, PW, PRW = [], [], [], []
                        for it in bp:
                            lang = it.lang
                            sl = slots_of(lang, it.ia, it.ib)
                            first_f, first_aff = sl[0]
                            second_f = sl[1][0]
                            v2, affs2 = (it.ia, lang["affA"]) if second_f == "a" else (it.ib, lang["affB"])
                            # (i) first realised affix replaced by the other value of its feature; second-slot forced choice
                            other_first = (lang["affA"] if first_f == "a" else lang["affB"])[1 - (it.ia if first_f == "a" else it.ib)]
                            pre, pr = fc_prefix_pair(lm, it.stem + "-" + other_first, affs2[v2], affs2[1 - v2])
                            if pr is not None:
                                PA.append(it.prompt + " " + pre)
                                PRA.append(pr)
                            # (ii) the SECOND-slot feature word replaced by its other value (bundle becomes attested); follow new word?
                            nia, nib = (1 - it.ia, it.ib) if second_f == "a" else (it.ia, 1 - it.ib)
                            line = f'{lang["wA"][nia]} {lang["wB"][nib]} {it.stem} ='
                            ctx = it.prompt[: it.prompt.rfind("\n") + 1]
                            pre, pr = fc_prefix_pair(lm, it.stem + "-" + first_aff, affs2[1 - v2], affs2[v2])
                            if pr is not None:
                                PW.append(ctx + line + " " + pre)
                                PRW.append(pr)
                        if PA:
                            swapA += fc_batch(PA, PRA)
                        if PW:
                            swapW += fc_batch(PW, PRW)
                    for name, heads in [("induction", top)] + [(f"random{d}", rd) for d, rd in enumerate(rnd_draws)]:
                        by_layer = {}
                        for _s, lyr, h in heads:
                            by_layer.setdefault(lyr, []).append(h)
                        enc_tmp = lm.tok(P, return_tensors="pt", padding=True)
                        T = enc_tmp["input_ids"].shape[1]
                        pos_mask = torch.zeros(len(P), T, dtype=torch.bool, device=lm.device)
                        if args.ablate_scope == "last":
                            pos_mask[:, -1] = True                      # ablate at the prediction position only
                        else:                                           # whole query line (from the last newline on)
                            for k, p in enumerate(P):
                                nq = len(lm.tok.encode(p[p.rfind("\n") + 1:]))
                                pos_mask[k, -nq:] = True
                        mk = ablate_hook(by_layer, n_heads, head_size, pos_mask, args.ablation)
                        abl.setdefault(name, [])
                        abl[name] += fc_batch(P, PR, pre_hooks=[(lyr, mk(lyr)) for lyr in by_layer])
                rnd_all = sum([abl[k] for k in abl if k.startswith("random")], [])
                rec["ablation"][task] = {"word_follow": round(float(np.mean(base)), 3),
                                         "ablate_induction": round(float(np.mean(abl["induction"])), 3),
                                         "ablate_random": round(float(np.mean(rnd_all)), 3),
                                         "ablate_random_draws": [round(float(np.mean(abl[k])), 3) for k in sorted(abl) if k.startswith("random")],
                                         "n": len(items), "mode": args.ablation}
                if swapA:
                    rec["ablation"][task]["swapA_word_follow"] = round(float(np.mean(swapA)), 3)
                    rec["ablation"][task]["swapWordB_follow"] = round(float(np.mean(swapW)), 3)
                if args.gen_ablation:
                    # free-generation version: ablate during decoding; score second-slot correctness / exact / copy
                    gsc = {}
                    for name, heads in [("none", None), ("induction", top)] + [(f"random{d}", rd) for d, rd in enumerate(rnd_draws)]:
                        by_layer = {}
                        for s_, lyr, h in (heads or []):
                            by_layer.setdefault(lyr, []).append(h)
                        mkf = ablate_hook_factory(by_layer, n_heads, head_size, args.ablation)
                        hooks = [(lyr, mkf(lyr)) for lyr in by_layer]
                        rows = []
                        for b in range(0, len(items), args.bs):
                            bp = items[b:b + args.bs]
                            gens = generate_with_prehooks(lm, [it.prompt for it in bp], hooks, [it.stem for it in bp])
                            for it, g in zip(bp, gens):
                                sc = score(it, g)
                                second = slots_of(it.lang, it.ia, it.ib)[-1][0]
                                rows.append((sc["exact"], sc["aff_" + second], sc["copy"]))
                        gsc[name] = {"exact": round(float(np.mean([r[0] for r in rows])), 3), "second_slot": round(float(np.mean([r[1] for r in rows])), 3),
                                     "copy": round(float(np.mean([r[2] for r in rows])), 3)}
                    rec["ablation"][task]["gen"] = gsc
            print(f"{tag} {step_key}: induction heads {rec['induction_heads'][:4]} | " + " | ".join(
                f"{t}: word {v['word_follow']:.2f} →abl-ind {v['ablate_induction']:.2f} abl-rand {v['ablate_random']:.2f}" + (f" swapA {v['swapA_word_follow']:.2f} swapW {v['swapWordB_follow']:.2f}" if 'swapA_word_follow' in v else "") + (" GEN " + " ".join(f"{n}:{g['exact']:.2f}/{g['second_slot']:.2f}/{g['copy']:.2f}" for n, g in v["gen"].items()) if "gen" in v else "") for t, v in rec["ablation"].items()) + f" [{time.time()-t0:.0f}s]", flush=True)

        # ---------- (3) selective-write DAS
        rec["das"] = {}
        if not args.skip_das:
            for task in args.tasks.split(","):
                feature = "a" if task == "bind_a" else ("b" if task == "bind_b" else "b")
                tr = build(task, args.n_train, seed=args.seed + 100, donor_feature=feature)
                te = build(task, args.n_test, seed=args.seed + 200, donor_feature=feature)
                if args.holdout_pair:
                    # word-pair generalisation: the DAS subspace is trained on languages whose intervened feature uses the OTHER
                    # word pairs and tested only on languages using the held-out pair (index into WORDS_A / WORDS_B)
                    pairs_all = WORDS_A if feature == "a" else WORDS_B
                    hp = list(pairs_all[args.holdout_pair - 1])
                    wkey = "wA" if feature == "a" else "wB"
                    tr_big = build(task, args.n_train * 2, seed=args.seed + 100, donor_feature=feature)
                    te_big = build(task, args.n_test * 4, seed=args.seed + 200, donor_feature=feature)
                    tr = [it for it in tr_big if list(it.lang[wkey]) != hp][: args.n_train]
                    te = [it for it in te_big if list(it.lang[wkey]) == hp][: args.n_test]

                def pos_char(it, prompt):
                    """char END offset of the intervention position in this prompt's query line."""
                    if args.position == "eq":
                        return len(prompt)                         # the "=" token
                    if args.position == "stem":
                        return len(prompt) - len(" =")             # last stem token
                    # feature word of the intervened feature (first word of the line for "a"; the B word otherwise)
                    line_start = prompt.rfind("\n") + 1
                    words = prompt[line_start:].split(" ")
                    idx = 0 if feature == "a" or it.ia is None else 1
                    return line_start + len(" ".join(words[:idx + 1]))

                def prep(items):
                    B = []
                    for b in range(0, len(items), args.bs):
                        bp = items[b:b + args.bs]
                        prompts, pairs, cends = [], [], []
                        for it in bp:
                            pre, pair = prefix_for(it, feature, lm)
                            prompts.append(it.prompt + pre)
                            pairs.append(pair)
                            cends.append([pos_char(it, it.prompt)])
                        enc_b, P_b = lm.encode(prompts, cends)
                        enc_d, P_d = lm.encode([it.donor_prompt for it in bp], [[pos_char(it, it.donor_prompt)] for it in bp])
                        enc_sv, P_sv = lm.encode([it.sv_prompt for it in bp], [[pos_char(it, it.sv_prompt)] for it in bp])
                        cid = torch.tensor([lm.first_token_id(c) for c, w in pairs], device=lm.device)
                        wid = torch.tensor([lm.first_token_id(w) for c, w in pairs], device=lm.device)
                        B.append({"items": bp, "enc_b": enc_b, "P_b": P_b[:, 0], "enc_d": enc_d, "P_d": P_d[:, 0], "enc_sv": enc_sv, "P_sv": P_sv[:, 0], "cid": cid, "wid": wid})
                    return B
                tr_b, te_b = prep(tr), prep(te)
                for layer in layers:
                    def grab(bs_, which):
                        H = []
                        with torch.no_grad():
                            for B in bs_:
                                _, a = lm.forward(B["enc_" + which], capture_layers=[layer], positions=B["P_" + which][:, None])
                                H.append(a[:, 0, 0, :].float())
                        return torch.cat(H).to(lm.device)
                    Hb_tr, Hd_tr = grab(tr_b, "b"), grab(tr_b, "d")
                    Hb_te, Hd_te, Hsv_te = grab(te_b, "b"), grab(te_b, "d"), grab(te_b, "sv")
                    # probe for the feature value at the position
                    def lab(items):
                        return torch.tensor([(it.ia if feature == "a" else it.ib) for it in items], device=lm.device)
                    ytr, yte = lab(tr), lab(te)
                    W_, b_, mu_, sd_ = fit_logistic(Hb_tr, ytr, 2)
                    probe = float((predict(W_, b_, mu_, sd_, Hb_te) == yte).float().mean())

                    def fn_factory(Q, Hs, P, full=False):
                        def fn(h, off=None):
                            if off is None:
                                off = [0]
                            nb = h.shape[0]
                            hs = Hs[off[0]:off[0] + nb]
                            p = P[off[0]:off[0] + nb]
                            off[0] += nb
                            T = h.shape[1]
                            mask = (torch.arange(T, device=h.device)[None, :] == p.to(h.device)[:, None])[:, :, None].float()
                            hb = h.float()
                            delta = (hs[:, None, :] - hb) if full else (hs[:, None, :] - hb) @ Q
                            return (hb + mask * delta).to(h.dtype)
                        return fn

                    def eval_fc(Q, Hs, full=False, bs_=te_b):
                        fn = fn_factory(Q, Hs, torch.cat([B["P_b"] for B in bs_]), full)
                        prefs = []
                        for B in bs_:
                            with torch.no_grad():
                                lg, _ = lm.forward(B["enc_b"], interventions={layer: fn})
                            prefs += (lg.gather(1, B["wid"][:, None]) > lg.gather(1, B["cid"][:, None])).squeeze(1).float().tolist()
                        return round(float(np.mean(prefs)), 3)

                    def eval_gen(Q, Hs, full=False, bs_=te_b):
                        """generation from the bare prompt with the intervention at the position; selective-write scoring."""
                        # positions in the bare prompt (no prefix) — re-encode
                        keys = ["stem_kept", "stem_donor", "flipped", "other_kept", "selective", "exact_cf", "exact_base"]
                        acc = {k: [] for k in keys}
                        off = 0
                        for B in bs_:
                            bp = B["items"]
                            prompts = [it.prompt for it in bp]
                            _, P = lm.encode(prompts, [[pos_char(it, it.prompt)] for it in bp])
                            hs = Hs[off:off + len(bp)]
                            off += len(bp)
                            fn = fn_factory(Q, hs, P[:, 0], full) if Q is not None or full else None
                            gens = generate_with_hook(lm, prompts, layer, fn)
                            for it, g in zip(bp, gens):
                                s = score_cf(it, g)
                                for k in keys:
                                    acc[k].append(float(s[k]))
                                if it.ia is not None and it.ib is not None:
                                    tgt_held = [it.donor_ia, it.donor_ib] == list(it.lang["held"])
                                    acc.setdefault("flipped_to_heldout" if tgt_held else "flipped_to_seen", []).append(float(s["flipped"]))
                        return {k: round(float(np.mean(v)), 3) for k, v in acc.items() if v}

                    base_gen = eval_gen(None, Hd_te)
                    full_fc = eval_fc(None, Hd_te, full=True)
                    full_gen = eval_gen(None, Hd_te, full=True)
                    if args.no_das:
                        key = f"{task}|{args.position}|L{layer}|full"
                        rec["das"][key] = {"probe": round(probe, 3), "no_intervention": base_gen, "full_patch_fc": full_fc, "full_patch_gen": full_gen}
                        print(f"   {key}: full-patch gen flipped {full_gen['flipped']} →heldout {full_gen.get('flipped_to_heldout')} →seen {full_gen.get('flipped_to_seen')} stem_kept {full_gen['stem_kept']} | no-int flipped {base_gen['flipped']}", flush=True)
                        continue
                    for rank in [int(r) for r in args.ranks.split(",")]:
                        def new_R():
                            return torch.nn.Parameter(torch.linalg.qr(torch.randn(lm.d_model, rank, device=lm.device))[0].T.contiguous())
                        R = new_R()
                        opt = torch.optim.Adam([R], lr=1e-2)
                        n_ep = max(1, args.steps_opt // max(1, len(tr_b)))
                        torch.cat([B["P_b"] for B in tr_b])
                        for _ep in range(n_ep):
                            for bi, B in enumerate(tr_b):
                                nb = len(B["items"])
                                hs = Hd_tr[bi * args.bs: bi * args.bs + nb]
                                Q = R.T @ R
                                fn = fn_factory(Q, hs, B["P_b"])
                                with torch.enable_grad():
                                    hd = lm.layers[layer].register_forward_hook(lambda m, i, o, fn=fn: ((fn(o[0]),) + tuple(o[1:])) if isinstance(o, tuple) else fn(o))
                                    try:
                                        out = lm.model(**B["enc_b"], logits_to_keep=1)
                                    finally:
                                        hd.remove()
                                    lg = out.logits[:, -1, :].float()
                                    two = torch.stack([lg.gather(1, B["cid"][:, None]).squeeze(1), lg.gather(1, B["wid"][:, None]).squeeze(1)], 1)
                                    loss = torch.nn.functional.cross_entropy(two, torch.ones(nb, dtype=torch.long, device=lm.device))
                                    opt.zero_grad()
                                    loss.backward()
                                    opt.step()
                                    with torch.no_grad():
                                        U_, _, Vh = torch.linalg.svd(R.data, full_matrices=False)
                                        R.data.copy_(U_ @ Vh)
                        Q = (R.T @ R).detach()
                        Qr = (lambda Rr: (Rr.T @ Rr).detach())(new_R())
                        out = {"probe": round(probe, 3), "no_intervention": base_gen,
                               "full_patch_fc": full_fc, "full_patch_gen": full_gen,
                               "iia_fc": eval_fc(Q, Hd_te), "iia_fc_random_rotation": eval_fc(Qr, Hd_te), "iia_fc_same_value_donor": eval_fc(Q, Hsv_te),
                               "gen": eval_gen(Q, Hd_te), "gen_random_rotation": eval_gen(Qr, Hd_te), "gen_same_value_donor": eval_gen(Q, Hsv_te)}
                        key = f"{task}|{args.position}|L{layer}|r{rank}"
                        rec["das"][key] = out
                        g = out["gen"]
                        print(f"   {key}: probe {probe:.2f} | fc IIA {out['iia_fc']} rand {out['iia_fc_random_rotation']} samev {out['iia_fc_same_value_donor']} full {full_fc} | gen selective {g['selective']} flipped {g['flipped']} stem_kept {g['stem_kept']} other_kept {g['other_kept']} | rand-sel {out['gen_random_rotation']['selective']} samev-flip {out['gen_same_value_donor']['flipped']} | full-gen sel {full_gen['selective']} stem_kept {full_gen['stem_kept']} stem_donor {full_gen['stem_donor']}", flush=True)
        res[step_key] = rec
        write_json(res, outp)
        lm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outp


# ---------------------------------------------------------------- layer sweep
PANEL = {"task": "compose_s", "seed_offset": 200, "donor_feature": "b", "source": "morph_causal.py word-patch panel (build(task, n_test, seed=seed+200, donor_feature=feature))"}


def build_panel(n: int, seed: int):
    return build(PANEL["task"], n, seed=seed + PANEL["seed_offset"], donor_feature=PANEL["donor_feature"])


def query_parts(it):
    """(line_start, [wordA, wordB, stem, '=']) of the query line of it.prompt"""
    ls = it.prompt.rfind("\n") + 1
    return ls, it.prompt[ls:].split(" ")


def word_char_end(prompt: str) -> int:
    """char END offset (exclusive) of the query's second feature word (words: wA wB stem =)"""
    ls = prompt.rfind("\n") + 1
    words = prompt[ls:].split(" ")
    return ls + len(" ".join(words[:2]))


def donor_prompt(it) -> str:
    """the same prompt with the query's second feature word replaced by the other value of feature B (stem unchanged)"""
    lang = Lang(**it.lang)
    ctx = it.prompt[: it.prompt.rfind("\n") + 1]
    return ctx + lang.phrase(it.stem, it.ia, 1 - it.ib) + it.prompt[len(ctx) + len(lang.phrase(it.stem, it.ia, it.ib)):]


def donor_cell(it):
    return (it.ia, 1 - it.ib)


def target_of(it) -> str:
    return "into_unattested" if list(donor_cell(it)) == list(it.lang["held"]) else "into_attested"


def judge(it, gen: str):
    """(followed, flipped, category) for one generation; followed = Appendix-A parse equals the donor combination"""
    stem, affs = parse_form(gen, it.lang.get("scheme", "morph"))
    cat = categorise(it, stem, affs)
    donor_affs = [a for _, a in Lang(**it.lang).slots(*donor_cell(it))]
    followed = bool(stem == it.stem and affs == donor_affs)
    assert followed == (cat == "competitor"), "followed must coincide with the 'competitor' category on the attested-base panel"
    return followed, bool(score_cf(it, gen)["flipped"]), cat


def single_token(enc_offsets_row, pos: int, word_start: int) -> bool:
    s = int(enc_offsets_row[pos][0])
    return s == word_start or s == word_start - 1   # some tokenizers include the leading space in the offset


def layer_sweep(model: str, seed: int, steps: str = "main", n: int = 160, bs: int = 16, layers: str = "", max_new: int = 8, tag: str = "_lsweep",
                out_dir: str | Path = ".", overwrite: bool = False, fp32: bool = False, device: str | None = None, lm=None) -> Path:
    """The word -> slot patch at every layer (shown-combination panel, donor feature b). Returns <out_dir>/patch_sweep_<model>_s<seed><tag>.json."""
    args = SimpleNamespace(model=model, seed=seed, steps=steps, n=n, bs=bs, layers=layers, max_new=max_new, tag=tag, out_dir=str(out_dir), overwrite=overwrite, fp32=fp32)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mtag = args.model.split("/")[-1]
    outp = Path(args.out_dir) / f"patch_sweep_{mtag}_s{args.seed}{args.tag}.json"
    if outp.exists() and not args.overwrite:
        print("exists", outp)
        return outp
    rev = None if args.steps in ("main", "") else (args.steps if args.steps.startswith("stage") else f"step{args.steps}")
    t0 = time.time()
    lm = lm if lm is not None else load_model(args.model, revision=rev, dtype="float32" if args.fp32 else "bfloat16", device=device)
    nL = lm.n_layers
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(nL))
    items = build_panel(args.n, args.seed)
    rec = {"model": args.model, "revision": rev or "main", "seed": args.seed, "n": len(items), "n_layers": nL, "panel": PANEL, "fingerprint": lm.fingerprint(), "dtype": "float32" if args.fp32 else "bfloat16",
           "max_new_tokens": args.max_new, "items": [], "baseline": {"followed": [], "flipped": [], "gen": []}, "identity_check": {"layer": nL // 2, "unchanged": []},
           "layers": {str(L): {"followed": [], "flipped": [], "category": [], "gen": []} for L in layers}}
    for it in items:
        rec["items"].append({"i": len(rec["items"]), "target": target_of(it), "base_cell": [it.ia, it.ib], "donor_cell": list(donor_cell(it)), "held": list(it.lang["held"])})
    for b0 in range(0, len(items), args.bs):
        bp = items[b0:b0 + args.bs]
        prompts = [it.prompt for it in bp]
        donors = [donor_prompt(it) for it in bp]
        for it, d in zip(bp, donors):
            assert d != it.prompt and d.count("\n") == it.prompt.count("\n") and d[: d.rfind("\n")] == it.prompt[: it.prompt.rfind("\n")], "donor must differ only in the query line"
        enc_b, P_b = lm.encode(prompts, [[word_char_end(p)] for p in prompts])
        enc_d, P_d = lm.encode(donors, [[word_char_end(d)] for d in donors])
        assert (P_b >= 0).all() and (P_d >= 0).all(), "feature-word position not found"
        offs_b = lm.tok(prompts, return_offsets_mapping=True, padding=True)["offset_mapping"]
        offs_d = lm.tok(donors, return_offsets_mapping=True, padding=True)["offset_mapping"]
        for j, it in enumerate(bp):
            ws_b = word_char_end(prompts[j]) - len(query_parts(it)[1][1])
            ws_d = word_char_end(donors[j]) - len(donors[j][donors[j].rfind("\n") + 1:].split(" ")[1])
            rec["items"][b0 + j]["word_single_token"] = bool(single_token(offs_b[j], int(P_b[j, 0]), ws_b) and single_token(offs_d[j], int(P_d[j, 0]), ws_d))
        # donor residuals at the word position, every layer, one forward
        _, acts_d = lm.forward(enc_d, capture_layers=layers, positions=P_d[:, :1])          # [B, len(layers), 1, d] bf16 cpu
        _, acts_b = lm.forward(enc_b, capture_layers=[nL // 2], positions=P_b[:, :1])
        pos_b = P_b[:, 0].to(lm.device)

        def patch_fn(vecs):
            vecs = vecs.to(lm.device)
            def fn(h):
                T = h.shape[1]
                m = (torch.arange(T, device=h.device)[None, :] == pos_b[:, None])[:, :, None].to(h.dtype)   # [B, T, 1]
                return h * (1 - m) + vecs[:, None, :].to(h.dtype) * m
            return fn
        gens0 = generate_with_hook(lm, prompts, 0, None, max_new=args.max_new)
        for it, g in zip(bp, gens0):
            f, fl, _ = judge(it, g)
            rec["baseline"]["followed"].append(f)
            rec["baseline"]["flipped"].append(fl)
            rec["baseline"]["gen"].append(g)
        gens_id = generate_with_hook(lm, prompts, nL // 2, patch_fn(acts_b[:, 0, 0, :]), max_new=args.max_new)
        rec["identity_check"]["unchanged"] += [g1 == g0 for g1, g0 in zip(gens_id, gens0)]
        for li, L in enumerate(layers):
            gens = generate_with_hook(lm, prompts, L, patch_fn(acts_d[:, li, 0, :]), max_new=args.max_new)
            for it, g in zip(bp, gens):
                f, fl, cat = judge(it, g)
                r = rec["layers"][str(L)]
                r["followed"].append(f)
                r["flipped"].append(fl)
                r["category"].append(cat)
                r["gen"].append(g)
        print(f"  batch {b0 // args.bs + 1}/{(len(items) + args.bs - 1) // args.bs} [{time.time() - t0:.0f}s]", flush=True)
    tg = np.array([x["target"] for x in rec["items"]])
    summ = {}
    for L in layers:
        f = np.array(rec["layers"][str(L)]["followed"], float)
        summ[L] = (float(f[tg == "into_unattested"].mean()) if (tg == "into_unattested").any() else float("nan"), float(f[tg == "into_attested"].mean()) if (tg == "into_attested").any() else float("nan"))
    rec["identity_check"]["share_unchanged"] = float(np.mean(rec["identity_check"]["unchanged"]))
    rec["summary"] = {"into_unattested_by_layer": {str(L): summ[L][0] for L in layers}, "into_attested_by_layer": {str(L): summ[L][1] for L in layers},
                      "n_into_unattested": int((tg == "into_unattested").sum()), "n_into_attested": int((tg == "into_attested").sum()),
                      "baseline_followed": float(np.mean(rec["baseline"]["followed"])), "word_single_token_share": float(np.mean([x["word_single_token"] for x in rec["items"]]))}
    best = max(layers, key=lambda L: (summ[L][0], -L))
    print(f"{mtag} {rev or 'main'} s{args.seed}: n_unatt {rec['summary']['n_into_unattested']} n_att {rec['summary']['n_into_attested']} | baseline followed {rec['summary']['baseline_followed']:.2f} | "
          f"identity unchanged {rec['identity_check']['share_unchanged']:.2f} | single-token words {rec['summary']['word_single_token_share']:.2f} | "
          f"peak into_unattested L{best} = {summ[best][0]:.2f} (into_attested there {summ[best][1]:.2f}) | " + " ".join(f"L{L}:{summ[L][0]:.2f}/{summ[L][1]:.2f}" for L in layers) + f" [{time.time() - t0:.0f}s]", flush=True)
    write_json(rec, outp, indent=None)
    print("wrote", outp)
    return outp


# ---------------------------------------------------------------- layer x position patch map
POSITIONS = ["wordA", "wordB", "stem", "hyphen", "marker1", "blank"]


def layers_of(lm):
    return lm.layers


def swapped_prompt(it):
    """The same prompt with the query's second feature word replaced by the feature's other value (demonstrations unchanged)."""
    lang = Lang(**it.lang)
    lines = it.prompt.split("\n")
    query = lines[-1]
    sl = slots_of(it.lang, it.ia, it.ib)
    second_f = sl[1][0]
    words = query.split(" ")
    if second_f == "a":
        new_ia, new_ib = 1 - it.ia, it.ib
        old, new = lang.wA[it.ia], lang.wA[1 - it.ia]
    else:
        new_ia, new_ib = it.ia, 1 - it.ib
        old, new = lang.wB[it.ib], lang.wB[1 - it.ib]
    idx = words.index(old)
    words[idx] = new
    lines[-1] = " ".join(words)
    return "\n".join(lines), (new_ia, new_ib), idx


def char_spans(it, prompt, form, word_idx_A, word_idx_B):
    """char spans (start, end) in prompt+' '+form for the six positions."""
    lines = prompt.split("\n")
    qstart = len(prompt) - len(lines[-1])
    q = lines[-1]
    words = q.split(" ")
    starts = []
    p = qstart
    for w in words:
        starts.append(p)
        p += len(w) + 1
    sp = {"wordA": (starts[word_idx_A], starts[word_idx_A] + len(words[word_idx_A])), "wordB": (starts[word_idx_B], starts[word_idx_B] + len(words[word_idx_B]))}
    # left-hand stem is the word before " =" in the query line
    eq = q.index(" =") if " =" in q else q.index(" ->")
    stem_word_start = qstart + q[:eq].rfind(" ") + 1
    sp["stem"] = (stem_word_start, qstart + eq)
    cont_start = len(prompt) + 1
    h1 = form.index("-")
    h2 = form.index("-", h1 + 1)
    sp["hyphen"] = (cont_start + h1, cont_start + h1 + 1)
    sp["marker1"] = (cont_start + h1 + 1, cont_start + h2)
    sp["blank"] = None  # resolved from tokens (the last token shared by the two continuations)
    return sp


def patch_map(model: str, seed: int = 2, n: int = 160, bs: int = 8, tag: str = "", revision: str | None = None, out_dir: str | Path = ".",
              device: str | None = None) -> Path:
    """Layer x position activation patching (six query positions) scored by the decision logit difference at the blank.
    Returns <out_dir>/patch_<model>_s<seed><tag>.json."""
    args = SimpleNamespace(model=model, seed=seed, n=n, bs=bs, tag=tag)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mtag = args.model.split("/")[-1]
    outp = Path(out_dir) / f"patch_{mtag}_s{args.seed}{args.tag}.json"
    if outp.exists():
        print("exists", outp)
        return outp
    t0 = time.time()
    lm = load_model(args.model, revision=revision, device=device)
    layers = layers_of(lm)
    nL = len(layers)
    items = build("compose_h", args.n, seed=args.seed + 11)
    rec = {"model": args.model, "seed": args.seed, "positions": POSITIONS, "n_layers": nL, "n": len(items), "fingerprint": lm.fingerprint(),
           "ell_clean": [], "ell_corrupt": [], "ell_patched": [[[] for _ in POSITIONS] for _ in range(nL)], "ell_patched_rev": [[[] for _ in POSITIONS] for _ in range(nL)]}

    def encode(texts_c, texts_k):
        enc_c = lm.tok(texts_c, return_offsets_mapping=True)
        enc_k = lm.tok(texts_k)
        prefixes, targets, offs = [], [], []
        for i in range(len(texts_c)):
            ic, ik = enc_c["input_ids"][i], enc_k["input_ids"][i]
            d = next(j for j in range(min(len(ic), len(ik))) if ic[j] != ik[j])
            prefixes.append(ic[:d])
            targets.append((ik[d], ic[d]))
            offs.append(enc_c["offset_mapping"][i][:d])
        T = max(len(p) for p in prefixes)
        pad = lm.tok.pad_token_id if lm.tok.pad_token_id is not None else 0
        ids = torch.full((len(prefixes), T), pad, dtype=torch.long)
        am = torch.zeros((len(prefixes), T), dtype=torch.long)
        for i, p in enumerate(prefixes):
            ids[i, :len(p)] = torch.tensor(p)
            am[i, :len(p)] = 1
        return ids.to(lm.device), am.to(lm.device), targets, offs, [len(p) - 1 for p in prefixes]

    def tok_positions(offs, span, blank):
        if span is None:
            return [blank]
        s, e = span
        return [t for t in range(len(offs)) if int(offs[t][1]) > 0 and int(offs[t][0]) < e and int(offs[t][1]) > s]

    def ell(ids, am, targets, blanks, patch=None):
        """patch = (layer, list of (row, positions), source tensor [B, T, D]) replaces the block output at those positions."""
        handles = []
        if patch is not None:
            L, pos_rows, src = patch
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                for r, ps in pos_rows:
                    for p in ps:
                        h[r, p] = src[r, p].to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            handles.append(layers[L].register_forward_hook(hook))
        with torch.no_grad():
            out = lm.model(input_ids=ids, attention_mask=am, output_hidden_states=True)
        for h in handles:
            h.remove()
        rows = torch.arange(ids.shape[0], device=lm.device)
        b = torch.tensor(blanks, device=lm.device)
        lg = out.logits[rows, b].float()
        tk = torch.tensor([t[0] for t in targets], device=lm.device)
        tc = torch.tensor([t[1] for t in targets], device=lm.device)
        return (lg[rows, tk] - lg[rows, tc]), out.hidden_states

    for b0 in range(0, len(items), args.bs):
        bp = items[b0:b0 + args.bs]
        tc_, tk_, tc2, tk2, spans_c, spans_k = [], [], [], [], [], []
        for it in bp:
            forms = dict(four_forms(it))
            tgt = (it.ia, it.ib)
            first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
            comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
            sw, new_tgt, widx = swapped_prompt(it)
            it.prompt.split("\n")[-1].split(" ")
            wa = 0
            wb = 1
            tc_.append(it.prompt + " " + forms[tgt])
            tk_.append(it.prompt + " " + forms[comp])
            spans_c.append(char_spans(it, it.prompt, forms[tgt], wa, wb))
            # corrupted prompt: same continuations/labels (clean labels fixed); its own correct form is forms[comp] by construction
            tc2.append(sw + " " + forms[tgt])
            tk2.append(sw + " " + forms[comp])
            spans_k.append(char_spans(it, sw, forms[tgt], wa, wb))
        ids_c, am_c, tg_c, offs_c, bl_c = encode(tc_, tk_)
        ids_k, am_k, tg_k, offs_k, bl_k = encode(tc2, tk2)
        if ids_c.shape != ids_k.shape or any(a != b for a, b in zip(bl_c, bl_k)):
            # token alignment must hold (single-token feature words); fall back to per-item handling by skipping misaligned items
            keep = [i for i in range(len(bp)) if bl_c[i] == bl_k[i] and ids_c.shape[1] == ids_k.shape[1]]
            if len(keep) < len(bp):
                raise SystemExit(f"token misalignment between clean and corrupted prompts for items {[b0 + i for i in range(len(bp)) if i not in keep]}")
        e_clean, hs_clean = ell(ids_c, am_c, tg_c, bl_c)
        e_corr, hs_corr = ell(ids_k, am_k, tg_k, bl_k)
        rec["ell_clean"] += e_clean.tolist()
        rec["ell_corrupt"] += e_corr.tolist()
        for L in range(nL):
            src_k = hs_corr[L + 1]
            src_c = hs_clean[L + 1]
            for pi, pname in enumerate(POSITIONS):
                pos_rows = [(i, tok_positions(offs_c[i], spans_c[i][pname], bl_c[i])) for i in range(len(bp))]
                e_p, _ = ell(ids_c, am_c, tg_c, bl_c, patch=(L, pos_rows, src_k))
                rec["ell_patched"][L][pi] += e_p.tolist()
                e_r, _ = ell(ids_k, am_k, tg_k, bl_k, patch=(L, pos_rows, src_c))
                rec["ell_patched_rev"][L][pi] += e_r.tolist()
        print(f"  batch {b0}: clean ℓ {e_clean.mean():+.2f} corrupt ℓ {e_corr.mean():+.2f}  [{time.time()-t0:.0f}s]", flush=True)
    P = np.array(rec["ell_patched"])
    cl = np.array(rec["ell_clean"])
    co = np.array(rec["ell_corrupt"])
    Tm = (P - cl[None, None, :]).mean(2) / (co - cl).mean()
    print(f"{mtag}: clean ℓ {cl.mean():+.2f}, corrupt ℓ {co.mean():+.2f}; transmission max per position " + ", ".join(f"{p}={Tm[:, i].max():.2f}@L{int(Tm[:, i].argmax())}" for i, p in enumerate(POSITIONS)) + f"  [{time.time()-t0:.0f}s]")
    write_json(rec, outp, indent=None)
    print("wrote", outp)
    return outp
