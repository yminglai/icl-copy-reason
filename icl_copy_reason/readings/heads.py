"""Head-level analyses of the decision at the blank: direct logit attribution, source-conditioned head ablation, cumulative
ablation curves with the OV-copy check, inference-time head steering, head dynamics along a fine-tuning run, and the attention
signatures of the head groups.

Decision quantity: ell = logit(copied candidate's ending) - logit(correct ending) at the blank, the first token at which the two
candidate forms diverge (right-padded prefix batch). Items = build("compose_h", n, seed + 11), the same items as the attention ablation.

  attribution    direct logit attribution of ell per head and per MLP through the final norm frozen at the observed residual, plus
                 each head's attention mass at the blank on four spans. Schema of the released dla_<model>[_<rev>]_s<seed>.json.
  ablate_sets    cross-fitted top-5 pro-copied (R5) / pro-correct (B5) head sets from the other seed's attribution file, zeroed from
                 the query line on, with 20 layer-matched random sets
                 ell and the span-ablation margins under each.
                 Schema of ablate_<model>[_<rev>]_s<seed>.json.
  cumulative     top-k curves (k = 1..K) ranked on the selection seed and evaluated on the other seed, with the paired span-ablation
                 effect under each ablated model and the OV-copy check of the R5 heads. Schema of mech_cumulative_<model>_sel<s>_eval<t>.json.
  steer          scale the top-k pro-copied heads (alpha) and/or pro-correct heads (beta) from the query line on, with layer-matched
                 random control sets
                 four-way accuracy, A_null, strict composed-form rate and diagnostics. Schema of steer_<model>[_<rev>]_s<seed>_steer.json.
  finetune_dynamics  attribution at every saved checkpoint of a fine-tuning run
  R(t) and B(t) of the step-0 / final head sets,
                 cross-fitted over two item seeds. Schema of mech_induce_traj_sel<s>_eval<t>.json.
  attention_signatures  the attention-signature block (per head group, per span) — see the functions at the end of the module.

Inputs : model name/revision, seeds, item counts, attribution files for the cross-fitted sets. Outputs: JSON records under out_dir.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..io import read_json, write_json
from ..tasks.prompts import AFFIX_POOL, Lang, build, four_forms, score, slots_of
from .ablation import line_sets, scored_logprobs
from .copy_share import induction_score, load_model, revision_key
from .patching import induction_heads


def route_sets(C, k: int = 5):
    """Top-k pro-copied and pro-correct heads by mean contribution (stable argsort; ties by head index). None if fewer than k qualify."""
    m = C.mean(0)
    order = np.argsort(-m, kind="stable")
    R = [int(i) for i in order[:k] if m[i] > 0]
    Bh = [int(i) for i in order[::-1][:k] if m[i] < 0]
    return m, (R if len(R) == k else None), (Bh if len(Bh) == k else None)


# ---------------------------------------------------------------- direct logit attribution
def arch_parts(lm):
    """(layers, final_norm, unembed_weight, attn_getter, mlp_getter, post_norm_flag) for the supported families."""
    m = lm.model
    name = type(m).__name__
    if hasattr(m, "gpt_neox"):
        core = m.gpt_neox
        layers = core.layers
        fnorm = core.final_layer_norm
        WU = m.get_output_embeddings().weight
        return layers, fnorm, WU, (lambda L: (L.attention, L.attention.dense)), (lambda L: L.mlp), None, None
    lang = m.model.language_model if hasattr(m.model, "language_model") else m.model
    layers = lang.layers
    fnorm = lang.norm
    WU = m.get_output_embeddings().weight
    post_attn = post_mlp = None
    if "Olmo2" in name or "Gemma3" in name:
        def post_attn(L):
            return L.post_attention_layernorm
        def post_mlp(L):
            return L.post_feedforward_layernorm
    return layers, fnorm, WU, (lambda L: (L.self_attn, L.self_attn.o_proj)), (lambda L: L.mlp), post_attn, post_mlp


def ratio_through_norm(norm, y):
    """per-dimension factor r with norm(y) = y * r for RMSNorm-type modules (Gemma's (1+w) included), evaluated at y ([B, D])."""
    with torch.no_grad():
        n = norm(y.unsqueeze(1)).squeeze(1) if y.dim() == 2 else norm(y)
    return torch.where(y.abs() > 1e-9, n / y, torch.zeros_like(y))


def final_projection(fnorm, x_final, u):
    """Return a function mapping a residual write c ([B, D]) to its logit-difference contribution ([B]) under the frozen final norm."""
    is_ln = isinstance(fnorm, torch.nn.LayerNorm) or ("RMS" not in type(fnorm).__name__ and hasattr(fnorm, "bias") and fnorm.bias is not None)
    if is_ln:
        eps = getattr(fnorm, "eps", 1e-5)
        g = fnorm.weight.float()
        sigma = torch.sqrt(x_final.float().var(-1, unbiased=False, keepdim=True) + eps)      # [B, 1]
        return lambda c: ((((c.float() - c.float().mean(-1, keepdim=True)) / sigma) * g) @ u.float())
    r = ratio_through_norm(fnorm, x_final.float())
    return lambda c: ((c.float() * r) @ u.float())


def attribution(model: str, steps: str = "main", seed: int = 2, n: int = 160, bs: int = 8, scheme: str = "morph", tag: str = "", fp32: bool = False,
                overwrite: bool = False, out_dir: str | Path = ".", device: str | None = None) -> list[Path]:
    """Direct logit attribution for `model` at every revision in `steps`. Returns the written paths (dla_<model>[_<rev>]_s<seed><tag>.json)."""
    args = SimpleNamespace(model=model, steps=steps, seed=seed, n=n, bs=bs, scheme=scheme, tag=tag, fp32=fp32, overwrite=overwrite)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    mtag = args.model.split("/")[-1]
    for step in args.steps.split(","):
        rev = revision_key(step)
        outp = Path(out_dir) / f"dla_{mtag}{'_' + rev if rev else ''}_s{args.seed}{args.tag}.json"
        if outp.exists() and not args.overwrite:
            print("exists", outp)
            written.append(outp)
            continue
        t0 = time.time()
        lm = load_model(args.model, revision=rev, dtype="float32" if args.fp32 else "bfloat16", device=device)
        lm.model.set_attn_implementation("eager")
        cfg = lm.text_cfg if hasattr(lm, "text_cfg") else lm.model.config
        layers, fnorm, WU, attn_get, mlp_get, post_attn, post_mlp = arch_parts(lm)
        nL = len(layers)
        nH = attn_get(layers[0])[1].in_features // (getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads))
        hd = attn_get(layers[0])[1].in_features // nH
        items = build("compose_h", args.n, seed=args.seed + 11, scheme=args.scheme)
        heads = [[L, h] for L in range(nL) for h in range(nH)]
        rec = {"model": args.model, "revision": rev or "main", "seed": args.seed, "n": len(items), "dtype": "float32" if args.fp32 else "bfloat16", "heads": heads, "n_layers": nL, "n_heads": nH, "fingerprint": lm.fingerprint(),
               "l_items": [], "head_contrib": [], "mlp_contrib": [], "attn_bias_contrib": [], "embed_contrib": [], "normbias_contrib": [], "identity_err": [], "blank_pos": [],
               "attn_share": {"competitor_marker": [], "wordB": [], "single2_marker": [], "wordA": []}}
        for b0 in range(0, len(items), args.bs):
            bp = items[b0:b0 + args.bs]
            texts_c, texts_k, spans_l = [], [], []
            for it in bp:
                forms = dict(four_forms(it))
                tgt = (it.ia, it.ib)
                first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
                comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
                texts_c.append(it.prompt + " " + forms[tgt])
                texts_k.append(it.prompt + " " + forms[comp])
                spans_l.append(line_sets(it)[0])
            enc_c = lm.tok(texts_c, return_offsets_mapping=True)
            enc_k = lm.tok(texts_k)
            prefixes, targets, blanks, offs_all = [], [], [], []
            for i in range(len(bp)):
                ic, ik = enc_c["input_ids"][i], enc_k["input_ids"][i]
                d = next(j for j in range(min(len(ic), len(ik))) if ic[j] != ik[j])
                prefixes.append(ic[:d])
                targets.append((ik[d], ic[d]))
                blanks.append(d - 1)
                offs_all.append(enc_c["offset_mapping"][i][:d])
            T = max(len(p) for p in prefixes)
            pad = lm.tok.pad_token_id if lm.tok.pad_token_id is not None else 0
            ids = torch.full((len(bp), T), pad, dtype=torch.long)
            am = torch.zeros((len(bp), T), dtype=torch.long)
            for i, p in enumerate(prefixes):
                ids[i, :len(p)] = torch.tensor(p)
                am[i, :len(p)] = 1
            ids, am = ids.to(lm.device), am.to(lm.device)
            bidx = torch.tensor(blanks, device=lm.device)
            rows = torch.arange(len(bp), device=lm.device)
            cap = {"z": {}, "attn_in": {}, "mlp": {}, "postattn": {}, "postmlp": {}, "final_in": None}
            handles = []
            for L, layer in enumerate(layers):
                attn_mod, o_proj = attn_get(layer)
                handles.append(o_proj.register_forward_hook(lambda mod, inp, out, L=L: cap["z"].__setitem__(L, inp[0][rows, bidx].detach().float())))
                if post_attn is None:
                    handles.append(o_proj.register_forward_hook(lambda mod, inp, out, L=L: cap["attn_in"].__setitem__(L, out[rows, bidx].detach().float())))
                else:
                    handles.append(post_attn(layer).register_forward_hook(lambda mod, inp, out, L=L: cap["postattn"].__setitem__(L, (inp[0][rows, bidx].detach().float(), out[rows, bidx].detach().float()))))
                if post_mlp is None:
                    handles.append(mlp_get(layer).register_forward_hook(lambda mod, inp, out, L=L: cap["mlp"].__setitem__(L, out[rows, bidx].detach().float())))
                else:
                    handles.append(post_mlp(layer).register_forward_hook(lambda mod, inp, out, L=L: cap["postmlp"].__setitem__(L, out[rows, bidx].detach().float())))
            handles.append(fnorm.register_forward_hook(lambda mod, inp, out: cap.__setitem__("final_in", inp[0][rows, bidx].detach().float())))
            with torch.no_grad():
                out = lm.model(input_ids=ids, attention_mask=am, output_attentions=True, output_hidden_states=True)
            assert out.attentions is not None and out.attentions[0] is not None, "attention weights not returned (attention implementation is not eager)"
            for h in handles:
                h.remove()
            logits = out.logits[rows, bidx].float()
            tk = torch.tensor([t[0] for t in targets], device=lm.device)
            tc = torch.tensor([t[1] for t in targets], device=lm.device)
            ell = (logits[rows, tk] - logits[rows, tc])                                   # [B]
            u = (WU[tk] - WU[tc]).float()                                                  # [B, D]
            x_final = cap["final_in"]
            proj_fn = final_projection(fnorm, x_final, u.T)     # maps [B, D] -> [B, B]; take diagonal below
            def proj(c):
                return torch.diagonal(proj_fn(c))
            emb = out.hidden_states[0][rows, bidx].float()
            head_c = torch.zeros(len(bp), nL * nH, device=lm.device)
            mlp_c = torch.zeros(len(bp), nL, device=lm.device)
            bias_c = torch.zeros(len(bp), device=lm.device)
            for L, layer in enumerate(layers):
                attn_mod, o_proj = attn_get(layer)
                W = o_proj.weight.float()             # [D, H*hd]
                z = cap["z"][L]                                                            # [B, H*hd]
                writes = torch.stack([z[:, h * hd:(h + 1) * hd] @ W[:, h * hd:(h + 1) * hd].T for h in range(nH)], 1)   # [B, H, D]
                bias = o_proj.bias.float()[None].expand(len(bp), -1) if o_proj.bias is not None else torch.zeros(len(bp), W.shape[0], device=lm.device)
                if post_attn is not None:
                    y, n = cap["postattn"][L]
                    r = torch.where(y.abs() > 1e-9, n / y, torch.zeros_like(y))
                    writes = writes * r[:, None, :]
                    bias = bias * r
                for h in range(nH):
                    head_c[:, L * nH + h] = proj(writes[:, h])
                bias_c += proj(bias)
                mlp_c[:, L] = proj(cap["postmlp"][L] if post_mlp is not None else cap["mlp"][L])
            emb_c = proj(emb)
            nb = getattr(fnorm, "bias", None)
            normbias_c = (nb.float()[None, :] * u).sum(1) if nb is not None else torch.zeros(len(bp), device=lm.device)   # final-norm bias, projected directly (it is added after scaling)
            total = head_c.sum(1) + mlp_c.sum(1) + bias_c + emb_c + normbias_c
            err = (total - ell).abs()
            # attention shares at the blank
            att = torch.stack([a[rows, :, bidx, :] for a in out.attentions], 1).float()      # [B, L, H, T]
            for i, it in enumerate(bp):
                offs = offs_all[i]
                def mass(key):
                    cols = [t for t in range(len(offs)) if int(offs[t][1]) > 0 and any(int(offs[t][0]) < e and int(offs[t][1]) > s for s, e in spans_l[i][key])]
                    return att[i][:, :, cols].sum(-1).reshape(-1).tolist() if cols else [0.0] * (nL * nH)
                for key in rec["attn_share"]:
                    rec["attn_share"][key].append(mass(key))
            rec["l_items"] += ell.tolist()
            rec["head_contrib"] += head_c.tolist()
            rec["mlp_contrib"] += mlp_c.tolist()
            rec["attn_bias_contrib"] += bias_c.tolist()
            rec["embed_contrib"] += emb_c.tolist()
            rec["normbias_contrib"] += normbias_c.tolist()
            rec["identity_err"] += err.tolist()
            rec["blank_pos"] += blanks
        m = np.array(rec["head_contrib"]).mean(0)
        order = np.argsort(-m)
        print(f"{mtag} {rev or 'main'}: ℓ mean {np.mean(rec['l_items']):+.2f} (P(comp>corr) {np.mean(np.array(rec['l_items'])>0):.2f}); identity err median {np.median(rec['identity_err']):.3f} max {np.max(rec['identity_err']):.3f}; "
              f"top pro-competitor heads {[heads[i] for i in order[:5]]} ({m[order[:5]].sum():+.2f}); top pro-correct {[heads[i] for i in order[::-1][:5]]} ({m[order[::-1][:5]].sum():+.2f}); "
              f"MLP total {np.array(rec['mlp_contrib']).mean(0).sum():+.2f}  [{time.time()-t0:.0f}s]", flush=True)
        write_json(rec, outp, indent=None)
        print("wrote", outp)
        written.append(outp)
        lm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return written


# ---------------------------------------------------------------- source-conditioned head ablation
N_RANDOM = 20


class HeadAblator:
    """Zero the outputs of a set of (layer, head) at positions >= qstart (per row) by hooking the o_proj/dense input."""

    def __init__(self, lm, layers, attn_get, nH, hd):
        self.lm, self.layers, self.attn_get, self.nH, self.hd = lm, layers, attn_get, nH, hd
        self.sets = None
        self.qrows = None
        self.handles = []

    def install(self):
        for L, layer in enumerate(self.layers):
            o_proj = self.attn_get(layer)[1]
            self.handles.append(o_proj.register_forward_pre_hook(self._make(L)))

    def _make(self, L):
        def pre(mod, inp):
            if self.sets is None:
                return None
            heads = [h for (lyr, h) in self.sets if lyr == L]
            if not heads:
                return None
            x = inp[0].clone()
            for r, q0 in enumerate(self.qrows):
                for h in heads:
                    x[r, q0:, h * self.hd:(h + 1) * self.hd] = 0
            return (x,) + tuple(inp[1:])
        return pre

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def ablate_sets(model: str, steps: str = "main", seed: int = 2, other_seed: int | None = None, n: int = 160, bs: int = 8, tag: str = "",
                out_dir: str | Path = ".", dla_dir: str | Path | None = None, device: str | None = None) -> list[Path]:
    """Cross-fitted R5 / B5 head ablation with 20 layer-matched random sets. Returns the written paths (ablate_<model>[_<rev>]_s<seed><tag>.json)."""
    args = SimpleNamespace(model=model, steps=steps, seed=seed, other_seed=other_seed, n=n, bs=bs, tag=tag)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    other = args.other_seed if args.other_seed is not None else {2: 3, 3: 2}.get(args.seed, args.seed + 1)
    mtag = args.model.split("/")[-1]
    for step in args.steps.split(","):
        rev = None if step in ("main", "") else (step if step.startswith("stage") else f"step{step}")
        suffix = f"{'_' + rev if rev else ''}"
        outp = Path(out_dir) / f"ablate_{mtag}{suffix}_s{args.seed}{args.tag}.json"
        if outp.exists():
            print("exists", outp)
            written.append(outp)
            continue
        dla_path = Path(dla_dir or out_dir) / f"dla_{mtag}{suffix}_s{other}.json"
        if not dla_path.exists():
            raise FileNotFoundError(f"cross-fitting needs {dla_path}")
        dla = read_json(dla_path)
        m = np.array(dla["head_contrib"]).mean(0)
        order = np.argsort(-m, kind="stable")
        R5 = [tuple(dla["heads"][i]) for i in order[:5] if m[i] > 0]
        B5 = [tuple(dla["heads"][i]) for i in order[::-1][:5] if m[i] < 0]
        if len(R5) < 5 or len(B5) < 5:
            print(f"{mtag} {rev}: fewer than 5 qualifying heads (R {len(R5)}, B {len(B5)}); recording the sets as-is (mech_analysis fails the rule)")
        _nL, nH = dla["n_layers"], dla["n_heads"]
        rng = np.random.default_rng(args.seed)
        def layer_matched(S):
            out = []
            for _ in range(N_RANDOM):
                s_ = []
                for (lyr, h) in S:
                    choices = [hh for hh in range(nH) if (lyr, hh) not in S and (lyr, hh) not in s_]
                    s_.append((lyr, int(rng.choice(choices))))
                out.append(s_)
            return out
        randR, randB = layer_matched(R5), layer_matched(B5)
        t0 = time.time()
        lm = load_model(args.model, revision=rev, device=device)
        lm.model.set_attn_implementation("eager")
        layers, fnorm, WU, attn_get, mlp_get, post_attn, post_mlp = arch_parts(lm)
        hd = attn_get(layers[0])[1].in_features // nH
        abl = HeadAblator(lm, layers, attn_get, nH, hd)
        abl.install()
        items = build("compose_h", args.n, seed=args.seed + 11)
        sets = {"none": None, "R5": R5, "B5": B5, **{f"randR_{k}": s_ for k, s_ in enumerate(randR)}, **{f"randB_{k}": s_ for k, s_ in enumerate(randB)}}
        rec = {"model": args.model, "revision": rev or "main", "seed": args.seed, "sets_from_seed": other, "n": len(items), "fingerprint": lm.fingerprint(),
               "sets": {"R5": [list(x) for x in R5], "B5": [list(x) for x in B5], "randR": [[list(x) for x in s_] for s_ in randR], "randB": [[list(x) for x in s_] for s_ in randB]},
               "ell": {k: [] for k in sets}, "knockout_base": {k: [] for k in ("none", "R5", "B5")},
               "knockout": {c: {k: [] for k in ("none", "R5", "B5")} for c in ("competitor_marker", "wordB")}}
        for b0 in range(0, len(items), args.bs):
            bp = items[b0:b0 + args.bs]
            texts_c, texts_k, spans_l, qstarts, prompts, conts_c, conts_k = [], [], [], [], [], [], []
            for it in bp:
                forms = dict(four_forms(it))
                tgt = (it.ia, it.ib)
                first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
                comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
                sp, q0 = line_sets(it)
                spans_l.append(sp)
                qstarts.append(q0)
                prompts.append(it.prompt)
                conts_c.append(forms[tgt])
                conts_k.append(forms[comp])
                texts_c.append(it.prompt + " " + forms[tgt])
                texts_k.append(it.prompt + " " + forms[comp])
            # ℓ at the blank (first divergent token), batched with right padding
            enc_c = lm.tok(texts_c, return_offsets_mapping=True)
            enc_k = lm.tok(texts_k)
            prefixes, targets, blanks, qtok = [], [], [], []
            for i in range(len(bp)):
                ic, ik = enc_c["input_ids"][i], enc_k["input_ids"][i]
                d = next(j for j in range(min(len(ic), len(ik))) if ic[j] != ik[j])
                prefixes.append(ic[:d])
                targets.append((ik[d], ic[d]))
                blanks.append(d - 1)
                offs = enc_c["offset_mapping"][i]
                qtok.append(next(t for t in range(d) if int(offs[t][1]) > 0 and int(offs[t][0]) >= qstarts[i]))
            T = max(len(p) for p in prefixes)
            pad = lm.tok.pad_token_id if lm.tok.pad_token_id is not None else 0
            ids = torch.full((len(bp), T), pad, dtype=torch.long)
            am = torch.zeros((len(bp), T), dtype=torch.long)
            for i, p in enumerate(prefixes):
                ids[i, :len(p)] = torch.tensor(p)
                am[i, :len(p)] = 1
            ids, am = ids.to(lm.device), am.to(lm.device)
            rows = torch.arange(len(bp), device=lm.device)
            bidx = torch.tensor(blanks, device=lm.device)
            tk = torch.tensor([t[0] for t in targets], device=lm.device)
            tc = torch.tensor([t[1] for t in targets], device=lm.device)
            for name, S in sets.items():
                abl.sets = S
                abl.qrows = qtok
                with torch.no_grad():
                    lg = lm.model(input_ids=ids, attention_mask=am).logits[rows, bidx].float()
                rec["ell"][name] += (lg[rows, tk] - lg[rows, tc]).tolist()
            # sequence-scored margins with/without span knockouts under none / R5 / B5 (scored_logprobs builds its own padded batch of the
            # four forms; the ablator needs the query start per row of that batch — 4 rows per item, same order)
            for name in ("none", "R5", "B5"):
                abl.sets = sets[name]
                full_prompts = [p for p in prompts for _ in range(2)]
                full_conts = [c for pair in zip(conts_c, conts_k) for c in pair]
                # query-start token per row: recompute from offsets of prompt+cont
                enc = lm.tok([p + " " + c for p, c in zip(full_prompts, full_conts)], return_offsets_mapping=True, padding=True)
                abl.qrows = [next(t for t in range(len(enc["offset_mapping"][r])) if int(enc["offset_mapping"][r][t][1]) > 0 and int(enc["offset_mapping"][r][t][0]) >= qstarts[r // 2]) for r in range(len(full_prompts))]
                for cond in ("none", "competitor_marker", "wordB"):
                    sp_list = [None if cond == "none" else spans_l[r // 2][cond] for r in range(len(full_prompts))]
                    lps = scored_logprobs(lm, full_prompts, full_conts, sp_list, [qstarts[r // 2] for r in range(len(full_prompts))])
                    margins = [lps[2 * i] - lps[2 * i + 1] for i in range(len(bp))]
                    (rec["knockout_base"][name] if cond == "none" else rec["knockout"][cond][name]).extend(margins)
            abl.sets = None
        e = {k: np.mean(v) for k, v in rec["ell"].items()}
        rr = np.percentile([abs(e[f"randR_{k}"] - e["none"]) for k in range(N_RANDOM)], 95)
        rb = np.percentile([abs(e[f"randB_{k}"] - e["none"]) for k in range(N_RANDOM)], 95)
        kb = {k: np.mean(v) for k, v in rec["knockout_base"].items()}
        eff = {c: {k: np.mean(rec["knockout"][c][k]) - kb[k] for k in kb} for c in rec["knockout"]}
        print(f"{mtag} {rev or 'main'}: ℓ none {e['none']:+.2f} | R5 ablated {e['R5'] - e['none']:+.2f} (rand p95 {rr:.2f}) | B5 ablated {e['B5'] - e['none']:+.2f} (rand p95 {rb:.2f}) | "
              f"Δ(competitor_marker) none {eff['competitor_marker']['none']:+.2f} R5 {eff['competitor_marker']['R5']:+.2f} B5 {eff['competitor_marker']['B5']:+.2f} | "
              f"Δ(wordB) none {eff['wordB']['none']:+.2f} R5 {eff['wordB']['R5']:+.2f} B5 {eff['wordB']['B5']:+.2f}  [{time.time()-t0:.0f}s]", flush=True)
        write_json(rec, outp, indent=None)
        print("wrote", outp)
        written.append(outp)
        abl.remove()
        lm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return written


# ---------------------------------------------------------------- cumulative ablation + OV-copy check
def rankings(dla, K):
    m = np.array(dla["head_contrib"]).mean(0)
    order = np.argsort(-m, kind="stable")
    R = [tuple(dla["heads"][i]) for i in order if m[i] > 0][:K]
    Bh = [tuple(dla["heads"][i]) for i in order[::-1] if m[i] < 0][:K]
    return R, Bh


def item_batch(lm, bp):
    """right-padded prefix batch up to the blank, per-item (competitor token, correct token), blank index, query-start token, spans"""
    texts_c, texts_k, spans_l, qstarts, conts_c, conts_k, comp_cells = [], [], [], [], [], [], []
    for it in bp:
        forms = dict(four_forms(it))
        tgt = (it.ia, it.ib)
        first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
        comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
        comp_cells.append(comp)
        sp, q0 = line_sets(it)
        spans_l.append(sp)
        qstarts.append(q0)
        conts_c.append(forms[tgt])
        conts_k.append(forms[comp])
        texts_c.append(it.prompt + " " + forms[tgt])
        texts_k.append(it.prompt + " " + forms[comp])
    enc_c = lm.tok(texts_c, return_offsets_mapping=True)
    enc_k = lm.tok(texts_k)
    prefixes, targets, blanks, qtok = [], [], [], []
    for i in range(len(bp)):
        ic, ik = enc_c["input_ids"][i], enc_k["input_ids"][i]
        d = next(j for j in range(min(len(ic), len(ik))) if ic[j] != ik[j])
        prefixes.append(ic[:d])
        targets.append((ik[d], ic[d]))
        blanks.append(d - 1)
        offs = enc_c["offset_mapping"][i]
        qtok.append(next(t for t in range(d) if int(offs[t][1]) > 0 and int(offs[t][0]) >= qstarts[i]))
    T = max(len(p) for p in prefixes)
    pad = lm.tok.pad_token_id if lm.tok.pad_token_id is not None else 0
    ids = torch.full((len(bp), T), pad, dtype=torch.long)
    am = torch.zeros((len(bp), T), dtype=torch.long)
    for i, p in enumerate(prefixes):
        ids[i, :len(p)] = torch.tensor(p)
        am[i, :len(p)] = 1
    return {"ids": ids.to(lm.device), "am": am.to(lm.device), "targets": targets, "blanks": blanks, "qtok": qtok, "spans": spans_l, "qstarts": qstarts,
            "prompts": [it.prompt for it in bp], "conts_c": conts_c, "conts_k": conts_k, "prefixes": prefixes, "comp_cells": comp_cells}


def pool_token_ids(lm, it, prefix_ids, comp_cell):
    """unique first-token ids of the 30 pool markers in the item's second slot, and the competitor's second-marker first token"""
    lang = Lang(**it.lang)
    d = len(prefix_ids)
    sl = slots_of(it.lang, it.ia, it.ib)
    first_f = sl[0][0]
    def form_with(m):
        first = lang.affA[it.ia] if first_f == "a" else lang.affB[it.ib]
        return it.prompt + " " + it.stem + "-" + first + "-" + m
    ids_by_marker, mism = {}, 0
    for m in AFFIX_POOL:
        ids_m = lm.tok.encode(form_with(m))
        if ids_m[:d] == list(prefix_ids) and len(ids_m) > d:
            ids_by_marker[m] = ids_m[d]
        else:
            mism += 1
            dd = next((j for j in range(min(len(ids_m), d)) if ids_m[j] != prefix_ids[j]), min(len(ids_m) - 1, d))
            ids_by_marker[m] = ids_m[dd]
    comp_second = [a for _, a in lang.slots(*comp_cell)][1]
    return sorted(set(ids_by_marker.values())), ids_by_marker[comp_second], mism


def cumulative(model: str, select_seed: int, eval_seed: int, steps: str = "main", n: int = 160, bs: int = 8, K: int = 20, ks: str = "", tag: str = "",
               overwrite: bool = False, fp32: bool = False, out_dir: str | Path = ".", dla_dir: str | Path | None = None, device: str | None = None) -> Path:
    """Cumulative ablation curves (ranked on select_seed, evaluated on eval_seed) + OV-copy check. Returns mech_cumulative_<model>_sel<s>_eval<t><tag>.json."""
    args = SimpleNamespace(model=model, steps=steps, select_seed=select_seed, eval_seed=eval_seed, n=n, bs=bs, K=K, ks=ks, tag=tag, overwrite=overwrite, fp32=fp32)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mtag = args.model.split("/")[-1]
    rev = None if args.steps in ("main", "") else (args.steps if args.steps.startswith("stage") else f"step{args.steps}")
    outp = Path(out_dir) / f"mech_cumulative_{mtag}_sel{args.select_seed}_eval{args.eval_seed}{args.tag}.json"
    if outp.exists() and not args.overwrite:
        print("exists", outp)
        return outp
    dla_path = Path(dla_dir or out_dir) / f"dla_{mtag}{'_' + rev if rev else ''}_s{args.select_seed}.json"
    if not dla_path.exists():
        raise FileNotFoundError(f"ranking needs the attribution file {dla_path}")
    dla = read_json(dla_path)
    rank_R, rank_B = rankings(dla, args.K)
    if len(rank_R) < args.K or len(rank_B) < args.K:
        print(f"WARNING: only {len(rank_R)} pro-competitor / {len(rank_B)} pro-correct heads qualify (K={args.K}); curves stop there (analysis fails the rule)")
    ks = [int(k) for k in args.ks.split(",")] if args.ks else list(range(1, args.K + 1))
    t0 = time.time()
    lm = load_model(args.model, revision=rev, dtype="float32" if args.fp32 else "bfloat16", device=device)
    lm.model.set_attn_implementation("eager")
    layers, fnorm, WU, attn_get, mlp_get, post_attn, post_mlp = arch_parts(lm)
    nL, nH = dla["n_layers"], dla["n_heads"]
    hd = attn_get(layers[0])[1].in_features // nH
    assert nL == len(layers), "layer count differs from the DLA file"
    abl = HeadAblator(lm, layers, attn_get, nH, hd)
    abl.install()
    R5 = rank_R[:5]
    items = build("compose_h", args.n, seed=args.eval_seed + 11)
    rec = {"model": args.model, "revision": rev or "main", "select_seed": args.select_seed, "eval_seed": args.eval_seed, "dla_file": dla_path.name, "n": len(items), "K": args.K, "ks": ks,
           "n_layers": nL, "n_heads": nH, "fingerprint": lm.fingerprint(), "dtype": "float32" if args.fp32 else "bfloat16", "rank_R": [list(h) for h in rank_R], "rank_B": [list(h) for h in rank_B], "R5": [list(h) for h in R5], "B5": [list(h) for h in rank_B[:5]],
           "ell": {"intact": [], "R": {str(k): [] for k in ks if k <= len(rank_R)}, "B": {str(k): [] for k in ks if k <= len(rank_B)}},
           "margin": {"intact": {"none": [], "competitor_marker": [], "wordB": []}, "R": {str(k): {"none": [], "competitor_marker": []} for k in ks if k <= len(rank_R)}, "B": {str(k): {"none": [], "wordB": []} for k in ks if k <= len(rank_B)}},
           "ov": {"heads": [list(h) for h in R5], "favours_competitor": [[] for _ in R5], "competitor_logit_rank": [[] for _ in R5], "n_unique_pool_tokens": [], "ov_prefix_mismatch": 0}}
    cap = {}
    def z_hook(L):
        return lambda mod, inp, out: cap.__setitem__(L, inp[0].detach())
    for b0 in range(0, len(items), args.bs):
        bp = items[b0:b0 + args.bs]
        Bt = item_batch(lm, bp)
        rows = torch.arange(len(bp), device=lm.device)
        bidx = torch.tensor(Bt["blanks"], device=lm.device)
        tk = torch.tensor([t[0] for t in Bt["targets"]], device=lm.device)
        tc = torch.tensor([t[1] for t in Bt["targets"]], device=lm.device)

        def ell_pass(S, capture=False):
            abl.sets = S
            abl.qrows = Bt["qtok"]
            handles = []
            if capture:
                cap.clear()
                for L, layer in enumerate(layers):
                    handles.append(attn_get(layer)[1].register_forward_hook(z_hook(L)))
                    if post_attn is not None:
                        handles.append(post_attn(layer).register_forward_hook(lambda mod, inp, out, L=L: cap.__setitem__(("post", L), (inp[0][rows, bidx].detach().float(), out[rows, bidx].detach().float()))))
                handles.append(fnorm.register_forward_hook(lambda mod, inp, out: cap.__setitem__("final_in", inp[0][rows, bidx].detach().float())))
            with torch.no_grad():
                lg = lm.model(input_ids=Bt["ids"], attention_mask=Bt["am"]).logits[rows, bidx].float()
            for h in handles:
                h.remove()
            return (lg[rows, tk] - lg[rows, tc]).tolist()
        rec["ell"]["intact"] += ell_pass(None, capture=True)
        # (b) OV-copy on the intact forward: R5 heads' writes at the blank through the frozen final norm and the unembedding
        x_final = cap["final_in"]
        pools = [pool_token_ids(lm, it, Bt["prefixes"][i], Bt["comp_cells"][i]) for i, it in enumerate(bp)]
        union = sorted(set(t for uniq, _, _ in pools for t in uniq))
        col = {t: j for j, t in enumerate(union)}
        proj = final_projection(fnorm, x_final, WU[union].float().T)                                       # c [B, D] -> logits over the union [B, |U|]
        for uniq, comp_tok, mism in pools:
            rec["ov"]["n_unique_pool_tokens"].append(len(uniq))
            rec["ov"]["ov_prefix_mismatch"] += mism
        for hi, (L, h) in enumerate(R5):
            W = attn_get(layers[L])[1].weight.float()
            z = cap[L][rows, bidx].float()
            write = z[:, h * hd:(h + 1) * hd] @ W[:, h * hd:(h + 1) * hd].T                                # [B, D]
            if post_attn is not None:
                y, n = cap[("post", L)]
                write = write * torch.where(y.abs() > 1e-9, n / y, torch.zeros_like(y))
            lg_u = proj(write)
            for i, (uniq, comp_tok, _) in enumerate(pools):
                vals = lg_u[i, [col[t] for t in uniq]]
                best = uniq[int(torch.argmax(vals))]
                rec["ov"]["favours_competitor"][hi].append(bool(best == comp_tok))
                rec["ov"]["competitor_logit_rank"][hi].append(int((vals > lg_u[i, col[comp_tok]]).sum()))
        # (a) cumulative ablation: ℓ_k and the paired knockout margins under each ablated model
        def margins(S, cond):
            abl.sets = S
            full_prompts = [p for p in Bt["prompts"] for _ in range(2)]
            full_conts = [c for pair in zip(Bt["conts_c"], Bt["conts_k"]) for c in pair]
            enc = lm.tok([p + " " + c for p, c in zip(full_prompts, full_conts)], return_offsets_mapping=True, padding=True)
            abl.qrows = [next(t for t in range(len(enc["offset_mapping"][r])) if int(enc["offset_mapping"][r][t][1]) > 0 and int(enc["offset_mapping"][r][t][0]) >= Bt["qstarts"][r // 2]) for r in range(len(full_prompts))]
            sp_list = [None if cond == "none" else Bt["spans"][r // 2][cond] for r in range(len(full_prompts))]
            lps = scored_logprobs(lm, full_prompts, full_conts, sp_list, [Bt["qstarts"][r // 2] for r in range(len(full_prompts))])
            return [lps[2 * i] - lps[2 * i + 1] for i in range(len(bp))]
        for cond in ("none", "competitor_marker", "wordB"):
            rec["margin"]["intact"][cond] += margins(None, cond)
        for curve, rank, cond in (("R", rank_R, "competitor_marker"), ("B", rank_B, "wordB")):
            for k in ks:
                if k > len(rank):
                    continue
                S = rank[:k]
                rec["ell"][curve][str(k)] += ell_pass(S)
                rec["margin"][curve][str(k)]["none"] += margins(S, "none")
                rec["margin"][curve][str(k)][cond] += margins(S, cond)
        abl.sets = None
        print(f"  batch {b0 // args.bs + 1}/{(len(items) + args.bs - 1) // args.bs} [{time.time() - t0:.0f}s]", flush=True)
    e0 = np.mean(rec["ell"]["intact"])
    def dl(curve, k):
        return np.mean(rec["ell"][curve][str(k)]) - e0
    def ko(curve, k, cond):
        m = rec["margin"][curve][str(k)] if k else rec["margin"]["intact"]
        return np.mean(m[cond]) - np.mean(m["none"])
    kR = [k for k in ks if str(k) in rec["ell"]["R"]]
    kB = [k for k in ks if str(k) in rec["ell"]["B"]]
    print(f"{mtag} {rev or 'main'} sel{args.select_seed}->eval{args.eval_seed}: ℓ intact {e0:+.2f} | Δℓ_k pro-competitor " + " ".join(f"{k}:{dl('R', k):+.2f}" for k in kR) +
          " | Δℓ_k pro-correct " + " ".join(f"{k}:{dl('B', k):+.2f}" for k in kB) +
          f" | Δ_KO(competitor_marker) intact {ko('R', 0, 'competitor_marker'):+.2f} " + " ".join(f"k{k}:{ko('R', k, 'competitor_marker'):+.2f}" for k in kR if k in (1, 5, 10, 20)) +
          f" | Δ_KO(wordB) intact {ko('B', 0, 'wordB'):+.2f} " + " ".join(f"k{k}:{ko('B', k, 'wordB'):+.2f}" for k in kB if k in (1, 5, 10, 20)) +
          f" | OV favours competitor per R5 head {[round(float(np.mean(v)), 2) for v in rec['ov']['favours_competitor']]} (unique pool tokens median {np.median(rec['ov']['n_unique_pool_tokens']):.0f}, prefix mismatches {rec['ov']['ov_prefix_mismatch']}) [{time.time() - t0:.0f}s]", flush=True)
    write_json(rec, outp, indent=None)
    print("wrote", outp)
    abl.remove()
    lm = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outp


# ---------------------------------------------------------------- inference-time head steering
KS = (10, 5)
SUPPRESS = (0.0, 0.5)
AMPLIFY = (1.5, 2.0)
BOTH = ((0.5, 1.5), (0.0, 2.0))
PRIMARY = {"sup_a0.0": ("R", 0.0, None), "amp_b2.0": ("B", None, 2.0), "both_a0.5_b1.5": ("RB", 0.5, 1.5)}


class HeadScaler:
    """Multiply the outputs of chosen (layer, head) pairs by a factor at positions >= qstart (prefill) and at every decode position."""

    def __init__(self, lm, layers, attn_get, nH, hd):
        self.layers, self.attn_get, self.nH, self.hd = layers, attn_get, nH, hd
        self.scales = None      # dict {(lyr, h): factor}
        self.qrows = None       # per-row first query-line token index (prefill)
        self.prefill_only = False   # diagnostic: leave decode steps (T == 1) untouched
        self.handles = []

    def install(self):
        for L, layer in enumerate(self.layers):
            o_proj = self.attn_get(layer)[1]
            self.handles.append(o_proj.register_forward_pre_hook(self._make(L)))

    def _make(self, L):
        def pre(mod, inp):
            if not self.scales:
                return None
            heads = [(h, f) for (lyr, h), f in self.scales.items() if lyr == L]
            if not heads:
                return None
            x = inp[0].clone()
            T = x.shape[1]
            if T == 1 and self.prefill_only:
                return None
            for r in range(x.shape[0]):
                q0 = 0 if T == 1 or self.qrows is None else self.qrows[r]
                for h, f in heads:
                    x[r, q0:, h * self.hd:(h + 1) * self.hd] *= f
            return (x,) + tuple(inp[1:])
        return pre

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def qrow_of(enc_offsets, qstart):
    return next(t for t in range(len(enc_offsets)) if int(enc_offsets[t][1]) > 0 and int(enc_offsets[t][0]) >= qstart)


def four_form_scores(lm, scaler, items, qstarts, bs):
    """summed log P(form | prompt) for the four forms of each item, steering hooks active from the query line onward"""
    out = []
    for b0 in range(0, len(items), bs):
        bp = items[b0:b0 + bs]
        qs = qstarts[b0:b0 + bs]
        prompts, conts, tags, q_rep = [], [], [], []
        for it, q0 in zip(bp, qs):
            for bundle, form in four_forms(it):
                prompts.append(it.prompt)
                conts.append(form)
                tags.append(bundle)
                q_rep.append(q0)
        full = [p + " " + c for p, c in zip(prompts, conts)]
        enc = lm.tok(full, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        scaler.qrows = [qrow_of(offs[r], q_rep[r]) for r in range(len(full))]
        ids = enc["input_ids"].to(lm.device)
        am = enc["attention_mask"].to(lm.device)
        with torch.no_grad():
            logits = lm.model(input_ids=ids, attention_mask=am).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(2, ids[:, 1:, None]).squeeze(-1)
        for i, p in enumerate(prompts):
            start = len(p)
            m = torch.tensor([int(offs[i, t + 1, 0]) >= start and int(offs[i, t + 1, 1]) > 0 for t in range(lp.shape[1])], device=lm.device)
            out.append(float((lp[i] * m).sum()))
    return [dict(zip([b for b, _ in four_forms(it)], out[4 * j:4 * j + 4])) for j, it in enumerate(items)]


def generate_steered(lm, scaler, items, qstarts, bs, max_new=8):
    gens = []
    for b0 in range(0, len(items), bs):
        bp = items[b0:b0 + bs]
        qs = qstarts[b0:b0 + bs]
        enc = lm.tok([it.prompt for it in bp], return_tensors="pt", padding=True, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        scaler.qrows = [qrow_of(offs[r], qs[r]) for r in range(len(bp))]
        enc = {k: v.to(lm.device) for k, v in enc.items()}
        with torch.no_grad():
            out = lm.model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=lm.tok.pad_token_id)
        gens += [lm.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]
    return gens


def ell_first_divergent(lm, scaler, items, qstarts, bs):
    """logit(competitor) - logit(correct) at the first token where the two forms' tokenisations diverge (the mech_dla convention)"""
    out = []
    for b0 in range(0, len(items), bs):
        bp = items[b0:b0 + bs]
        qs = qstarts[b0:b0 + bs]
        texts_c, texts_k = [], []
        for it in bp:
            forms = dict(four_forms(it))
            tgt = (it.ia, it.ib)
            first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
            comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
            texts_c.append(it.prompt + " " + forms[tgt])
            texts_k.append(it.prompt + " " + forms[comp])
        enc_c = lm.tok(texts_c, return_offsets_mapping=True)
        enc_k = lm.tok(texts_k)
        prefixes, targets, blanks, qrows = [], [], [], []
        for i in range(len(bp)):
            ic, ik = enc_c["input_ids"][i], enc_k["input_ids"][i]
            d = next(j for j in range(min(len(ic), len(ik))) if ic[j] != ik[j])
            prefixes.append(ic[:d])
            targets.append((ik[d], ic[d]))
            blanks.append(d - 1)
            qrows.append(qrow_of(enc_c["offset_mapping"][i][:d], qs[i]))
        T = max(len(p) for p in prefixes)
        pad = lm.tok.pad_token_id if lm.tok.pad_token_id is not None else 0
        ids = torch.full((len(bp), T), pad, dtype=torch.long)
        am = torch.zeros((len(bp), T), dtype=torch.long)
        for i, p in enumerate(prefixes):
            ids[i, :len(p)] = torch.tensor(p)
            am[i, :len(p)] = 1
        scaler.qrows = qrows
        with torch.no_grad():
            lg = lm.model(input_ids=ids.to(lm.device), attention_mask=am.to(lm.device)).logits.float()
        rows = torch.arange(len(bp), device=lm.device)
        bidx = torch.tensor(blanks, device=lm.device)
        tk = torch.tensor([t[0] for t in targets], device=lm.device)
        tc = torch.tensor([t[1] for t in targets], device=lm.device)
        out += (lg[rows, bidx, tk] - lg[rows, bidx, tc]).tolist()
    return out


def evaluate(lm, scaler, items, qstarts, bs, scales, prefill_only=False):
    scaler.scales = scales
    scaler.prefill_only = prefill_only
    lps = four_form_scores(lm, scaler, items, qstarts, bs)
    ells = ell_first_divergent(lm, scaler, items, qstarts, bs)
    gens = generate_steered(lm, scaler, items, qstarts, bs)
    per = []
    for j, (it, cand, g, ell) in enumerate(zip(items, lps, gens, ells)):
        tgt = (it.ia, it.ib)
        first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
        comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
        best = max(cand, key=cand.get)
        margin = cand[tgt] - cand[comp]
        sc = score(it, g)
        cat = sc.get("category")
        per.append({"item": j, "acc": int(best == tgt), "competitor": int(best == comp), "margin": float(margin), "a_null": float(1.0 / (1.0 + np.exp(margin))),
                    "ell": float(ell), "exact": int(bool(sc.get("exact", False))), "category": cat, "parsed_valid": int(cat not in ("malformed", "stem_error")),
                    "truncated": int(cat == "single_affix"), "gen_tokens": len(lm.tok(g)["input_ids"]), "gen": g[:40]})
    def mean(k):
        return float(np.mean([x[k] for x in per]))
    return {"four_way_acc": mean("acc"), "a_null": mean("a_null"), "exact": mean("exact"), "ell": mean("ell"), "parsed_valid": mean("parsed_valid"),
            "truncated": mean("truncated"), "gen_tokens": mean("gen_tokens"), "prefill_only": prefill_only, "items": per}


def steer(model: str, steps: str = "main", seed: int = 2, other_seed: int | None = None, n: int = 160, bs: int = 8, n_random: int = 19, tag: str = "_steer",
          fp32: bool = False, out_dir: str | Path = ".", dla_dir: str | Path | None = None, device: str | None = None) -> list[Path]:
    """Inference-time head steering with layer-matched random controls (n_random = 0 for models already at ceiling). Returns steer_<model>[_<rev>]_s<seed><tag>.json."""
    args = SimpleNamespace(model=model, steps=steps, seed=seed, other_seed=other_seed, n=n, bs=bs, n_random=n_random, tag=tag, fp32=fp32)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    other = args.other_seed if args.other_seed is not None else {2: 3, 3: 2}.get(args.seed, args.seed + 1)
    mtag = args.model.split("/")[-1]
    for step in args.steps.split(","):
        rev = None if step in ("main", "") else (step if step.startswith("stage") else f"step{step}")
        suffix = f"{'_' + rev if rev else ''}"
        outp = Path(out_dir) / f"steer_{mtag}{suffix}_s{args.seed}{args.tag}.json"
        if outp.exists():
            print("exists", outp)
            written.append(outp)
            continue
        dla_path = Path(dla_dir or out_dir) / f"dla_{mtag}{suffix}_s{other}.json"
        if not dla_path.exists():
            raise FileNotFoundError(f"cross-fitting needs {dla_path}")
        dla = read_json(dla_path)
        m = np.array(dla["head_contrib"]).mean(0)
        order = np.argsort(-m, kind="stable")
        heads = [tuple(h) for h in dla["heads"]]
        nH = dla["n_heads"]
        topR = {k: [heads[i] for i in order[:k] if m[i] > 0] for k in KS}
        topB = {k: [heads[i] for i in order[::-1][:k] if m[i] < 0] for k in KS}
        k_actual = {"R": {k: len(v) for k, v in topR.items()}, "B": {k: len(v) for k, v in topB.items()}}
        for k in KS:
            if len(topR[k]) < k or len(topB[k]) < k:
                print(f"WARNING {mtag} s{args.seed}: fewer than {k} qualifying heads (R {len(topR[k])}, B {len(topB[k])}); recorded as k_actual, the rule fails closed")
        rng = np.random.default_rng(1000 + args.seed)
        def layer_matched_pair(SR, SB, n_sets):
            """n_sets pairs (randR, randB): layer-matched to SR and SB, disjoint from both selected sets and from each other"""
            out = []
            for _ in range(n_sets):
                taken = set(SR) | set(SB)
                rR, rB = [], []
                for (lyr, h) in SR:
                    choices = [hh for hh in range(nH) if (lyr, hh) not in taken]
                    c = (lyr, int(rng.choice(choices)))
                    rR.append(c)
                    taken.add(c)
                for (lyr, h) in SB:
                    choices = [hh for hh in range(nH) if (lyr, hh) not in taken]
                    c = (lyr, int(rng.choice(choices)))
                    rB.append(c)
                    taken.add(c)
                out.append((rR, rB))
            return out
        pairs = layer_matched_pair(topR[10], topB[10], args.n_random)
        randR = [p[0] for p in pairs]
        randB = [p[1] for p in pairs]
        conditions = {"none": {}}
        for k in KS:
            for a in SUPPRESS:
                conditions[f"k{k}_sup_a{a}"] = {h: a for h in topR[k]}
            for b in AMPLIFY:
                conditions[f"k{k}_amp_b{b}"] = {h: b for h in topB[k]}
            for a, b in BOTH:
                conditions[f"k{k}_both_a{a}_b{b}"] = {**{h: a for h in topR[k]}, **{h: b for h in topB[k]}}
        for j, (sR, sB) in enumerate(zip(randR, randB)):
            conditions[f"rand{j}_sup_a0.0"] = {h: 0.0 for h in sR}
            conditions[f"rand{j}_amp_b2.0"] = {h: 2.0 for h in sB}
            conditions[f"rand{j}_both_a0.5_b1.5"] = {**{h: 0.5 for h in sR}, **{h: 1.5 for h in sB}}
        conditions["k10_both_a0.5_b1.5_prefill_only"] = conditions["k10_both_a0.5_b1.5"]
        t0 = time.time()
        lm = load_model(args.model, revision=rev, dtype="float32" if (args.fp32 or device == "cpu") else "bfloat16", device=device)
        lm.model.set_attn_implementation("eager")
        layers, fnorm, WU, attn_get, mlp_get, post_attn, post_mlp = arch_parts(lm)
        hd = attn_get(layers[0])[1].in_features // nH
        scaler = HeadScaler(lm, layers, attn_get, nH, hd)
        scaler.install()
        items = build("compose_h", args.n, seed=args.seed + 11)
        qstarts = [line_sets(it)[1] for it in items]
        rec = {"model": args.model, "revision": rev or "main", "seed": args.seed, "sets_from_seed": other, "n": len(items), "fingerprint": lm.fingerprint(),
               "dtype": str(next(lm.model.parameters()).dtype), "k_actual": k_actual, "n_random": args.n_random,
               "sets": {"topR": {str(k): [list(h) for h in v] for k, v in topR.items()}, "topB": {str(k): [list(h) for h in v] for k, v in topB.items()},
                        "randR": [[list(h) for h in s_] for s_ in randR], "randB": [[list(h) for h in s_] for s_ in randB]},
               "conditions": {}}
        for name, scales in conditions.items():
            rec["conditions"][name] = evaluate(lm, scaler, items, qstarts, args.bs, scales, prefill_only=name.endswith("_prefill_only"))
            r = rec["conditions"][name]
            print(f"{mtag} s{args.seed} {name:30s} 4way {r['four_way_acc']:.3f} A_null {r['a_null']:.3f} exact {r['exact']:.3f} ell {r['ell']:+.2f} len {r['gen_tokens']:.1f} ({time.time() - t0:.0f}s)", flush=True)
        rec["seconds"] = round(time.time() - t0)
        write_json(rec, outp, indent=None)
        print("wrote", outp)
        written.append(outp)
        scaler.remove()
        lm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return written


# ---------------------------------------------------------------- head dynamics along a fine-tuning run
COARSE_STEPS = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
FINE_STEPS = [10, 20, 30, 40, 50, 75, 100, 150, 200]


def checkpoint_dir(root: str | Path, step: int, run: str = "compose_s0", fine_run: str = "compose_s0_fine") -> Path:
    return Path(root) / (run if step in COARSE_STEPS else fine_run) / f"step{step}"


def finetune_dynamics(checkpoint_root: str | Path, out_dir: str | Path, n: int = 160, bs: int = 8, seeds=(2, 3), steps=None, fp32: bool = False,
                      tag: str = "", device: str | None = None) -> list[Path]:
    """Attribution at every saved checkpoint of the direct fine-tune (compose arm), then R(t) / B(t) of the step-0 pro-copied and
    final-step pro-correct head sets, cross-fitted over the two item seeds. Returns the two mech_induce_traj_sel<s>_eval<t>.json paths."""
    out_dir = Path(out_dir)
    steps = sorted(steps or COARSE_STEPS + FINE_STEPS)
    seeds = list(seeds)
    if len(seeds) != 2:
        raise ValueError("exactly two item seeds (cross-fitting)")
    for s in steps:
        if not checkpoint_dir(checkpoint_root, s).is_dir():
            raise FileNotFoundError(f"missing checkpoint {checkpoint_dir(checkpoint_root, s)}")
    files = {s: {} for s in seeds}
    for seed in seeds:
        for st in steps:
            ck = checkpoint_dir(checkpoint_root, st)
            odir = out_dir / "finetune_attribution" / ck.parent.name
            outp = odir / f"dla_{ck.name}_s{seed}.json"
            if not outp.exists():
                attribution(str(ck), "main", seed, n, bs, fp32=fp32, out_dir=odir, device=device)
            if not outp.exists():
                raise RuntimeError(f"attribution did not write {outp}")
            files[seed][st] = outp
    recs = {seed: {st: read_json(files[seed][st]) for st in steps} for seed in seeds}
    for seed in seeds:
        for st in steps:
            r = recs[seed][st]
            if len(r["l_items"]) != n or r["heads"] != recs[seeds[0]][steps[0]]["heads"]:
                raise RuntimeError(f"fail-closed: {files[seed][st]} has n={len(r['l_items'])} or a different head list")
    if 0 not in steps or 2000 not in steps:
        print("WARNING: steps 0 and 2000 are needed for the R5/B5 selection; attribution files written only")
        return []
    heads = recs[seeds[0]][0]["heads"]
    written = []
    for sel, ev in ((seeds[0], seeds[1]), (seeds[1], seeds[0])):
        _, R5, _ = route_sets(np.array(recs[sel][0]["head_contrib"]))
        _, _, B5 = route_sets(np.array(recs[sel][2000]["head_contrib"]))
        out = {"select_seed": sel, "eval_seed": ev, "n": n, "steps": steps, "R5": None if R5 is None else [heads[i] for i in R5], "B5": None if B5 is None else [heads[i] for i in B5],
               "R_t": [], "B_t": [], "R_items": {}, "B_items": {}, "l_mean": {}, "identity_err_median": {}, "dla_files": {str(st): str(files[ev][st].relative_to(out_dir)) for st in steps},
               "fingerprints": {str(st): recs[ev][st]["fingerprint"] for st in steps}}
        for st in steps:
            C = np.array(recs[ev][st]["head_contrib"])
            Ri = C[:, R5].sum(1) if R5 is not None else np.full(C.shape[0], np.nan)
            Bi = C[:, B5].sum(1) if B5 is not None else np.full(C.shape[0], np.nan)
            out["R_items"][str(st)] = Ri.tolist()
            out["B_items"][str(st)] = Bi.tolist()
            out["R_t"].append(float(Ri.mean()))
            out["B_t"].append(float(Bi.mean()))
            out["l_mean"][str(st)] = float(np.mean(recs[ev][st]["l_items"]))
            out["identity_err_median"][str(st)] = float(np.median(recs[ev][st]["identity_err"]))
        p = out_dir / f"mech_induce_traj_sel{sel}_eval{ev}{tag}.json"
        write_json(out, p, indent=None)
        written.append(p)
        print(f"  sel{sel}->eval{ev}: R5 {out['R5']} B5 {out['B5']}", flush=True)
    return written


# ---------------------------------------------------------------- attention signatures at the answer slot
ATTN_KEYS = ["single2_marker", "competitor_marker", "wordB", "wordA", "single2", "competitor", "sham", "otherbundles"]  # the last two are controls
RULE_KEYS = ("single2_marker", "competitor_marker")  # must map to at least one token for every item, else the checkpoint fails


class SpanFailure(Exception):
    pass


def resolved_commit(lm, model, commit):
    c = getattr(lm.model.config, "_commit_hash", None)
    if c is None and not os.path.isdir(model):
        try:
            from huggingface_hub import HfApi
            c = HfApi(token=os.environ.get("HF_TOKEN") or None).model_info(model, revision=commit).sha
        except Exception:
            c = None
    return c


def local_weights_identity(model_dir):
    p = os.path.join(model_dir, "model.safetensors")
    if not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return {"file": "model.safetensors", "sha256": h.hexdigest(), "size": os.path.getsize(p)}


def forward_batch(lm, bp, bs_try):
    """returns (att [B, L, H, T], logits at the slot [B, V], prefixes, blanks, offsets, spans); retries smaller batches on OOM"""
    texts_c, texts_k, spans_l = [], [], []
    for it in bp:
        forms = dict(four_forms(it))
        tgt = (it.ia, it.ib)
        first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
        comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
        texts_c.append(it.prompt + " " + forms[tgt])
        texts_k.append(it.prompt + " " + forms[comp])
        spans_l.append(line_sets(it)[0])
    enc_c = lm.tok(texts_c, return_offsets_mapping=True)
    enc_k = lm.tok(texts_k)
    prefixes, targets, blanks, offs_all = [], [], [], []
    for i in range(len(bp)):
        ic, ik = enc_c["input_ids"][i], enc_k["input_ids"][i]
        d = next(j for j in range(min(len(ic), len(ik))) if ic[j] != ik[j])
        prefixes.append(ic[:d])
        targets.append((ik[d], ic[d]))
        blanks.append(d - 1)
        offs_all.append(enc_c["offset_mapping"][i][:d])
    T = max(len(p) for p in prefixes)
    pad = lm.tok.pad_token_id if lm.tok.pad_token_id is not None else 0
    ids = torch.full((len(bp), T), pad, dtype=torch.long)
    am = torch.zeros((len(bp), T), dtype=torch.long)
    for i, p in enumerate(prefixes):
        ids[i, :len(p)] = torch.tensor(p)
        am[i, :len(p)] = 1
    ids, am = ids.to(lm.device), am.to(lm.device)
    bidx = torch.tensor(blanks, device=lm.device)
    torch.arange(len(bp), device=lm.device)
    atts, logits = [], []
    for s0 in range(0, len(bp), bs_try):
        sl = slice(s0, s0 + bs_try)
        with torch.no_grad():
            out = lm.model(input_ids=ids[sl], attention_mask=am[sl], output_attentions=True)
        assert out.attentions is not None and out.attentions[0] is not None, "attention weights not returned (attention implementation is not eager)"
        r = torch.arange(ids[sl].shape[0], device=lm.device)
        atts.append(torch.stack([a[r, :, bidx[sl], :] for a in out.attentions], 1).float().cpu())       # [b, L, H, T]
        logits.append(out.logits[r, bidx[sl]].float().cpu())
        del out
    return torch.cat(atts), torch.cat(logits), prefixes, targets, blanks, offs_all, spans_l


def measure_attention(lm, args, seed, outp, identity):
    t0 = time.time()
    items = build("compose_h", args.n, seed=seed + 11, scheme=args.scheme)
    nL = nH = None
    rec = {"model": args.model, "revision": args.revision or "main", "commit": identity.get("commit"), "weights": identity.get("weights"), "seed": seed, "n": len(items),
           "dtype": "bfloat16", "fingerprint": lm.fingerprint(), "keys": ATTN_KEYS, "rule_keys": list(RULE_KEYS),
           "attn_share": {k: [] for k in ATTN_KEYS}, "span_tokens": {k: [] for k in ATTN_KEYS}, "span_dist": {k: [] for k in ATTN_KEYS},
           "blank_pos": [], "prefix_len": [], "blank_is_prompt_end": [], "l_items": [], "answer_is_competitor": []}
    bs = args.bs
    for b0 in range(0, len(items), args.bs):
        bp = items[b0:b0 + args.bs]
        while True:
            try:
                att, logits, prefixes, targets, blanks, offs_all, spans_l = forward_batch(lm, bp, bs)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs //= 2
                print(f"OOM: batch size -> {bs}", flush=True)
        if nL is None:
            nL, nH = att.shape[1], att.shape[2]
            rec["n_layers"], rec["n_heads"] = nL, nH
            rec["heads"] = [[L, h] for L in range(nL) for h in range(nH)]
        if not (torch.isfinite(att).all() and torch.isfinite(logits).all()):
            raise SpanFailure(f"non-finite attention or logits in items {b0}..{b0 + len(bp) - 1}")
        tk = torch.tensor([t[0] for t in targets])
        tc = torch.tensor([t[1] for t in targets])
        rows = torch.arange(len(bp))
        ell = (logits[rows, tk] - logits[rows, tc]).tolist()                              # logit(competitor) - logit(correct) for the next token, as in mech_dla
        rec["l_items"] += ell
        rec["answer_is_competitor"] += [int(x > 0) for x in ell]
        for i, it in enumerate(bp):
            offs = offs_all[i]
            # the blank is the last token shared by the two forms; it equals the prompt's final token unless the two markers share a leading token
            rec["blank_is_prompt_end"].append(int(int(offs[blanks[i]][1]) <= len(it.prompt)))
            for key in ATTN_KEYS:
                cols = [t for t in range(len(offs)) if int(offs[t][1]) > 0 and any(int(offs[t][0]) < e and int(offs[t][1]) > s for s, e in spans_l[i][key])]
                if key in RULE_KEYS and not cols:
                    raise SpanFailure(f"item {b0 + i} (seed {seed}): span {key} maps to no token under this tokenizer")
                rec["attn_share"][key].append(att[i][:, :, cols].sum(-1).reshape(-1).tolist() if cols else [0.0] * (nL * nH))   # raw summed mass; per-token normalisation in the analysis
                rec["span_tokens"][key].append(len(cols))
                rec["span_dist"][key].append(float(np.mean([blanks[i] - t for t in cols])) if cols else None)
        rec["blank_pos"] += blanks
        rec["prefix_len"] += [len(p) for p in prefixes]
    # induction: per-head repeat-token attention [L, H] and the behavioural copying score, frozen defaults of the existing functions
    rec["induction_heads"] = induction_heads(lm).tolist()
    rec["induction_score"] = induction_score(lm)
    b = np.array(rec["attn_share"]["single2_marker"])
    c = np.array(rec["attn_share"]["competitor_marker"])
    print(f"{os.path.basename(args.model.rstrip('/'))} {args.revision or 'main'} seed {seed}: raw mass to the defining marker {b.mean():.4f} / competitor marker {c.mean():.4f}  "
          f"P(comp>corr) {np.mean(rec['answer_is_competitor']):.2f}  induction {rec['induction_score']:.2f} (max head {np.max(rec['induction_heads']):.2f})  "
          f"blank_is_prompt_end {np.mean(rec['blank_is_prompt_end']):.2f}  [{time.time()-t0:.0f}s]", flush=True)
    write_json(rec, outp, indent=None)
    print("wrote", outp, flush=True)


def attention_signatures(model: str, seeds=(2, 3), revision: str | None = None, commit: str | None = None, n: int = 160, bs: int = 8, scheme: str = "morph",
                         tag: str = "", overwrite: bool = False, out_dir: str | Path = ".", device: str | None = None) -> list[Path]:
    """Per-head attention masses at the answer slot on eight spans (+ induction scores) for every item seed, one model load.
    HF models may be loaded by a pinned `commit` (recorded)
    a local checkpoint directory records its weights' sha256. Returns the written
    paths attn_<model>[_<revision>]_s<seed><tag>.json. Raises SpanFailure (nothing written) when a rule span maps to no token."""
    args = SimpleNamespace(model=model, revision=None if revision in (None, "", "main") else revision, commit=commit, n=n, bs=bs, scheme=scheme, tag=tag)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mtag = os.path.basename(model.rstrip("/"))
    outs = {sd: out_dir / f"attn_{mtag}{'_' + args.revision if args.revision else ''}_s{sd}{tag}.json" for sd in seeds}
    todo = [sd for sd in seeds if overwrite or not outs[sd].exists()]
    if not todo:
        return [outs[sd] for sd in seeds]
    lm = load_model(model, revision=commit or args.revision, dtype="bfloat16", device=device)
    lm.model.set_attn_implementation("eager")
    identity = {"commit": resolved_commit(lm, model, commit or args.revision), "weights": local_weights_identity(model) if os.path.isdir(model) else None}
    if commit and identity["commit"] and identity["commit"] != commit:
        raise RuntimeError(f"commit mismatch: loaded {identity['commit']} but pinned {commit}")
    for sd in todo:
        measure_attention(lm, args, sd, outs[sd], identity)
    lm = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [outs[sd] for sd in seeds]


def pin_commits(repos_and_revisions: list[tuple[str, str | None]]) -> dict[str, str]:
    """Resolve mutable Hub references (repo, revision) to commit shas once, before a measurement; the result is recorded with each run."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    return {f"{repo}@{rev or 'main'}": api.model_info(repo, revision=rev).sha for repo, rev in repos_and_revisions}
