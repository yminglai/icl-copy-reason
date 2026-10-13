"""Direct fine-tuning of a released model on diagnostic prompts of the ALTERNATE template, loss on the answer tokens only.

Arms (identical data, order, seeds and hyper-parameters):
  compose  every training prompt is a withheld-combination query whose target is the composed form
  control  every training prompt is an attested query whose target is the attested form
and, for the guided variant with a regulariser on top of the same recipe:
  A        plain fine-tuning
  B        + lambda_att * mean attention mass of the selected heads from the blank to the copied form's marker tokens (lambda_att 2)
  C        - lambda_dla * mean direct contribution of the selected heads to logit(correct) - logit(copied) at the blank, through FROZEN
           step-0 copies of o_proj / final norm / unembedding (lambda_dla 0.1)
  B_rand / C_rand  the same penalties on 5 layer-matched random heads (default_rng(2000 + select_seed))
Recipe of record: full fine-tuning, fp32 master weights with bf16 autocast, AdamW lr 1e-5 betas (.9, .95) weight decay 0, batch 32 prompts,
100 warm-up steps then cosine to 10 % over the schedule length, gradient clipping 1.0. Training items: a fresh generator seed per step
(100000 + train_seed * 10000 + step). Checkpoints are saved in HF format at the requested steps so every reading scores them unmodified.
Inputs : model id + revision, arm, train seed (and select seed + attribution file for the guided arms), steps, save steps.
Outputs: <out>/<run>/step<k>/, config.json, train.log, reg.jsonl (guided arms).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any

import numpy as np

LAMBDA_ATT = 2.0
LAMBDA_DLA = 0.1


def device() -> str:
    return os.environ.get("ICR_DEVICE", "cuda")


def cosine_to_tenth(opt, warmup: int, total: int):
    import torch

    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) if s < warmup else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))


def head_sets(dla_path: str, k: int = 5) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """Top-k heads towards the copied form (R) and towards the correct form (B) from a frozen attribution file of the selection seed."""
    dla = json.load(open(dla_path))
    m = np.array(dla["head_contrib"]).mean(0)
    order = np.argsort(-m, kind="stable")
    heads = [tuple(h) for h in dla["heads"]]
    r_set = [heads[i] for i in order[:k] if m[i] > 0]
    b_set = [heads[i] for i in order[::-1][:k] if m[i] < 0]
    assert len(r_set) == k and len(b_set) == k, f"fewer than {k} qualifying heads in {dla_path} (R {len(r_set)}, B {len(b_set)})"
    return r_set, b_set, dla["n_heads"]


def layer_matched_random(sel, other, n_heads: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    taken = set(sel) | set(other)
    out = []
    for layer, _ in sel:
        choices = [hh for hh in range(n_heads) if (layer, hh) not in taken]
        c = (layer, int(rng.choice(choices)))
        out.append(c)
        taken.add(c)
    return out


def token_span_cols(offsets, spans) -> list[int]:
    return [t for t in range(len(offsets)) if int(offsets[t][1]) > 0 and any(int(offsets[t][0]) < e and int(offsets[t][1]) > s for s, e in spans)]


class Regulariser:
    """Captures the selected heads' attention probabilities (arm B) or per-head o_proj inputs at the blank (arm C) and returns a differentiable penalty."""

    def __init__(self, lm, arm: str, r_set, b_set):
        import torch

        from icl_copy_reason.readings.heads import arch_parts

        self.lm, self.arm, self.R, self.B = lm, arm, r_set, b_set
        self.layers, self.fnorm, self.WU, self.attn_get, _, _, _ = arch_parts(lm)
        n_heads = lm.text_cfg.num_attention_heads
        self.nH = n_heads
        self.hd = self.attn_get(self.layers[0])[1].in_features // n_heads
        self.cap: dict[int, Any] = {}
        self.handles = []
        self.blank = None
        self.comp_cols = None
        self.targets = None
        self.final_in = None
        self.W0 = {layer: self.attn_get(self.layers[layer])[1].weight.detach().clone().float() for layer in {lay for (lay, _) in b_set}}
        self.fnorm_w0 = self.fnorm.weight.detach().clone().float()
        self.WU0 = self.WU.detach().clone().float()
        self._torch = torch
        if arm == "C":
            for layer in {lay for (lay, _) in b_set}:
                self.handles.append(self.attn_get(self.layers[layer])[1].register_forward_pre_hook(lambda mod, inp, layer=layer: self.cap.__setitem__(layer, inp[0])))

    def penalty(self):
        torch = self._torch
        rows = torch.arange(len(self.blank), device=self.lm.device)
        b = torch.tensor(self.blank, device=self.lm.device)
        if self.arm == "B":
            masses = []
            for layer, h in self.R:
                att = self.cap[layer]
                if att is None:
                    raise RuntimeError("attention weights not returned; eager attention with output_attentions is required")
                row = att[rows, h, b]
                masses.append(torch.stack([row[i, self.comp_cols[i]].sum() if self.comp_cols[i] else row[i, :1].sum() * 0 for i in range(len(self.blank))]))
            m = torch.stack(masses).mean()
            return LAMBDA_ATT * m, {"att_mass": float(m.detach())}
        contribs = []
        u = self.WU0[[t[1] for t in self.targets]] - self.WU0[[t[0] for t in self.targets]]
        for layer, h in self.B:
            z = self.cap[layer][rows, b]
            w = self.W0[layer]
            write = z[:, h * self.hd : (h + 1) * self.hd].float() @ w[:, h * self.hd : (h + 1) * self.hd].T
            contribs.append((self._through_norm(write) * u).sum(-1))
        c = torch.stack(contribs).mean()
        return -LAMBDA_DLA * c, {"dla": float(c.detach())}

    def _through_norm(self, write):
        torch = self._torch
        xf = self.final_in
        fn = self.fnorm
        if isinstance(fn, torch.nn.LayerNorm):
            sigma = xf.float().var(-1, unbiased=False, keepdim=True).add(fn.eps).sqrt()
            centred = write - write.mean(-1, keepdim=True)
            return centred / sigma * self.fnorm_w0
        rms = xf.float().pow(2).mean(-1, keepdim=True).add(getattr(fn, "variance_epsilon", getattr(fn, "eps", 1e-6))).sqrt()
        w = self.fnorm_w0
        w = (1 + w) if "Gemma" in type(fn).__name__ else w
        return write / rms * w

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def _answer_labels(tok, items, enc, offs=None):
    """Mask every token before the answer (left padding: the prompt occupies [n_pad, n_pad + prompt_len))."""
    labels = enc["input_ids"].clone()
    n_pad = (enc["attention_mask"] == 0).sum(1)
    for i, it in enumerate(items):
        if offs is not None:
            first_ans = next(t for t in range(enc["input_ids"].shape[1]) if int(offs[i][t][1]) > 0 and int(offs[i][t][0]) >= len(it.prompt))
            labels[i, :first_ans] = -100
        else:
            labels[i, : int(n_pad[i]) + len(tok(it.prompt)["input_ids"])] = -100
    labels[enc["attention_mask"] == 0] = -100
    assert int((labels != -100).sum()) > 0, "no answer tokens in the batch"
    return labels


def train(arm: str, train_seed: int, out: str, model: str = "EleutherAI/pythia-1.4b", revision: str | None = "step143000", steps: int = 2000, eval_every: int = 250,
          bs: int = 32, lr: float = 1e-5, warmup: int = 100, save_steps: list[int] | None = None, out_suffix: str = "", sched_steps: int | None = None,
          select_seed: int = 3, dla_path: str | None = None, accum: int = 1, template: str = "alt") -> str:
    """Train one arm; returns the run directory. Plain arms: compose | control. Guided arms: A | B | C | B_rand | C_rand (need dla_path)."""
    import torch

    from icl_copy_reason.readings.lm import LM
    from icl_copy_reason.tasks.prompts import build, set_template

    guided = arm in ("A", "B", "C", "B_rand", "C_rand")
    dev = device()
    if guided:
        r_set, b_set, n_heads = head_sets(dla_path)
        r_used, b_used = r_set, b_set
        if arm == "B_rand":
            r_used = layer_matched_random(r_set, b_set, n_heads, 2000 + select_seed)
        if arm == "C_rand":
            b_used = layer_matched_random(b_set, r_set, n_heads, 2000 + select_seed)
        base_arm = arm[0]
        run = f"guided_{arm}_s{train_seed}_f{select_seed}"
        task = "compose_h"
        sched_total = sched_steps or 2000
        save_set = sorted(set(save_steps or [0, 10, 20, 30, 40, 50, 75, 100, 150, 200]))
    else:
        run = f"{arm}_s{train_seed}{out_suffix}"
        task = "compose_h" if arm == "compose" else "compose_s"
        sched_total = sched_steps or steps
        save_set = sorted(set(save_steps)) if save_steps else None
        base_arm = None
    outdir = os.path.join(out, run)
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(train_seed)
    random.seed(train_seed)
    lm = LM(model, device=dev, dtype=torch.float32, revision=revision)
    net = lm.model
    tok = lm.tok
    if guided:
        net.set_attn_implementation("eager")
    for p in net.parameters():
        p.requires_grad_(True)
    net.train()
    if not guided:
        net.gradient_checkpointing_enable()  # the regularised arms read hooked activations that must carry gradients, so they do not checkpoint
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = cosine_to_tenth(opt, warmup, sched_total)
    set_template(template)
    reg = Regulariser(lm, base_arm, r_used, b_used) if guided and arm != "A" else None
    cfg: dict[str, Any] = {"arm": arm, "train_seed": train_seed, "model": model, "revision": revision, "steps": steps, "bs": bs, "lr": lr, "warmup": warmup, "sched_steps": sched_total,
                           "template": template, "accum": accum}
    if guided:
        cfg.update({"R": [list(h) for h in r_set], "B": [list(h) for h in b_set], "R_used": [list(h) for h in r_used], "B_used": [list(h) for h in b_used], "dla": dla_path,
                    "lambda_att": LAMBDA_ATT, "lambda_dla": LAMBDA_DLA, "select_seed": select_seed})
    json.dump(cfg, open(os.path.join(outdir, "config.json"), "w"), indent=1)

    def batch(step: int):
        items = build(task, bs, seed=100000 + train_seed * 10000 + step)
        texts = [it.prompt + " " + it.answer for it in items]
        if not guided:
            enc = tok(texts, return_tensors="pt", padding=True)
            labels = _answer_labels(tok, items, enc)
            return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), labels.to(dev), None, None, None
        from icl_copy_reason.readings.ablation import line_sets
        from icl_copy_reason.tasks.prompts import four_forms, slots_of

        enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        labels = _answer_labels(tok, items, enc, offs)
        n_pad = (enc["attention_mask"] == 0).sum(1)
        blank, comp_cols, targets = [], [], []
        for i, it in enumerate(items):
            spans, _ = line_sets(it)
            comp_cols.append(token_span_cols(offs[i], spans["competitor_marker"]))
            forms = dict(four_forms(it))
            first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
            comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
            ic = enc["input_ids"][i][int(n_pad[i]) :].tolist()
            ik = tok(it.prompt + " " + forms[comp])["input_ids"]
            assert forms[(it.ia, it.ib)] == it.answer
            d = next(j for j in range(min(len(ic), len(ik))) if ic[j] != ik[j])
            blank.append(int(n_pad[i]) + d - 1)
            targets.append((ik[d], ic[d]))
        return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), labels.to(dev), blank, comp_cols, targets

    def save(step: int) -> None:
        d = os.path.join(outdir, f"step{step}")
        net.save_pretrained(d, safe_serialization=True)
        tok.save_pretrained(d)
        print(f"saved {d}", flush=True)

    if save_set is None or 0 in save_set:
        save(0)
    t0 = time.time()
    log = open(os.path.join(outdir, "train.log"), "a")
    regf = open(os.path.join(outdir, "reg.jsonl"), "a") if guided else None
    probe = next(net.parameters())
    probe0 = probe.detach().clone()
    final_cap: dict[str, Any] = {}
    if reg is not None and base_arm == "C":
        reg.handles.append(reg.fnorm.register_forward_hook(lambda mod, inp, out_: final_cap.__setitem__("x", inp[0].detach())))
    micro = bs // accum
    last = (save_set[-1] if save_set else steps)
    for step in range(1, steps + 1):
        ids, am, labels, blank, comp_cols, targets = batch(step)
        tot_loss = tot_lm = 0.0
        info_acc: dict[str, float] = {}
        for a0 in range(0, bs, micro):
            sl = slice(a0, a0 + micro)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dev != "cpu" else torch.autocast("cpu", enabled=False)
            with ctx:
                out = net(input_ids=ids[sl], attention_mask=am[sl], labels=labels[sl], output_attentions=(reg is not None and base_arm == "B"))
                loss = out.loss
                info: dict[str, float] = {}
                if reg is not None:
                    reg.blank, reg.comp_cols, reg.targets = blank[sl], comp_cols[sl], targets[sl]
                    if base_arm == "B":
                        assert out.attentions is not None and out.attentions[0].requires_grad, "attention weights must be returned with gradient"
                        for layer in {lay for (lay, _) in r_used}:
                            reg.cap[layer] = out.attentions[layer]
                    else:
                        rows = torch.arange(len(blank[sl]), device=lm.device)
                        reg.final_in = final_cap["x"][rows, torch.tensor(blank[sl], device=lm.device)]
                    pen, info = reg.penalty()
                    loss = loss + pen
            (loss / accum).backward()
            tot_loss += float(loss.detach()) / accum
            tot_lm += float(out.loss.detach()) / accum
            for k_, v_ in info.items():
                info_acc[k_] = info_acc.get(k_, 0.0) + v_ / accum
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step == 1:  # fail closed: refuse to continue if the parameters did not move
            assert float(gnorm) > 0 and not torch.equal(probe.detach(), probe0), "parameters did not change after the first step"
        if regf is not None:
            regf.write(json.dumps({"step": step, "loss": tot_loss, "lm_loss": tot_lm, "preclip_gnorm": float(gnorm), "clipped": bool(float(gnorm) > 1.0), "seconds": round(time.time() - t0, 1), **info_acc}) + "\n")
            regf.flush()
        if step % (10 if guided else 25) == 0 or step == 1:
            msg = f"{run} step {step} loss {tot_loss:.4f} lm {tot_lm:.4f} {info_acc} gnorm {float(gnorm):.3f} lr {sched.get_last_lr()[0]:.2e} [{time.time() - t0:.0f}s]"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if (save_set is None and step % eval_every == 0) or (save_set is not None and step in save_set):
            save(step)
        if save_set is not None and step >= last:
            break
    if reg is not None:
        reg.remove()
    print("TRAIN DONE", run, f"[{time.time() - t0:.0f}s]")
    return outdir
