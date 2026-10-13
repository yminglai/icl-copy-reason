"""Model loading (any Hugging Face causal LM at a revision), tokenisation with character-to-token mapping, residual capture and
residual intervention hooks.

Inputs : model name, revision, dtype, device. Outputs: an LM object (model, tokenizer, decoder layers, forward with hooks, a weight
fingerprint recorded in every result file so a silently mis-resolved revision is detectable).

Residual stream convention: "layer L" = output of decoder block L (hidden_states[L+1] in HF numbering), which is the hook
point of both Qwen-Scope (`model.layers[L]` output) and Gemma Scope 2 (`model.layers.L.output`).
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ALIASES = {
    "qwen17": "Qwen/Qwen3-1.7B-Base",
    "qwen06": "Qwen/Qwen3-0.6B-Base",
    "qwen4b": "Qwen/Qwen3-4B-Base",
    "gemma4": "unsloth/gemma-3-4b-pt",
    "gemma1": "unsloth/gemma-3-1b-pt",
}


def _clean_snapshot(name: str, revision: str | None):
    """Some EleutherAI/pythia step branches (2.8b) carry a stray single-file model.safetensors, identical on every branch, next to the
    real sharded checkpoint
    transformers prefers the single file and silently loads the wrong weights. Return a symlink directory
    without the stray file in that case, else None."""
    import os
    try:
        from huggingface_hub import constants
        repo = os.path.join(constants.HF_HUB_CACHE, "models--" + name.replace("/", "--"))
        ref = os.path.join(repo, "refs", revision or "main")
        commit = open(ref).read().strip() if os.path.exists(ref) else (revision or "")
        p = os.path.join(repo, "snapshots", commit)
        files = os.listdir(p)
    except Exception:
        return None
    if "model.safetensors" in files and "model.safetensors.index.json" in files and any(f.startswith("model-0000") for f in files):
        clean = os.path.join(constants.HF_HUB_CACHE, "clean-snapshots", os.path.basename(p))
        os.makedirs(clean, exist_ok=True)
        for f in files:
            if f != "model.safetensors" and not os.path.lexists(os.path.join(clean, f)):
                os.symlink(os.path.join(p, f), os.path.join(clean, f))
        return clean
    return None


class LM:
    def __init__(self, name: str, device: str = "cuda", dtype=torch.bfloat16, revision: str | None = None):
        name = MODEL_ALIASES.get(name, name)
        self.name = name
        self.revision = revision
        clean = _clean_snapshot(name, revision)
        if clean is not None:
            name, revision = clean, None
        try:
            self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        except ValueError as e:
            if "trust_remote_code" in str(e):
                self.tok = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
            else:
                # intermediate-checkpoint repos/tags that ship only a slow tokenizer (BLOOM global_step tags): the family's released tokenizer
                # is identical; take it from the same repo at main, else from the repo without the "-intermediate" suffix
                import re as _re
                base = _re.sub(r"(_step_\d+|-step-\d+[kK]?(-token-\d+B)?|-intermediate-checkpoints|-intermediate)$", "", name)
                try:
                    self.tok = AutoTokenizer.from_pretrained(name)
                except Exception:
                    self.tok = AutoTokenizer.from_pretrained(base)   # separate step repos (OpenLLaMA _step_N, INTELLECT-1-step-N) and -intermediate repos
        self.tok.padding_side = "left"
        if self.tok.eos_token is None and "<|endoftext|>" in self.tok.get_vocab():
            self.tok.eos_token = "<|endoftext|>"  # some Pythia intermediate revisions ship a tokenizer config without special tokens
        if self.tok.pad_token is None:
            self.tok.pad_token = "<|padding|>" if "<|padding|>" in self.tok.get_vocab() else self.tok.eos_token
        try:
            cfg = AutoConfig.from_pretrained(name, revision=revision)
        except ValueError:
            cfg = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=True)
        if "GPTNeoXForCausalLM" in (cfg.architectures or []):
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device, revision=revision).eval()
            self.text_cfg = self.model.config
            self.layers = self.model.gpt_neox.layers
        elif "BloomForCausalLM" in (cfg.architectures or []):
            # BLOOM's ALiBi builder reads (batch, seq) from the attention mask; the knockout passes a 4-D additive mask. Derive the 2-D padding
            # mask from the 4-D mask's diagonal (a real token may attend to itself; a pad position is fully masked) for the ALiBi tensor only.
            import transformers.models.bloom.modeling_bloom as _bloom
            if not getattr(_bloom, "_alibi_mask_patched", False):
                _orig = _bloom.build_alibi_tensor

                def _build(attention_mask, num_heads, dtype):
                    if attention_mask.dim() == 4:
                        diag = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
                        attention_mask = (diag > torch.finfo(attention_mask.dtype).min / 2).to(attention_mask.dtype if attention_mask.dtype.is_floating_point else torch.float32)
                    return _orig(attention_mask, num_heads, dtype)
                _bloom.build_alibi_tensor = _build
                _bloom._alibi_mask_patched = True
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device, revision=revision).eval()
            self.text_cfg = self.model.config
            self.layers = self.model.transformer.h
        elif "Gemma3ForConditionalGeneration" in (cfg.architectures or []):
            from transformers import Gemma3ForConditionalGeneration
            self.model = Gemma3ForConditionalGeneration.from_pretrained(name, dtype=dtype, device_map=device, revision=revision).eval()
            self.text_cfg = cfg.text_config
            self.layers = self.model.model.language_model.layers
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device, revision=revision).eval()
            except ValueError as e:
                if "trust_remote_code" not in str(e):
                    raise
                # remote-code architectures (Baichuan2 intermediate checkpoints)
                self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device, revision=revision, trust_remote_code=True).eval()
            self.text_cfg = self.model.config
            self.layers = self.model.model.layers
        self.n_layers = self.text_cfg.num_hidden_layers
        self.d_model = self.text_cfg.hidden_size
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ tokenisation with char→token positions
    def encode(self, prompts: Sequence[str], char_ends: Sequence[Sequence[int | None]]):
        """Tokenise a batch (left padded). char_ends[i] = list of character END offsets (exclusive) whose token index we
        need
        returns enc, positions [B, len(char_ends[i])] (index of the token containing char end-1), and last index."""
        enc = self.tok(list(prompts), return_tensors="pt", padding=True, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        B, T = enc["input_ids"].shape
        pos = torch.full((B, len(char_ends[0])), -1, dtype=torch.long)
        for i in range(B):
            om = offs[i]
            for j, ce in enumerate(char_ends[i]):
                if ce is None:
                    continue
                target = ce - 1
                # token whose span covers `target`
                found = -1
                for t in range(T - 1, -1, -1):
                    s, e = int(om[t, 0]), int(om[t, 1])
                    if e == 0 and s == 0:
                        continue  # padding / special
                    if s <= target < e:
                        found = t
                        break
                    if e <= target:
                        found = t  # target falls in whitespace after this token (shouldn't happen for expression ends)
                        break
                pos[i, j] = found
        return {k: v.to(self.device) for k, v in enc.items()}, pos

    # ------------------------------------------------------------------ forward with capture / intervention
    @torch.no_grad()
    def forward(self, enc, capture_layers: Sequence[int | None] = None, positions: torch.Tensor | None = None,
                interventions: dict[int, Callable[[torch.Tensor | None, torch.Tensor]]] = None):
        """Run the model. If capture_layers is given, return residuals at `positions` ([B, P]) for each layer as a tensor
        [B, L, P, d] (bf16 on CPU). interventions: {layer: fn(resid [B, T, d]) -> resid} applied to block outputs."""
        captured = {}
        handles = []
        interventions = interventions or {}
        cap = set(capture_layers or [])

        def make_hook(L):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                if L in interventions:
                    h = interventions[L](h)
                if L in cap:
                    if positions is None:
                        captured[L] = h.detach()
                    else:
                        idx = positions.to(h.device).clamp(min=0)
                        captured[L] = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1])).detach()
                if isinstance(output, tuple):
                    return (h,) + tuple(output[1:])
                return h
            return hook

        for L in sorted(cap | set(interventions.keys())):
            handles.append(self.layers[L].register_forward_hook(make_hook(L)))
        try:
            try:
                out = self.model(**enc, logits_to_keep=1)
            except TypeError:
                out = self.model(**enc)
        finally:
            for hd in handles:
                hd.remove()
        logits = out.logits[:, -1, :].float()
        acts = None
        if cap:
            acts = torch.stack([captured[L] for L in sorted(cap)], dim=1).to(torch.bfloat16).cpu()  # [B, L, P, d]
        return logits, acts

    def fingerprint(self) -> float:
        """cheap weight fingerprint (detects a revision that silently resolved to other weights)"""
        w = next(self.layers[0].parameters())
        return round(float(w.float().flatten()[:4096].sum()), 4)

    def first_token_id(self, text: str) -> int:
        return self.tok.encode(text, add_special_tokens=False)[0]


@contextlib.contextmanager
def no_grad_bf16():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        yield


def load_model(name: str, revision: str | None = None, dtype: str | torch.dtype = "bfloat16", device: str | None = None) -> LM:
    """Load a causal LM for the readings. dtype: "bfloat16" (the paper's default) or "float32"; device defaults to CUDA when present."""
    if isinstance(dtype, str):
        dtype = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32, "float16": torch.float16}[dtype]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return LM(name, device=device, dtype=dtype, revision=revision)


def wrap_model(model, tokenizer, device: str = "cpu") -> LM:
    """Wrap an already-constructed Hugging Face model (used by the CPU tests and by callers that load weights themselves)."""
    lm = LM.__new__(LM)
    lm.name, lm.revision, lm.model, lm.tok, lm.device = getattr(model.config, "_name_or_path", "model"), None, model.eval(), tokenizer, device
    lm.tok.padding_side = "left"
    if lm.tok.pad_token is None:
        lm.tok.pad_token = lm.tok.eos_token
    lm.text_cfg = getattr(model.config, "text_config", model.config)
    for attr in ("gpt_neox", "transformer", "model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            lm.layers = getattr(inner, "layers", None) or getattr(inner, "h", None) or inner.language_model.layers
            break
    lm.n_layers, lm.d_model = lm.text_cfg.num_hidden_layers, lm.text_cfg.hidden_size
    for p in lm.model.parameters():
        p.requires_grad_(False)
    lm.model.to(device)
    return lm
