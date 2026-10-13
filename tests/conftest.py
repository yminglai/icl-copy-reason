"""Shared fixtures: a tiny randomly initialised GPT-2 (no download) and a byte-level tokenizer for CPU smoke tests."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def tiny_model_and_tokenizer():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    torch.manual_seed(0)
    try:  # the public tiny checkpoint when it is cached; otherwise an in-process random model with the GPT-2 tokenizer
        tok = GPT2TokenizerFast.from_pretrained("sshleifer/tiny-gpt2")
        model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    except Exception:  # noqa: BLE001 - offline fallback
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=tok.vocab_size, n_positions=256)
        model = GPT2LMHeadModel(cfg)
    model.eval()
    _ = transformers  # keep the import visible for skip logic
    return model, tok
