"""Item sets of record and the fresh-item sampler.

The paper's item sets are deterministic functions of (task, n, seed, controls) through tasks.prompts.build; this module writes them
as JSON lines under data/diagnostic/ and loads them back, so a reader can inspect the exact prompts without running the generator.
Seeds of record: item seeds 0 and 1 for the first measurements (prompt generator seed = seed for the copy-probability share, seed + 11
for the attention ablation / deletion / patching panels), item seeds 2 and 3 for the fresh-item re-scoring and the head-level analyses.

Inputs : task, n, seed, controls. Outputs: list[MItem]; files data/diagnostic/items_<task>_n<n>_seed<seed>[_<controls>].jsonl.gz.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..config import data_root
from ..io import read_jsonl, write_jsonl
from .prompts import MItem, build, set_template

ABLATION_SEED_OFFSET = 11  # the ablation / deletion / layer-sweep panels use build("compose_h", n, seed + 11)
PATCH_TRAIN_OFFSET, PATCH_TEST_OFFSET, HEAD_PANEL_OFFSET = 100, 200, 7


def item_file(task: str, n: int, seed: int, controls: str = "", root: str | Path | None = None) -> Path:
    root = Path(root) if root else data_root() / "diagnostic"
    return root / f"items_{task}_n{n}_seed{seed}{('_' + controls) if controls else ''}.jsonl.gz"


def items_of_record(task: str, n: int = 160, seed: int = 0, offset: int = 0, scheme: str = "morph", template: str = "default", order: str = "ab",
                    n_single: int = 2, n_combo: int = 2, donor_feature: str | None = None) -> list[MItem]:
    """The exact items a measurement used: build(task, n, seed + offset, ...) under the given template."""
    set_template(template)  # "default" restores the standard pools after an alternate-template build
    return build(task, n, seed=seed + offset, n_single=n_single, n_combo=n_combo, order=order, scheme=scheme, donor_feature=donor_feature)


def write_items(items: list[MItem], path: str | Path) -> Path:
    return write_jsonl((it.to_json() for it in items), path)


def load_items(path: str | Path) -> list[MItem]:
    return [MItem(**row) for row in read_jsonl(path)]


def fresh_items(task: str, n: int, seed: int, **kw) -> list[MItem]:
    """A new item set that shares no prompt with the sets of record: seeds >= 1000 are reserved for fresh draws."""
    if seed < 1000:
        raise ValueError("fresh item sets use seeds >= 1000 so they never coincide with the sets of record")
    return build(task, n, seed=seed, **kw)


def sample_items(items: list[MItem], k: int, seed: int = 0) -> list[MItem]:
    """A random subset of k items (for quick checks)."""
    rng = random.Random(seed)
    return rng.sample(items, k)
