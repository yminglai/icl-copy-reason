"""One configuration system for the package.

Inputs : a YAML file under configs/ (model names, data manifests, hyper-parameters, seeds, evaluation sets) and
         environment variables ICR_API_BASE / ICR_API_KEY (OpenAI-compatible endpoint used by the generator/verifier),
         ICR_DATA_ROOT (where data/ lives; default = the repository's data/), ICR_RESULTS_ROOT (where new per-run
         outputs are written; default = <ICR_DATA_ROOT>/../runs), ICR_HF_HOME (weights cache; default HF_HOME).
Outputs: a Config object (attribute access + dict); every CLI accepts --config <file> and --set key=value overrides.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def data_root() -> Path:
    return Path(os.environ.get("ICR_DATA_ROOT", REPO_ROOT / "data")).resolve()


def results_root() -> Path:
    return Path(os.environ.get("ICR_RESULTS_ROOT", data_root().parent / "runs")).resolve()


def hf_home() -> str | None:
    return os.environ.get("ICR_HF_HOME") or os.environ.get("HF_HOME")


@dataclass
class ApiSettings:
    """OpenAI-compatible endpoint for the exercise generator / verifier / auditor."""

    base_url: str | None = None
    api_key: str | None = None
    generator_model: str = "claude-sonnet-5"
    verifier_model: str = "gpt-5.5"
    auditor_model: str = "deepseek-v4-pro"
    max_concurrency: int = 32
    timeout_s: float = 180.0

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> ApiSettings:
        s = cls(base_url=os.environ.get("ICR_API_BASE"), api_key=os.environ.get("ICR_API_KEY"))
        for k, v in (overrides or {}).items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, v)
        return s


@dataclass
class Config:
    """A loaded experiment configuration (dictionary semantics plus a few typed helpers)."""

    name: str
    values: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def __getattr__(self, item: str) -> Any:
        values = self.__dict__.get("values", {})
        if item in values:
            return values[item]
        raise AttributeError(item)

    def get(self, key: str, default: Any = None) -> Any:
        cur: Any = self.values
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, key: str, value: Any) -> None:
        cur = self.values
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    @property
    def api(self) -> ApiSettings:
        return ApiSettings.from_env(self.values.get("api"))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


def _coerce(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(name_or_path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load configs/<name>.yaml (or an explicit path) and apply key=value overrides."""
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no config {name_or_path!r} (looked in {CONFIG_DIR})")
    with open(p) as fh:
        values = yaml.safe_load(fh) or {}
    base = values.pop("extends", None)
    if base:
        parent = load_config(base)
        merged = parent.values
        _deep_update(merged, values)
        values = merged
    cfg = Config(name=p.stem, values=values, path=p)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        cfg.set(key.strip(), _coerce(raw))
    return cfg


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def add_config_args(parser) -> None:
    """Attach --config / --set to an argparse parser (shared by every CLI)."""
    parser.add_argument("--config", help="config name under configs/ or a YAML path")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")


def config_from_args(args, default: str | None = None) -> Config | None:
    name = getattr(args, "config", None) or default
    if name is None:
        return None
    return load_config(name, getattr(args, "set", None))
