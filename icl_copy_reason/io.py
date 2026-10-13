"""Small shared I/O helpers (JSON, JSON lines, file hashes).

Inputs : paths (str or Path). Outputs: parsed objects / written files. No other module re-implements these.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def _open(path: str | Path, mode: str):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode + ("t" if "b" not in mode else ""), encoding=None if "b" in mode else "utf-8")
    return open(p, mode, encoding=None if "b" in mode else "utf-8")


def read_json(path: str | Path) -> Any:
    with _open(path, "r") as fh:
        return json.load(fh)


def write_json(obj: Any, path: str | Path, indent: int | None = 1) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _open(p, "w") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
    return p


def read_jsonl(path: str | Path) -> Iterator[Any]:
    with _open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
