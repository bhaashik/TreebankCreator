"""Atomic I/O, deterministic hashing and resumable JSONL stage storage."""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_json(path: str | Path, value: Any) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=dest.parent, encoding="utf-8", delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        staging = f.name
    os.replace(staging, dest)


def jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL {path}:{number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL record must be an object at {path}:{number}")
                yield value


def repair_jsonl(path: str | Path) -> None:
    """Discard a partly written final line (e.g. interrupted GPU VM)."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, "rb+") as f:
        f.seek(0, os.SEEK_END)
        n = f.tell()
        if n == 0:
            return
        f.seek(n - 1)
        if f.read(1) == b"\n":
            return
        block = min(n, 1_000_000)
        f.seek(n - block)
        content = f.read(block)
        last_newline = content.rfind(b"\n")
        f.truncate(n - block + last_newline + 1 if last_newline >= 0 else 0)


def append_jsonl(path: str | Path, item: dict) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def done_ids(path: str | Path) -> set[str]:
    repair_jsonl(path)
    return {str(row["id"]) for row in jsonl(path)} if Path(path).exists() else set()


def fingerprint(*items: Any) -> str:
    """Hash configs and full input file bytes for invalidation."""
    h = hashlib.sha256()
    for item in items:
        if isinstance(item, Path):
            if not item.is_file():
                raise FileNotFoundError(item)
            h.update(str(item.resolve()).encode("utf-8"))
            with item.open("rb") as f:
                for part in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(part)
        else:
            h.update(json.dumps(item, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def write_lines(path: str | Path, lines: Iterable[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=p.parent, delete=False) as f:
        f.writelines(lines)
        name = f.name
    os.replace(name, p)


def seed_everything(seed: int = 17) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
