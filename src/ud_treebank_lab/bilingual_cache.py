"""Resumable sharded frozen transformer word-embedding caches.

A frozen encoder is used only under inference_mode; it never enters the trainer
or optimizer. Shards permit resumption after an interrupted Colab/Kaggle session.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import torch

from .utils import fingerprint, write_json


def cache_signature(rows: list[tuple[str, list[str]]], model: str, max_subtokens: int,
                    shard_size: int) -> str:
    return fingerprint(rows, model, max_subtokens, shard_size, "first-subword-v1")


def load_manifest(directory: str | Path) -> dict | None:
    p = Path(directory) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = str(path) + ".tmp"
    torch.save(value, staging)
    os.replace(staging, path)


def cache_embeddings(rows: list[tuple[str, list[str]]], directory: str | Path,
                     *, model: str = "xlm-roberta-base", device: str = "auto",
                     batch_size: int = 12, shard_size: int = 256,
                     max_subtokens: int = 256) -> dict:
    """Cache one embedding matrix [words,dimension] per sentence ID.

    If an interrupted shard is missing, recompute that *whole* shard. Existing
    completed shards and their manifest remain untouched on resumption.
    """
    if not rows or batch_size < 1 or shard_size < 1:
        raise ValueError("Need rows and positive batch_size/shard_size")
    if len(set(sid for sid, _ in rows)) != len(rows):
        raise ValueError("Duplicate embedding cache sentence ID")
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(rows, model, max_subtokens, shard_size)
    manifest = load_manifest(dest)
    if manifest and manifest.get("signature") != signature:
        raise ValueError(f"Cache signature mismatch for {dest}; use a new cache directory")
    if manifest is None:
        manifest = {"signature": signature, "model": model, "max_subtokens": max_subtokens,
                    "shard_size": shard_size, "num_sentences": len(rows),
                    "pooling": "first-subword-v1"}
        write_json(dest / "manifest.json", manifest)
    shards = [(i, rows[i:i + shard_size]) for i in range(0, len(rows), shard_size)]
    missing = [(i, subset) for i, subset in shards
               if not (dest / f"part_{i // shard_size:05d}.pt").exists()]
    if not missing:
        return {**manifest, "directory": str(dest), "new_shards": 0}
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Need transformers: pip install -e '.[hf]'") from exc
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Encoder requires a HF fast tokenizer with word_ids()")
    encoder = AutoModel.from_pretrained(model).to(device).eval()
    for start, subset in missing:
        output = {}
        for b in range(0, len(subset), batch_size):
            batch = subset[b:b + batch_size]
            enc = tokenizer([words for _, words in batch], is_split_into_words=True,
                            padding=True, truncation=True, max_length=max_subtokens,
                            return_tensors="pt")
            with torch.inference_mode():
                h = encoder(input_ids=enc["input_ids"].to(device),
                            attention_mask=enc["attention_mask"].to(device)).last_hidden_state
            for i, (sid, words) in enumerate(batch):
                ids = enc.word_ids(batch_index=i)
                first: dict[int, int] = {}
                for pos, idx in enumerate(ids):
                    if idx is not None and idx not in first:
                        first[idx] = pos
                if set(first) != set(range(len(words))):
                    raise ValueError(f"Subtoken limit truncated {sid}; increase max_subtokens")
                emb = h[i, [first[j] for j in range(len(words))]].float().cpu().to(torch.float16)
                output[sid] = emb
        _atomic_torch_save(output, dest / f"part_{start // shard_size:05d}.pt")
    del encoder
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {**manifest, "directory": str(dest), "new_shards": len(missing)}


def read_embeddings(rows: list[tuple[str, list[str]]], directory: str | Path,
                    *, model: str = "xlm-roberta-base", max_subtokens: int = 256,
                    shard_size: int = 256) -> dict[str, torch.Tensor]:
    dest = Path(directory)
    expected = cache_signature(rows, model, max_subtokens, shard_size)
    manifest = load_manifest(dest)
    if manifest is None or manifest.get("signature") != expected:
        raise ValueError(f"Missing or invalid embedding cache {dest}. Run cache-bilingual first")
    embeddings = {}
    for start in range(0, len(rows), shard_size):
        f = dest / f"part_{start // shard_size:05d}.pt"
        if not f.is_file():
            raise ValueError(f"Missing embedding shard {f}")
        data = torch.load(f, map_location="cpu", weights_only=True)
        embeddings.update(data)
    if set(embeddings) != {sid for sid, _ in rows}:
        raise ValueError("Cache rows do not match the requested sentence IDs")
    for sid, tokens in rows:
        if embeddings[sid].ndim != 2 or embeddings[sid].size(0) != len(tokens):
            raise ValueError(f"Invalid cached embedding shape for {sid}")
    return embeddings
