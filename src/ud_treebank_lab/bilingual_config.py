"""Validated YAML configuration for bilingual experiments."""
from __future__ import annotations

from pathlib import Path
import yaml

REQUIRED = {"language", "source_ud", "targets", "alignments", "source_cache", "target_cache", "output_dir"}
PATHS = {"source_ud", "targets", "alignments", "source_cache", "target_cache", "output_dir", "gold_dev"}
DEFAULTS = {"model": "xlm-roberta-base", "epochs": 6, "batch_size": 4,
            "accumulate": 4, "lr": 0.0003, "hidden": 192, "layers": 2,
            "dropout": 0.15, "alignment_dropout": 0.15,
            "alignment_bias": 0.6, "min_strength": 0.02,
            "label_weight": 0.5, "max_words": 70, "max_subtokens": 256,
            "cache_batch_size": 12, "shard_size": 256,
            "dev_fraction": 0.1, "seed": 17, "save_steps": 100,
            "early_stopping": 3, "max_pairs": None, "device": "auto"}


def read_config(filename: str | Path) -> dict:
    filepath = Path(filename).expanduser().resolve()
    raw = yaml.safe_load(filepath.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Bilingual configuration must be a YAML mapping")
    missing = REQUIRED - set(raw)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")
    unknown = set(raw) - (REQUIRED | set(DEFAULTS) | {"gold_dev"})
    if unknown:
        raise ValueError(f"Unsupported bilingual config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **raw}
    for p in PATHS & cfg.keys():
        v = Path(str(cfg[p])).expanduser()
        cfg[p] = str(v if v.is_absolute() else filepath.parent / v)
        cfg[p] = str(Path(cfg[p]).resolve())
    for k in ("epochs", "batch_size", "accumulate", "hidden", "layers", "save_steps",
              "early_stopping", "max_words", "max_subtokens", "shard_size", "cache_batch_size"):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if cfg["max_pairs"] is not None and (type(cfg["max_pairs"]) is not int or cfg["max_pairs"] < 2):
        raise ValueError("max_pairs must be >=2 or null")
    for key in ("dropout", "alignment_dropout"):
        if not 0 <= cfg[key] < 1:
            raise ValueError(f"{key} must be in [0,1)")
    if not 0 < cfg["dev_fraction"] < 1:
        raise ValueError("dev_fraction must be in (0,1)")
    if cfg["lr"] <= 0 or cfg["label_weight"] < 0 or cfg["min_strength"] < 0:
        raise ValueError("lr positive, min_strength and label_weight nonnegative")
    return cfg
