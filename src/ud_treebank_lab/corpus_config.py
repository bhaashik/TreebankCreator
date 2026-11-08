"""Strict, non-destructive resource configuration for corpus-wide preprocessing.

Resource modes intentionally default to 'skip'. An existing path is used as
'supplied' unless its mode explicitly says 'generate'; nothing invokes SimAlign,
Apertium, HF, or network access implicitly.
"""
from __future__ import annotations

from pathlib import Path

import yaml


RESOURCE_KEYS = {"alignment", "apertium", "unimorph"}
RESOURCE_OPTIONS = {
    "alignment": {"mode", "path", "backend", "model", "method", "device", "lexicon"},
    "apertium": {"mode", "path", "analyzer_bin", "binary", "batch_size"},
    "unimorph": {"mode", "path"},
}
DEFAULT_REFINEMENT = {"iterations": 3, "max_words": 90, "anchor_similarity": 0.72,
                      "min_link_confidence": 0.52, "temperature": 0.72,
                      "max_alternatives": 3, "min_affix_support": 3,
                      "affix_max_length": 4, "seed": 17}


def read_corpus_config(filename: str | Path) -> dict:
    path = Path(filename).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Corpus preprocessing configuration must be a YAML mapping")
    if set(cfg) - {"source_ud", "targets", "output_dir", "resources", "refinement"}:
        raise ValueError(f"Unknown corpus preprocessing keys: {sorted(set(cfg) - {'source_ud', 'targets', 'output_dir', 'resources', 'refinement'})}")
    for field in ("source_ud", "targets", "output_dir"):
        if not isinstance(cfg.get(field), str) or not cfg[field].strip():
            raise ValueError(f"Required nonempty path: {field}")
        v = Path(cfg[field]).expanduser()
        cfg[field] = str((v if v.is_absolute() else path.parent / v).resolve())
    sources = cfg.get("resources") or {}
    if not isinstance(sources, dict) or set(sources) - RESOURCE_KEYS:
        raise ValueError("resources accepts only alignment, apertium, unimorph")
    normalized = {}
    for name in RESOURCE_KEYS:
        options = sources.get(name) or {}
        if not isinstance(options, dict) or set(options) - RESOURCE_OPTIONS[name]:
            raise ValueError(f"Unsupported options for {name}")
        mode = options.get("mode", "supplied" if options.get("path") else "skip")
        if mode not in ({"supplied", "generate", "skip"} if name != "unimorph" else {"supplied", "skip"}):
            raise ValueError(f"Invalid {name}.mode: {mode}")
        if mode == "supplied" and not options.get("path"):
            raise ValueError(f"{name}.mode=supplied needs path")
        if mode == "generate" and name == "apertium" and not options.get("analyzer_bin"):
            raise ValueError("apertium.mode=generate needs analyzer_bin")
        merged = {**options, "mode": mode}
        for k in ("path", "lexicon", "analyzer_bin"):
            if merged.get(k):
                p = Path(merged[k]).expanduser()
                merged[k] = str((p if p.is_absolute() else path.parent / p).resolve())
        if mode == "supplied" and not Path(merged["path"]).is_file():
            raise FileNotFoundError(merged["path"])
        if mode == "generate" and merged.get("path"):
            # Users may point generated stages at any *new* file; supplied data
            # must never be overwritten by a generator.
            raise ValueError(f"{name}.mode=generate must omit path; output is managed")
        normalized[name] = merged
    cfg["resources"] = normalized
    refinement = cfg.get("refinement") or {}
    if not isinstance(refinement, dict) or set(refinement) - set(DEFAULT_REFINEMENT):
        raise ValueError("Invalid refinement options")
    refine = {**DEFAULT_REFINEMENT, **refinement}
    for key in ("iterations", "max_words", "max_alternatives", "min_affix_support",
                "affix_max_length"):
        if type(refine[key]) is not int or refine[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("anchor_similarity", "min_link_confidence"):
        if not 0 <= refine[key] <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    if not 0 < refine["temperature"] <= 4:
        raise ValueError("temperature must lie in (0,4]")
    cfg["refinement"] = refine
    return cfg
