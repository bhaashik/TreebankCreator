"""Strict, non-destructive resource specifications for corpus preprocessing.

The default is skip. The optional 'auto' mode only generates a missing resource
when ``if_missing: generate`` is explicitly configured. No implicit package
installation, executable invocation, or user-input modification is permitted.
"""
from __future__ import annotations

from pathlib import Path
import yaml

RESOURCE_KEYS = {"alignment", "apertium", "unimorph"}
RESOURCE_OPTIONS = {
    "alignment": {"mode", "path", "format", "available", "if_missing", "output",
                  "backend", "model", "method", "device", "lexicon", "setup",
                  "probabilities", "sentence_order", "a3_source_side", "verify_forms"},
    "apertium": {"mode", "path", "format", "available", "if_missing", "output",
                 "analyzer_bin", "binary", "batch_size", "setup", "layout", "verify_forms"},
    "unimorph": {"mode", "path", "available", "if_missing"},
}
DEFAULT_REFINEMENT = {"iterations": 3, "max_words": 90, "anchor_similarity": 0.72,
                      "min_link_confidence": 0.52, "temperature": 0.72,
                      "max_alternatives": 3, "min_affix_support": 3,
                      "affix_max_length": 4, "seed": 17}


def _path(value: str, base: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty path string")
    p = Path(value).expanduser()
    return str((p if p.is_absolute() else base / p).resolve())


def _setup(value: dict | None) -> dict:
    """A declared install command is never executed unless on_missing=install."""
    value = value or {}
    allowed = {"on_missing", "install_command", "check_module", "check_executable"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"setup accepts only {sorted(allowed)}")
    mode = value.get("on_missing", "error")
    if mode not in {"error", "skip", "install"}:
        raise ValueError("setup.on_missing must be error, skip, or install")
    command = value.get("install_command")
    if command is not None and (not isinstance(command, list) or not command or
                                any(not isinstance(a, str) or not a for a in command)):
        raise ValueError("setup.install_command must be a nonempty list of arguments")
    if mode == "install" and not command:
        raise ValueError("setup.on_missing=install requires install_command")
    for key in ("check_module", "check_executable"):
        if key in value and (not isinstance(value[key], str) or not value[key].strip()):
            raise ValueError(f"setup.{key} must be nonempty text")
    return {**value, "on_missing": mode}


def read_corpus_config(filename: str | Path) -> dict:
    path = Path(filename).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Corpus preprocessing configuration must be a YAML mapping")
    valid = {"source_ud", "targets", "output_dir", "resources", "refinement"}
    if set(cfg) - valid:
        raise ValueError(f"Unknown corpus preprocessing keys: {sorted(set(cfg) - valid)}")
    for field in ("source_ud", "targets", "output_dir"):
        cfg[field] = _path(cfg.get(field), path.parent)
    sources = cfg.get("resources") or {}
    if not isinstance(sources, dict) or set(sources) - RESOURCE_KEYS:
        raise ValueError("resources accepts only alignment, apertium, unimorph")
    normalized = {}
    for name in sorted(RESOURCE_KEYS):
        opts = sources.get(name) or {}
        if not isinstance(opts, dict) or set(opts) - RESOURCE_OPTIONS[name]:
            raise ValueError(f"Unsupported options for {name}")
        mode = opts.get("mode", "supplied" if opts.get("path") else "skip")
        valid_modes = {"supplied", "auto", "skip"} if name == "unimorph" else {
            "supplied", "auto", "generate", "skip"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid {name}.mode: {mode}")
        available = opts.get("available", "auto")
        if available not in (True, False, "auto") or (type(available) is int):
            raise ValueError(f"{name}.available must be true, false or auto")
        if_missing = opts.get("if_missing", "error")
        if if_missing not in ({"skip", "error"} if name == "unimorph" else
                           {"skip", "error", "generate"}):
            raise ValueError(f"Invalid {name}.if_missing: {if_missing}")
        if mode == "supplied" and (not opts.get("path") or available is False):
            raise ValueError(f"{name}.mode=supplied needs path and available must not be false")
        if mode == "generate" and opts.get("path"):
            raise ValueError(f"{name}.mode=generate must omit path; output is managed")
        if mode in {"generate", "skip"} and available != "auto":
            # boolean values are valid in YAML; do not silently ignore them
            raise ValueError(f"{name}.available is only relevant for supplied/auto modes")
        if mode == "auto" and available is True and not opts.get("path"):
            raise ValueError(f"{name}.available=true requires path")
        if mode == "auto" and available is False and opts.get("path"):
            raise ValueError(f"{name}.available=false must omit path")
        if mode == "auto" and not opts.get("path") and if_missing == "error":
            raise ValueError(f"{name}.mode=auto without path needs if_missing=generate or skip")
        if mode == "supplied" and not Path(_path(opts["path"], path.parent)).is_file():
            raise FileNotFoundError(_path(opts["path"], path.parent))
        fmt = opts.get("format", "auto")
        formats = {"auto", "jsonl", "awesome", "a3"} if name == "alignment" else (
            {"auto", "jsonl", "raw"} if name == "apertium" else {"auto"})
        if fmt not in formats:
            raise ValueError(f"Invalid {name}.format: {fmt}")
        merged = {**opts, "mode": mode, "available": available, "if_missing": if_missing,
                  "format": fmt}
        if name != "unimorph":
            merged["setup"] = _setup(opts.get("setup"))
        for k in ("path", "output", "lexicon", "analyzer_bin", "probabilities"):
            if merged.get(k):
                merged[k] = _path(merged[k], path.parent)
        if merged.get("path") and merged.get("output") and merged["path"] == merged["output"]:
            raise ValueError(f"{name}.output must differ from source path")
        if merged.get("batch_size") is not None and (
                type(merged["batch_size"]) is not int or merged["batch_size"] < 1):
            raise ValueError("apertium.batch_size must be positive")
        if name == "apertium" and mode == "generate" and not opts.get("analyzer_bin"):
            raise ValueError("apertium.mode=generate needs analyzer_bin")
        if name == "alignment" and opts.get("sentence_order", "source_ud") not in ("source_ud", "targets"):
            raise ValueError("alignment.sentence_order must be source_ud or targets")
        if name == "alignment" and opts.get("a3_source_side", "auto") not in ("auto", "annotated", "plain"):
            raise ValueError("alignment.a3_source_side must be auto, annotated or plain")
        if name == "apertium" and opts.get("layout", "auto") not in (
                "auto", "sentence-lines", "token-lines", "stream", "word-list"):
            raise ValueError("apertium.layout is unsupported")
        normalized[name] = merged
    cfg["resources"] = normalized
    refinement = cfg.get("refinement") or {}
    if not isinstance(refinement, dict) or set(refinement) - set(DEFAULT_REFINEMENT):
        raise ValueError("Invalid refinement options")
    refine = {**DEFAULT_REFINEMENT, **refinement}
    for key in ("iterations", "max_words", "max_alternatives", "min_affix_support", "affix_max_length"):
        if type(refine[key]) is not int or refine[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("anchor_similarity", "min_link_confidence"):
        if not 0 <= refine[key] <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    if not 0 < refine["temperature"] <= 4:
        raise ValueError("temperature must lie in (0,4]")
    cfg["refinement"] = refine
    return cfg
