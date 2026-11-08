"""Generate one bilingual YAML experiment config for each target language."""
from __future__ import annotations

from pathlib import Path
import re

import yaml

from .utils import jsonl


def make_configs(manifest: str, source: str, config_dir: str, cache_root: str,
                 runs_root: str, *, model: str = "xlm-roberta-base",
                 epochs: int = 6, batch_size: int = 4, max_pairs: int | None = None) -> dict:
    manifest_p = Path(manifest).expanduser().resolve()
    source_p = Path(source).expanduser().resolve()
    if not source_p.is_file():
        raise FileNotFoundError(source_p)
    out = Path(config_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_root).expanduser().resolve()
    runs = Path(runs_root).expanduser().resolve()
    result = []
    seen = set()
    for item in jsonl(manifest_p):
        code = str(item.get("language", ""))
        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,15}", code):
            raise ValueError(f"Invalid language code in manifest: {code!r}")
        if code in seen:
            raise ValueError(f"Duplicate language {code}")
        seen.add(code)
        if "targets" not in item or ("alignments" not in item and "corpus_preprocess" not in item):
            raise ValueError(f"Manifest entry {code} requires targets and either alignments or corpus_preprocess")
        cfg = {"language": code, "source_ud": str(source_p),
               "source_cache": str(cache / "shared_hindi"),
               "target_cache": str(cache / code),
               "output_dir": str(runs / code), "model": model,
               "epochs": epochs, "batch_size": batch_size}
        if max_pairs is not None:
            cfg["max_pairs"] = max_pairs
        for key in ("targets", "alignments", "gold_dev", "corpus_preprocess"):
            if key in item:
                name = Path(str(item[key])).expanduser()
                cfg[key] = str((name if name.is_absolute() else manifest_p.parent / name).resolve())
        if "corpus_preprocess" in cfg:
            from .corpus_config import read_corpus_config
            corpus = read_corpus_config(cfg["corpus_preprocess"])
            expected = str(Path(corpus["output_dir"]) / "alignments.refined.jsonl")
            if "alignments" in cfg and cfg["alignments"] != expected:
                raise ValueError(f"Manifest {code} alignments disagree with corpus output")
            if Path(cfg["targets"]) != Path(corpus["targets"]) or source_p != Path(corpus["source_ud"]):
                raise ValueError(f"Manifest {code} source/targets disagree with corpus config")
            cfg["alignments"] = expected
        path = out / f"hindi_to_{code}_bilingual.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        result.append(str(path))
    if not result:
        raise ValueError("Manifest has no language entries")
    return {"configs": result, "languages": len(result),
            "shared_source_cache": str(cache / "shared_hindi")}
