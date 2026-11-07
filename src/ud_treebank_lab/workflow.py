"""One-command project runner with input-aware stage invalidation.

Data stages resume per sentence; training backend behavior is documented per backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .alignment import run_alignment
from .projection import run_projection
from .training import prepare_data, train_trankit
from .translate import translate_ud
from .utils import fingerprint, write_json


def _check_stage(manifest: Path, signature: str, generated: list[Path]):
    """Invalidate only generated outputs for changed inputs/config. Keep user data."""
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("signature") == signature:
            return
    for p in generated:
        if p.is_file():
            p.unlink()
    if manifest.exists():
        manifest.unlink()


def _path(config_dir: Path, s: str | Path | None) -> Path | None:
    if s is None:
        return None
    p = Path(s).expanduser()
    return p if p.is_absolute() else (config_dir / p).resolve()


def run_workflow(config_file: str | Path, train_override: str | None = None) -> dict:
    path = Path(config_file).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("YAML config root must be a mapping")
    base = path.parent
    proj = cfg.get("project", {})
    out = _path(base, proj.get("output_dir", "./runs/default"))
    assert out is not None
    out.mkdir(parents=True, exist_ok=True)
    pairs = cfg.get("pairs", [])
    if not pairs:
        raise ValueError("Config requires at least one pairs item")
    align_cfg = cfg.get("alignment", {"backend": "simalign"})
    pro_cfg = cfg.get("projection", {})
    seed = int(proj.get("seed", 17))
    run_info = {"target": proj.get("target", "unspecified"), "pairs": []}
    silver = []
    metas = []
    for idx, item in enumerate(pairs, 1):
        stage = out / f"pair_{idx:02}"
        stage.mkdir(exist_ok=True)
        src = _path(base, item["source_ud"])
        assert src is not None
        targets = _path(base, item.get("targets"))
        tcfg = item.get("translate")
        if targets is None:
            if not tcfg:
                raise ValueError(f"Pair {idx}: supply targets or translate")
            targets = stage / "target.jsonl"
            signature = fingerprint(src, tcfg)
            state = stage / "translate.state.json"
            _check_stage(state, signature, [targets])
            transl = translate_ud(src, targets, src_lang=tcfg.get("source_language", "hin_Deva"),
                                  tgt_lang=tcfg["target_language"], model_name=tcfg.get("model",
                                  "ai4bharat/indictrans2-indic-indic-dist-320M"),
                                  batch_size=int(tcfg.get("batch_size", 8)),
                                  device=tcfg.get("device", "auto"))
            write_json(state, {"signature": signature, "output": str(targets)})
        assert targets is not None
        alignment = _path(base, item.get("alignments"))
        if alignment is None:
            alignment = stage / "alignments.jsonl"
            signature = fingerprint(src, targets, align_cfg)
            state = stage / "align.state.json"
            _check_stage(state, signature, [alignment])
            run_alignment(src, targets, alignment, backend=align_cfg.get("backend", "simalign"),
                          model=align_cfg.get("model", "xlmr"),
                          method=align_cfg.get("method", "itermax"),
                          device=align_cfg.get("device", "auto"),
                          lexicon=align_cfg.get("lexicon"))
            write_json(state, {"signature": signature, "output": str(alignment)})
        morph = _path(base, item.get("morphology"))
        conllu = stage / "silver.conllu"
        meta = stage / "metadata.jsonl"
        cache = stage / "projection.stage.jsonl"
        rules = pro_cfg.get("rules", {})
        signature = fingerprint(src, targets, alignment, morph if morph else "no-morph", rules,
                                {"prefix": f"p{idx:02}:"})
        state = stage / "project.state.json"
        _check_stage(state, signature, [conllu, meta, cache])
        proj_stats = run_projection(src, targets, alignment, conllu, meta,
                                   morph_file=morph, rules=rules, cache=cache,
                                   prefix=f"p{idx:02}:")
        write_json(state, {"signature": signature, "output": str(conllu)})
        silver.append(conllu)
        metas.append(meta)
        run_info["pairs"].append({"pair": idx, "source_ud": str(src),
                                  "targets": str(targets), "silver": proj_stats})
    train_cfg = cfg.get("training", {})
    backend = train_override or train_cfg.get("backend", "none")
    if backend not in ("none", "trankit", "hf"):
        raise ValueError(f"Unknown training backend: {backend}")
    split_dir = out / "training_data"
    def optional_file(k):
        return str(_path(base, train_cfg.get(k))) if train_cfg.get(k) else None
    report = prepare_data(silver, metas, split_dir,
                          min_quality=float(pro_cfg.get("min_quality", 0.45)),
                          dev_fraction=float(pro_cfg.get("dev_fraction", .1)), seed=seed,
                          gold_train=optional_file("gold_train"),
                          gold_dev=optional_file("gold_dev"),
                          gold_test=optional_file("gold_test"),
                          gold_repeat=int(train_cfg.get("gold_repeat", 3)))
    run_info["data"] = report
    if backend == "trankit":
        run_info["training"] = train_trankit(str(split_dir / "train.conllu"),
                                            str(split_dir / "dev.conllu"),
                                            str(out / "model_trankit"),
                                            extra=train_cfg.get("extra"))
    elif backend == "hf":
        from .hf_parser import train_hf
        extra = train_cfg.get("hf", {})
        run_info["training"] = train_hf(str(split_dir / "train.conllu"),
                                        str(split_dir / "dev.conllu"), str(out / "model_hf"),
                                        seed=seed, **extra)
    else:
        run_info["training"] = {"backend": "none", "note": "Prepared silver treebank only"}
    write_json(out / "run_manifest.json", run_info)
    return run_info
