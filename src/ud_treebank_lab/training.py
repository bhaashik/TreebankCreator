"""Silver/gold dataset preparation and official Trankit integration."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .conllu import read_conllu
from .utils import jsonl, write_json, write_lines


def prepare_data(silver_files: list[str | Path], metadata_files: list[str | Path],
                 out_dir: str | Path, min_quality: float = 0.45,
                 dev_fraction: float = 0.1, seed: int = 17,
                 gold_train: str | None = None, gold_dev: str | None = None,
                 gold_test: str | None = None, gold_repeat: int = 3) -> dict:
    """Hold out gold data strictly. Silver dev is *not* an accuracy benchmark."""
    if len(silver_files) != len(metadata_files):
        raise ValueError("Need one metadata file per silver treebank")
    if not 0 <= min_quality <= 1 or not 0 <= dev_fraction < 1:
        raise ValueError("min_quality and dev_fraction must be between 0 and 1")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gold_tr = list(read_conllu(gold_train)) if gold_train else []
    gold_dv = list(read_conllu(gold_dev)) if gold_dev else []
    gold_ts = list(read_conllu(gold_test)) if gold_test else []
    # Text matching is a conservative leakage guard; duplicate paraphrases
    # require external near-duplicate detection before publication.
    protected = {s.text.strip() for s in gold_dv + gold_ts}
    protected |= {" ".join(w.form for w in s.words) for s in gold_dv + gold_ts}
    silver = []
    counts = {"total_silver": 0, "rejected_low_quality": 0,
              "rejected_gold_overlap": 0, "accepted": 0}
    seen = set()
    for sf, mf in zip(silver_files, metadata_files):
        meta = {str(d["id"]): d for d in jsonl(mf)}
        for s in read_conllu(sf):
            counts["total_silver"] += 1
            if s.sent_id not in meta:
                raise ValueError(f"Missing metadata for projected sentence {s.sent_id}")
            if float(meta[s.sent_id]["quality"]) < min_quality:
                counts["rejected_low_quality"] += 1
                continue
            txt = s.text.strip() or " ".join(w.form for w in s.words)
            if txt in protected:
                counts["rejected_gold_overlap"] += 1
                continue
            signature = (s.sent_id, txt)
            if signature in seen:
                continue
            seen.add(signature)
            silver.append(s)
    counts["accepted"] = len(silver)
    if not silver and not gold_tr:
        raise ValueError("No trainable sentences after filtering. Try min_quality=0.0")
    random.Random(seed).shuffle(silver)
    if gold_dv:
        silver_dev = []
    else:
        n_dev = max(1, round(len(silver) * dev_fraction)) if len(silver) >= 2 else 0
        n_dev = min(n_dev, len(silver) - 1)
        silver_dev = silver[:n_dev]
        silver = silver[n_dev:]
    train = silver + [s for _ in range(gold_repeat) for s in gold_tr]
    if not train:
        raise ValueError("Training split is empty")
    if not gold_dv and not silver_dev:
        # Trankit needs nonempty dev; if only one silver sentence, repeat it
        # with a clear warning (no independent validation).
        silver_dev = [train[0]]
        dev_is_train = True
    else:
        dev_is_train = False
    dev = gold_dv or silver_dev
    write_lines(out / "train.conllu", (s.to_conllu() for s in train))
    write_lines(out / "dev.conllu", (s.to_conllu() for s in dev))
    if gold_ts:
        write_lines(out / "test.conllu", (s.to_conllu() for s in gold_ts))
    report = {**counts, "train_sentences": len(train), "silver_train_sentences": len(silver),
              "dev_sentences": len(dev), "test_gold_sentences": len(gold_ts),
              "gold_train_sentences": len(gold_tr), "dev_is_gold": bool(gold_dv),
              "dev_is_train": dev_is_train,
              "warning": None if gold_dv else "Dev set is silver pseudo-gold; LAS is not true target-language quality"}
    write_json(out / "split_report.json", report)
    return report


def train_trankit(train_file: str, dev_file: str, model_dir: str,
                  task: str = "posdep", category: str = "customized", resume: bool = True,
                  extra: dict | None = None) -> dict:
    """Use Trankit's documented custom training API.

    NOTE: stage-level resumption only. Trankit internal mid-epoch restoration
    depends on its installed version; we make no such guarantee.
    """
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    from .utils import fingerprint
    signature = fingerprint(Path(train_file), Path(dev_file), {"task": task, "category": category,
                                                              "extra": extra or {}})
    done = out / f"{task}.complete.json"
    if resume and done.exists():
        with open(done, encoding="utf-8") as f:
            previous = json.load(f)
        if previous.get("signature") == signature:
            return {"status": "skipped-completed", "output": str(out)}
    try:
        import trankit
    except ImportError as exc:
        raise RuntimeError("Install the optional Trankit backend: pip install -e '.[trankit]'") from exc
    cfg = {"category": category, "task": task, "save_dir": str(out),
           "train_conllu_fpath": str(Path(train_file).resolve()),
           "dev_conllu_fpath": str(Path(dev_file).resolve())}
    cfg.update(extra or {})
    trainer = trankit.TPipeline(training_config=cfg)
    trainer.train()
    write_json(done, {"signature": signature, "task": task, "backend": "trankit"})
    return {"status": "trained", "output": str(out)}


def export_stanza(train_file: str, dev_file: str, out_dir: str,
                  language: str, treebank: str = "Silver") -> dict:
    """Export Stanza's official naming/layout without claiming models trained."""
    root = Path(out_dir) / f"UD_{language}-{treebank}"
    root.mkdir(parents=True, exist_ok=True)
    short = language.lower() + "_" + treebank.lower()
    for split, source in (("train", train_file), ("dev", dev_file)):
        dest = root / f"{short}-ud-{split}.conllu"
        dest.write_bytes(Path(source).read_bytes())
    instructions = (f"Stanza dataset prepared at: {root}\n"
                    "Use Stanza's official prepare_pos_treebank and prepare_depparse_treebank --gold "
                    "then run_pos and run_depparse scripts.\n"
                    "See docs/TRAINING.md for required environment variables and options.\n")
    (root / "STANZA_README.txt").write_text(instructions, encoding="utf-8")
    return {"path": str(root), "short_name": short,
            "note": "Dataset export only; Stanza training invoked separately"}
