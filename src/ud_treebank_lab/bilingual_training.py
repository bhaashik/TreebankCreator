"""Run checkpointable bilingual weak-supervision training and source-guided annotation.

NOTE: dev weak-projection agreement is NOT target-language parsing accuracy.
Only independently annotated target dev trees support meaningful LAS selection.
"""
from __future__ import annotations

from collections import Counter
import json
import os
import random
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .bilingual_cache import cache_embeddings, read_embeddings
from .bilingual_config import read_config, ensure_corpus_preprocessing
from .bilingual_data import Pair, label_inventory, load_pairs, soft_structural_targets, split_pairs
from .bilingual_model import ParallelParser, weak_loss
from .conllu import Word, Sentence, read_conllu, validate_tree
from .hf_parser import _decode
from .utils import fingerprint, seed_everything, write_json, write_lines


def _source_rows(cfg: dict) -> list[tuple[str, list[str]]]:
    rows = []
    seen = set()
    for s in read_conllu(cfg["source_ud"]):
        if s.sent_id in seen:
            raise ValueError(f"Duplicate source sentence ID {s.sent_id}")
        seen.add(s.sent_id)
        if len(s.words) <= cfg["max_words"]:
            rows.append((str(s.sent_id), [w.form for w in s.words]))
    return rows


def _pairs(cfg: dict) -> list[Pair]:
    return load_pairs(cfg["source_ud"], cfg["targets"], cfg["alignments"],
                      max_words=cfg["max_words"], limit=cfg["max_pairs"])


def cache_for_config(config_file: str, side: str = "both", device: str | None = None) -> dict:
    cfg = read_config(config_file)
    if side in ("target", "both"):
        ensure_corpus_preprocessing(cfg)
    if side not in ("source", "target", "both"):
        raise ValueError("side must be source, target or both")
    params = dict(model=cfg["model"], device=device or cfg["device"],
                  batch_size=cfg["cache_batch_size"], shard_size=cfg["shard_size"],
                  max_subtokens=cfg["max_subtokens"])
    result = {}
    if side in ("source", "both"):
        result["source"] = cache_embeddings(_source_rows(cfg), cfg["source_cache"], **params)
    if side in ("target", "both"):
        result["target"] = cache_embeddings([(p.sid, p.tokens) for p in _pairs(cfg)],
                                            cfg["target_cache"], **params)
    return result


def _load_data(cfg: dict):
    pairs = _pairs(cfg)
    source_embeddings = read_embeddings(_source_rows(cfg), cfg["source_cache"],
                                        model=cfg["model"], max_subtokens=cfg["max_subtokens"],
                                        shard_size=cfg["shard_size"])
    target_embeddings = read_embeddings([(p.sid, p.tokens) for p in pairs], cfg["target_cache"],
                                        model=cfg["model"], max_subtokens=cfg["max_subtokens"],
                                        shard_size=cfg["shard_size"])
    dims = {v.size(1) for v in source_embeddings.values()} | {v.size(1) for v in target_embeddings.values()}
    if len(dims) != 1:
        raise ValueError("Source and target caches must have the same encoder width")
    return pairs, source_embeddings, target_embeddings, next(iter(dims))


def _batch(rows: list[Pair], source_cache: dict, target_cache: dict,
           labels: dict[str, int], device: str, min_strength: float):
    b = len(rows)
    m = max(len(p.source.words) for p in rows)
    n = max(len(p.tokens) for p in rows)
    d = source_cache[rows[0].sid].size(1)
    s = torch.zeros((b, m, d), dtype=torch.float32)
    t = torch.zeros((b, n, d), dtype=torch.float32)
    heads = torch.zeros((b, m), dtype=torch.long)
    sm = torch.zeros((b, m), dtype=torch.bool)
    tm = torch.zeros((b, n), dtype=torch.bool)
    a = torch.zeros((b, n, m), dtype=torch.float32)
    reliab = torch.zeros((b, n), dtype=torch.float32)
    q = torch.zeros((b, n, n + 1, len(labels)), dtype=torch.float32)
    weights = torch.zeros((b, n), dtype=torch.float32)
    for i, p in enumerate(rows):
        si, ti = len(p.source.words), len(p.tokens)
        s[i, :si] = source_cache[p.sid].float()
        t[i, :ti] = target_cache[p.sid].float()
        heads[i, :si] = torch.tensor([w.head for w in p.source.words])
        sm[i, :si] = True
        tm[i, :ti] = True
        a[i, :ti, :si] = p.align
        reliab[i, :ti] = p.reliability
        joint, _, weight = soft_structural_targets(p, labels, min_strength=min_strength)
        q[i, :ti, :ti + 1] = joint
        weights[i, :ti] = weight
    return tuple(x.to(device) for x in (s, t, heads, sm, tm, a, reliab, q, weights))


def _model_forward(model, batch):
    s, t, heads, sm, tm, a, reliab, q, w = batch
    return model(s, t, heads, sm, tm, a, reliab)


def _decode_tree(scores) -> list[int]:
    """Fast-path greedy single-root acyclic trees; otherwise exact MST fallback.

    The fallback uses the existing one-root maximum spanning arborescence
    implementation. This avoids running NetworkX for already-valid trees.
    """
    import numpy as np
    scores = np.asarray(scores)
    n = scores.shape[0]
    if scores.shape != (n, n + 1):
        raise ValueError("Arc score matrix must be [target words, heads+root]")
    heads = scores.argmax(axis=-1).astype(int).tolist()
    if heads.count(0) == 1 and all(h != i + 1 for i, h in enumerate(heads)):
        good = True
        for i in range(n):
            seen = {i + 1}
            head = heads[i]
            while head:
                if head < 0 or head > n or head in seen:
                    good = False
                    break
                seen.add(head)
                head = heads[head - 1]
            if not good:
                break
        if good:
            return heads
    return _decode(scores)


def _validate_gold(gold_path: str | None, dev: list[Pair]) -> dict[str, Sentence]:
    if not gold_path:
        return {}
    dev_by_id = {p.sid: p for p in dev}
    gold: dict[str, Sentence] = {}
    for sent in read_conllu(gold_path):
        if sent.sent_id not in dev_by_id:
            raise ValueError(f"Gold-dev ID {sent.sent_id} is not in held-out dev pairs")
        if [w.form for w in sent.words] != dev_by_id[sent.sent_id].tokens:
            raise ValueError(f"Gold-dev tokenization differs for {sent.sent_id}")
        if validate_tree(sent):
            raise ValueError(f"Invalid independent gold-dev tree {sent.sent_id}")
        gold[sent.sent_id] = sent
    if not gold:
        raise ValueError("Gold-dev file contains no held-out example")
    return gold


def _evaluate(model, dev, gold, src_emb, tgt_emb, labels, cfg, device):
    model.eval()
    val_sum = 0.0
    val_mass = 0.0
    uas = las = words = 0
    inv = {v: k for k, v in labels.items()}
    with torch.inference_mode():
        for start in range(0, len(dev), cfg["batch_size"]):
            rows = dev[start:start + cfg["batch_size"]]
            batch = _batch(rows, src_emb, tgt_emb, labels, device, cfg["min_strength"])
            arcs, rel, _ = _model_forward(model, batch)
            loss, details = weak_loss(arcs, rel, batch[-2], batch[-1],
                                      label_weight=cfg["label_weight"])
            if details["mass"]:
                val_sum += float(loss) * details["mass"]
                val_mass += details["mass"]
            for i, p in enumerate(rows):
                if p.sid not in gold:
                    continue
                n = len(p.tokens)
                predicted_heads = _decode_tree(arcs[i, :n, :n + 1].float().cpu().numpy())
                for dep, head in enumerate(predicted_heads):
                    g = gold[p.sid].words[dep]
                    predicted_rel = "root" if head == 0 else inv[int(rel[i, dep, head].argmax())]
                    if predicted_rel == "root" and head != 0:
                        predicted_rel = "dep"
                    uas += int(head == g.head)
                    las += int(head == g.head and predicted_rel == g.deprel)
                    words += 1
    if val_mass <= 0:
        raise ValueError("No projectable dev arcs; adjust alignments/quality filtering")
    result = {"weak_dev_loss": val_sum / val_mass, "weak_dev_mass": val_mass}
    if gold:
        result.update({"gold_dev_uas": uas / words, "gold_dev_las": las / words,
                       "gold_dev_words": words})
    return result


def _path_signature(cfg: dict) -> str:
    items = [{k: v for k, v in cfg.items() if k != "device"}, Path(cfg["source_ud"]), Path(cfg["targets"]), Path(cfg["alignments"])]
    if cfg.get("gold_dev"):
        items.append(Path(cfg["gold_dev"]))
    return fingerprint(*items)


def _atomic_save(data: dict, p: Path):
    tmp = str(p) + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, p)


def train_bilingual(config_file: str, *, device: str | None = None,
                    max_steps: int | None = None, auto_cache: bool = True) -> dict:
    cfg = read_config(config_file)
    ensure_corpus_preprocessing(cfg)
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be a positive optimizer-step count")
    if device:
        cfg["device"] = device
    selected_device = cfg["device"]
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # This is idempotent and stages only missing cache shards. In a two-GPU
    # launch, the launcher stages the shared Hindi cache in advance.
    if auto_cache:
        cache_for_config(config_file, device=selected_device)
    pairs, src, tgt, width = _load_data(cfg)
    train, dev = split_pairs(pairs, cfg["dev_fraction"], cfg["seed"])
    gold = _validate_gold(cfg.get("gold_dev"), dev)
    labels = {x: i for i, x in enumerate(label_inventory(train))}
    # Relation inventories are drawn only from TRAIN source trees, never gold-dev.
    seed_everything(cfg["seed"])
    model = ParallelParser(width, len(labels), hidden=cfg["hidden"], layers=cfg["layers"],
                           dropout=cfg["dropout"], alignment_dropout=cfg["alignment_dropout"],
                           alignment_bias=cfg["alignment_bias"]).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    signature = _path_signature(cfg)
    done_file = output / "done.json"
    if done_file.exists():
        done = json.loads(done_file.read_text(encoding="utf-8"))
        if done.get("signature") == signature and max_steps is None:
            return {"status": "already-complete", "output_dir": str(output),
                    "epochs": done["epochs"], "global_steps": done["steps"]}
        if done.get("signature") != signature:
            raise ValueError("Completed run signature differs; choose a new output directory")
    last = output / "last.pt"
    best = output / "best.pt"
    epoch0 = batch0 = global_steps = bad_epochs = 0
    best_metric = -float("inf")
    history = []
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["signature"] != signature or saved["label_ids"] != labels:
            raise ValueError("Incompatible checkpoint, use a different output directory")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for item in optimizer.state.values():
            for key, tensor in item.items():
                if isinstance(tensor, torch.Tensor):
                    item[key] = tensor.to(selected_device)
        epoch0, batch0 = saved["epoch"], saved["batch"]
        global_steps = saved["global_steps"]
        best_metric = saved["best_metric"]
        bad_epochs = saved["bad_epochs"]
        history = saved["history"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if selected_device.startswith("cuda") and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device=selected_device)

    def state(epoch, batch):
        return {"signature": signature, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "label_ids": labels,
                "epoch": epoch, "batch": batch, "global_steps": global_steps,
                "best_metric": best_metric, "bad_epochs": bad_epochs,
                "history": history, "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state(device=selected_device)
                if selected_device.startswith("cuda") else None}

    if not any(float(soft_structural_targets(p, labels, min_strength=cfg["min_strength"])[2].sum()) > 0
               for p in train):
        raise ValueError("No projectable weak training arcs. Inspect alignment files")
    for epoch in range(epoch0, cfg["epochs"]):
        ordered = train[:]
        random.Random(cfg["seed"] + epoch).shuffle(ordered)
        batches = [ordered[i:i + cfg["batch_size"]] for i in range(0, len(ordered), cfg["batch_size"])]
        from_batch = batch0 if epoch == epoch0 else 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        start_time = time.monotonic()
        for bi in range(from_batch, len(batches)):
            batch = _batch(batches[bi], src, tgt, labels, selected_device, cfg["min_strength"])
            arc_logits, rel_logits, _ = _model_forward(model, batch)
            loss, details = weak_loss(arc_logits, rel_logits, batch[-2], batch[-1],
                                      label_weight=cfg["label_weight"])
            (loss / cfg["accumulate"]).backward()
            loss_total += float(loss.detach())
            # Allow clean checkpoint boundaries even when batches are not divisible
            # by accumulate. A partial last accumulation still gets an optimizer step.
            if ((bi - from_batch + 1) % cfg["accumulate"] == 0) or bi + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_steps += 1
                if global_steps % cfg["save_steps"] == 0:
                    _atomic_save(state(epoch, bi + 1), last)
                if max_steps is not None and global_steps >= max_steps:
                    _atomic_save(state(epoch, bi + 1), last)
                    return {"status": "paused", "global_steps": global_steps,
                            "epoch": epoch, "next_batch": bi + 1, "output_dir": str(output)}
        val = _evaluate(model, dev, gold, src, tgt, labels, cfg, selected_device)
        # Without independent target gold, optimizing weak-dev loss is NOT a LAS metric.
        metric = val["gold_dev_las"] if gold else -val["weak_dev_loss"]
        improvement = metric > best_metric + 1e-8
        if improvement:
            best_metric = metric
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({"epoch": epoch + 1, "train_loss_sum": loss_total,
                        "epoch_seconds": round(time.monotonic() - start_time, 2), **val})
        if improvement:
            _atomic_save(state(epoch + 1, 0), best)
        _atomic_save(state(epoch + 1, 0), last)
        write_json(output / "history.json", history)
        print(f"[{cfg['language']}] epoch={epoch+1}/{cfg['epochs']} step={global_steps} "
              f"weak_dev_loss={val['weak_dev_loss']:.4f} "
              + (f"gold_dev_las={val['gold_dev_las']:.4f}" if gold else "(no gold LAS)"), flush=True)
        if bad_epochs >= cfg["early_stopping"]:
            break
    write_json(output / "labels.json", labels)
    done = {"signature": signature, "epochs": len(history), "steps": global_steps,
            "model_selection": "independent_gold_dev_las" if gold else "weak_projection_dev_loss",
            "best_metric": best_metric, "config": cfg}
    write_json(done_file, done)
    return {"status": "complete", "output_dir": str(output), "global_steps": global_steps,
            "epochs": len(history), "model_selection": done["model_selection"]}


def predict_bilingual(config_file: str, *, out_file: str | None = None,
                      checkpoint: str = "best.pt", device: str = "auto") -> dict:
    cfg = read_config(config_file)
    ensure_corpus_preprocessing(cfg)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs, src, tgt, width = _load_data(cfg)
    ckpt = Path(cfg["output_dir"]) / checkpoint
    saved = torch.load(ckpt, map_location="cpu", weights_only=True)
    if saved["signature"] != _path_signature(cfg):
        raise ValueError("Checkpoint does not match input data/config")
    label_ids = saved["label_ids"]
    labels = {int(v): str(k) for k, v in label_ids.items()}
    model = ParallelParser(width, len(label_ids), hidden=cfg["hidden"], layers=cfg["layers"],
                           dropout=cfg["dropout"], alignment_dropout=cfg["alignment_dropout"],
                           alignment_bias=cfg["alignment_bias"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    result = []
    with torch.inference_mode():
        for begin in range(0, len(pairs), cfg["batch_size"]):
            rows = pairs[begin:begin + cfg["batch_size"]]
            b = _batch(rows, src, tgt, label_ids, device, cfg["min_strength"])
            arcs, rels, _ = _model_forward(model, b)
            for i, pair in enumerate(rows):
                n = len(pair.tokens)
                heads = _decode_tree(arcs[i, :n, :n + 1].float().cpu().numpy())
                words = []
                for dep, (token, head) in enumerate(zip(pair.tokens, heads)):
                    # Weak UPOS vote from aligned source (NOT target gold/morphology).
                    counts = Counter()
                    for j, src_word in enumerate(pair.source.words):
                        counts[src_word.upos] += float(pair.align[dep, j])
                    upos = counts.most_common(1)[0][0] if counts and pair.reliability[dep] > 0 else "X"
                    deprel = "root" if head == 0 else labels[int(rels[i, dep, head].argmax())]
                    if head != 0 and deprel == "root":
                        deprel = "dep"
                    words.append(Word(id=dep + 1, form=token, upos=upos,
                                      head=head, deprel=deprel))
                sent = Sentence(words, pair.sid, " ".join(pair.tokens),
                                ["# annotation = bilingual_silver; upos = weak_hindi_transfer"])
                invalid = validate_tree(sent)
                if invalid:
                    raise RuntimeError(f"Decoder produced invalid basic UD tree {pair.sid}: {invalid}")
                result.append(sent.to_conllu())
    path = out_file or str(Path(cfg["output_dir"]) / "refined_silver.conllu")
    write_lines(path, result)
    return {"path": path, "sentences": len(result), "kind": "silver_target_dependency"}
