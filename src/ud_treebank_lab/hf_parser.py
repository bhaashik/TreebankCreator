"""Optional lightweight XLM-R/any HF fast-tokenizer UD POS + dependency parser.

The HF backend provides exact optimizer/epoch/batch checkpoint resumption.
It is an *alternative* to Trankit, not a replacement for its morph/lemma modules.
No generative model is used for parsing.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import networkx as nx

from .conllu import Sentence, Word, read_conllu, validate_tree
from .utils import seed_everything, write_json, write_lines, fingerprint


def _deps():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
        return torch, nn, F, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and transformers: pip install -e '.[hf]' "
                           "or pip install torch 'transformers>=4.39,<5'") from exc


def _build_model(pretrained: str, n_pos: int, n_rel: int, freeze: bool, dropout: float = 0.2):
    torch, nn, F, AutoModel, AutoTokenizer = _deps()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(pretrained)
            dim = int(self.encoder.config.hidden_size)
            self.root = nn.Parameter(torch.zeros(dim))
            self.drop = nn.Dropout(dropout)
            self.pos = nn.Linear(dim, n_pos)
            self.dep_arc = nn.Linear(dim, 256)
            self.head_arc = nn.Linear(dim, 256)
            self.dep_rel = nn.Linear(dim, 128)
            self.head_rel = nn.Linear(dim, 128)
            self.rel = nn.Linear(256, n_rel)
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad = False
                self.encoder.eval()

        def train(self, mode=True):
            super().train(mode)
            if freeze:
                self.encoder.eval()
            return self

        def forward(self, ids, attn, word_positions, word_mask, gold_heads=None):
            hidden = self.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
            b, n = word_positions.shape
            selected = hidden.gather(1, word_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
            selected = self.drop(selected)
            hiddens = torch.cat([self.root.view(1, 1, -1).expand(b, -1, -1), selected], dim=1)
            arc_scores = torch.matmul(self.dep_arc(selected), self.head_arc(hiddens).transpose(1, 2)) / 16
            arc_scores = arc_scores.masked_fill(~torch.cat([
                torch.ones((b, 1), dtype=torch.bool, device=word_mask.device), word_mask], dim=1).unsqueeze(1), -1e4)
            self_ids = torch.arange(n, device=word_mask.device)
            arc_scores[:, self_ids, self_ids + 1] = -1e4
            pos_scores = self.pos(selected)
            if gold_heads is None:
                heads = arc_scores.argmax(dim=-1)
            else:
                heads = gold_heads.clamp(min=0)
            selected_heads = hiddens.gather(1, heads.unsqueeze(-1).expand(-1, -1, hiddens.size(-1)))
            rel_scores = self.rel(torch.cat([self.dep_rel(selected),
                                             self.head_rel(selected_heads)], dim=-1))
            return pos_scores, arc_scores, rel_scores

        def label_given_heads(self, ids, attn, word_positions, heads):
            # Use a second pass for labels conditioned on globally decoded MST heads.
            # Small datasets permit this without material overhead.
            m = torch.ones_like(word_positions, dtype=torch.bool)
            return self.forward(ids, attn, word_positions, m, heads)[2]

    return Model()


def _batch(sentences, tokenizer, pos_ids, rel_ids, max_subtokens, device, label=True):
    torch, *_ = _deps()
    tokens = [[w.form for w in s.words] for s in sentences]
    enc = tokenizer(tokens, is_split_into_words=True, padding=True, truncation=True,
                    max_length=max_subtokens, return_tensors="pt")
    positions = []
    for i, sent in enumerate(sentences):
        wids = enc.word_ids(batch_index=i)
        first = {}
        for at, word_id in enumerate(wids):
            if word_id is not None and word_id not in first:
                first[word_id] = at
        if set(first) != set(range(len(sent.words))):
            raise ValueError(f"Sentence {sent.sent_id} exceeds max_subtokens={max_subtokens}. "
                             "Raise max_subtokens or filter long sentences.")
        positions.append([first[j] for j in range(len(sent.words))])
    maxw = max(len(p) for p in positions)
    pos = torch.zeros((len(sentences), maxw), dtype=torch.long)
    mask = torch.zeros((len(sentences), maxw), dtype=torch.bool)
    if label:
        upos = torch.full((len(sentences), maxw), -100, dtype=torch.long)
        heads = torch.full((len(sentences), maxw), -100, dtype=torch.long)
        rels = torch.full((len(sentences), maxw), -100, dtype=torch.long)
    else:
        upos = heads = rels = None
    for b, sent in enumerate(sentences):
        n = len(sent.words)
        pos[b, :n] = torch.tensor(positions[b], dtype=torch.long)
        mask[b, :n] = True
        if label:
            upos[b, :n] = torch.tensor([pos_ids.get(w.upos, pos_ids["X"]) for w in sent.words])
            heads[b, :n] = torch.tensor([w.head for w in sent.words])
            rels[b, :n] = torch.tensor([rel_ids.get(w.deprel, rel_ids["dep"]) for w in sent.words])
    return (enc["input_ids"].to(device), enc["attention_mask"].to(device), pos.to(device),
            mask.to(device), upos.to(device) if label else None,
            heads.to(device) if label else None, rels.to(device) if label else None)


def _loss(model, batch):
    torch, nn, F, *_ = _deps()
    ids, attn, words, mask, gold_pos, gold_head, gold_rel = batch
    pos_logits, arc_logits, rel_logits = model(ids, attn, words, mask, gold_head)
    return (F.cross_entropy(pos_logits.reshape(-1, pos_logits.size(-1)), gold_pos.reshape(-1)) +
            F.cross_entropy(arc_logits.reshape(-1, arc_logits.size(-1)), gold_head.reshape(-1)) +
            F.cross_entropy(rel_logits.reshape(-1, rel_logits.size(-1)), gold_rel.reshape(-1)))


def _atomic_torch_save(torch, value, path):
    tmp = str(path) + ".tmp"
    torch.save(value, tmp)
    os.replace(tmp, path)


def _decode(scores):
    """Return 1-based heads (0=root) with exactly one root; scores[dep,head]."""
    n = int(scores.shape[0])
    if n == 1:
        return [0]
    ranked = sorted(range(1, n + 1), key=lambda r: -float(scores[r - 1, 0]))[:3]
    best = None
    for root in ranked:
        g = nx.DiGraph()
        g.add_nodes_from(range(n + 1))
        g.add_edge(0, root, weight=float(scores[root - 1, 0]))
        for dep in range(1, n + 1):
            if dep == root:
                continue
            for head in range(1, n + 1):
                if head != dep:
                    g.add_edge(head, dep, weight=float(scores[dep - 1, head]))
        tree = nx.maximum_spanning_arborescence(g)
        weight = sum(d["weight"] for _, _, d in tree.edges(data=True))
        if best is None or weight > best[0]:
            best = (weight, tree)
    assert best is not None
    return [next(iter(best[1].predecessors(i))) for i in range(1, n + 1)]


def _infer_one(model, tokenizer, sentence, pos_ids, rel_ids, max_subtokens, device):
    torch, *_ = _deps()
    ids, attn, words, mask, _, _, _ = _batch([sentence], tokenizer, pos_ids, rel_ids,
                                            max_subtokens, device, False)
    model.eval()
    with torch.inference_mode():
        pos_logits, arc_logits, _ = model(ids, attn, words, mask)
        n = len(sentence.words)
        heads = _decode(arc_logits[0, :n, :n+1].float().cpu().numpy())
        rel_logits = model.label_given_heads(ids, attn, words,
                                             torch.tensor([heads], device=device, dtype=torch.long))
    inv_pos = {v: k for k, v in pos_ids.items()}
    inv_rel = {v: k for k, v in rel_ids.items()}
    result = []
    for i, w in enumerate(sentence.words):
        head = heads[i]
        rel = inv_rel[int(rel_logits[0, i].argmax())] if head != 0 else "root"
        if rel == "root" and head != 0:
            rel = "dep"
        result.append(Word(id=i+1, form=w.form, lemma="_", upos=inv_pos[int(pos_logits[0, i].argmax())],
                           feats="_", head=head, deprel=rel))
    predicted = Sentence(result, sentence.sent_id, sentence.text)
    assert not validate_tree(predicted)
    return predicted


def train_hf(train_file: str, dev_file: str, out_dir: str,
             pretrained: str = "xlm-roberta-base", epochs: int = 5,
             batch_size: int = 4, accumulate: int = 2, lr: float = 3e-4,
             freeze_encoder: bool = True, max_words: int = 70,
             max_subtokens: int = 256, seed: int = 17,
             save_steps: int = 200, resume: bool = True,
             device: str = "auto") -> dict:
    torch, nn, F, AutoModel, AutoTokenizer = _deps()
    seed_everything(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = [s for s in read_conllu(train_file) if len(s.words) <= max_words]
    dev = [s for s in read_conllu(dev_file) if len(s.words) <= max_words]
    if not train or not dev:
        raise ValueError("Need nonempty train/dev sentences under max_words")
    pos_tags = sorted({"X"} | {w.upos for s in train for w in s.words})
    rel_tags = sorted({"dep", "root"} | {w.deprel for s in train for w in s.words})
    pos_ids = {p: i for i, p in enumerate(pos_tags)}
    rel_ids = {r: i for i, r in enumerate(rel_tags)}
    args = {"pretrained": pretrained, "freeze_encoder": freeze_encoder, "max_words": max_words,
            "max_subtokens": max_subtokens, "seed": seed, "batch_size": batch_size,
            "accumulate": accumulate, "lr": lr, "pos_ids": pos_ids, "rel_ids": rel_ids,
            "data_signature": fingerprint(Path(train_file), Path(dev_file))}
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("HF parser needs a fast tokenizer with word_ids()")
    model = _build_model(pretrained, len(pos_ids), len(rel_ids), freeze_encoder).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    last = out / "last.pt"
    best_path = out / "best.pt"
    epoch0 = batch0 = step = 0
    best_las = -1.0
    if resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["config"] != args:
            raise ValueError("Checkpoint configuration differs. Use a new output dir or --no-resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # torch loads tensors on CPU, optimizer can be moved by PyTorch when needed.
        for value in optimizer.state.values():
            for k, v in value.items():
                if isinstance(v, torch.Tensor): value[k] = v.to(device)
        epoch0, batch0, step = state["epoch"], state["batch"], state["step"]
        best_las = state.get("best_las", -1.0)
        random.setstate(state["python_random"])
        torch.set_rng_state(state["torch_random"])
        if device == "cuda" and state.get("cuda_random") is not None:
            torch.cuda.set_rng_state_all(state["cuda_random"])
    def checkpoint(epoch, batch):
        _atomic_torch_save(torch, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "batch": batch, "step": step, "best_las": best_las,
                        "python_random": random.getstate(), "torch_random": torch.get_rng_state(),
                        "cuda_random": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                        "config": args}, last)
    for epoch in range(epoch0, epochs):
        order = list(range(len(train)))
        random.Random(seed + epoch).shuffle(order)
        chunks = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = batch0 if epoch == epoch0 else 0
        for i in range(start, len(chunks)):
            batch = _batch([train[j] for j in chunks[i]], tokenizer, pos_ids, rel_ids,
                           max_subtokens, device)
            loss = _loss(model, batch) / accumulate
            loss.backward()
            if (i + 1) % accumulate == 0 or i == len(chunks) - 1:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if save_steps and step % save_steps == 0:
                    checkpoint(epoch, i+1)
        # Use actual decoded dependency trees for validation (silver dev is proxy only).
        correct_arc = correct_label = tokens = correct_pos = 0
        for sent in dev:
            prediction = _infer_one(model, tokenizer, sent, pos_ids, rel_ids, max_subtokens, device)
            for gold, pred in zip(sent.words, prediction.words):
                tokens += 1
                correct_pos += gold.upos == pred.upos
                correct_arc += gold.head == pred.head
                correct_label += gold.head == pred.head and gold.deprel == pred.deprel
        las = correct_label / max(1, tokens)
        if las > best_las:
            best_las = las
            _atomic_torch_save(torch, {"model": model.state_dict(), "config": args,
                                     "epoch": epoch, "dev_las": las}, best_path)
        checkpoint(epoch+1, 0)
        write_json(out / "history.json", {"completed_epochs": epoch + 1, "best_dev_las": best_las,
                                           "latest_dev": {"LAS": las, "UAS": correct_arc/max(1,tokens),
                                                          "UPOS": correct_pos/max(1,tokens)},
                                           "dev_is_silver_warning": "See split_report.json"})
        batch0 = 0
    return {"model_dir": str(out), "completed_epochs": max(epochs, epoch0),
            "best_dev_las": best_las, "resume": str(last)}


def predict_hf(model_dir: str, input_conllu: str, output_conllu: str,
               checkpoint: str = "best.pt", device: str = "auto") -> dict:
    torch, nn, F, AutoModel, AutoTokenizer = _deps()
    path = Path(model_dir) / checkpoint
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained"], use_fast=True)
    model = _build_model(cfg["pretrained"], len(cfg["pos_ids"]), len(cfg["rel_ids"]),
                         cfg["freeze_encoder"]).to(device)
    model.load_state_dict(state["model"])
    n = 0
    def predict():
        nonlocal n
        for sent in read_conllu(input_conllu):
            if len(sent.words) > cfg["max_words"]:
                raise ValueError(f"Sentence {sent.sent_id} exceeds trained max_words")
            result = _infer_one(model, tokenizer, sent, cfg["pos_ids"], cfg["rel_ids"],
                                cfg["max_subtokens"], device)
            n += 1
            yield result.to_conllu()
    write_lines(output_conllu, predict())
    return {"predicted_sentences": n, "output": output_conllu}
