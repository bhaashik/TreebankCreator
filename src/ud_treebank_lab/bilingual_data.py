"""Paired UD examples and uncertain alignment evidence for bilingual dependency parsing.

Source UD is annotated; target tokens need not have any annotation. Hard links are
*heuristic evidence*, not probabilities or gold dependencies. All indices in
alignment JSONL are zero-based (source index, target index).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from .alignment import read_targets
from .conllu import Sentence, read_conllu, validate_tree
from .utils import jsonl


@dataclass
class Pair:
    sid: str
    source: Sentence
    tokens: list[str]
    align: torch.Tensor          # [target words, source words] row-normalized
    reliability: torch.Tensor    # [target words] uncalibrated heuristic confidence


def soft_alignment(record: dict, m: int, n: int,
                   hard_link_weight: float = 0.55) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept {pairs:[[source,target,optional_weight],...]}, scored links, or a matrix.

    Probability-like scores from aligners are *not* automatically calibrated. This
    routine preserves relative weights, records row reliability separately, and
    never treats a hard aligner output as certain correspondence.
    """
    if m < 1 or n < 1:
        raise ValueError("Nonempty source and target tokens are required")
    a = torch.zeros((n, m), dtype=torch.float32)
    if "matrix" in record:
        value = torch.as_tensor(record["matrix"], dtype=torch.float32)
        if value.shape != a.shape:
            raise ValueError(f"Alignment matrix must be target x source ({n},{m})")
        if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("Alignment matrix requires finite values in [0,1]")
        a = value.clone()
    else:
        for link in record.get("links", record.get("pairs", [])):
            if isinstance(link, dict):
                src = link.get("src", link.get("source"))
                tgt = link.get("tgt", link.get("target"))
                strength = link.get("score", link.get("confidence", hard_link_weight))
            else:
                if not isinstance(link, (list, tuple)) or len(link) not in (2, 3):
                    raise ValueError("Alignment links need [src,tgt] or [src,tgt,score]")
                src, tgt = link[:2]
                strength = link[2] if len(link) == 3 else hard_link_weight
            if isinstance(src, bool) or isinstance(tgt, bool):
                raise ValueError("Alignment indices must be integers")
            i, j = int(src), int(tgt)
            w = float(strength)
            if i < 0 or i >= m or j < 0 or j >= n or not 0 <= w <= 1:
                raise ValueError(f"Out-of-range/invalid link: {(i,j,w)}")
            if not torch.isfinite(torch.tensor(w)):
                raise ValueError("Alignment score must be finite")
            a[j, i] = max(a[j, i].item(), w)
    row_sum = a.sum(dim=-1, keepdim=True)
    conditional = a / row_sum.clamp(min=1e-8)
    maximum = a.amax(dim=-1)
    # Ambiguity penalty: an n-to-1 source link is less reliable than a singleton.
    entropy = -(conditional * conditional.clamp(min=1e-12).log()).sum(dim=-1)
    reliability = maximum * torch.exp(-0.5 * entropy)
    reliability = torch.where(row_sum.squeeze(-1) > 0, reliability, 0.)
    return conditional, reliability


def load_pairs(source_ud: str | Path, targets_jsonl: str | Path,
               alignments_jsonl: str | Path, *, max_words: int = 70,
               limit: int | None = None) -> list[Pair]:
    targets = read_targets(targets_jsonl)
    alignments: dict[str, dict] = {}
    for item in jsonl(alignments_jsonl):
        sid = str(item["id"])
        if sid in alignments:
            raise ValueError(f"Duplicate alignment sent_id {sid}")
        alignments[sid] = item
    result: list[Pair] = []
    seen: set[str] = set()
    for sent in read_conllu(source_ud):
        sid = str(sent.sent_id)
        if sid in seen:
            raise ValueError(f"Duplicate source sent_id {sid}")
        seen.add(sid)
        if sid not in targets:
            continue
        if sid not in alignments:
            raise ValueError(f"No alignment for {sid}")
        tokens = targets[sid]["tokens"]
        if len(tokens) > max_words or len(sent.words) > max_words:
            continue
        errors = validate_tree(sent)
        if errors:
            raise ValueError(f"Invalid Hindi gold source tree for {sid}: {errors}")
        a, confidence = soft_alignment(alignments[sid], len(sent.words), len(tokens))
        result.append(Pair(sid, sent, tokens, a, confidence))
        if limit is not None and len(result) >= limit:
            break
    if not result:
        raise ValueError("No suitable aligned pairs; check sent_ids, max_words and paths")
    missing_targets = set(targets) - seen
    if missing_targets and limit is None:
        raise ValueError(f"Target sent_ids not found in source UD: {sorted(missing_targets)[:5]}")
    return result


def split_pairs(pairs: list[Pair], dev_fraction: float, seed: int) -> tuple[list[Pair], list[Pair]]:
    if len(pairs) < 2 or not 0 < dev_fraction < 1:
        raise ValueError("Need >=2 pairs and 0 < dev_fraction < 1")
    ranked = sorted(pairs, key=lambda p: hashlib.sha256(f"{seed}:{p.sid}".encode()).digest())
    num_dev = min(len(pairs) - 1, max(1, round(dev_fraction * len(pairs))))
    dev_ids = {p.sid for p in ranked[:num_dev]}
    return [p for p in pairs if p.sid not in dev_ids], [p for p in pairs if p.sid in dev_ids]


def label_inventory(pairs: list[Pair]) -> list[str]:
    return sorted({"dep", "root"} | {w.deprel for p in pairs for w in p.source.words})


def soft_structural_targets(pair: Pair, label_ids: dict[str, int],
                            *, min_strength: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-target-token joint q(dep, head, relation) plus arc and reliability.

    The source is gold UD, but projected structures are only *weak labels*. We
    exclude target self loops and give zero training weight when no source arc
    can be projected; we never invent confident root heads for such words.
    """
    n, m = len(pair.tokens), len(pair.source.words)
    joint = torch.zeros((n, n + 1, len(label_ids)), dtype=torch.float32)
    a = pair.align
    rel = pair.reliability
    # All source dependents are known from Hindi gold UD. Construct a
    # source-dependent -> possible target head probability table, then project
    # with small dense matrix multiplications. This avoids O(n^2*m) Python loops
    # on 17k-sentence corpora and preserves many-to-many uncertainty.
    possible_heads = torch.zeros((m, n + 1), dtype=torch.float32)
    source_labels = {}
    for sw in pair.source.words:
        sd = sw.id - 1
        source_labels.setdefault(label_ids.get(sw.deprel, label_ids["dep"]), []).append(sd)
        if sw.head == 0:
            possible_heads[sd, 0] = 1.0
        else:
            possible_heads[sd, 1:] = a[:, sw.head - 1] * rel
    dep_mass = a * rel[:, None]
    for lid, columns in source_labels.items():
        joint[:, :, lid] = dep_mass[:, columns] @ possible_heads[columns, :]
    # A dependent cannot be its own head even if multiple Hindi tokens merge
    # into one target token. The remaining mass is intentionally not replaced.
    dep_idx = torch.arange(n)
    joint[dep_idx, dep_idx + 1, :] = 0
    raw_mass = joint.sum((1, 2))
    weight = raw_mass.clamp(max=1.0)
    # weak supervision can be dropped entirely for uncertain token projections.
    weight = torch.where(weight >= min_strength, weight, torch.zeros_like(weight))
    q_joint = joint / raw_mass[:, None, None].clamp(min=1e-8)
    q_arc = q_joint.sum(dim=-1)
    return q_joint, q_arc, weight
