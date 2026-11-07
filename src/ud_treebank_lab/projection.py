"""Alignment-based UD projection with weighted single-root arborescence repair.

This implements a deliberately conservative first-pass transfer system. It avoids
invalid trees, never fabricates language-specific lemmas/morphology from Hindi,
and records provenance/confidence for review or filtering.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import unicodedata

import networkx as nx

from .alignment import read_alignments, read_targets
from .conllu import Sentence, Word, read_conllu, validate_tree
from .morphology import read_morph
from .utils import append_jsonl, done_ids, jsonl, write_lines


@dataclass(frozen=True)
class Arc:
    score: float
    label: str
    origin: str
    confidence: float


DEFAULT_RULES = {
    # Very weak preferences: these are NOT universal syntactic laws.
    "direction": {},
    "direction_bonus": 0.08,
    "root_upos": ["VERB", "AUX"],
    "root_verb_bonus": 0.12,
    "max_root_candidates": 3,
    "min_score": 0.0,
    "min_coverage": 0.0,
}


def _punct(token: str) -> bool:
    return bool(token) and all(unicodedata.category(c).startswith("P") for c in token)


def _bonus(dep: int, head: int, label: str, rules: dict) -> float:
    direction = (rules.get("direction") or {}).get(label, "either")
    if direction == "right":
        return float(rules.get("direction_bonus", 0.08)) * (1 if head > dep else -1)
    if direction == "left":
        return float(rules.get("direction_bonus", 0.08)) * (1 if head < dep else -1)
    return 0.0


def _choose_upos(tindex: int, source: Sentence, backwards: dict[int, list[int]],
                 morph: list[dict]) -> tuple[str, str, str]:
    m = morph[tindex] if tindex < len(morph) else {}
    pos_votes = Counter(source.words[si].upos for si in backwards.get(tindex, [])
                        if source.words[si].upos not in ("_", ""))
    pos = m.get("upos") or (pos_votes.most_common(1)[0][0] if pos_votes else "X")
    lemma = m.get("lemma") or "_"
    feats = m.get("feats", {})
    return pos, lemma, "|".join(f"{k}={v}" for k, v in sorted(feats.items())) or "_"


def project_sentence(source: Sentence, target_tokens: list[str],
                     links: list[tuple[int, int]], morph: list[dict] | None = None,
                     rules: dict | None = None, target_id: str | None = None) -> tuple[Sentence, dict]:
    """Project one dependency sentence, using 0-based alignment link indices.

    Source: 1-based UD indices and HEAD=0 for root. Target: 0-based link indices.
    Returns a valid, single-root CoNLL-U basic dependency tree plus diagnostics.
    """
    cfg = {**DEFAULT_RULES, **(rules or {})}
    n = len(target_tokens)
    if not n or not source.words:
        raise ValueError("Source and target sentences must be nonempty")
    errors = validate_tree(source)
    if errors:
        raise ValueError(f"Source sentence {source.sent_id} invalid: {errors}")
    for si, ti in links:
        if not 0 <= si < len(source.words) or not 0 <= ti < n:
            raise ValueError(f"Out-of-range alignment {si}-{ti} in {source.sent_id}")
    links = sorted(set(links))
    fw: dict[int, list[int]] = defaultdict(list)
    bw: dict[int, list[int]] = defaultdict(list)
    for si, ti in links:
        fw[si].append(ti)
        bw[ti].append(si)
    for value in fw.values(): value.sort()
    for value in bw.values(): value.sort()
    md = morph or [{} for _ in target_tokens]
    tags = [_choose_upos(i, source, bw, md) for i in range(n)]
    for i, tok in enumerate(target_tokens):
        if _punct(tok):
            tags[i] = ("PUNCT", tok, "_")
        elif tok.isdecimal() and tags[i][0] == "X":
            tags[i] = ("NUM", "_", "NumType=Card")

    # A source node may project onto multiple target tokens. Choose an anchor
    # (not an arbitrary expanding root) and use weak edges for extra tokens.
    def anchor(i: int) -> int | None:
        targets = fw.get(i, [])
        if not targets:
            return None
        desired = source.words[i].upos
        matches = [t for t in targets if tags[t][0] == desired]
        return (matches or targets)[0] + 1

    candidates: dict[tuple[int, int], Arc] = {}   # (head, dependent), 1-based
    root_voting = Counter()

    def propose(h: int, d: int, arc: Arc) -> None:
        if h == d or d < 1 or d > n or h < 0 or h > n:
            return
        k = (h, d)
        score = arc.score + (_bonus(d, h, arc.label, cfg) if h else 0)
        old = candidates.get(k)
        candidate = Arc(score, arc.label, arc.origin, arc.confidence)
        if old is None or score > old.score:
            candidates[k] = candidate

    for si, sw in enumerate(source.words):
        dep_targets = fw.get(si, [])
        if not dep_targets:
            continue
        tdep = anchor(si)
        assert tdep is not None
        if sw.head == 0:
            root_voting[tdep] += 2.0
        else:
            ancestor = sw.head
            hops = 0
            visited = set()
            while ancestor and (ancestor - 1) not in fw:
                if ancestor in visited:
                    ancestor = 0
                    break
                visited.add(ancestor)
                ancestor = source.words[ancestor - 1].head
                hops += 1
            thead = anchor(ancestor - 1) if ancestor else None
            if thead and thead != tdep:
                ambiguous = max(len(dep_targets), len(fw.get(ancestor - 1, [])),
                                len(bw.get(tdep - 1, [])))
                conf = min(0.96, max(0.28, .94 - .10 * (ambiguous - 1) - .09 * hops))
                score = 1.0 + conf - 0.08 * hops
                propose(thead, tdep, Arc(score, sw.deprel if sw.deprel != "root" else "dep",
                                          "projection", conf))
        for extra in dep_targets:
            t = extra + 1
            if t != tdep:
                propose(tdep, t, Arc(0.92, "dep", "many-to-one-expansion", 0.34))

    for ti, token in enumerate(target_tokens):
        if not _punct(token):
            continue
        tdep = ti + 1
        heads = [j + 1 for j in range(n) if j != ti and not _punct(target_tokens[j])]
        if heads:
            head = min(heads, key=lambda h: (abs(h - tdep), h))
            propose(head, tdep, Arc(0.95, "punct", "punctuation", 0.55))

    # Root candidates: projection-based first, with verb fallback. Each run
    # forces one single root (only node receiving an edge from artificial 0).
    ranks = Counter(root_voting)
    for i, (upos, _, _) in enumerate(tags, 1):
        if upos in cfg.get("root_upos", ["VERB", "AUX"]):
            ranks[i] += float(cfg.get("root_verb_bonus", 0.12))
    if not ranks:
        ranks[1] = 0.01
    root_candidates = [i for i, _ in sorted(ranks.items(), key=lambda x: (-x[1], x[0]))
                       ][:max(1, int(cfg.get("max_root_candidates", 3)))]

    # We need one fixed root and a maximally scored directed arborescence. The
    # complete low-score fallback graph ensures connectivity even without links.
    best = None
    for root in root_candidates:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n + 1))
        graph.add_edge(0, root, weight=2.0 + root_voting[root],
                       label="root", origin="root-projection" if root in root_voting else "root-guess",
                       confidence=.90 if root in root_voting else .35)
        for dep in range(1, n + 1):
            if dep == root:
                continue
            for head in range(1, n + 1):
                if head == dep:
                    continue
                arc = candidates.get((head, dep))
                if arc is None:
                    # Prefer a shorter fallback edge. Don't force SOV/SVO across languages.
                    arc = Arc(0.04 - abs(head - dep) * 0.001, "dep", "fallback", 0.14)
                graph.add_edge(head, dep, weight=arc.score, label=arc.label,
                               origin=arc.origin, confidence=arc.confidence)
        tree = nx.maximum_spanning_arborescence(graph, preserve_attrs=True)
        if tree.number_of_edges() != n or tree.in_degree(0) != 0:
            raise RuntimeError("Could not construct single-root arborescence")
        quality = sum(float(info["weight"]) for _, _, info in tree.edges(data=True))
        if best is None or quality > best[0]:
            best = (quality, tree)
    assert best is not None
    tree = best[1]
    words = []
    arc_meta = []
    for i, token in enumerate(target_tokens, 1):
        incoming = list(tree.in_edges(i, data=True))
        if len(incoming) != 1:
            raise RuntimeError(f"Graph invalid: {i} has {len(incoming)} incoming arcs")
        head, _, info = incoming[0]
        pos, lemma, feats = tags[i - 1]
        label = "root" if head == 0 else info["label"]
        conf = float(info["confidence"])
        words.append(Word(id=i, form=token, lemma=lemma, upos=pos, feats=feats,
                          head=head, deprel=label, misc=f"ProjConf={conf:.2f}"))
        arc_meta.append({"id": i, "head": head, "label": label, "confidence": round(conf, 3),
                         "origin": info["origin"], "aligned": bool(bw.get(i - 1))})
    coverage = len(bw) / n
    projected = sum(arc["origin"] == "projection" for arc in arc_meta) / n
    mean_conf = sum(a["confidence"] for a in arc_meta) / n
    quality = round(0.4 * coverage + 0.35 * projected + 0.25 * mean_conf, 4)
    result = Sentence(words, str(target_id or source.sent_id), " ".join(target_tokens),
                      [f"# source_sent_id = {source.sent_id}",
                       f"# projection_quality = {quality:.4f}",
                       f"# alignment_coverage = {coverage:.4f}"])
    errors = validate_tree(result)
    if errors:
        raise AssertionError(f"Projected invalid tree: {errors}")
    diagnostics = {"id": result.sent_id, "source_id": source.sent_id,
                   "quality": quality, "alignment_coverage": round(coverage, 4),
                   "projected_arc_fraction": round(projected, 4),
                   "mean_confidence": round(mean_conf, 4),
                   "arc_provenance": arc_meta, "n_words": n}
    return result, diagnostics


def run_projection(source_ud: str | Path, targets_file: str | Path,
                   alignments_file: str | Path, output_conllu: str | Path,
                   output_metadata: str | Path, morph_file: str | Path | None = None,
                   rules: dict | None = None, cache: str | Path | None = None,
                   prefix: str = "") -> dict:
    output_conllu = Path(output_conllu)
    output_metadata = Path(output_metadata)
    cache = Path(cache or str(output_conllu) + ".stage.jsonl")
    targets = read_targets(targets_file)
    links = read_alignments(alignments_file)
    morph = read_morph(morph_file)
    done = done_ids(cache)
    processed = 0
    for source in read_conllu(source_ud):
        sid = str(source.sent_id)
        if sid in done:
            continue
        if sid not in targets or sid not in links:
            raise ValueError(f"Missing target or alignment for {sid}; ensure parallel sentence IDs")
        target_id = prefix + sid
        result, diag = project_sentence(source, targets[sid]["tokens"], links[sid],
                                        morph.get(sid), rules, target_id=target_id)
        append_jsonl(cache, {"id": sid, "conllu": result.to_conllu(), "metrics": diag})
        processed += 1
    # Materialize completed files atomically. The stage JSONL survives interruption.
    write_lines(output_conllu, (record["conllu"] for record in jsonl(cache)))
    write_lines(output_metadata, (__import__("json").dumps(row["metrics"], ensure_ascii=False) + "\n"
                                  for row in jsonl(cache)))
    return {"processed_this_call": processed, "total": len(done_ids(cache)),
            "conllu": str(output_conllu), "metadata": str(output_metadata)}
