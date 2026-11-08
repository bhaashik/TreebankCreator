"""Dataset-level, uncertainty-preserving Hindi-to-target correspondence induction.

CPU-only. Inputs are imperfect; this produces *hypotheses*, not gold word
alignments, morphological analyses, or UD annotations. All outputs are
provenance-tracked and regenerated atomically on input change. Supplied
alignments/Apertium/UniMorph inputs are never modified.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import json
import math
from pathlib import Path
import unicodedata

from .alignment import read_targets, run_alignment
from .conllu import Sentence, read_conllu
from .corpus_config import read_corpus_config
from .corpus_morph import (generate_apertium_lattice, read_apertium_candidates,
                           read_unimorph, merge_candidates)
from .utils import fingerprint, jsonl, write_json, write_lines

FUNCTION_POS = {"ADP", "AUX", "PART", "DET", "SCONJ", "CCONJ"}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "PRON"}


def _norm(word: str) -> str:
    return unicodedata.normalize("NFC", word).casefold().strip()


@lru_cache(maxsize=250_000)
def _similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.
    if a == b:
        return 1.
    # Orthographic hints work within scripts; for different scripts rely on
    # supplied alignments/morphological resources rather than pretending
    # Unicode edit similarity is cross-lingual semantic correspondence.
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _load_initial(path: str | None) -> dict[str, dict]:
    if path is None:
        return {}
    result = {}
    for record in jsonl(path):
        sid = str(record["id"])
        if sid in result:
            raise ValueError(f"Duplicate initial alignment sent_id {sid}")
        result[sid] = record
    return result


def _initial_matrix(record: dict | None, source_count: int, target_count: int) -> list[list[float]]:
    matrix = [[0.0] * source_count for _ in range(target_count)]
    if record is None:
        return matrix
    if "matrix" in record:
        rows = record["matrix"]
        if len(rows) != target_count or any(len(row) != source_count for row in rows):
            raise ValueError("Initial alignment matrix must be target x source")
        for j, row in enumerate(rows):
            for i, value in enumerate(row):
                v = float(value)
                if not math.isfinite(v) or v < 0 or v > 1:
                    raise ValueError("Initial alignment scores must be finite in [0,1]")
                matrix[j][i] = v
    else:
        for entry in record.get("links", record.get("pairs", [])):
            if isinstance(entry, dict):
                i = entry.get("src", entry.get("source"))
                j = entry.get("tgt", entry.get("target"))
                v = entry.get("score", entry.get("confidence", .55))
            else:
                if not isinstance(entry, (list, tuple)) or len(entry) not in (2, 3):
                    raise ValueError("Alignment link must be [src,tgt,score?]")
                i, j = entry[:2]
                v = entry[2] if len(entry) == 3 else .55
            if type(i) is not int or type(j) is not int or not (0 <= i < source_count
                                                                and 0 <= j < target_count):
                raise ValueError(f"Invalid alignment index: {i}, {j}")
            v = float(v)
            if not math.isfinite(v) or not (0 <= v <= 1):
                raise ValueError("Alignment strength must be finite in [0,1]")
            matrix[j][i] = max(matrix[j][i], v)
    return matrix


@dataclass
class Evidence:
    source: Sentence
    sid: str
    target: list[str]
    initial: list[list[float]]
    morphology: list[list[dict]]


def _load_pairs(source_file: str, targets_file: str, align_path: str | None,
                apertium_path: str | None, unimorph_path: str | None,
                max_words: int) -> tuple[list[Evidence], dict[str, list[dict]]]:
    targets = read_targets(targets_file)
    aligned = _load_initial(align_path)
    apertium = read_apertium_candidates(apertium_path, targets)
    unimorph = read_unimorph(unimorph_path)
    result, seen = [], set()
    for source in read_conllu(source_file):
        sid = str(source.sent_id)
        if sid in seen:
            raise ValueError(f"Duplicate source sent_id {sid}")
        seen.add(sid)
        if sid not in targets:
            continue
        words = targets[sid]["tokens"]
        if not words or len(words) > max_words or len(source.words) > max_words:
            continue
        matrix = _initial_matrix(aligned.get(sid), len(source.words), len(words))
        lattice = merge_candidates(words, apertium.get(sid), unimorph)
        result.append(Evidence(source, sid, words, matrix, lattice))
    if not result:
        raise ValueError("No paired sentences within max_words")
    if set(targets) - seen:
        raise ValueError(f"Targets absent from Hindi UD: {sorted(set(targets) - seen)[:5]}")
    if set(aligned) - seen:
        raise ValueError(f"Initial alignment IDs absent from Hindi UD: {sorted(set(aligned)-seen)[:5]}")
    return result, unimorph


def _best_pos(analyses: list[dict]) -> set[str]:
    return {a["upos"] for a in analyses if a.get("upos")}


def _seed_lexicon(data: list[Evidence], anchor_threshold: float):
    """Use strong orthography/resource anchors; weak raw links cannot dominate.

    String similarities are language-dependent and explicitly NOT calibrated.
    Per-example counts later receive leave-one-out adjustments.
    """
    lex = Counter()
    target_pos = defaultdict(Counter)
    for row in data:
        for j, target in enumerate(row.target):
            options = row.morphology[j]
            for o in options:
                if o.get("upos"):
                    target_pos[_norm(target)][o["upos"]] += .7
            for i, src in enumerate(row.source.words):
                sim = _similarity(src.form, target)
                morph_match = any(_norm(o.get("lemma") or "") in
                                  {_norm(src.form), _norm(src.lemma)} for o in options)
                if sim >= anchor_threshold and (src.upos in CONTENT_POS or sim == 1):
                    w = 1.0 + (sim == 1) * .7
                elif morph_match and src.upos in CONTENT_POS:
                    w = .8
                else:
                    # Keep weak original observations as hints, not ground truth.
                    w = .055 * row.initial[j][i]
                if w:
                    lex[(_norm(src.lemma if src.lemma != "_" else src.form),
                         _norm(target))] += w
                    if src.upos != "_" and w > .5:
                        target_pos[_norm(target)][src.upos] += w
    return lex, target_pos


def _sentence_cooccurrence(data: list[Evidence]) -> tuple[Counter, Counter, Counter]:
    """Count sentence-level cooccurrence, regardless of aligner correctness.

    Especially valuable for recurrent proper names in different scripts. This
    is not a translation dictionary: association is used only as a feature.
    Per-sentence deduplication prevents repeated function words dominating.
    """
    pairs, source_freq, target_freq = Counter(), Counter(), Counter()
    for record in data:
        src = {_norm(w.lemma if w.lemma != "_" else w.form) for w in record.source.words}
        tgt = {_norm(w) for w in record.target}
        source_freq.update(src)
        target_freq.update(tgt)
        pairs.update((s, t) for s in src for t in tgt)
    return pairs, source_freq, target_freq


def _cooccurrence_score(s: str, t: str, counts: tuple[Counter, Counter, Counter]) -> float:
    both, src_freq, tgt_freq = counts
    # Leave-one-sentence-out: a unique accidental cooccurrence has no force.
    joint = max(0, both[(s, t)] - 1)
    if not joint:
        return 0.
    a = joint / (max(0, src_freq[s] - 1) + 2.)
    b = joint / (max(0, tgt_freq[t] - 1) + 2.)
    return math.sqrt(a * b) * min(1., joint / 3.)


def _lex_stats(lex: Counter):
    src_total, tgt_total = Counter(), Counter()
    for (src, tgt), w in lex.items():
        src_total[src] += w
        tgt_total[tgt] += w
    return src_total, tgt_total


def _pair_score(row: Evidence, j: int, i: int, lex: Counter,
                totals: tuple[Counter, Counter], coocc: tuple[Counter, Counter, Counter],
                target_pos: dict, config: dict,
                *, loo: bool = True) -> tuple[float, dict]:
    sw, tw = row.source.words[i], row.target[j]
    sim = _similarity(sw.form, tw)
    raw = row.initial[j][i]
    options = row.morphology[j]
    source_key = _norm(sw.lemma if sw.lemma != "_" else sw.form)
    target_key = _norm(tw)
    s_total, t_total = totals
    observed = lex.get((source_key, target_key), 0.)
    # A conservative leave-one-out proxy: this sentence's approximate seed
    # contribution cannot count as confirmation of its own alignment.
    if loo:
        own = 0.
        if sim >= config["anchor_similarity"] and (sw.upos in CONTENT_POS or sim == 1):
            own = 1.0 + (sim == 1) * .7
        elif any(_norm(o.get("lemma") or "") in {_norm(sw.form), source_key}
                 for o in options) and sw.upos in CONTENT_POS:
            own = .8
        else:
            own = .055 * raw
        observed = max(0., observed - own)
        src_denom = max(0., s_total[source_key] - own)
        tgt_denom = max(0., t_total[target_key] - own)
    else:
        src_denom, tgt_denom = s_total[source_key], t_total[target_key]
    lexical = (observed / (src_denom + .3) * observed / (tgt_denom + .3)) ** .5
    # Direct orthographic agreement, corpus lexical consistency, and optional
    # independent morph analyses contribute separately.
    association = _cooccurrence_score(source_key, target_key, coocc)
    score = (-1.55 + 3.9 * (sim ** 2.2) + 1.65 * lexical
             + 3.3 * association + .55 * raw)
    if sim == 1:
        score += .65
    pos = _best_pos(options)
    inferred = target_pos.get(target_key, Counter())
    if sum(inferred.values()) >= 1.5:
        rank = inferred.most_common(1)[0]
        if rank[1] / sum(inferred.values()) >= .72:
            # Corpus anchors can rank genuinely ambiguous analyses. This is
            # probabilistic disambiguation evidence, not gold morph tagging.
            if not pos or rank[0] in pos:
                pos = {rank[0]}
    if pos:
        if sw.upos in pos:
            score += .43
        elif sw.upos in FUNCTION_POS and pos.isdisjoint(FUNCTION_POS):
            score -= 2.45  # ADP -> PROPN should be unlikely, not impossible
        elif sw.upos == "PROPN" and "PROPN" not in pos:
            score -= .8
        else:
            score -= .35
    if sw.upos in FUNCTION_POS and sim < .4:
        score -= .42
    if any(_norm(opt.get("lemma") or "") in (source_key, _norm(sw.form))
           for opt in options):
        score += .65
    # Position is only a very weak tie-breaker; local reordering is legitimate.
    if len(row.target) > 1 and len(row.source.words) > 1:
        score += .18 * (1 - abs(j / (len(row.target)-1) - i / (len(row.source.words)-1)))
    return score, {"char_similarity": round(sim, 4), "original_link": round(raw, 4),
                   "corpus_lexical": round(lexical, 4),
                   "sentence_cooccurrence": round(association, 4),
                   "candidate_upos": sorted(pos)}


def _refine_once(data: list[Evidence], lex: Counter,
                 coocc: tuple[Counter, Counter, Counter], target_pos: dict, cfg: dict):
    totals = _lex_stats(lex)
    outputs, improved_lex = [], Counter()
    for row in data:
        n, m = len(row.target), len(row.source.words)
        matrix = [[0.] * m for _ in range(n)]
        chosen, reasons, accepted_pairs = [], [], []
        for j in range(n):
            scores = [_pair_score(row, j, i, lex, totals, coocc, target_pos, cfg) for i in range(m)]
            ranked = sorted(range(m), key=lambda i: scores[i][0], reverse=True)
            best = scores[ranked[0]][0]
            # NULL is always available; weak or disputed alternatives must not
            # get fabricated hard links. Soft weights retain top alternatives.
            if best < -.22:
                # If all orthographic/resource evidence is weak (especially
                # cross-script), retain the original alignment as *low-weight*
                # supervision, not a confident hard link. Without this,
                # a different-script language would lose all training arcs.
                original_sum = sum(row.initial[j])
                raw_links = [i for i, w in enumerate(row.initial[j]) if w > 0]
                # Preserve supplied 1:n and n:1 links unless independent
                # evidence contradicts their POS type. A retained link is
                # explicitly marked unverified and gets lower soft weight.
                # With repeated parallel names, corpus-wide association may
                # favor a source token even when script/orthography differ.
                # Do not require the absolute score of that token to be high
                # before allowing a conservative correction.
                raw_best = max(raw_links, key=lambda i: row.initial[j][i]) if raw_links else None
                corpus_recovery = (raw_best is not None and ranked[0] != raw_best
                    and scores[ranked[0]][1]["sentence_cooccurrence"] >= .24
                    and scores[ranked[0]][0] >= scores[raw_best][0] + .3
                    and row.source.words[ranked[0]].upos in CONTENT_POS)
                contradicts = any(
                    row.source.words[i].upos in FUNCTION_POS
                    and bool(scores[i][1]["candidate_upos"])
                    and set(scores[i][1]["candidate_upos"]).isdisjoint(FUNCTION_POS)
                    for i in raw_links)
                if corpus_recovery:
                    chosen.append(ranked[0])
                    accepted_pairs.append((ranked[0], j))
                    matrix[j][ranked[0]] = .53  # uncalibrated, deliberately modest
                    status = "corpus_consistency_recovered"
                elif raw_links and not contradicts:
                    for i in raw_links:
                        matrix[j][i] = round(.44 * row.initial[j][i]
                                              / max(original_sum, 1e-8), 6)
                        accepted_pairs.append((i, j))
                    chosen.append(max(raw_links, key=lambda i: row.initial[j][i]))
                    status = "retained_supplied_unverified"
                else:
                    if original_sum:
                        for i, w in enumerate(row.initial[j]):
                            matrix[j][i] = round(.18 * w / original_sum, 6)
                    chosen.append(None)
                    status = "contradicted_original_soft_only" if contradicts else "null_or_unresolved"
                reasons.append({"target_index": j, "status": status,
                                "best_score": round(best, 4),
                                "alternative_source": ranked[0],
                                "original_soft_weight": round(sum(matrix[j]), 4)})
                continue
            top = ranked[:cfg["max_alternatives"]]
            close = [i for i in top if scores[i][0] >= best - .9]
            denominator = sum(math.exp((scores[i][0] - best) / cfg["temperature"])
                              for i in close)
            strength = min(.89, .52 + .095 * max(0., best))
            if len(close) > 1:
                strength *= .83
            for i in close:
                matrix[j][i] = round(strength * math.exp(
                    (scores[i][0] - best) / cfg["temperature"]) / denominator, 6)
            top_idx = ranked[0]
            # Preserve additional initial alignments as weak soft hypotheses,
            # useful for Hindi multiword -> target single-word phenomena.
            for other, raw_value in enumerate(row.initial[j]):
                if other == top_idx or raw_value <= 0 or other in close:
                    continue
                candidate_pos = set(scores[other][1]["candidate_upos"])
                if (row.source.words[other].upos in FUNCTION_POS and candidate_pos
                    and candidate_pos.isdisjoint(FUNCTION_POS)):
                    continue
                matrix[j][other] = round(min(.10, .10 * raw_value / .55), 6)
            chosen.append(top_idx if matrix[j][top_idx] >= cfg["min_link_confidence"] else None)
            status = "scored_proposal"
            if chosen[-1] is not None:
                accepted_pairs.append((top_idx, j))
            else:
                # Multiple close candidates can dilute max soft weight below
                # the hard threshold. Retain a plausible supplied *content*
                # anchor instead of losing it merely through ambiguity.
                original_content = [i for i, v in enumerate(row.initial[j])
                                    if v > 0 and row.source.words[i].upos in CONTENT_POS
                                    and scores[i][0] >= best - 1.0]
                if original_content:
                    best_initial = max(original_content, key=lambda i: row.initial[j][i])
                    accepted_pairs.append((best_initial, j))
                    chosen[-1] = best_initial
                    status = "ambiguous_original_retained"
            reasons.append({"target_index": j, "status": status,
                            "best_score": round(best, 4),
                            "selected_source": chosen[-1], "alternatives": [
                                {"source": i, "score": round(matrix[j][i], 4),
                                 **scores[i][1]} for i in close]})
            # Prevent self-reinforcing counts: take only high-confidence links
            # backed by independent orthographic or morphology evidence.
            sim = scores[top_idx][1]["char_similarity"]
            if chosen[-1] is not None and sim >= cfg["anchor_similarity"]:
                sw = row.source.words[top_idx]
                source_key = _norm(sw.lemma if sw.lemma != "_" else sw.form)
                improved_lex[(source_key, _norm(row.target[j]))] += min(.7, matrix[j][top_idx])
        old_pairs = {(i, j) for j in range(n) for i in range(m) if row.initial[j][i] >= .45}
        new_pairs = sorted(set(accepted_pairs))
        # A source word can remain unaligned or contribute to a phrase/morpheme.
        # Null/unresolved refers to no accepted hard word link; low-weight
        # alternatives remain in matrix and are not silently discarded.
        accepted = {i for i, j in new_pairs}
        source_null = [i for i in range(m) if i not in accepted]
        outputs.append({"id": row.sid, "matrix": matrix,
                        "pairs": [list(p) for p in new_pairs],
                        "null_source": source_null,
                        "null_target": [j for j, i in enumerate(chosen) if i is None],
                        "old_pairs": sorted(map(list, old_pairs)), "evidence": reasons,
                        "changed": old_pairs != set(new_pairs)})
    return outputs, improved_lex


def _affix_families(data: list[Evidence], maxlen: int) -> dict[str, tuple[str, str]]:
    """Very conservative *unlabeled* stem/suffix families, not proven morphology."""
    forms = {w for pair in data for w in pair.target}
    by_prefix = defaultdict(list)
    for word in forms:
        if len(word) >= 5:
            by_prefix[word[:4]].append(word)
    result = {}
    for words in by_prefix.values():
        if len(words) < 2:
            continue
        for word in words:
            possibilities = []
            for other in words:
                if word == other:
                    continue
                k = 0
                for x, y in zip(word, other):
                    if x != y:
                        break
                    k += 1
                if k >= 4 and k < len(word) and (len(word)-k <= maxlen and
                                                   len(other)-k <= maxlen):
                    possibilities.append(k)
            if possibilities:
                stemlen = max(possibilities)
                result[word] = (word[:stemlen], word[stemlen:])
    return result


def _function_patterns(data: list[Evidence], output: list[dict], cfg: dict) -> dict:
    """Describe word/NULL correspondences and *contrast-supported* affix hints.

    An affix association is reported only when the same host lemma/family was
    observed both with and without the source function word. Even then it is
    observational evidence, not a learned grammatical rule.
    """
    families = _affix_families(data, cfg["affix_max_length"])
    stats = defaultdict(lambda: {"word": Counter(), "null": 0, "total": 0})
    # For each source governor occurrence, keep grammatical function children.
    host_occurrences = defaultdict(lambda: defaultdict(list))
    for row, refined in zip(data, output):
        target_for_src = defaultdict(list)
        for src_idx, tgt_idx in refined["pairs"]:
            target_for_src[src_idx].append(tgt_idx)
        for sw in row.source.words:
            if sw.upos not in FUNCTION_POS:
                continue
            key = f"{_norm(sw.lemma if sw.lemma != '_' else sw.form)}|{sw.upos}|{sw.deprel}"
            stats[key]["total"] += 1
            mappings = target_for_src.get(sw.id - 1, [])
            if mappings:
                for j in mappings:
                    stats[key]["word"][_norm(row.target[j])] += 1
            else:
                stats[key]["null"] += 1
        for head in row.source.words:
            if head.upos not in CONTENT_POS:
                continue
            tgt_for_head = target_for_src.get(head.id - 1, [])
            if not tgt_for_head:
                continue
            j = tgt_for_head[0]
            target_form = row.target[j]
            known_lemma = next((c["lemma"] for c in row.morphology[j] if c.get("lemma")), None)
            stem, suffix = families.get(target_form, (target_form, ""))
            if (known_lemma and target_form.startswith(known_lemma)
                    and 0 < len(target_form)-len(known_lemma) <= cfg["affix_max_length"]):
                stem, suffix = known_lemma, target_form[len(known_lemma):]
            family = _norm(known_lemma or stem)
            attached = [c for c in row.source.words if c.head == head.id and c.upos in FUNCTION_POS]
            keys_present = {f"{_norm(c.lemma if c.lemma != '_' else c.form)}|{c.upos}|{c.deprel}"
                            for c in attached}
            host_occurrences[_norm(head.lemma if head.lemma != "_" else head.form)][family].append(
                (keys_present, suffix))
    affix = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    compared_totals = defaultdict(lambda: [0, 0])
    keys = set(stats)
    for by_family in host_occurrences.values():
        for forms in by_family.values():
            if len(forms) < 2:
                continue
            observed_keys = set().union(*(present for present, _ in forms))
            for key in observed_keys & keys:
                has = sum(key in present for present, _ in forms)
                lacks = len(forms) - has
                if not has or not lacks:
                    continue
                for present, suffix in forms:
                    flag = int(key in present)
                    compared_totals[key][flag] += 1
                    if suffix:
                        affix[key][suffix][flag] += 1
    output_patterns = {}
    for key, val in sorted(stats.items()):
        hints = []
        absent_total, present_total = compared_totals[key]
        for suffix, (negative, positive) in affix[key].items():
            if positive < cfg["min_affix_support"] or not absent_total or not present_total:
                continue
            # Contrast is WITHIN attested host-lemma/family groups, not merely
            # function-word cooccurrence in unrelated sentences.
            p_positive = (positive + .5) / (present_total + 1.)
            p_negative = (negative + .5) / (absent_total + 1.)
            if p_positive <= 1.8 * p_negative:
                continue
            hints.append({"suffix": suffix, "positive_hosts": positive,
                          "negative_hosts": negative,
                          "positive_contexts": present_total,
                          "negative_contexts": absent_total,
                          "association_ratio": round(p_positive / p_negative, 3),
                          "association_only": True})
        output_patterns[key] = {"instances": val["total"], "null_or_bound": val["null"],
                                "target_words": dict(val["word"].most_common(20)),
                                "candidate_affixes": sorted(hints, key=lambda x: -x["positive_hosts"]),
                                "warning": "NULL cannot yet be distinguished from bound or constructional realization"}
    return output_patterns


def _infer_bound_candidates(data: list[Evidence], refined: list[dict], patterns: dict,
                            cfg: dict) -> list[dict]:
    """Annotate *hypothetical* affix realization of unmatched source function words.

    This never inserts a fake target word or a hard syntactic arc. Evidence is
    only reported if a contrast-supported candidate suffix is seen on the
    target token aligned to the function word's Hindi governor.
    """
    families = _affix_families(data, cfg["affix_max_length"])
    results = []
    for row, record in zip(data, refined):
        target_for_source = defaultdict(list)
        for si, ti in record["pairs"]:
            target_for_source[si].append(ti)
        proposals = []
        for sw in row.source.words:
            if sw.upos not in FUNCTION_POS or (sw.id - 1) not in record["null_source"]:
                continue
            if sw.head < 1:
                continue
            key = f"{_norm(sw.lemma if sw.lemma != '_' else sw.form)}|{sw.upos}|{sw.deprel}"
            candidates = patterns.get(key, {}).get("candidate_affixes", [])
            for ti in target_for_source.get(sw.head - 1, []):
                word = row.target[ti]
                known = [o.get("lemma") for o in row.morphology[ti] if o.get("lemma")]
                candidates_suffix = [word[len(lemma):] for lemma in known if
                                     word.startswith(lemma) and word != lemma]
                if word in families:
                    candidates_suffix.append(families[word][1])
                for candidate in candidates:
                    suffix = candidate["suffix"]
                    if suffix in candidates_suffix and suffix:
                        proposals.append({"source_index": sw.id - 1, "source_form": sw.form,
                                          "target_host_index": ti, "target_host_form": word,
                                          "candidate_suffix": suffix,
                                          "association_ratio": candidate["association_ratio"],
                                          "status": "unverified_bound_morpheme_hypothesis"})
        results.append({"id": row.sid, "candidates": proposals})
    return results


def _morph_hypotheses(data: list[Evidence], target_pos: dict, cfg: dict) -> list[dict]:
    result = []
    families = _affix_families(data, cfg["affix_max_length"])
    for pair in data:
        words = []
        for form, options in zip(pair.target, pair.morphology):
            if options:
                inferred = target_pos.get(_norm(form), Counter())
                ranked = []
                for option in options:
                    prior = inferred.get(option.get("upos"), 0.)
                    # Agreement of independent sources (Apertium/UniMorph)
                    # is evidence for a candidate, not automatic correctness.
                    agreement = sum(other.get("lemma") == option.get("lemma") and
                                    other.get("upos") == option.get("upos")
                                    for other in options if other is not option)
                    ranked.append({**option, "corpus_support": round(prior, 4),
                                   "agreement_count": agreement,
                                   "ranking_score": round(math.log1p(prior) + .5*agreement, 4)})
                ranked.sort(key=lambda item: -item["ranking_score"])
                words.append({"form": form, "candidates": ranked,
                              "status": "resource_ambiguous" if len(options) > 1 else "resource_candidate"})
            else:
                ranks = target_pos.get(_norm(form), Counter())
                possible = []
                if ranks:
                    total = sum(ranks.values())
                    possible = [{"upos": pos, "lemma": None, "feats": {},
                                 "score": round(count / total, 4), "resource": "corpus_anchor"}
                                for pos, count in ranks.most_common(3)]
                if form in families:
                    stem, suffix = families[form]
                    possible.append({"upos": None, "lemma": stem, "feats": {},
                                     "candidate_suffix": suffix, "resource": "unsupervised_family",
                                     "warning": "Not an established morphological analysis"})
                words.append({"form": form, "candidates": possible,
                              "status": "inferred_uncertain" if possible else "unresolved"})
        result.append({"id": pair.sid, "words": words})
    return result


def _output_paths(directory: Path) -> dict[str, Path]:
    return {"alignments": directory / "alignments.refined.jsonl",
            "morphology": directory / "morph_candidates.jsonl",
            "corrections": directory / "corrections.jsonl",
            "bound_morphemes": directory / "bound_morpheme_hypotheses.jsonl",
            "lexicon": directory / "bilingual_lexicon.json",
            "functions": directory / "function_realizations.json",
            "report": directory / "report.json", "manifest": directory / "manifest.json"}


def run_corpus_refinement(filename: str | Path, *, force: bool = False) -> dict:
    cfg = read_corpus_config(filename)
    from .corpus_resources import prepare_corpus_resources
    # Resolve existing input, import raw formats, or explicitly generate a
    # missing resource. This also allows a separate `prepare-resources` run.
    prepared = prepare_corpus_resources(cfg, force=force)
    selected = prepared["selected_inputs"]
    dest = Path(cfg["output_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    resources = cfg["resources"]
    outputs = _output_paths(dest)
    ingredients = [Path(cfg["source_ud"]), Path(cfg["targets"]),
                   Path(prepared["manifest"])]
    ingredients += [Path(source) for source in selected.values() if source]
    signature = fingerprint({"algorithm": "corpus-refine-0.3.2", **cfg}, *ingredients)
    old_manifest = json.loads(outputs["manifest"].read_text()) if outputs["manifest"].exists() else None
    complete = all(p.is_file() for p in outputs.values())
    if complete and old_manifest and old_manifest.get("signature") == signature and not force:
        return {"status": "already-complete", "output_dir": str(dest),
                "alignments": str(outputs["alignments"]), "signature": signature,
                "resources": prepared["stages"]}
    data, unimorph = _load_pairs(cfg["source_ud"], cfg["targets"], selected["alignment"],
                                 selected["apertium"], selected["unimorph"],
                                 cfg["refinement"]["max_words"])
    lex, target_pos = _seed_lexicon(data, cfg["refinement"]["anchor_similarity"])
    coocc = _sentence_cooccurrence(data)
    for iteration in range(cfg["refinement"]["iterations"]):
        refined, added = _refine_once(data, lex, coocc, target_pos, cfg["refinement"])
        if iteration + 1 < cfg["refinement"]["iterations"]:
            lex.update(added)
    patterns = _function_patterns(data, refined, cfg["refinement"])
    bound = _infer_bound_candidates(data, refined, patterns, cfg["refinement"])
    hypotheses = _morph_hypotheses(data, target_pos, cfg["refinement"])
    changed = sum(record["changed"] for record in refined)
    total_null = sum(len(record["null_target"]) for record in refined)
    report = {"status": "complete", "sentence_pairs": len(data),
              "initial_alignment_mode": resources["alignment"]["mode"],
              "resource_stages": prepared["stages"],
              "apertium_mode": resources["apertium"]["mode"],
              "unimorph_mode": resources["unimorph"]["mode"],
              "changed_sentences": changed, "null_or_unresolved_target_tokens": total_null,
              "sentence_level_cooccurrence_pairs": len(coocc[0]),
              "candidate_affix_warning": "Unsupervised suffixes are hypotheses, not verified inflection rules",
              "note": "Alignment confidence is an uncalibrated heuristic, not gold correctness",
              "paths": {k: str(v) for k, v in outputs.items() if k != "manifest"}}
    # Atomic replace; a rerun after interruption recomputes finished outputs
    # without corrupting supplied input resources.
    write_lines(outputs["alignments"], (
        json.dumps({k: v for k, v in record.items() if k not in ("old_pairs", "evidence", "changed")},
                   ensure_ascii=False) + "\n" for record in refined))
    write_lines(outputs["corrections"], (
        json.dumps({"id": record["id"], "changed": record["changed"],
                    "initial_pairs": record["old_pairs"], "new_pairs": record["pairs"],
                    "evidence": record["evidence"]}, ensure_ascii=False) + "\n" for record in refined))
    write_lines(outputs["morphology"], (json.dumps(item, ensure_ascii=False) + "\n"
                                         for item in hypotheses))
    write_lines(outputs["bound_morphemes"], (json.dumps(item, ensure_ascii=False) + "\n"
                                              for item in bound))
    totals = _lex_stats(lex)
    mappings = defaultdict(list)
    for (s, t), count in lex.items():
        if count > .04:
            mappings[s].append({"target_form": t, "evidence": round(count, 4),
                                "p_target_given_source": round(count/(totals[0][s]+.3), 5),
                                "p_source_given_target": round(count/(totals[1][t]+.3), 5)})
    write_json(outputs["lexicon"], {s: sorted(rows, key=lambda x: -x["evidence"])[:30]
                                    for s, rows in sorted(mappings.items())})
    write_json(outputs["functions"], patterns)
    write_json(outputs["report"], report)
    write_json(outputs["manifest"], {"signature": signature, "resource_modes": {
        name: spec["mode"] for name, spec in resources.items()},
        "selected_inputs": selected, "output_files": {k: str(p) for k, p in outputs.items()}})
    return {**report, "alignments": str(outputs["alignments"]), "signature": signature}
