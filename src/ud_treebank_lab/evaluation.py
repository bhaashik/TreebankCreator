"""Strict gold-tokenized UD metrics and reproducible error analyses.

For official word/segmentation F1, MLAS and BLEX use UD conll18 evaluator separately.
We deliberately fail on differing tokenization rather than claiming meaningful LAS.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .conllu import Sentence, read_conllu, validate_tree
from .utils import write_json


def evaluate(gold_file: str | Path, predicted_file: str | Path,
             output_json: str | Path | None = None) -> dict:
    gold = list(read_conllu(gold_file))
    pred = list(read_conllu(predicted_file))
    if len(gold) != len(pred):
        raise ValueError(f"Sentence-count mismatch: gold={len(gold)}, predicted={len(pred)}")
    n = len(gold)
    counts = Counter()
    confusion_pos = Counter()
    confusion_deprel = Counter()
    rel_stats = defaultdict(Counter)
    buckets = defaultdict(Counter)
    errors = []
    for sidx, (g, p) in enumerate(zip(gold, pred)):
        if len(g.words) != len(p.words) or any(gw.form != pw.form for gw, pw in zip(g.words, p.words)):
            raise ValueError(f"Tokenization differs at sentence #{sidx+1} ({g.sent_id}). "
                             "Use the official CoNLL 2018 evaluator for tokenization differences.")
        if validate_tree(p):
            raise ValueError(f"Predicted tree invalid at sentence #{sidx+1}: {validate_tree(p)}")
        exact_las = True
        for gw, pw in zip(g.words, p.words):
            if gw.upos == "PUNCT":
                # UAS/LAS standard typically ignore punctuation (UPOS-based).
                is_parse_word = False
            else:
                is_parse_word = True
            upos_ok = gw.upos == pw.upos
            feats_ok = gw.feats == pw.feats
            lemma_ok = gw.lemma == pw.lemma
            head_ok = gw.head == pw.head
            rel_ok = gw.deprel.split(":")[0] == pw.deprel.split(":")[0]
            counts["words"] += 1
            counts["pos"] += upos_ok
            counts["feats"] += feats_ok
            counts["lemma"] += lemma_ok
            confusion_pos[(gw.upos, pw.upos)] += 1
            if is_parse_word:
                counts["parse_words"] += 1
                counts["uas"] += head_ok
                counts["las"] += head_ok and rel_ok
                counts["root_gold"] += gw.head == 0
                counts["root_correct"] += gw.head == 0 and pw.head == 0
                confusion_deprel[(gw.deprel, pw.deprel)] += 1
                rel_stats[gw.deprel]["gold"] += 1
                rel_stats[gw.deprel]["correct"] += head_ok and rel_ok
                dist = abs(gw.id - gw.head) if gw.head else 0
                bucket = "0(root)" if dist == 0 else "1" if dist == 1 else "2-3" if dist <= 3 else "4-7" if dist <= 7 else "8+"
                buckets[bucket]["total"] += 1
                buckets[bucket]["uas"] += head_ok
                buckets[bucket]["las"] += head_ok and rel_ok
                if not (head_ok and rel_ok):
                    exact_las = False
            if not (upos_ok and head_ok and rel_ok):
                if len(errors) < 1000:
                    errors.append({"id": g.sent_id, "token_id": gw.id, "form": gw.form,
                                   "gold_upos": gw.upos, "pred_upos": pw.upos,
                                   "gold_head": gw.head, "pred_head": pw.head,
                                   "gold_deprel": gw.deprel, "pred_deprel": pw.deprel})
        counts["exact_sentence_LAS"] += exact_las
    def ratio(a, b):
        return round(float(a) / b, 5) if b else None
    results = {
        "n_sentences": n, "n_words": counts["words"], "n_parse_words": counts["parse_words"],
        "scores": {"UPOS_accuracy": ratio(counts["pos"], counts["words"]),
                   "UFeats_exact_accuracy": ratio(counts["feats"], counts["words"]),
                   "Lemma_exact_accuracy": ratio(counts["lemma"], counts["words"]),
                   "UAS": ratio(counts["uas"], counts["parse_words"]),
                   "LAS": ratio(counts["las"], counts["parse_words"]),
                   "Root_recall": ratio(counts["root_correct"], counts["root_gold"]),
                   "Whole_sentence_LAS": ratio(counts["exact_sentence_LAS"], n)},
        "per_deprel_recall": {rel: {"count": c["gold"], "recall": ratio(c["correct"], c["gold"])}
                              for rel, c in sorted(rel_stats.items())},
        "distance_buckets": {b: {"words": c["total"], "UAS": ratio(c["uas"], c["total"]),
                                  "LAS": ratio(c["las"], c["total"])}
                             for b, c in sorted(buckets.items())},
        "upos_confusion": [{"gold": a, "pred": b, "count": v} for (a,b),v in
                           confusion_pos.most_common(70)],
        "deprel_confusion": [{"gold": a, "pred": b, "count": v} for (a,b),v in
                             confusion_deprel.most_common(70)],
        "errors": errors,
        "warnings": ["Evaluation assumes identical gold tokenization; punctuation excluded from UAS/LAS.",
                     "UPOS/FEATS/Lemma accuracy includes punctuation and '_' values.",
                     "For official CoNLL18 Words, MLAS, BLEX and LAS, use the UD evaluator."]}
    if output_json:
        write_json(output_json, results)
    return results
