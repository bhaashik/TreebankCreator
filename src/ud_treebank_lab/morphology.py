"""Apertium finite-state analysis and optional external morphology mappings.

Conservative ambiguous analysis: do not pretend that selecting the first morphological
reading is gold. Only deterministic/known features are exported by default.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .alignment import read_targets
from .utils import append_jsonl, done_ids, jsonl

TAG_UPOS = {
    "n": "NOUN", "np": "PROPN", "vblex": "VERB", "vbser": "AUX", "vbmod": "AUX",
    "adj": "ADJ", "adv": "ADV", "pr": "ADP", "post": "ADP", "det": "DET", "prn": "PRON",
    "num": "NUM", "cnjcoo": "CCONJ", "cnjsub": "SCONJ", "ij": "INTJ",
}
TAG_FEATS = {"sg": ("Number", "Sing"), "pl": ("Number", "Plur"),
             "m": ("Gender", "Masc"), "f": ("Gender", "Fem"),
             "p1": ("Person", "1"), "p2": ("Person", "2"), "p3": ("Person", "3"),
             "past": ("Tense", "Past"), "pres": ("Tense", "Pres"),
             "fut": ("Tense", "Fut")}


def parse_apertium(text: str) -> dict:
    """Parse 1 lexical analysis, use features only if unambiguous across readings."""
    units = re.findall(r"\^([^$]+)\$", text)
    if not units:
        return {}
    value = units[0]
    readings = value.split("/")[1:]
    analyses = []
    for reading in readings:
        if not reading or reading.startswith("*"):
            continue
        lemma = reading.split("<", 1)[0]
        tags = re.findall(r"<([^<>]+)>", reading)
        upos = next((TAG_UPOS[t] for t in tags if t in TAG_UPOS), None)
        feats = dict(TAG_FEATS[t] for t in tags if t in TAG_FEATS)
        analyses.append({"lemma": lemma, "upos": upos, "feats": feats})
    if not analyses:
        return {}
    answer = {}
    for key in ("lemma", "upos"):
        values = {a[key] for a in analyses}
        if len(values) == 1 and next(iter(values)):
            answer[key] = next(iter(values))
    common = set.intersection(*(set(a["feats"].items()) for a in analyses))
    if common:
        answer["feats"] = dict(sorted(common))
    return answer


def run_apertium(target_jsonl: str | Path, output: str | Path,
                 analyzer_bin: str, binary: str = "lt-proc", batch_size: int = 1000) -> dict:
    if not Path(analyzer_bin).exists():
        raise FileNotFoundError(analyzer_bin)
    done = done_ids(output)
    targets = read_targets(target_jsonl)
    pending = [(sid, row) for sid, row in targets.items() if sid not in done]
    # Analyze unique surface forms in batches (not one subprocess per token).
    unique = list(dict.fromkeys(w for _, row in pending for w in row["tokens"]))
    cache = {}
    for start in range(0, len(unique), batch_size):
        chunk = unique[start:start + batch_size]
        raw = "\n".join(chunk) + "\n"
        result = subprocess.run([binary, "-a", analyzer_bin], input=raw,
                                text=True, capture_output=True, check=True)
        lines = result.stdout.splitlines()
        if len(lines) != len(chunk):
            raise RuntimeError("Unexpected lt-proc output: expected one analysis line per input token. "
                               "Check analyzer/transducer configuration and token escaping.")
        for word, analysis in zip(chunk, lines):
            cache[word] = parse_apertium(analysis)
    for sid, row in pending:
        append_jsonl(output, {"id": sid, "analyses": [cache[w] for w in row["tokens"]]})
    return {"processed_this_call": len(pending), "unique_forms": len(unique), "path": str(output)}


def read_morph(path: str | Path | None) -> dict[str, list[dict]]:
    if not path:
        return {}
    return {str(row["id"]): row["analyses"] for row in jsonl(path)}
