"""Ambiguity-preserving Apertium and UniMorph resource readers.

CoNLL-U FEATS and UniMorph tags are not identical tagsets. Mappings here are
conservative hints; unknown tags remain in a 'raw_tags' field for audit.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import subprocess

from .alignment import read_targets
from .morphology import TAG_UPOS, TAG_FEATS
from .utils import jsonl, write_lines


UNIMORPH = {
    "SG": ("Number", "Sing"), "PL": ("Number", "Plur"),
    "MASC": ("Gender", "Masc"), "FEM": ("Gender", "Fem"),
    "PST": ("Tense", "Past"), "PRS": ("Tense", "Pres"),
    "FUT": ("Tense", "Fut"), "1": ("Person", "1"),
    "2": ("Person", "2"), "3": ("Person", "3"),
}
UM_POS = {"N": "NOUN", "V": "VERB", "ADJ": "ADJ", "ADV": "ADV", "PRO": "PRON",
          "PRON": "PRON", "PROPN": "PROPN", "NUM": "NUM"}


def parse_apertium_candidates(text: str) -> list[dict]:
    """Parse one lt-proc -a line; return *all* readings for its first lexical unit."""
    units = re.findall(r"\^([^$]+)\$", text)
    if not units:
        return []
    candidates = []
    # Apertium lexical units: ^surface/lemma<tag><tag>/lemma<tag>$
    for reading in units[0].split("/")[1:]:
        if not reading or reading.startswith("*"):
            continue
        lemma = reading.split("<", 1)[0]
        tags = re.findall(r"<([^<>]+)>", reading)
        features = dict(TAG_FEATS[x] for x in tags if x in TAG_FEATS)
        upos = next((TAG_UPOS[x] for x in tags if x in TAG_UPOS), None)
        record = {"lemma": lemma or None, "upos": upos, "feats": features,
                  "raw_tags": tags, "resource": "apertium"}
        if record not in candidates:
            candidates.append(record)
    return candidates


def generate_apertium_lattice(targets_path: str, output: str, analyzer_bin: str,
                              *, binary: str = "lt-proc", batch_size: int = 1000) -> dict:
    if not Path(analyzer_bin).is_file():
        raise FileNotFoundError(analyzer_bin)
    targets = read_targets(targets_path)
    unique = list(dict.fromkeys(word for t in targets.values() for word in t["tokens"]))
    by_word = {}
    for start in range(0, len(unique), batch_size):
        chunk = unique[start:start + batch_size]
        proc = subprocess.run([binary, "-a", analyzer_bin], input="\n".join(chunk) + "\n",
                              text=True, capture_output=True, check=True)
        lines = proc.stdout.splitlines()
        if len(lines) != len(chunk):
            raise ValueError("Apertium output count mismatched input forms. Check analyzer encoding")
        by_word.update((w, parse_apertium_candidates(line)) for w, line in zip(chunk, lines))
    write_lines(output, (
        __import__("json").dumps({"id": sid,
            "analyses": [by_word[w] for w in t["tokens"]]}, ensure_ascii=False) + "\n"
        for sid, t in targets.items()
    ))
    return {"sentences": len(targets), "unique_forms": len(unique), "path": output}


def read_apertium_candidates(path: str | None, targets: dict[str, dict]) -> dict[str, list[list[dict]]]:
    """Accept both legacy 0.2 collapsed per-token dictionaries and full lattices."""
    if not path:
        return {}
    result = {}
    for record in jsonl(path):
        sid = str(record["id"])
        if sid in result or sid not in targets:
            raise ValueError(f"Duplicate or unknown Apertium sentence ID: {sid}")
        analyses = record["analyses"]
        if len(analyses) != len(targets[sid]["tokens"]):
            raise ValueError(f"Apertium/token count mismatch: {sid}")
        normalized = []
        for token in analyses:
            options = [token] if isinstance(token, dict) else token
            if not isinstance(options, list) or not all(isinstance(a, dict) for a in options):
                raise ValueError(f"Invalid morphological alternatives for {sid}")
            normalized.append([{"lemma": opt.get("lemma"), "upos": opt.get("upos"),
                                "feats": opt.get("feats") or {},
                                "resource": opt.get("resource", "apertium")}
                               for opt in options if opt])
        result[sid] = normalized
    return result


def read_unimorph(path: str | None) -> dict[str, list[dict]]:
    """Parse UniMorph TSV: lemma TAB surface TAB tags, allowing repeated forms."""
    if not path:
        return {}
    result = defaultdict(list)
    for ln, text in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not text.strip() or text.lstrip().startswith("#"):
            continue
        fields = text.split("\t")
        if len(fields) < 3:
            raise ValueError(f"UniMorph {path}:{ln}: expected lemma, surface, features")
        lemma, form, tag_string = fields[:3]
        if not form or not lemma:
            continue
        tags = tag_string.split(";")
        feats = dict(UNIMORPH[t] for t in tags if t in UNIMORPH)
        upos = next((UM_POS[t] for t in tags if t in UM_POS), None)
        item = {"lemma": lemma, "upos": upos, "feats": feats,
                "raw_tags": tags, "resource": "unimorph"}
        if item not in result[form]:
            result[form].append(item)
    return dict(result)


def merge_candidates(tokens: list[str], apertium: list[list[dict]] | None,
                     unimorph: dict[str, list[dict]]) -> list[list[dict]]:
    result = []
    for i, form in enumerate(tokens):
        options = []
        for item in (apertium[i] if apertium else []) + unimorph.get(form, []):
            normalized = {"lemma": item.get("lemma"), "upos": item.get("upos"),
                          "feats": item.get("feats") or {},
                          "resource": item.get("resource", "unimorph")}
            if normalized not in options:
                options.append(normalized)
        result.append(options)
    return result
