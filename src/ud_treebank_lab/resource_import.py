"""Import *existing* alignment and Apertium output into project JSONL schemas.

Never invokes an aligner or morphological analyzer. Sentence and token identities
are checked against the parallel corpus; original input files remain untouched.
"""
from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
import re
import tempfile
from collections import OrderedDict
from typing import Iterable

import yaml

from .alignment import read_targets
from .conllu import read_conllu
from .morphology import TAG_FEATS, TAG_UPOS

_PAIR = re.compile(r"(\d+)-(\d+)\Z")
_A3_HEADER = re.compile(
    r"#\s*Sentence pair\s*\((\d+)\)\s+source length\s+(\d+)\s+"
    r"target length\s+(\d+)\s+alignment score\s*:\s*(\S+)", re.IGNORECASE
)
_A3_ALIGN = re.compile(r"([^\s]+)\s+\(\{\s*([\d\s]*)\s*\}\)")


def _open_text(path: str | Path):
    p = Path(path)
    return gzip.open(p, "rt", encoding="utf-8-sig") if p.suffix.lower() == ".gz" else p.open(
        "r", encoding="utf-8-sig"
    )


def _pairs_from_corpus(source_ud: str | Path, targets_file: str | Path,
                       sentence_order: str = "source_ud") -> list[dict]:
    if sentence_order not in {"source_ud", "targets"}:
        raise ValueError("sentence_order must be source_ud or targets")
    source = OrderedDict()
    for sent in read_conllu(source_ud):
        sid = str(sent.sent_id)
        if sid in source:
            raise ValueError(f"Duplicate source ID {sid}")
        source[sid] = [w.form for w in sent.words]
    targets = read_targets(targets_file)
    if set(source) != set(targets):
        missing = list(set(source) - set(targets))[:3]
        extra = list(set(targets) - set(source))[:3]
        raise ValueError(f"Source/target sentence IDs differ: source-only={missing}, target-only={extra}")
    order = source if sentence_order == "source_ud" else targets
    if not order:
        raise ValueError("No source/target sentence pairs")
    return [{"id": sid, "source": source[sid], "target": targets[sid]["tokens"]}
            for sid in order]


def _assert_output(input_paths: Iterable[str | Path], out: str | Path, overwrite: bool):
    dest = Path(out).expanduser().resolve()
    for p in input_paths:
        if p and dest == Path(p).expanduser().resolve():
            raise ValueError(f"Refusing to replace an input file: {dest}")
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {dest} (use --overwrite explicitly)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def _atomic_jsonl(destination: Path, records: Iterable[dict]) -> int:
    staging = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=".import-", delete=False) as f:
            staging = Path(f.name)
            n = 0
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                n += 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(staging, destination)
        return n
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def _pharaoh_links(text: str, src_len: int, tgt_len: int, sid: str,
                   probabilities: str | None = None) -> dict:
    entries = text.strip().split()
    probs = probabilities.strip().split() if probabilities is not None else None
    if probs is not None and len(probs) != len(entries):
        raise ValueError(f"Sentence {sid}: {len(entries)} links but {len(probs)} probabilities")
    scored = {}
    for n, pair in enumerate(entries):
        m = _PAIR.fullmatch(pair)
        if not m:
            raise ValueError(f"Sentence {sid}: invalid Pharaoh link {pair!r}; expected i-j")
        i, j = int(m[1]), int(m[2])
        if i >= src_len or j >= tgt_len:
            raise ValueError(f"Sentence {sid}: link {i}-{j} outside {src_len}x{tgt_len} tokens")
        if probs is None:
            scored[(i, j)] = None
        else:
            try:
                score = float(probs[n])
            except ValueError as exc:
                raise ValueError(f"Sentence {sid}: invalid alignment probability {probs[n]!r}") from exc
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Sentence {sid}: probability must be in [0,1]")
            scored[(i, j)] = max(score, scored.get((i, j), 0.0))
    pairs = sorted(scored)
    result = {"id": sid, "pairs": [list(p) for p in pairs], "backend": "awesome-align"}
    if probs is not None:
        result["links"] = [{"src": i, "tgt": j, "score": scored[i, j]} for i, j in pairs]
    return result


def _iter_a3_blocks(path: str | Path):
    header = None
    content = []
    with _open_text(path) as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.lstrip().startswith("#"):
                if not _A3_HEADER.fullmatch(line.strip()):
                    raise ValueError(f"Unexpected A3 header: {line[:100]}")
                if header is not None:
                    yield header, content
                header, content = line.strip(), []
            elif line.strip():
                if header is None:
                    raise ValueError("GIZA A3 data before the first '# Sentence pair' header")
                content.append(line)
    if header is None:
        raise ValueError("No GIZA A3 sentence blocks found")
    yield header, content


def _a3_links(header: str, content: list[str], sentence: dict,
              record_number: int, source_side: str, verify_text: bool) -> dict:
    match = _A3_HEADER.fullmatch(header)
    if match is None:
        raise ValueError(f"Invalid GIZA A3 header: {header}")
    pair_number = int(match[1])
    sid = sentence["id"]
    if pair_number != record_number:
        raise ValueError(f"GIZA A3 sentence pair index {pair_number} at record {record_number}; "
                         "input order must match corpus order")
    if len(content) < 2:
        raise ValueError(f"A3 sentence {sid}: expected plain and annotated word lines")
    plain = content[0].split()
    annotated_text = " ".join(content[1:])
    annotated = []
    position = 0
    for a in _A3_ALIGN.finditer(annotated_text):
        if annotated_text[position:a.start()].strip():
            raise ValueError(f"A3 sentence {sid}: unrecognized annotation text")
        annotated.append((a[1], [int(x) for x in a[2].split()]))
        position = a.end()
    if annotated_text[position:].strip() or not annotated or annotated[0][0].upper() != "NULL":
        raise ValueError(f"A3 sentence {sid}: invalid NULL/word alignment block")
    annotated_words = [w for w, _ in annotated[1:]]
    s, t = sentence["source"], sentence["target"]
    valid_annotated = annotated_words == s and plain == t
    valid_plain = annotated_words == t and plain == s
    if source_side == "auto":
        if valid_annotated == valid_plain:
            raise ValueError(f"A3 sentence {sid}: cannot determine Hindi source side; "
                             "choose --a3-source-side annotated|plain and verify tokenization")
        source_side = "annotated" if valid_annotated else "plain"
    if source_side == "annotated":
        if len(annotated_words) != len(s) or len(plain) != len(t):
            raise ValueError(f"A3 sentence {sid}: annotated/plain token counts do not match corpus")
        if verify_text and not valid_annotated:
            raise ValueError(f"A3 sentence {sid}: source/target token text mismatch "
                             "(source is annotated); fix tokenization or use --no-verify-forms")
    elif source_side == "plain":
        if len(plain) != len(s) or len(annotated_words) != len(t):
            raise ValueError(f"A3 sentence {sid}: plain/annotated token counts do not match corpus")
        if verify_text and not valid_plain:
            raise ValueError(f"A3 sentence {sid}: source/target token text mismatch "
                             "(source is plain); fix tokenization or use --no-verify-forms")
    else:
        raise ValueError("a3_source_side must be auto, annotated or plain")
    # GIZA positions are ONE-BASED and refer to the words on the plain line.
    pairs = set()
    for annotated_index, (_, plain_indices) in enumerate(annotated[1:]):
        for one_based in plain_indices:
            if one_based <= 0 or one_based > len(plain):
                raise ValueError(f"A3 sentence {sid}: invalid target position {one_based}")
            source_index, target_index = (
                (annotated_index, one_based - 1) if source_side == "annotated"
                else (one_based - 1, annotated_index)
            )
            pairs.add((source_index, target_index))
    result = {"id": sid, "pairs": [list(p) for p in sorted(pairs)], "backend": "giza-a3"}
    explicit_null = [number - 1 for number in annotated[0][1]]
    if any(index < 0 or index >= len(plain) for index in explicit_null):
        raise ValueError(f"A3 sentence {sid}: invalid NULL index")
    result["null_target" if source_side == "annotated" else "null_source"] = sorted(set(explicit_null))
    # A3 alignment score is a *sentence* score, not a per-word link confidence.
    try:
        val = float(match[4])
    except ValueError as exc:
        raise ValueError(f"A3 sentence {sid}: invalid alignment score") from exc
    if not math.isfinite(val):
        raise ValueError(f"A3 sentence {sid}: non-finite alignment score")
    result["a3_sentence_score"] = val
    return result


def import_alignments(*, source_ud: str, targets: str, input_path: str, output: str,
                      format: str = "auto", probabilities: str | None = None,
                      sentence_order: str = "source_ud", a3_source_side: str = "auto",
                      verify_forms: bool = True, overwrite: bool = False) -> dict:
    """Import AWESOME Pharaoh or GIZA A3 alignments, assigning real sentence IDs."""
    if format not in {"auto", "awesome", "a3"}:
        raise ValueError("format must be auto, awesome or a3")
    if a3_source_side not in {"auto", "annotated", "plain"}:
        raise ValueError("a3_source_side must be auto, annotated or plain")
    sentences = _pairs_from_corpus(source_ud, targets, sentence_order)
    dest = _assert_output([source_ud, targets, input_path, probabilities], output, overwrite)
    with _open_text(input_path) as f:
        first = next((line for line in f if line.strip()), "").lstrip()
    actual = ("a3" if first.startswith("# Sentence pair") else "awesome") if format == "auto" else format
    if actual == "a3" and probabilities:
        raise ValueError("--probabilities applies only to AWESOME-align Pharaoh output")

    def records():
        if actual == "awesome":
            with _open_text(input_path) as f:
                lines = f.readlines()
            if len(lines) != len(sentences):
                raise ValueError(f"AWESOME-align has {len(lines)} lines; expected {len(sentences)} "
                                 "(including empty alignment lines)")
            probability_lines = None
            if probabilities:
                with _open_text(probabilities) as f:
                    probability_lines = f.readlines()
                if len(probability_lines) != len(sentences):
                    raise ValueError("AWESOME alignment/probability line count mismatch")
            for k, pair in enumerate(sentences):
                yield _pharaoh_links(lines[k], len(pair["source"]), len(pair["target"]),
                                     pair["id"], probability_lines[k] if probability_lines else None)
        else:
            seen = 0
            for seen, (header, content) in enumerate(_iter_a3_blocks(input_path), 1):
                if seen > len(sentences):
                    raise ValueError("A3 contains more sentence pairs than the corpus")
                yield _a3_links(header, content, sentences[seen - 1], seen,
                                a3_source_side, verify_forms)
            if seen != len(sentences):
                raise ValueError(f"A3 contains {seen} pairs; expected {len(sentences)}")

    n = _atomic_jsonl(dest, records())
    return {"kind": "alignment", "format": actual, "sentences": n, "path": str(dest),
            "original_unchanged": True}


def _split_unescaped(text: str, marker: str) -> list[str]:
    parts, buffer, escaped = [], [], False
    for c in text:
        if escaped:
            buffer.extend(("\\", c))
            escaped = False
        elif c == "\\":
            escaped = True
        elif c == marker:
            parts.append("".join(buffer))
            buffer = []
        else:
            buffer.append(c)
    if escaped:
        raise ValueError("Unterminated Apertium escape")
    parts.append("".join(buffer))
    return parts


def _unescape(text: str) -> str:
    return re.sub(r"\\(.)", r"\1", text)


def _lexical_units(text: str) -> list[str]:
    """Collect ^...$ units; ignore formatting superblanks and escaped delimiters."""
    found = []
    i = 0
    while i < len(text):
        if text[i] == "\\":
            i += 2
        elif text[i] == "[":
            # Superblanks are formatted payloads, not analyzed surface tokens.
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == "]":
                    i += 1
                    break
                else:
                    i += 1
            else:
                raise ValueError("Unterminated Apertium superblank")
        elif text[i] == "^":
            start = i + 1
            i = start
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == "$":
                    found.append(text[start:i])
                    i += 1
                    break
                else:
                    i += 1
            else:
                raise ValueError("Unterminated Apertium lexical unit starting with ^")
        else:
            i += 1
    return found


def _candidate(reading: str) -> dict | None:
    if not reading or reading[0] in {"*", "@"}:
        return None
    lemma_buffer, tags = [], []
    pos = 0
    in_lemma = True
    while pos < len(reading):
        c = reading[pos]
        if c == "\\" and pos + 1 < len(reading):
            if in_lemma:
                lemma_buffer.append(reading[pos + 1])
            pos += 2
        elif c == "<":
            in_lemma = False
            close = pos + 1
            while close < len(reading):
                if reading[close] == "\\":
                    close += 2
                elif reading[close] == ">":
                    break
                else:
                    close += 1
            if close == len(reading):
                raise ValueError("Unclosed morphological tag in Apertium analysis")
            tags.append(_unescape(reading[pos + 1:close]))
            pos = close + 1
        else:
            if in_lemma:
                lemma_buffer.append(c)
            pos += 1
    lemma = "".join(lemma_buffer)
    return {"lemma": lemma or None,
            "upos": next((TAG_UPOS[t] for t in tags if t in TAG_UPOS), None),
            "feats": dict(TAG_FEATS[t] for t in tags if t in TAG_FEATS),
            "raw_tags": tags, "raw_analysis": reading, "resource": "apertium"}


def _analyze_unit(text: str) -> tuple[str | None, list[dict]]:
    parts = _split_unescaped(text, "/")
    if len(parts) == 1:
        # Disambiguated lexical streams may omit the surface; no form verification.
        surface = None
        readings = parts
    else:
        surface = _unescape(parts[0])
        readings = parts[1:]
    candidates = []
    for reading in readings:
        cand = _candidate(reading)
        if cand is not None and cand not in candidates:
            candidates.append(cand)
    return surface, candidates


def _validate_units(units: list[str], tokens: list[str], sid: str,
                    verify_forms: bool) -> list[list[dict]]:
    if len(units) != len(tokens):
        raise ValueError(f"Apertium sentence {sid}: {len(units)} units but {len(tokens)} target tokens")
    analyses = []
    for i, (unit, expected) in enumerate(zip(units, tokens)):
        observed, candidates = _analyze_unit(unit)
        if verify_forms and observed is not None and observed != expected:
            raise ValueError(f"Apertium sentence {sid}, token {i}: {observed!r} != "
                             f"{expected!r}; check corpus order/tokenization")
        analyses.append(candidates)
    return analyses


def import_apertium(*, targets: str, input_path: str, output: str,
                    layout: str = "auto", verify_forms: bool = True,
                    overwrite: bool = False) -> dict:
    """Convert a supplied Apertium lexical stream to the full analyses lattice."""
    if layout not in {"auto", "sentence-lines", "token-lines", "stream", "word-list"}:
        raise ValueError("layout must be auto, sentence-lines, token-lines, stream or word-list")
    targets_by_id = read_targets(targets)
    ordered = list(targets_by_id.values())
    dest = _assert_output([targets, input_path], output, overwrite)
    with _open_text(input_path) as f:
        data = f.read()
    physical = [line for line in data.splitlines() if line.strip()]
    # The mapping is by *surface form*, so a unique-word analyzer dump is
    # imported even when its rows do not follow sentence/token order.
    if layout == "word-list":
        by_surface = {}
        for line in physical:
            units = _lexical_units(line)
            if len(units) != 1:
                raise ValueError("Apertium word-list expects one lexical unit per nonempty line")
            surface, options = _analyze_unit(units[0])
            if surface is None:
                raise ValueError("Apertium word-list requires explicit ^surface/analysis$ units")
            previous = by_surface.setdefault(surface, [])
            for opt in options:
                if opt not in previous:
                    previous.append(opt)
        def records():
            for sent in ordered:
                yield {"id": sent["id"], "analyses": [by_surface.get(w, []) for w in sent["tokens"]],
                       "backend": "apertium-import"}
        count = _atomic_jsonl(dest, records())
        unmatched = sum(1 for sent in ordered for w in sent["tokens"] if w not in by_surface)
        return {"kind": "apertium", "layout": "word-list", "sentences": count,
                "missing_forms": unmatched, "path": str(dest), "original_unchanged": True}

    total_words = sum(len(t["tokens"]) for t in ordered)
    modes = [layout] if layout != "auto" else ["sentence-lines", "token-lines", "stream"]
    errors = []
    for mode in modes:
        try:
            lattices = []
            if mode == "sentence-lines":
                if len(physical) != len(ordered):
                    raise ValueError(f"expected {len(ordered)} nonblank sentence lines, found {len(physical)}")
                for line, sent in zip(physical, ordered):
                    lattices.append(_validate_units(_lexical_units(line), sent["tokens"],
                                                    sent["id"], verify_forms))
            elif mode == "token-lines":
                if len(physical) != total_words:
                    raise ValueError(f"expected {total_words} nonblank token lines, found {len(physical)}")
                cursor = 0
                for sent in ordered:
                    units = []
                    for line in physical[cursor:cursor + len(sent["tokens"])]:
                        parsed = _lexical_units(line)
                        if len(parsed) != 1:
                            raise ValueError("expected exactly one lexical unit per token line")
                        units.extend(parsed)
                    cursor += len(sent["tokens"])
                    lattices.append(_validate_units(units, sent["tokens"], sent["id"], verify_forms))
            else:
                units = _lexical_units(data)
                if len(units) != total_words:
                    raise ValueError(f"expected {total_words} units, found {len(units)}")
                cursor = 0
                for sent in ordered:
                    length = len(sent["tokens"])
                    lattices.append(_validate_units(units[cursor:cursor + length],
                                                    sent["tokens"], sent["id"], verify_forms))
                    cursor += length
            resolved = mode
            break
        except ValueError as exc:
            errors.append(f"{mode}: {exc}")
    else:
        raise ValueError("Apertium layout could not be reconciled with targets: " + "; ".join(errors))

    def records():
        for sent, lattice in zip(ordered, lattices):
            yield {"id": sent["id"], "analyses": lattice, "backend": "apertium-import"}
    count = _atomic_jsonl(dest, records())
    return {"kind": "apertium", "layout": resolved, "sentences": count,
            "path": str(dest), "original_unchanged": True}


def import_resources(config: str | Path, *, overwrite: bool = False) -> dict:
    """Run only explicitly described conversions; unrelated resources stay idle."""
    config_path = Path(config).expanduser().resolve()
    doc = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or set(doc) - {"source_ud", "targets", "alignment", "apertium"}:
        raise ValueError("Import YAML accepts source_ud, targets, alignment and apertium")
    if not doc.get("alignment") and not doc.get("apertium"):
        raise ValueError("Import config must specify at least one conversion")
    if not doc.get("targets") or (doc.get("alignment") and not doc.get("source_ud")):
        raise ValueError("targets required; source_ud required for alignment import")
    def resolve(name):
        p = Path(name).expanduser()
        return str((p if p.is_absolute() else config_path.parent / p).resolve())
    target_path = resolve(doc["targets"])
    source_path = resolve(doc["source_ud"]) if doc.get("source_ud") else None
    result = {}
    for section in ("alignment", "apertium"):
        item = doc.get(section)
        if item is None:
            continue
        if not isinstance(item, dict) or not item.get("input") or not item.get("output"):
            raise ValueError(f"{section} requires input and output")
        if section == "alignment":
            valid = {"input", "output", "format", "probabilities", "sentence_order",
                     "a3_source_side", "verify_forms"}
            if set(item) - valid:
                raise ValueError("Unknown alignment importer options")
            params = {k: v for k, v in item.items() if k in valid - {"input", "output", "probabilities"}}
            if item.get("probabilities"):
                params["probabilities"] = resolve(item["probabilities"])
            result[section] = import_alignments(source_ud=source_path, targets=target_path,
                                                input_path=resolve(item["input"]),
                                                output=resolve(item["output"]),
                                                overwrite=overwrite, **params)
        else:
            valid = {"input", "output", "layout", "verify_forms"}
            if set(item) - valid:
                raise ValueError("Unknown Apertium importer options")
            result[section] = import_apertium(targets=target_path,
                                              input_path=resolve(item["input"]),
                                              output=resolve(item["output"]),
                                              overwrite=overwrite,
                                              **{k: v for k, v in item.items()
                                                 if k not in {"input", "output"}})
    return result
