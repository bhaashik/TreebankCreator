"""Alignment adapters (SimAlign, supplied alignments, or test-only identity)."""
from __future__ import annotations

import re
import importlib.util
import unicodedata
from pathlib import Path
from typing import Iterator

from .conllu import Sentence, read_conllu
from .utils import append_jsonl, done_ids, jsonl

PUNCT = "।॥،؛؟.!?,;:()[]{}\"'“”‘’"


def tokenize(text: str) -> list[str]:
    """Conservative whitespace tokenizer with edge punctuation splitting.

    NOT a substitute for language-specific UD word segmentation. Prefer JSONL tokens.
    """
    result: list[str] = []
    for item in text.split():
        while item and item[0] in PUNCT:
            result.append(item[0]); item = item[1:]
        suffix = []
        while item and item[-1] in PUNCT:
            suffix.append(item[-1]); item = item[:-1]
        if item:
            result.append(item)
        result.extend(reversed(suffix))
    return result


def _fold(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text.casefold())
                   if not unicodedata.combining(ch)).strip()


class Aligner:
    def __init__(self, backend: str = "simalign", model: str = "xlmr", method: str = "itermax",
                 device: str = "auto", lexicon: dict | None = None):
        self.backend = backend
        self.method = method
        self.lexicon = {str(k).casefold(): {str(v).casefold()} if isinstance(v, str)
                        else {str(x).casefold() for x in v}
                        for k, v in (lexicon or {}).items()}
        self.engine = None
        if backend == "simalign":
            try:
                from simalign import SentenceAligner
                if device == "auto":
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                self.engine = SentenceAligner(model=model, token_type="bpe",
                                             matching_methods="mai", device=device)
            except ImportError as exc:
                raise RuntimeError("Install word alignment dependencies: pip install -e '.[align]'") from exc
        elif backend not in ("identity", "lexicon"):
            raise ValueError(f"Unknown alignment backend: {backend}")

    def align(self, source: list[str], target: list[str]) -> list[tuple[int, int]]:
        if not source or not target:
            return []
        if self.engine is not None:
            result = self.engine.get_word_aligns(source, target)
            if self.method not in result:
                raise ValueError(f"Method {self.method} not returned by SimAlign; choose {list(result)}")
            return sorted({(int(i), int(j)) for i, j in result[self.method]})
        if self.backend == "identity":
            # Smoke test ONLY: unsuitable for translation/reordering.
            return [(i, i) for i in range(min(len(source), len(target)))]
        alignments = []
        for i, s in enumerate(source):
            sf = _fold(s)
            candidates = self.lexicon.get(s.casefold(), set()) | {sf}
            for j, t in enumerate(target):
                if _fold(t) in candidates or t.casefold() in candidates:
                    alignments.append((i, j))
        return sorted(set(alignments))


def read_targets(path: str | Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in jsonl(path):
        sid = str(item["id"])
        if sid in result:
            raise ValueError(f"Duplicate target ID: {sid}")
        tokens = item.get("tokens")
        if tokens is None:
            tokens = tokenize(item["text"])
        if not tokens or not isinstance(tokens, list) or not all(isinstance(t, str) and t for t in tokens):
            raise ValueError(f"Bad token sequence in target ID: {sid}")
        result[sid] = {**item, "id": sid, "tokens": tokens}
    return result


def run_alignment(source_ud: str | Path, target_jsonl: str | Path, output: str | Path,
                  backend: str = "simalign", model: str = "xlmr", method: str = "itermax",
                  device: str = "auto", lexicon: dict | None = None,
                  binary: str = "awesome-align") -> dict:
    if backend == "auto":
        # Explicitly requested fallback; keep supplied resources untouched.
        if importlib.util.find_spec("simalign") is not None:
            backend = "simalign"
        else:
            from .awesome_adapter import _executable
            try:
                _executable(binary)
            except RuntimeError as exc:
                raise RuntimeError("Neither SimAlign nor AWESOME-align is installed; "
                                   "run scripts/install_linguistic_tools.sh --aligner auto") from exc
            backend = "awesome"
    if backend == "awesome":
        from .awesome_adapter import run_awesome_alignment
        return run_awesome_alignment(source_ud, target_jsonl, output, model=model,
                                     extraction=("softmax" if method == "itermax" else method),
                                     device=device, binary=binary)
    targets = read_targets(target_jsonl)
    done = done_ids(output)
    engine = None
    completed = 0
    for sent in read_conllu(source_ud):
        sid = str(sent.sent_id)
        if sid in done:
            continue
        if sid not in targets:
            raise ValueError(f"No target sentence for source {sid}; IDs must match")
        if engine is None:
            engine = Aligner(backend, model, method, device, lexicon)
        tgt = targets[sid]
        links = engine.align([w.form for w in sent.words], tgt["tokens"])
        append_jsonl(output, {"id": sid, "pairs": links, "backend": backend})
        completed += 1
    return {"processed_this_call": completed, "total": len(done_ids(output)), "path": str(output)}


def read_alignments(path: str | Path) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for item in jsonl(path):
        sid = str(item["id"])
        if sid in result:
            raise ValueError(f"Duplicate alignment ID {sid}")
        pairs = item["pairs"]
        if not isinstance(pairs, list) or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in pairs):
            raise ValueError(f"Bad pairs for ID {sid}")
        result[sid] = sorted(set((int(a), int(b)) for a, b in pairs))
    return result
