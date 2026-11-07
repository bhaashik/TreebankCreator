"""Small dependency-focused CoNLL-U reader/writer. Word IDs are always 1..N.

We deliberately ignore MWT range lines and empty-node lines while projecting basic UD.
This module is not an enhanced-dependency parser or raw-text tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

COLS = ("id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc")


@dataclass
class Word:
    id: int
    form: str
    lemma: str = "_"
    upos: str = "_"
    xpos: str = "_"
    feats: str = "_"
    head: int = 0
    deprel: str = "dep"
    deps: str = "_"
    misc: str = "_"

    def to_line(self) -> str:
        return "\t".join(map(str, (self.id, self.form, self.lemma, self.upos, self.xpos,
            self.feats, self.head, self.deprel, self.deps, self.misc)))


@dataclass
class Sentence:
    words: list[Word]
    sent_id: str
    text: str = ""
    comments: list[str] = field(default_factory=list)

    def to_conllu(self) -> str:
        comments = [f"# sent_id = {self.sent_id}", f"# text = {self.text or ' '.join(w.form for w in self.words)}"]
        comments += [c for c in self.comments if not c.startswith(("# sent_id =", "# text ="))]
        return "\n".join(comments + [w.to_line() for w in self.words]) + "\n\n"


def _parse_block(lines: list[str], fallback_id: str) -> Sentence | None:
    if not lines:
        return None
    words = []
    comments = [line for line in lines if line.startswith("#")]
    for line in lines:
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            raise ValueError(f"Expected 10 CoNLL-U columns, got {len(cols)}: {line!r}")
        if not cols[0].isdigit():
            # Multi-word tokens (1-2) and enhanced empty nodes (2.1).
            continue
        head = 0 if cols[6] == "_" else int(cols[6])
        words.append(Word(int(cols[0]), cols[1], cols[2], cols[3], cols[4], cols[5],
                          head, cols[7], cols[8], cols[9]))
    if not words:
        return None
    words.sort(key=lambda w: w.id)
    # Projection relies on contiguous integer word IDs. MWT ranges are excluded.
    if [w.id for w in words] != list(range(1, len(words) + 1)):
        raise ValueError(f"Non-contiguous integer word IDs in {fallback_id}")
    meta = {}
    for c in comments:
        if " = " in c:
            k, v = c[2:].split(" = ", 1)
            meta[k] = v
    return Sentence(words, meta.get("sent_id", fallback_id), meta.get("text", ""), comments)


def read_conllu(path: str | Path) -> Iterator[Sentence]:
    block: list[str] = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                if block:
                    n += 1
                    sent = _parse_block(block, f"s{n:07}")
                    if sent is not None:
                        yield sent
                    block = []
            else:
                block.append(line)
        if block:
            n += 1
            sent = _parse_block(block, f"s{n:07}")
            if sent is not None:
                yield sent


def parse_conllu_block(block: str) -> Sentence:
    sent = _parse_block([x for x in block.splitlines() if x.strip()], "in-memory")
    if sent is None:
        raise ValueError("Empty CoNLL-U sentence")
    return sent


def validate_tree(sent: Sentence) -> list[str]:
    errors: list[str] = []
    n = len(sent.words)
    if [w.id for w in sent.words] != list(range(1, n + 1)):
        errors.append("bad-id-sequence")
        return errors
    roots = [w.id for w in sent.words if w.head == 0]
    if len(roots) != 1:
        errors.append(f"roots={len(roots)}")
    for w in sent.words:
        if w.head < 0 or w.head > n or w.head == w.id:
            errors.append(f"bad-head:{w.id}->{w.head}")
        if w.head == 0 and w.deprel != "root":
            errors.append(f"root-label:{w.id}")
        if w.head != 0 and w.deprel == "root":
            errors.append(f"nonroot-label:{w.id}")
        seen = {w.id}
        h = w.head
        while 1 <= h <= n:
            if h in seen:
                errors.append(f"cycle:{w.id}")
                break
            seen.add(h)
            h = sent.words[h - 1].head
    return sorted(set(errors))


def copy_with_id(sent: Sentence, new_id: str) -> Sentence:
    return replace(sent, sent_id=new_id, words=[replace(w) for w in sent.words])
