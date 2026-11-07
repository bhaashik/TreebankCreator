"""Expand *linguist-authored* aligned UD sentence templates.

Requires an explicit inflected source/target realization per variant; no LLM
or uncontrolled synthetic generation. Output is SILVER until reviewed.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from .conllu import Sentence, Word, validate_tree
from .utils import write_lines


def create_from_templates(template_json: str, out_dir: str, max_examples: int = 10000) -> dict:
    """Template format described in docs/DATA.md.

    A source UD skeleton and target alignment are held fixed; variants only
    replace full word forms, not word order or dependency/morphology labels.
    """
    templates = json.loads(Path(template_json).read_text(encoding="utf-8"))
    if not isinstance(templates, list):
        raise ValueError("Templates JSON must be an array")
    source, target, align = [], [], []
    seen = set()
    for item in templates:
        slots = item.get("slots", {})
        if not isinstance(slots, dict):
            raise ValueError("slots must map names to lists of inflected bilingual variants")
        variants = [slots[name] for name in slots]
        for choice in itertools.product(*variants):
            if len(source) >= max_examples:
                break
            values = dict(zip(slots, choice))
            def realize(pattern: str, side: str) -> str:
                substitutions = {k: v[side] for k, v in values.items()}
                return pattern.format(**substitutions)
            sid = f"synthetic-{len(source):07}"
            src = [Word(id=i+1, form=realize(w["form"], "source"),
                        lemma="_", upos=w.get("upos", "X"), head=int(w["head"]),
                        deprel=w["deprel"], feats=w.get("feats", "_"))
                   for i, w in enumerate(item["source_words"])]
            words = [realize(w, "target") for w in item["target_tokens"]]
            if any(not x or any(char.isspace() for char in x) for x in words + [w.form for w in src]):
                raise ValueError("Template variant changed tokenization or produced empty word")
            s = Sentence(src, sid)
            if validate_tree(s):
                raise ValueError(f"Invalid synthetic template {item.get('id')}: {validate_tree(s)}")
            signature = (tuple(w.form for w in src), tuple(words))
            if signature in seen:
                continue
            seen.add(signature)
            links = item["alignments"]
            for si, ti in links:
                if not 0 <= si < len(src) or not 0 <= ti < len(words):
                    raise ValueError(f"Invalid link {si}-{ti} in template {item.get('id')}")
            source.append(s.to_conllu())
            target.append(json.dumps({"id": sid, "text": " ".join(words), "tokens": words},
                                     ensure_ascii=False) + "\n")
            align.append(json.dumps({"id": sid, "pairs": links, "backend": "human-template"},
                                    ensure_ascii=False) + "\n")
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_lines(directory / "source.conllu", source)
    write_lines(directory / "targets.jsonl", target)
    write_lines(directory / "alignments.jsonl", align)
    return {"generated": len(source), "source_ud": str(directory / "source.conllu"),
            "targets": str(directory / "targets.jsonl"),
            "alignments": str(directory / "alignments.jsonl"),
            "warning": "Templates must be linguistically reviewed; not automatically gold"}
