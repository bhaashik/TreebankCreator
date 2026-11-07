"""Optional Stanza parsing and constituency inference adapters.

Constituency is independent of UD dependency projection; .mrg predictions are
separate silver phrase-structure annotations, not guaranteed to exist for every language.
"""
from __future__ import annotations

from pathlib import Path

from .alignment import read_targets
from .conllu import Sentence, Word
from .utils import append_jsonl, done_ids, write_lines


def _require_stanza():
    try:
        import stanza
        return stanza
    except ImportError as exc:
        raise RuntimeError("Install Stanza support: pip install -e '.[stanza]'") from exc


def parse_parallel_source(source_jsonl: str, out_conllu: str, lang: str = "hi",
                         package: str = "default", use_gpu: bool = True) -> dict:
    stanza = _require_stanza()
    targets = read_targets(source_jsonl)
    nlp = stanza.Pipeline(lang=lang, processors="tokenize,pos,lemma,depparse", package=package,
                          tokenize_no_ssplit=True, use_gpu=use_gpu)
    output = Path(out_conllu)
    # Parse cache stores CoNLL-U per id so interruption cannot corrupt finished output.
    cache = output.with_suffix(".source.stage.jsonl")
    done = done_ids(cache)
    count = 0
    for sid, row in targets.items():
        if sid in done: continue
        doc = nlp(row["text"])
        if len(doc.sentences) != 1:
            raise ValueError(f"Stanza split {sid} into {len(doc.sentences)} sentences")
        words = []
        for w in doc.sentences[0].words:
            words.append(Word(id=int(w.id), form=w.text, lemma=w.lemma or "_",
                              upos=w.upos or "X", xpos=w.xpos or "_", feats=w.feats or "_",
                              head=int(w.head or 0), deprel=w.deprel or "dep"))
        sentence = Sentence(words, sent_id=sid, text=row["text"])
        append_jsonl(cache, {"id": sid, "conllu": sentence.to_conllu()})
        count += 1
    from .utils import jsonl
    write_lines(output, (row["conllu"] for row in jsonl(cache)))
    return {"processed_this_call": count, "output": str(output)}


def constituency_infer(input_jsonl: str, out_mrg: str, lang: str = "en",
                       package: str = "default", use_gpu: bool = True) -> dict:
    stanza = _require_stanza()
    pipe = stanza.Pipeline(lang=lang, processors="tokenize,pos,constituency", package=package,
                           tokenize_no_ssplit=True, use_gpu=use_gpu)
    targets = read_targets(input_jsonl)
    cache = Path(out_mrg).with_suffix(".mrg.stage.jsonl")
    done = done_ids(cache)
    count = 0
    for sid, row in targets.items():
        if sid in done: continue
        doc = pipe(row["text"])
        if len(doc.sentences) != 1 or doc.sentences[0].constituency is None:
            raise RuntimeError(f"Stanza returned no constituency tree for {sid}")
        append_jsonl(cache, {"id": sid, "mrg": str(doc.sentences[0].constituency)})
        count += 1
    from .utils import jsonl
    write_lines(out_mrg, (row["mrg"] + "\n" for row in jsonl(cache)))
    return {"processed_this_call": count, "output": out_mrg,
            "warning": "Predicted silver constituency trees, not derived from UD"}
