"""Optional non-LLM neural machine translation with IndicTrans2.

User-supplied parallel data are preferable where possible. Runtime trusts model code
only when the user explicitly enables this optional third-party adapter.
"""
from __future__ import annotations

from pathlib import Path

from .conllu import read_conllu
from .utils import append_jsonl, done_ids

DEFAULT_MODEL = "ai4bharat/indictrans2-indic-indic-dist-320M"


def translate_ud(source_ud: str | Path, out: str | Path, src_lang: str, tgt_lang: str,
                 model_name: str = DEFAULT_MODEL, batch_size: int = 8,
                 num_beams: int = 4, max_length: int = 256, device: str = "auto") -> dict:
    """Generate one target record per UD sentence, resumable by sent_id.

    Target *word* segmentation after MT is only heuristic; inspect/replace tokens for
    production. The model never generates parse trees or linguistic annotations.
    """
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError:
            from IndicTransToolkit.processor import IndicProcessor
    except ImportError as exc:
        raise RuntimeError("Install MT dependencies: pip install -e '.[translate]' "
                           "(and see docs/INSTALL.md for IndicTransToolkit)") from exc
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    done = done_ids(out)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    processor = IndicProcessor(inference=True)
    processed = 0
    todo = (s for s in read_conllu(source_ud) if str(s.sent_id) not in done)
    from itertools import islice
    while True:
        batch = list(islice(todo, batch_size))
        if not batch:
            break
        lines = [s.text or " ".join(w.form for w in s.words) for s in batch]
        inputs = processor.preprocess_batch(lines, src_lang=src_lang, tgt_lang=tgt_lang)
        tokenized = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            ids = model.generate(**tokenized, num_beams=num_beams, max_length=max_length)
        translations = processor.postprocess_batch(tokenizer.batch_decode(ids, skip_special_tokens=True),
                                                    lang=tgt_lang)
        for s, trans in zip(batch, translations):
            # Tokenization is explicitly left revisable. No sentence-level reordering of records.
            append_jsonl(out, {"id": str(s.sent_id), "text": trans,
                               "mt_model": model_name, "source_id": str(s.sent_id)})
            processed += 1
    return {"processed_this_call": processed, "path": str(out)}
