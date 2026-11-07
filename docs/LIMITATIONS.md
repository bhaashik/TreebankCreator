# Current limitations and prioritized roadmap

This package is a **tested offline projection prototype**, plus code integration points for GPU training and pretrained NLP systems. It is **not** a validated Hindi→22-languages production treebank pipeline yet.

## Functional status

**Works in the offline test environment:** CoNLL-U basic dependency parsing and serialization (no MWT/empty-node preservation); manual/lexicon/identity alignments; single-root arborescence projection; confidence/provenance recording; YAML multi-source orchestration; split/filter; synthetic templates; strict evaluation; HTML SVG visualization; JSONL restart; repeated-run skipping.

**Implemented but not GPU/integration-tested in this environment:** SimAlign model downloads/inference, IndicTrans2+IndicTransToolkit MT, Apertium `lt-proc` with a real compiled analyzer, Trankit custom trainer, Stanza source-side or constituency inference, HF transformer training/prediction. These require validating package versions, model access and hardware on the user's VM.

**Not implemented:** automatic target-language MWT tokenization, full lemma/FEATS prediction by the custom HF model, full UD enhanced dependency graphs, neural model ensemble/teacher-student self-training, Stanza end-to-end custom training wrapper, native constituency projection from dependencies, user-facing manual annotation editor, uncertainty calibration with gold labels, grammatical error constraints beyond soft head-direction, semantic near-duplicate detection, multi-source agreement for the *same* target sentence, automatic language-specific morphology mappings for every scheduled language, direct BPCC/HF datasets import, fully automated 22-language benchmarks. A user-supplied aligned parallel dataset is currently normalized into JSONL outside the package.

## Highest-value follow-up engineering work

1. **Dataset ingest:** consistent BPCC/ILCI/OPUS/FLORES adapters, sentence-level semantic similarity checks, dedup and alignment diagnostics.
2. **Target word tokenization:** script-aware Indic NLP or UDTok adapters for clitic segmentation, multiword expressions and gold-text matching.
3. **Multi-source consensus:** project each same-target sentence from multiple parallel UD trees; combine normalized arc scores, trust, and alignment entropy.
4. **Model distillation/refinement:** train/validate target Trankit/Stanza with careful silver/gold weighting, self-training on unlabeled target text, confidence calibration, nonparametric error correction. Keep annotation provenance.
5. **Morphology and linguistics:** per-language finite-state dictionaries/tag maps, feature disambiguation, case/agreement constraints and auxiliary/complex predicate handling.
6. **Linguist review UI:** sentence/edge corrections, change history, inter-annotator agreement, adjudication, active review queue.
7. **Constituency branch:** true phrase-structure training datasets and independent constituency parser fine-tuning for supported targets. Projected dependencies alone do not contain enough information.
8. **Research rigor:** official UD evaluation, held-out raw tokenization, bootstrap CI, out-of-domain tests, model cards, ablations and reproducible benchmarking.

## Scientific caveats

- Machine translations preserve meaning imperfectly and may change argument structure, grammatical function, voice, case and clause structure. Projection is only a starting point.
- Hindi HDTB is news-domain, and many target-language datasets are not; domain transfer can dominate structural improvements.
- Root preferences and UD label distributions vary by language and UD treebank; soft universal rules are heuristic only.
- Coverage-based confidence is **not** validated; do not calibrate manually observed error rates by assuming it is a probability.
- Tree well-formedness is a syntactic graph property, not an indicator of linguistic correctness.
- High-quality UD for an under-resourced language usually requires human linguists and explicit annotation guidelines.
- 22 scheduled languages do **not** imply 22 comparable data resources or results. Both model coverage and available gold evaluation must be checked.

## Reproducibility and user safety

Do not run unknown downloaded HF custom model code without checking publisher/revision (`trust_remote_code=True` required for IndicTrans2). The optional HF parser loads PyTorch checkpoint files (pickle); never load untrusted checkpoints. Never place HF tokens in YAML. `udtl fetch-ud` downloads external source datasets; audit licenses. The project's MIT license only covers project code.
