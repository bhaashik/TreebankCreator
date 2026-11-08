# Corpus-wide lexical, morphological and alignment refinement (v0.3.0)

This is a CPU-only **pre-neural** stage. It consumes Hindi gold UD and parallel
**target tokens**, optionally pre-existing word alignments and morphological
resources, and generates *uncertain* repaired alignments and linguistic hints.
It **does not** train a morphology generator, neural aligner, XLM-R or Trankit.
It never edits input files. No downloading or expensive tool runs occur unless `mode: generate`
or `mode: auto` with `if_missing: generate` is explicitly selected in YAML.
For automatic input detection and setup, see [Conditional resource setup](CONDITIONAL_RESOURCE_SETUP.md).

## Installation and first smoke test

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
bash scripts/demo_corpus.sh
udtl refine-corpus --config configs/corpus_marathi_supplied.yaml
```

The bundled Marathi smoke example is three tiny sentence pairs, **not** a
performance benchmark. Input is Hindi UD CoNLL-U (10 columns, stable `sent_id`),
JSONL target sentences with matching `id` and `tokens`, and optional JSONL
alignment records. All alignment indexes are **0-based** in `[source,target]`
order. Sentence/token IDs must match exactly. A target CoNLL-U containing bad
projected `LEMMA`/UPOS/FEATS fields should not be used as an authority; supply
actual target token forms separately.

## Configuration and resource lifecycle

Create `configs/hindi_bhojpuri_corpus.yaml` (paths resolve **relative to this
YAML**, so cloud and Ubuntu installations work the same way):

```yaml
source_ud: ../data/hi_train.conllu
targets: ../data/bho_tokens.jsonl
output_dir: ../runs/bho/corpus
resources:
  alignment:
    mode: supplied
    path: ../data/automatic_alignments.jsonl
  apertium:
    mode: supplied
    path: ../data/apertium_analyses.jsonl
  unimorph:
    mode: supplied
    path: ../data/unimorph_bho.tsv
refinement:
  iterations: 3
  max_words: 90
  anchor_similarity: 0.72
  min_link_confidence: 0.52
  max_alternatives: 3
  min_affix_support: 3
```

Only include resource entries you have. The default mode of every omitted
resource is **`skip`**, not `generate`. If you provide `path` without `mode`,
that resource is inferred to be **`supplied`**. Missing supplied files cause
an error before processing. No supplied path is overwritten or repaired. Only
specific outputs under `output_dir` are written by this stage.

| Resource | `supplied` | `skip` (default) | `generate` (explicit opt-in) |
|---|---|---|---|
| `alignment` | Read `pairs`, scored `links`, or target×source `matrix` JSONL | Use orthographic/contextual corpus evidence only; may be sparse | Run existing `udtl align` adapter (default SimAlign, model download/GPU may be needed) |
| `apertium` | Read per-token analyses JSONL (ambiguous lists or legacy dictionaries) | Infer weak form families only | Invoke `lt-proc -a` using configured analyzer and preserve **all** readings |
| `unimorph` | Read UniMorph lemma/surface/feature TSV | No paradigm lexicon | Not supported: UniMorph is a dataset; it is not automatically generated |

To **explicitly** generate alignments:

```yaml
resources:
  alignment:
    mode: generate
    backend: simalign
    model: xlmr
    method: itermax
    device: auto
  apertium:
    mode: skip
  unimorph:
    mode: skip
```

To explicitly invoke an installed Apertium analyzer:

```yaml
resources:
  apertium:
    mode: generate
    analyzer_bin: /opt/apertium/languages/apertium-bho.bho.automorf.bin
    binary: lt-proc
```

The configured binary and analyzer must actually exist and support the target
language; the example path is **illustrative**, not an assertion of an
available Bhojpuri model. `mode: generate` **must not** include a `path`, so it
cannot accidentally overwrite a pre-existing resource. Produced files live in
`output_dir/initial_alignments.generated.jsonl` and
`output_dir/apertium.generated.jsonl`. If inputs/config change, generated
files are invalidated; supplied files are not.

## Outputs

```text
runs/bho/corpus/
  alignments.refined.jsonl           # JSONL: id, pairs, matrix, null_source, null_target
  corrections.jsonl                  # original-vs-proposed links, evidence, status
  bilingual_lexicon.json             # global token correspondence hypotheses
  function_realizations.json         # word / NULL-or-bound / candidate-affix counts
  morph_candidates.jsonl             # target morphological candidate lattices
  bound_morpheme_hypotheses.jsonl     # source function -> target-host suffix, if supported
  report.json                        # counts and explicit caveats
  manifest.json                      # full input/config fingerprint, modes, provenance
```

`pairs` is the **hard best** hypothesis for legacy `udtl project` and
`read_alignments`. `matrix` contains conservative **uncalibrated** soft scores
for the v0.2 bilingual neural trainer. Both are **silver** and may remain wrong.
`null_source` means no accepted *hard* token correspondence, not proof of
phonological zero realization; a source token may contribute to an affix or an
unmodeled grammatical construction. `null_target` indicates no hard match.

Multiple possible Apertium analyses are preserved, as are UniMorph paradigms,
including their original disambiguation uncertainty. The UniMorph-to-UD tag
mapping handles a **small subset** of tags; other tags are retained as raw
morphological labels when imported. UniMorph is not a context-sensitive
morphological disambiguator, and never silently overrides a conflict.

## How it infers corpus-wide evidence

1. Aggregate highly reliable surface matches, available morphological analyses,
   and lower-weight initial alignment observations. Hard initial links are
   **not** assumed gold.
2. Independently count *sentence-level* Hindi-lemma/target-form co-occurrences
   across the whole available corpus, including words not initially aligned.
   Use source and target sentence frequencies and a leave-one-sentence-out
   correction to avoid letting one example confirm itself.
3. Rerank token pairs with bidirectional lexical association, co-occurrence,
   orthographic similarity (where scripts permit), POS compatibility,
   optional morphology, and modest original-alignment evidence.
4. Permit no match, more than one soft match, and many source words aligning to
   one target. **Preserve** supplied links with reduced confidence when there
   is insufficient independent evidence to repair them. A known implausible
   function-word-to-proper-name link may be rejected.
5. Analyze Hindi function words (`ADP`, `PART`, auxiliaries, etc.) and their
   target *word*, *unresolved/NULL*, and potential *bound* realizations.
   Candidate suffix links are reported **only** when the same Hindi host lemma
   and target family recur both with and without the source function word.
   These are association hypotheses, not grammatical facts.
6. Run a bounded number of refinements and save everything for audit. A final
   Apertium or UniMorph lemma is a target-language candidate; **Hindi lemmas
   must not be copied into the target CoNLL-U lemma column**.

### Example: shifted Bhojpuri alignment

The test fixture uses a noisy link `ने → शाहजेहन`. In that example, the
Hindi name `शाहजहाँ` is a better match for Bhojpuri `शाहजेहन`; the former
link is withdrawn, with the Hindi postposition remaining unmatched as a
separate target word. The tests also cover cross-script proper-name repetition,
where character similarity is impossible and corpus co-occurrence becomes the
main clue. This **does not prove** robust correction across actual target
corpora or guarantee discovery of function-word-to-affix mappings.

## Use before bilingual training (opt-in)

The original v0.2.0 bilingual config continues to work unchanged. To
**explicitly** enable corpus preparation in a new bilingual config:

```yaml
language: bho
source_ud: ../data/hi_train.conllu
targets: ../data/bho_tokens.jsonl
alignments: ../runs/bho/corpus/alignments.refined.jsonl
corpus_preprocess: hindi_bhojpuri_corpus.yaml
source_cache: ../cache/shared_hindi
target_cache: ../cache/bho
output_dir: ../runs/bho/model
model: xlm-roberta-base
epochs: 6
batch_size: 4
```

Now any of these commands reuses the matching corpus preprocessing output if
it exists or creates it if missing/stale:

```bash
udtl refine-corpus --config configs/hindi_bhojpuri_corpus.yaml
udtl cache-bilingual --config configs/bilingual_bho.yaml --side both
udtl train-bilingual --config configs/bilingual_bho.yaml
udtl predict-bilingual --config configs/bilingual_bho.yaml
```

`corpus_preprocess` must point to a corpus YAML whose source, targets, and
**exact output alignment path** match the bilingual YAML. The tokenizer caches,
GPU training and optimizer checkpoints are otherwise unchanged. The generator
modes within the corpus YAML still control whether costly tools can run.

For 13 languages, the JSONL manifest accepted by `udtl make-bilingual-configs`
now supports records such as:

```json
{"language":"bho","targets":"../data/bho_tokens.jsonl","corpus_preprocess":"corpus_bho.yaml"}
```

The produced bilingual config points at the derived refined alignment path.
No corpus refinement runs during manifest creation. Stage and inspect corpus
outputs **before** launching two GPU workers on Kaggle, especially if initial
alignment `generate` uses GPU time. The CPU-only stage can run on Ubuntu,
Colab, or Kaggle without optional ML dependencies when resources are supplied.

## Scientific limitations and evaluations

- This is a **heuristic research prototype**, not a calibrated statistical
  aligner, a validated morphology generator, an Apertium replacement, or a
  jointly trained alignment/morphology model.
- Frequent systematic incorrect alignments can also contaminate global
  statistics. Source/target *sentence co-occurrence* strengthens evidence but
  does not identify every correct word correspondence; incidental topic words
  can co-occur consistently. Different-script rare names and paraphrases can
  remain unresolved.
- Function-word-to-affix mappings are merely proposed when a contrast-supported
  surface suffix can be observed. The model currently does **not** discover
  arbitrary nonconcatenative morphology, allomorphs, agreement dependencies, or
  zero-vs-bound distinctions reliably.
- The `morph_candidates.jsonl`/`function_realizations.json` outputs are not yet
  consumed as *differentiable neural losses*; v0.3 integrates via corrected
  alignments and their soft confidence. Joint morphology/alignment learning is
  a separate experimental step.
- The corpus stage uses **all supplied unlabeled parallel pairs** to learn
  statistics. If a target-language dev/test set is included among those pairs,
  this is transductive preprocessing; do not claim fully inductive evaluation.
  Prefer reserve held-out gold evaluation sentences outside this corpus.
- Measure alignment precision/recall and NULL/function-word behavior against
  manually checked word/morpheme links; compare actual target gold UPOS/UAS/LAS
  on independent held-out sentences. Ablate original alignments, Apertium,
  UniMorph, corpus association, and function-affix hypotheses.
- Keep licenses/provenance of all Hindi UD, parallel corpus, Apertium, UniMorph
  and derived data separate from the project's MIT software license.
