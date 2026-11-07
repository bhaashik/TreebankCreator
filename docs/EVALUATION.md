# Gold evaluation, error analysis and visualization

## Correct evaluation protocol

**Gold**: target sentence text with independently annotated target-language UD words/POS/HEAD/DEPREL/FEATS/lemmas, reviewed by a qualified annotator.

**Silver**: machine-produced label projection or target-parser predictions. Held-out silver does *not* estimate real accuracy. Its scores estimate how well a model reproduces the annotation process that created it.

For treebank creation, sample target gold examples from several genres, sentence lengths and grammatical constructions. Keep gold test examples separate, including translation variants and near duplicates of source data. `prepare_data` removes exact matching target-text overlaps when `gold_dev`/`gold_test` is supplied; semantic near-duplicate matching remains a responsibility of the experimenter.

## Built-in strict metrics

```bash
udtl evaluate --gold GOLD.conllu --pred PRED.conllu --out metrics.json
```

Output has:
- **UPOS accuracy** (exact word-level UPOS)
- **UAS** (unlabeled attachment score, punctuation excluded)
- **LAS** (attachment + coarse relation, punctuation excluded)
- **Exact UFeats accuracy**, **exact lemma accuracy**, whole-sentence LAS, root recall
- Relation-wise recall and confusion matrices, dependency-distance buckets, up to 1,000 misanalysis rows

The built-in scorer **requires identical sentence count, word count and token forms** between gold and predicted files. It errors instead of pairing different tokenizations as if they were identical. UAS/LAS are computed on matching word indices; this internal score is useful for error analysis and sanity checks, but is **not identical to the official CoNLL18 scoring protocol** for all corner cases, multi-word tokens and punctuation definitions. UPOS and lemma/FEATS exact accuracy may be inflated by `_` placeholders. For research publication use the official evaluator.

## Official UD evaluation (recommended)

The official CoNLL 2018 evaluator computes **token/word/sentence F1, UPOS, UFeats, UAS/LAS, CLAS, MLAS and BLEX** with its own token matching. Download `conll18_ud_eval.py` from the official UD resource page:

https://universaldependencies.org/conll18/evaluation.html

Then:

```bash
udtl official-eval --script /path/to/conll18_ud_eval.py \
  --gold GOLD.conllu --pred PRED.conllu --out official_metrics.txt
```

You can also run that script directly. The project does **not** bundle it (different code license); read its MPL-2.0 license and author credits.

## HTML report and dependency arc diagrams

```bash
udtl visualize --input PRED.conllu --out view.html \
  --metrics metrics.json --metadata runs/marathi_full/pair_01/metadata.jsonl \
  --limit 100
```

The output is a **standalone UTF-8 HTML** file with inline SVG trees (no external JavaScript/CDN, no internet connection). It shows UPOS labels, head-dependent curves, deprel, projection-quality histogram and frequent relation confusions. Each record also expands its original CoNLL-U. The default preview shows only 40 trees to avoid a huge HTML file; the full corpus remains in `silver.conllu`.

### Diagnostics worth tracking

- Alignment coverage and directionality by sentence and POS group
- Projected versus fallback arc ratio
- LAS/UAS by dependency length, head/dependent relation, script/word segmentation, OOV rate
- Cases where the source root aligns to no target word
- Morphological feature ambiguity and agreement failures
- Coverage/quality tradeoff across `min_quality` thresholds
- Sentence-domain differences (e.g. news Hindi HDTB vs conversational target translations)

A simple human review strategy is to sort `metadata.jsonl` by `quality` ascending, prioritizing low coverage or high fallback fractions. A later iteration could integrate a browser annotation editor, inter-annotator agreement (IAA) tools, and active-learning sampling.

## Interpreting the example

The bundled `examples/marathi_gold_sample.conllu` has three intentionally simple sentences with matching manually written links. It exists **only to test the evaluation code**. Perfect results from this fixture should **never** be reported as genuine cross-lingual parsing accuracy.
