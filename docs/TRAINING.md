# Training parsers and resuming interrupted jobs

For the complete **Lightning AI Pro Studio** training lifecycle, start/sleep controls,
Google Drive/Dropbox model staging, checkpoint backups and cloud restoration, see
[`LIGHTNING_AI_PRO.md`](LIGHTNING_AI_PRO.md). Studio sleeping stops a running Python
process; resume starts at the last **completed** saved checkpoint or stage.

## The two available dependency backends

**Trankit (recommended to start with when dependency versions permit):** calls the documented `trankit.TPipeline(training_config={category:'customized', task:'posdep', save_dir, train_conllu_fpath, dev_conllu_fpath}).train()` API. Uses Trankit's own pretrained multilingual transformer and adapters. Trankit is an independent model implementation; installation, language support and extra training flags are **version-specific**. It trains a combined POS/morphological-tag/dependency module and has its own released model files.

**HF parser (best checkpoint/resume control):** loads `xlm-roberta-base` (or another HF `AutoModel` with a fast tokenizer), pools **the first subword** of each UD word, trains a target UPOS head, scored head attachments, and dependency-label head. Inference decodes one-root maximum-weight arborescences. Encoder frozen by default (faster, less GPU memory); unfreeze optional. This is a **lightweight baseline**, not a claim to match the most advanced biaffine/adapted parsers. HF parser **does not predict lemmas, morphological FEATS, MWT segmentation, or constituency trees**.

## Quick path: Trankit

```bash
pip install -e '.[trankit]'
udtl train-trankit --train runs/marathi_demo/training_data/train.conllu \
  --dev runs/marathi_demo/training_data/dev.conllu \
  --out runs/marathi_demo/trankit_model
```

Choose `training: backend: trankit` to run this from YAML. The `train-trankit` wrapper writes `posdep.complete.json` **only after successful completion**. Re-executing with identical input data skips this completed task. **Mid-epoch and optimizer-state resume depend on Trankit itself; the wrapper does not implement them.** If using `customized`, the POS/dep module alone does not create a full raw-text tokenizer pipeline; train or provide a compatible tokenizer/other modules for full pipeline inference.

Trankit official customization guide: https://trankit.readthedocs.io/en/latest/training.html

## Quick path: HF train / resume

```bash
pip install -e '.[hf]'     # torch + transformers; separate from translation dependencies
udtl train-hf \
  --train runs/marathi_demo/training_data/train.conllu \
  --dev runs/marathi_demo/training_data/dev.conllu \
  --out runs/marathi_demo/model_hf \
  --pretrained xlm-roberta-base --epochs 6 \
  --batch-size 4 --accumulate 4 --save-steps 200
```

The next command resumes from `last.pt` with saved model, optimizer, random states, epoch and batch offset:

```bash
udtl train-hf --train runs/marathi_demo/training_data/train.conllu \
  --dev runs/marathi_demo/training_data/dev.conllu \
  --out runs/marathi_demo/model_hf --pretrained xlm-roberta-base \
  --epochs 6 --batch-size 4 --accumulate 4 --save-steps 200
```

Resuming after interruption may repeat up to `save_steps × accumulate` batches. The output directory needs durable storage. Re-running with changed training file contents, label inventory, model config, batch settings or learning rate is intentionally rejected; start a new output directory or `--no-resume` (the latter replaces state with a fresh run). Never load `last.pt` or `best.pt` from untrusted sources: PyTorch checkpoint loading can execute code.

To update the transformer itself (requires more GPU RAM): add `--unfreeze-encoder`. Default encoder freezing trades capacity for speed and reduces overfitting to noisy silver labels. Choose `max_words` and `max_subtokens` to avoid truncated records; the training command explicitly errors rather than silently training on truncated UD words.

**If `dev.conllu` is held-out silver rather than hand-annotated gold**, validation LAS merely measures imitation of projected labels. It should not be treated as true parsing accuracy or used to make strong model-selection claims.

## HF prediction

```bash
udtl predict-hf --model-dir runs/marathi_demo/model_hf \
  --input data/target_tokenized.conllu \
  --out runs/marathi_demo/pred.conllu
```

`target_tokenized.conllu` must contain one word per UD integer-token line, but gold HEAD/UPOS are not required. The HF backend emits only basic heads, UPOS and relations; others are `_` (lemmatization and morphology should be done by an additional module).

## Preparing Stanza data

```bash
udtl export-stanza --train runs/marathi_demo/training_data/train.conllu \
  --dev runs/marathi_demo/training_data/dev.conllu \
  --out stanza_data --language mr --treebank Silver
```

This writes standard `mr_silver-ud-{train,dev}.conllu` files under `UD_mr-Silver/`. To train Stanza, follow Stanza's **version-specific** official preparation commands, using `UDBASE`, `DATA_ROOT`, `POS_DATA_DIR`, and `DEPPARSE_DATA_DIR` as required. Illustrative commands from the Stanza docs are:

```bash
python -m stanza.utils.datasets.prepare_pos_treebank UD_mr-Silver
python -m stanza.utils.training.run_pos UD_mr-Silver --use_bert --bert_model xlm-roberta-base
python -m stanza.utils.datasets.prepare_depparse_treebank UD_mr-Silver --gold
python -m stanza.utils.training.run_depparse UD_mr-Silver --use_bert --bert_model xlm-roberta-base
```

**These Stanza training commands are not automated/tested here.** The particular Stanza version may require different naming or data-preparation flags, and unknown treebank IDs may require Stanza configuration changes. Reference: https://stanfordnlp.github.io/stanza/training_and_evaluation.html

## PSG / constituency trees are a separate learning problem

UD's basic `HEAD` and `DEPREL` fields **do not uniquely determine a phrase-structure tree**. Stanza's constituency processor is supported for selected languages (not all 22 scheduled languages). If a pretrained constituency model for the target language exists:

```bash
udtl constituency --lang en --input examples/EN_TARGETS.jsonl --out english_predicted.mrg
```

This generates **predicted** `.mrg` trees (silver phrase structure); they are not equivalent to human-annotated constituency treebanks. True constituency **fine-tuning** requires compatible bracketed constituency *training trees* and Stanza's constituency trainer, with independent bracketing gold test data. Generating reliable constituency annotations for Hindi/Marathi/etc from UD alone is **not implemented**. See https://stanfordnlp.github.io/stanza/new_language_constituency.html

## Recommended experimental design

1. Fix a small **real target-language gold** dev and test set *before* bootstrapping. Test must remain untouched during all modeling decisions.
2. Train/evaluate source-free zero-shot baseline, projection-only baseline, Trankit/HF silver-trained model, and mixed silver+gold model separately.
3. Use `min_quality` ablations (e.g. 0.30, 0.50, 0.70) and plot coverage versus gold LAS.
4. Run multiple random seeds and confidence intervals on the final independent test set.
5. Compare manual versus SimAlign/awesome-align links, test cases with many-to-one/missing alignment, and use genuine target morphology if possible.
6. Report UPOS, UAS, LAS, MLAS/BLEX (official evaluator), tokenization and sentence segmentation for *raw text* pipelines. HF backend currently assumes pretokenized input.

## Hardware time plan (experimental only)

Pilot 500–2,000 translated sentences; align, project, train 1–2 epochs, evaluate on target gold. Adjust to 5k–15k if it improves LAS, then compare languages. A reasonable *goal* on a single modern GPU is one target language within hours to a few days, but actual runtimes depend on model download/access, GPU VRAM, CPU graph projection, sequence length, data volume and training epochs. No all-22-days guarantee.
