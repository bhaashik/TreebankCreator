# UD Treebank Lab

**Resumable cross-lingual Universal Dependencies (UD) treebank bootstrapping, using alignment projection, optional finite-state morphology, grammar-informed soft constraints, and trainable multilingual parsers.**

Status: **research prototype v0.1.1, runnable offline end to end on the included demonstration.** The core projection, splitting, evaluation, visualization, and tests were run. Model downloads and GPU fine-tuning were **not** run in the development environment; verify each optional backend on your VM before launching a large project.

This tool **does not use LLMs to generate annotations**. An explicitly optional, conventional neural machine-translation adapter (IndicTrans2) can produce target *sentences* from a UD source treebank; it does not produce trees. Real parallel data may be supplied instead.

## First success in five commands (generic Ubuntu VM, CPU, no model download)

Python **3.10–3.12 recommended**. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest
bash scripts/demo.sh
```

Open **`runs/marathi_demo/preview.html`** in a browser. You will find projected Marathi CoNLL-U, annotation metadata, training splits, metrics, and SVG dependency graphs.

The **3 demonstration sentences** have *human-authored alignments* and deliberately simple structures. High scores on the demo have **zero statistical significance** for general treebank accuracy.

## Lightning AI Pro: setup, cloud data/model storage, start/sleep/resume

**Detailed guide:** [`docs/LIGHTNING_AI_PRO.md`](docs/LIGHTNING_AI_PRO.md).

For **Lightning AI Studios**, use the existing managed Python environment: **do not run
`python -m venv` or `scripts/setup_vm.sh` in a Studio**. Once you have opened a CPU Studio
and uploaded/extracted this ZIP into `/teamspace/studios/this_studio/`:

```bash
cd /teamspace/studios/this_studio/ud-treebank-lab
bash scripts/setup_lightning.sh
# Switch to a GPU only when the models and data are ready.
```

If you have **Google Drive/Dropbox remotes configured with rclone** (or have copied your
existing `rclone.conf` from Ubuntu into `~/.config/rclone/`), use:

```bash
bash scripts/cloud_io.sh pull-data drive:UDTreebankLab/data
bash scripts/cloud_io.sh pull-models dropbox:UDTreebankLab/models
# On a completed/paused experiment:
bash scripts/cloud_io.sh push-runs dropbox:UDTreebankLab/backups/run_2026-11-06
```

Use one or both services; substitute your remote/folder names. This uses **non-deleting
`rclone copy`** and *local* working files rather than training through a network mount.
Downloaded local HF model snapshots can be referenced by relative YAML fields such as
`../assets/models/xlm-roberta-base`; this release resolves such model directories
relative to the configuration file. A ready-to-edit staging preset is provided at
`configs/hindi_to_marathi_lightning_cloud.yaml`. Save a full `runs/` backup before **Sleep** to
protect checkpoint state. Lightning Studios persist files/packages across Sleep,
but running training processes are stopped. See the guide for detailed browser/CLI
instructions, OAuth/headless setup, billing checks, and restart steps.

## Actual Hindi → target-language workflow

1. Get Hindi HDTB (and read the original treebank license):

   ```bash
   udtl fetch-ud --treebank UD_Hindi-HDTB --short-name hi_hdtb --out data/ud_hindi_hdtb
   ```

2. On a GPU VM, install optional backends:

   ```bash
   # Choose a torch/CUDA wheel matching the VM FIRST: https://pytorch.org/get-started/locally/
   pip install -e '.[align,translate,hf]'
   # Optionally install parser backend (in a compatible environment):
   pip install -e '.[trankit]'
   ```

3. Edit `configs/hindi_to_marathi.yaml` to use either your own target `targets` JSONL / `alignments` JSONL, **or** automatic `translate` plus SimAlign.

4. Run (including train when configured):

   ```bash
   udtl run configs/hindi_to_marathi.yaml
   # Repeat this identical command to resume after interruption.
   ```

5. Check `runs/marathi_full/pair_01/silver.conllu`, `metadata.jsonl`, `training_data/`, and the model directory. Provide an *independent manually annotated target-language* dev/test corpus before quoting real LAS.

The `configs/hindi_to_marathi.yaml` preset uses the checkpointable HF parser, which learns UPOS and dependency labels/heads from **XLM-RoBERTa-base** features. Trankit is a documented alternative, with a joint POS/morph/dependency head; its **internal** mid-epoch restart depends on the Trankit version. See `docs/TRAINING.md`.

**Important:** IndicTrans2 model downloads can require Hugging Face access conditions and may not be available on an anonymous VM. Its `trust_remote_code=True` setting executes third-party model code; use only models you trust. Model/torch/transformers compatibility may need a dedicated environment.

## What is implemented

| Function | Support |
|---|---|
| Source UD, one or multiple treebanks | CoNLL-U; pooling multiple source-target pairs in one YAML run |
| Sentence-level pairing | Stable `sent_id`; alternative: parse sentence-aligned text with Stanza |
| Word alignment | SimAlign mBERT/XLM-R, external alignments (e.g. awesome-align/eflomal), lexicon baseline, demo-only identity |
| Automatic translation | Optional IndicTrans2 Indic→Indic model; alternative pretranslated pairs |
| UD annotation transfer | UPOS, deprels/heads, partial morphology from target analyzer; language-specific soft grammar rules |
| Many-to-many / missing alignments | Source-anchor selection, ancestor projection, weak fallback attachments |
| Structural correctness | Maximum-weight *single-root arborescence*, checked against UD tree invariants |
| Source/target morphology | Limited target-side Apertium `lt-proc` FST analyzer bridge; external morphology JSONL |
| Synthetic sentences | Inflected, linguist-authored bilingual templates with fixed UD structure |
| Training | Trankit custom `posdep`; HF XLM-R encoder + POS/dependency head, optional encoder tuning; Stanza dataset export |
| Constituency trees | **Separate**, optional Stanza constituency *inference* for languages with a pretrained model; not derived from UD |
| Checkpoint/resume | Per-sentence JSONL for translation/alignment/projection; HF trainer step/optimizer/epoch checkpoints; Trankit completed-stage skip |
| Evaluation | Strict UPOS, UAS, LAS, FEATS/lemma exact accuracy, root, relation confusion, distance buckets; optional official UD scorer |
| Visualization | Offline HTML with inline SVG dependency arcs, projection-quality distribution and errors |
| Reproducibility | YAML configs, seed, input fingerprints, run manifest, source metadata, tests |

## Main commands

```bash
udtl --help
udtl languages
udtl fetch-ud --out data/ud_hindi_hdtb
udtl run examples/demo.yaml
udtl align --source SOURCE.conllu --targets TARGETS.jsonl --out LINKS.jsonl
udtl project --source SOURCE.conllu --targets TARGETS.jsonl \
  --alignments LINKS.jsonl --out SILVER.conllu --metadata PROVENANCE.jsonl
udtl prepare --silver SILVER.conllu --metadata PROVENANCE.jsonl --out train_data
udtl train-trankit --train train_data/train.conllu --dev train_data/dev.conllu --out model_trankit
udtl train-hf --train train_data/train.conllu --dev train_data/dev.conllu --out model_hf
udtl predict-hf --model-dir model_hf --input TARGET_INPUT.conllu --out PRED.conllu
udtl evaluate --gold GOLD_TEST.conllu --pred PRED.conllu --out metrics.json
udtl visualize --input PRED.conllu --metrics metrics.json --out view.html
udtl synthesize --templates examples/synthetic_templates.json --out synth_data
```

> `SOURCE.conllu` needs gold/basic UD heads. `TARGET_INPUT.conllu` for `predict-hf` needs tokenized words but does not require gold annotations. The official UD scorer is strongly recommended for publishable results, especially when tokenization differs.

## Documentation

- [`docs/INSTALL.md`](docs/INSTALL.md): generic Ubuntu/GPU VM installation and resumption.
- [`docs/LIGHTNING_AI_PRO.md`](docs/LIGHTNING_AI_PRO.md): complete Lightning AI Pro Studio creation, GPU start/sleep, Google Drive/Dropbox, remote model snapshots, backup and resume.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): dependency projection algorithm, constraints, confidence, reference approaches.
- [`docs/DATA.md`](docs/DATA.md): exact input formats; parallel corpora, Apertium, multi-source, synthetic templates.
- [`docs/TRAINING.md`](docs/TRAINING.md): Trankit, Hugging Face and Stanza, checkpoint guarantees, evaluation design.
- [`docs/EVALUATION.md`](docs/EVALUATION.md): gold vs silver, metrics, leakage, error analysis, offline visualization.
- [`docs/LANGUAGES.md`](docs/LANGUAGES.md): all 22 Eighth Schedule languages and IndicTrans codes; priorities and risks.
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md): scientific/engineering limitations and prioritized improvements.
- [`docs/REFERENCES.md`](docs/REFERENCES.md): papers, repositories, model cards, UD license notes.
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md): release changes, including Lightning deployment additions.

## Outputs

```text
runs/<target>/
  pair_01/
    target.jsonl               # when translation was requested
    alignments.jsonl           # when alignment was requested
    projection.stage.jsonl     # resume cache (do not delete during a run)
    silver.conllu              # UD basic dependencies (silver)
    metadata.jsonl             # confidence, coverage, arc provenance
    *.state.json               # input-aware stage signature
  pair_02/ ...                 # when configured
  training_data/
    train.conllu
    dev.conllu
    test.conllu                # only if independent gold test supplied
    split_report.json
  model_hf/
    last.pt                    # optimizer, model, RNG, epoch, batch position
    best.pt
    history.json
  model_trankit/               # if Trankit chosen
  run_manifest.json
```

## Research integrity, usage, and licenses

This is a *starter corpus builder*, not an automatic generator of gold-quality treebanks. A silver split is not a gold benchmark. **Neither a perfect demo result nor a held-out silver dev LAS establishes target-language accuracy.** Linguistically checked data, error review, appropriate licensing and independent gold evaluation are necessary.

**Hindi HDTB is CC BY-NC-SA 4.0** (noncommercial/share-alike restrictions). This project's own **code is MIT**; that does **not** relicense Hindi HDTB, translations, derived annotations, or other model/dataset artifacts. Review the licenses and redistribution terms of every source corpus and model before releasing derivative treebanks or checkpoints.

The selected model licenses and actual training runtime on a VM have **not** been independently verified by executing full GPU training here. The package is designed to make those checks and experiments straightforward.
