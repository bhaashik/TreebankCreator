# Bilingual Hindi-UD-guided dependency parser (v0.2.0)

## Executive distinction

**This is a new experimental bilingual trainer, NOT Trankit's `TPipeline`.** It
uses a frozen, shared Hugging Face `AutoModel` encoder such as XLM-R-base, followed
by trainable Hindi-source dependency-graph encoding, target-to-Hindi alignment-
guided cross-attention (factorized bilinear scoring), and a **target-only**
biaffine arc and relation parser. The Hindi gold tree is passed *during* both
training and source-guided annotation/inference. There is no Hindi parsing head.

These components are implemented and CPU-tested, but **they have not yet been
benchmarked for 17,000 sentence pairs, run on a Kaggle T4, Titan V, or compared
against Trankit on independent gold test data**. Earlier quoted 2–12-hour ranges
are planning estimates, not measured performance. This experimental model is not
a drop-in Trankit checkpoint and has no claim of equivalent accuracy.

## Architecture

```text
Hindi UD sentence + GOLD heads       Target pretokenized sentence
          |                                   |
  shared frozen XLM-R                  same frozen XLM-R
          |                                   |
 cached word representations       cached word representations
          |                                   |
  trainable source-graph               target projection
  parent/child encoder                      |
          +------------+--------------------+
                       |
     automatic alignment (soft, imperfect; 0-based links)
                       |
   1–2 trainable cross-attention layers, alignment bias + dropout
                       |
              target-only representations
                       |
        biaffine target arc and relation scores
                       |
            single-root tree decoding (MST)
                       |
             target silver CoNLL-U output
```

**XLM-R weights never change.** Its word-level first-subtoken vectors can be
cached in sharded float16 tensors for reuse across epochs. All trainable
parameters (including source-tree encoder, parallel layers and target parsing
head) reside outside XLM-R and are optimized in float32.

The cross-lingual query/key attention score is a factorized bilinear function
of target and Hindi representations. It combines a learned cross-language
score with a log soft-alignment bias. A reliability-conditioned gate controls
information transfer. The final target-to-target arc scorer is fully biaffine.

### What "noise-aware" does and does not mean

The Hindi gold dependency tree is reliable *as a Hindi tree*, but automatic word
links and syntactic projections are uncertain. The system:

1. Accepts **hard links** (existing `udtl align` JSONL), scored links or a
   target-by-Hindi matrix. Hard links receive a conservative **uncalibrated**
   heuristic weight (default 0.55), not certainty.
2. Row-normalizes candidate correspondences, penalizes ambiguous many-to-one
   links and assigns zero weight to unaligned target words.
3. Projects source arcs to a **soft joint distribution** over target heads and
   dependency relations; excludes target self-loops.
4. Uses confidence-weighted soft supervision, **not hard projected tree gold**.
5. Randomly drops some alignment rows during training so the model has to use
   target sentence context when links are absent/unreliable.
6. Enforces a single root and no cycles during final tree decoding.

**Limit:** confidence heuristics and dropout cannot guarantee correcting
*systematic* alignment and projection errors. Most learning signals still come
from the same aligned Hindi source. Better soft predictions are not automatically
correct syntax. Always compare against independently annotated target-language
sentences, including cases of reordering, affixation, null alignment, dropped
arguments and many-to-many correspondence. Do not report weak dev loss or
agreement with projected Hindi as LAS.

## Data format and indexing

Source: regular UD CoNLL-U with correct Hindi `HEAD`, `DEPREL`, plus `sent_id`.
Only integer IDs 1...N are used; MWT and empty node lines are excluded. **Do
not train on Hindi test sentences that are intended for source-side evaluation.**

Target: JSONL with exactly the same `id` as Hindi `sent_id`. Use **UD word-level
segmentation** rather than arbitrary whitespace to match downstream CoNLL-U.

```json
{"id":"hi000001","text":"...","tokens":["...","...","..."]}
```

Alignment: one line per target sentence, with **zero-based** indices into Hindi
words and target words respectively. Three accepted representations:

```json
{"id":"hi000001","pairs":[[0,0],[1,0],[2,1]]}
{"id":"hi000002","links":[{"src":0,"tgt":0,"score":0.83},{"src":1,"tgt":1,"score":0.65}]}
{"id":"hi000003","matrix":[[0.9,0.1],[0.2,0.8],[0.0,0.0]]}
```

The matrix has **target words in rows and Hindi words in columns**. A score
supplied by an automatic aligner is not assumed to be calibrated. Uncertain or
missing links are allowed. The existing `udtl align` command outputs hard
`pairs` compatible with this trainer. To use calibrated probabilities, add
scored links from a separate aligner/calibration process.

## One-language run

From an extracted v0.2.0 project on Ubuntu, Colab, Kaggle or Lightning:

```bash
python -m pip install -e '.[bilingual]'

# Test-only 3-pair Hindi->Marathi config; downloads XLM-R-base.
udtl cache-bilingual --config configs/bilingual_marathi_demo.yaml --side both
udtl train-bilingual --config configs/bilingual_marathi_demo.yaml
udtl predict-bilingual --config configs/bilingual_marathi_demo.yaml
udtl validate runs/bilingual_mr/refined_silver.conllu
```

For a real experiment, create your own config file using this format:

```yaml
language: mr
source_ud: /data/hindi_train.conllu
targets: /data/mr_targets.jsonl
alignments: /data/mr_alignments.jsonl
source_cache: /cache/shared_hindi
target_cache: /cache/mr
output_dir: /runs/mr
model: xlm-roberta-base
epochs: 6
batch_size: 4
accumulate: 4
save_steps: 100
max_words: 70
max_subtokens: 256
hidden: 192
layers: 2
alignment_dropout: 0.15
alignment_bias: 0.6
min_strength: 0.02
# optional: gold_dev: /data/mr_gold_development.conllu
```

`source_cache` is shared by all 13 language experiments **only if the exact
Hindi token sequence, pretrained model, word limits and cache settings match**.
`target_cache` and `output_dir` must be distinct for every language.

`gold_dev` must contain genuinely independent gold target UD for a **subset of
the model's deterministic held-out development IDs**; the importer rejects
inconsistent tokens, IDs, or invalid trees. `gold_dev` must not be the target
language test set. Without gold dev, checkpoint selection uses only weak
projection dev loss and the final output must be evaluated externally.

### 13-language manifest

To avoid writing 13 YAML files manually, make a JSONL manifest of records with
`language`, `targets`, `alignments` (and optionally `gold_dev`), e.g.
`examples/bilingual_manifest_example.jsonl`. For Kaggle paths, edit the input
dataset slug to match your own dataset.

```bash
udtl make-bilingual-configs \
  --manifest data/languages.jsonl \
  --source data/hindi_train.conllu \
  --config-dir configs/generated \
  --cache-root caches/embeddings \
  --runs-root runs/bilingual \
  --model xlm-roberta-base --epochs 6 --batch-size 4
```

The generated YAMLs share a single Hindi embedding cache, while keeping a
different target embedding cache, training directory, checkpoint set and log
for each language.

### Dual GPUs on Kaggle or Ubuntu

```bash
# Dry run: no GPU required; inspect plan and paths.
udtl launch-bilingual --configs configs/generated/*.yaml \
  --gpus 0,1 --dry-run

# Actual execution: prepares the shared Hindi cache in a separate process,
# then launches at most two independent language jobs, one per GPU.
udtl launch-bilingual --configs configs/generated/*.yaml --gpus 0,1
```

Each worker runs in its own Python process with `CUDA_VISIBLE_DEVICES=0` or `1`.
Inside a worker, the selected device is `cuda`, thus each worker sees only
one T4. This **does not combine** two 16-GB T4s into a 32-GB device. Other
resource limits (CPU cores, RAM, disk, quota and 12-hour session) remain shared.
On a Titan V desktop, use `--gpus 0` (one language per worker).

With 13 configs, the launcher automatically queues the next language when a
GPU slot becomes free. Failed jobs are reported, and other jobs may complete.
The command returns a nonzero exit status if any worker failed.

**Check your Kaggle quota:** official limits can vary and GPU availability is
not guaranteed. A two-GPU notebook is not necessarily twice as fast if its
four shared CPU cores, data I/O, or RAM become bottlenecks.

## Checkpoints and resumption

A checkpoint `last.pt` is written **after optimizer updates** (not halfway
through gradient accumulation), every `save_steps` updates and after each
epoch. The checkpoint contains the trainable model, optimizer, global update
count, epoch, batch offset, Python/PyTorch RNG state, dev metric and labels.
`best.pt` stores the best dev checkpoint, judged by gold dev LAS if available,
otherwise by the weak-projection dev loss. `done.json` marks a finished run.

Re-run **the identical `udtl launch-bilingual ...` command** to skip completed
languages and resume interrupted ones. The most recent unsaved work may need
to be repeated. Input/config checksums prevent silent reuse after changing
training settings or corpora. The device (GPU or CPU) may change on resumption.

The `--max-steps N` flag on `train-bilingual` deliberately pauses after N
optimizer steps; useful for verifying checkpoint/resume behavior. It is not
required for normal operation.

Kaggle's `/kaggle/working` is *not durable across separate sessions* unless
outputs are saved and then reintroduced as notebook inputs. Save versions or
upload checkpoints to your own cloud storage, then **restore files to the same
cache/run paths** before invoking the launcher again. Missing embedding shards
are recomputed. Restoring only `last.pt` without a matching dataset, config and
labels will be refused. Load PyTorch checkpoint files only from trusted sources.

## Export and evaluation

```bash
udtl predict-bilingual --config configs/generated/hindi_to_mr_bilingual.yaml \
  --out runs/bilingual/mr/refined_silver.conllu
udtl validate runs/bilingual/mr/refined_silver.conllu
udtl evaluate --gold data/mr_independent_gold.conllu \
  --pred runs/bilingual/mr/refined_silver.conllu \
  --out runs/bilingual/mr/gold_metrics.json
udtl visualize --input runs/bilingual/mr/refined_silver.conllu \
  --out runs/bilingual/mr/trees.html
```

The evaluator requires **matching tokenization and sentence IDs**. The exported
UD contains target dependency heads and relations, but **UPOS currently comes
from weak alignment-weighted Hindi tag votes**, not a supervised target UPOS
head. LEMMA and FEATS remain `_` until a genuine target morphology component
(e.g. Apertium, vetted morphological model) is added. This is not a validated
complete morphosyntactic UD treebank. Existing Apertium/rule preprocessing can
be run independently, but is **not yet coupled into the neural loss**.

Recommended ablation: compare (i) hard projection + ordinary Trankit,
(ii) this model without alignment bias, (iii) with bias, (iv) with dropout,
(v) with independent target-language morphological constraints, after that
component is implemented. Include at least 200–500 independent target gold
sentences per language where practical, with a separate untouched test set.

## Resource notes

For 17k sentences at 25 words each, a float16 XLM-R-base word embedding cache
requires around **650 MB per side**, plus shard, metadata and model files. A
13-language target-cache collection can consume many GB. The same Hindi cache
can be shared across languages. Cached embeddings load into CPU RAM, then a
minibatch is transferred to its assigned GPU. The full pretrained transformer
is **not loaded during neural training** after caches are complete.

**Time:** no Titan V/T4 benchmarking has been done for this implementation. The
previous 2–10 h (Titan V) / 3–12 h (T4) estimates are targets to test with a
2k-pair pilot. The precise runtime depends on sentence length, projection
coverage, caching time, local I/O, batch size, source-graph layers and other
resource contention. Do not schedule a 12-hour Kaggle notebook on estimates
alone: verify by measuring 1–2 epochs and using checkpoints.
