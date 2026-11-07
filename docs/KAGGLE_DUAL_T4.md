# Running two target-language jobs simultaneously on Kaggle Free T4×2

Read [BILINGUAL_TRAINING.md](BILINGUAL_TRAINING.md) first for file formats,
actual neural architecture, weak annotation caveats and checkpoint semantics.

## 1. Create the notebook

1. Sign in to Kaggle and select **Code > New Notebook**.
2. In notebook settings choose **Accelerator: GPU T4×2** (if offered). Kaggle's
   September 2026 announcement says P100 has been retired, leaving T4×2.
3. Enable **Internet** only if permitted/available and needed for installing
   dependencies or downloading `xlm-roberta-base`. If internet is not available,
   upload a Hugging Face snapshot as a private Kaggle Dataset and set `model`
   in both configs to its local directory.
4. Execute `!nvidia-smi -L` or `torch.cuda.device_count()`: there should be
   **two T4 devices of 16 GB each**, not 32 GB combined.

Official overview: https://www.kaggle.com/docs/notebooks ;
P100 retirement: https://www.kaggle.com/product-announcements/735239

## 2. Stage the project and input dataset

Create a **private Kaggle Dataset** with this project ZIP and another dataset
with precomputed input files. Example input layout:

```text
/kaggle/input/my-ud-data/
  hindi_train.conllu
  languages.jsonl
  mr_targets.jsonl
  mr_alignments.jsonl
  bn_targets.jsonl
  bn_alignments.jsonl
  ...
```

Every target JSONL and alignment JSONL uses the Hindi `sent_id` and exact UD
word indices. The manifest can contain 13 records. `alignments` may be the
existing `udtl align` hard JSONL (uncertain weak links), or scored alignments.

Extract the ZIP into `/kaggle/working`, change into the project directory, then
install in Kaggle's Python environment:

```python
from pathlib import Path
import os, subprocess, zipfile
root = Path('/kaggle/working')
archive = next(Path('/kaggle/input').rglob('ud-treebank-lab-v0.2.0.zip'))
with zipfile.ZipFile(archive) as z:
    z.extractall(root)
os.chdir(root / 'ud-treebank-lab')
subprocess.run(['python','-m','pip','install','-e','.[bilingual]'], check=True)
subprocess.run(['python','-m','ud_treebank_lab','--version'], check=True)
```

**Kaggle notebook code:** See `notebooks/Kaggle_Dual_T4.ipynb`, included in the
ZIP. It has an input-dataset path cell, config generation, dry run and launch.

## 3. Generate configs once for all target languages

```bash
udtl make-bilingual-configs \
 --manifest /kaggle/input/my-ud-data/languages.jsonl \
 --source /kaggle/input/my-ud-data/hindi_train.conllu \
 --config-dir /kaggle/working/ud-treebank-lab/configs/generated \
 --cache-root /kaggle/working/bilingual_cache \
 --runs-root /kaggle/working/bilingual_runs \
 --epochs 6 --batch-size 4
```

You may edit configs to lower cache batch size on out-of-memory errors.
For each run the launcher will first cache shared Hindi embeddings once on
GPU 0, *release its process*, and then run two target jobs simultaneously.

```bash
udtl launch-bilingual --configs configs/generated/*.yaml --gpus 0,1 --dry-run
udtl launch-bilingual --configs configs/generated/*.yaml --gpus 0,1
```

The launcher starts the next queued language when a worker finishes. Logs:
`/kaggle/working/bilingual_runs/<lang>/gpu_worker.log`. Check real-time:
`!nvidia-smi` and `!tail -n 20 /kaggle/working/bilingual_runs/mr/gpu_worker.log`.

For a one-language pilot, run `udtl train-bilingual --config ... --device cuda`.
The shared Hindi cache alone takes storage; target caches for 13 languages may
not all fit in Kaggle's 20-GB saved-output area. Archive/cache only the two
active languages, or persist caches in successive datasets or Google Drive /
Dropbox and restore them when needed.

## 4. Preserving results across Kaggle sessions

Kaggle `/kaggle/working` is session-local unless saved as notebook output by
**Save Version** / **Save & Run All** and then reattached to the next session.
A Save & Run All run must finish within the documented session limit, so do
not rely on it to protect mid-run work. Save during the active session by
publishing a new private dataset or by copying files to Google Drive/Dropbox
(using `rclone` credentials from Kaggle Secrets). Do not publish OAuth tokens
in notebooks or datasets.

On a new session, **restore checkpoints and caches** from the saved dataset
under `/kaggle/input/...` to `/kaggle/working/bilingual_runs` and
`/kaggle/working/bilingual_cache`. Then run the same config-generation and
launcher commands; config paths and input files must match the checkpoint's
training signature.

Example transfer commands if you configured rclone beforehand:

```bash
rclone copy /kaggle/working/bilingual_runs drive:UDTreebankLab/runs -P
rclone copy /kaggle/working/bilingual_cache dropbox:UDTreebankLab/caches -P

# After restarting: restore before invoking the launcher
rclone copy drive:UDTreebankLab/runs /kaggle/working/bilingual_runs -P
rclone copy dropbox:UDTreebankLab/caches /kaggle/working/bilingual_cache -P
```

These are **copies**; they do not delete unmatched files at the destination.
Rclone remotes named `drive` and `dropbox` must have been configured in advance.
Drive mounts on Colab are not available automatically inside Kaggle.

## 5. Quota and session limits

Kaggle's notebook documentation lists T4×2 with 4 shared CPU cores, roughly
29 GB RAM, 20 GB saved working space and up to 12 hours per session. GPU
quota/availability are variable: check your own Kaggle account settings. Two
concurrent workers share the Kaggle instance's CPU and I/O and do not promise
an exact factor-of-two improvement.

If one job fails with CUDA out-of-memory, reduce per-job `batch_size`,
`cache_batch_size`, `hidden`, or `max_words` and **use a new run directory or
start again with consistent checkpoints** when changing training settings.

## 6. Quick validation

```bash
udtl predict-bilingual \
 --config configs/generated/hindi_to_mr_bilingual.yaml \
 --out /kaggle/working/bilingual_runs/mr/refined_silver.conllu
udtl validate /kaggle/working/bilingual_runs/mr/refined_silver.conllu
```

This creates *source-guided silver UD* only for the parallel sentences. It is
**not** a standalone Marathi/Bengali parser for arbitrary new sentences; that
requires target-only distillation/fine-tuning in a separate step.
