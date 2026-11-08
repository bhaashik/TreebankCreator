# TreebankCreator on NVIDIA TITAN V (Ubuntu)

The Titan V is a **Volta sm_70** device with 12 GB HBM2. It can run
SimAlign XLM-R / mBERT inference and the frozen-XLM-R bilingual parser. Its
main challenge in current installations is **binary compatibility**, not lack
of CUDA acceleration. A conventional `pip install torch` may select a wheel
built for CUDA 13, which no longer supports Volta. PyTorch's CUDA 12.6 wheels
retain `sm_70` through the 2.14 release line.

## Prerequisites

- An Ubuntu desktop with a working NVIDIA driver (`nvidia-smi -L` lists
  `TITAN V`). A compatible driver must already be installed; this script
  does not change drivers, apt packages or system CUDA libraries.
- Python **3.10–3.12** (prefer 3.12), with working `venv` and `pip`.
  If Ubuntu's default Python is newer, choose a dedicated compatible
  Python interpreter, e.g. one from your existing Conda installation.
- Internet access for the initial PyTorch, Python-package and model downloads.
  Hugging Face weights are downloaded only on first model use; cached weights
  can be reused when running offline.

A system-wide CUDA Toolkit installation is generally **not required** for the
official PyTorch wheel. `nvidia-smi` must work, however, and the driver must
support the CUDA runtime in that wheel. When PyTorch reports an inadequate
NVIDIA driver, update the driver through your distribution rather than
installing random CUDA packages in the project environment.

## Installing and validating

From the project root:

```bash
# Safe read-only preview
bash scripts/setup_titanv.sh --dry-run

# Creates an isolated .venv-titanv; reuses it on later calls
bash scripts/setup_titanv.sh --python python3.12

# Activate and enforce the torch pin for FUTURE pip operations too
source scripts/activate_titanv.sh
python scripts/check_titanv.py

# Optional development environment
bash scripts/setup_titanv.sh --with-tests
python -m pytest -q
```

Use `--check` to validate the existing environment **without installing**:

```bash
bash scripts/setup_titanv.sh --check
```

If using a different venv location, pass `--venv /path/to/venv` during setup,
then set `TREEBANK_TITANV_VENV=/path/to/venv` before sourcing
`scripts/activate_titanv.sh`.

The profile installs **torch==2.14.1 from the `cu126` index**, then uses
`constraints/titanv.txt` to pin the PyTorch version when installing
`ud-treebank-lab[align,bilingual]`. The activation helper exports
`PIP_CONSTRAINT`, so subsequent normal pip installs in the active venv cannot
silently select PyTorch 2.15+ or a different public version. As with any pip
constraint, a direct `--force-reinstall` from the wrong wheel index could still
replace the local CUDA variant; rerun `check_titanv.py` after such changes.

**Do not run the generic `setup_vm.sh` or unqualified `pip install torch` in
this environment.** Both are for other machines. Existing Apertium + lttoolbox
are installed separately using `scripts/install_linguistic_tools.sh --apertium`
when needed; target-language analyzers remain distinct resources.

### Optional AWESOME-align

```bash
bash scripts/setup_titanv.sh --with-awesome
```

The legacy upstream AWESOME-align code may not work with every current
Transformers/Python package combination. If its installation fails, keep
SimAlign in `.venv-titanv` and test AWESOME-align in a separate compatible
environment. The installer will **not** relax the PyTorch constraint to make
AWESOME-align installable.

## Corpus configuration and conditional reuse

The example [Titan V config](../configs/corpus_titanv_example.yaml) selects
SimAlign (`xlmr`, `itermax`, `device: cuda`) only when its alignment data is
missing. Supplied alignment or morphology files are reused or imported and
neither model inference nor setup is invoked for them.

```bash
udtl prepare-resources --config configs/corpus_titanv_example.yaml --dry-run
udtl prepare-resources --config configs/corpus_titanv_example.yaml
udtl refine-corpus --config configs/corpus_titanv_example.yaml
```

In the YAML, `setup.on_missing: error` is deliberate: an absent CUDA-ready
SimAlign is reported clearly instead of creating another environment from a
running Python process. **Run the Titan V installation profile explicitly
first**, then activate it before running `udtl`.

## Rough runtime for 17,000 sentence pairs in one language

These are **planning ranges**, not results measured on this Titan V or this
codebase. They assume short-to-medium sentences, GPU model inference and
locally cached weights. First-time downloads can add time.

| Stage | Approximate time | Scope |
|---|---:|---|
| SimAlign, XLM-R base, IterMax | 15–90 min | GPU initial alignment |
| Apertium + CPU corpus refinement | 5–30 min | Analysis and frequency/affix statistics |
| AWESOME-align mBERT inference | 20–90 min | Optional comparison |
| AWESOME-align fine-tuning (1–3 epochs) | 1–5 h | Optional experimental training |
| Cached embeddings + bilingual parser training | 2–8 h | Depends heavily on epochs/model settings |

The original SimAlign paper reports **22 seconds per 500 English–German pairs
with XLM-R on a GTX 1080**, which extrapolates to ~12.5 minutes for 17,000
pairs in its specific benchmark. Throughput on Hindi/Indic data and the current
sentence-by-sentence implementation may be lower. Do not treat this as a
Titan V timing measurement.

**A first full language experiment (SimAlign + corpus refinement + bilingual
parser) can be budgeted at about 3–10 hours**, excluding optional AWESOME
fine-tuning. These estimates are intentionally broad; benchmarking a sample
will produce a much more useful runtime for your dataset.

### Simple benchmark using your corpus

Time a run on a representative **500- or 1,000-sentence subset** of the
source CoNLL-U and matching target JSONL, using the same config and backend:

```bash
/usr/bin/time -p udtl prepare-resources --config configs/your_small_titanv.yaml
```

Do not time a previously cached result: `prepare-resources` deliberately
skips work when its outputs are already present. To extrapolate from a fresh
sample, use `T_full ≈ T_sample × 17000 / N_sample`, allowing for model
initialization, downloads, caching and larger sentences. Keep the sample
sentences representative and include source and target tokenization checks.

## Verification scope

- CPU-only automated tests cover safe preflight, pinning, incompatible wheels,
  incorrect GPU identity and the basic CLI; they do not download CUDA wheels.
- GPU kernel execution, driver compatibility and actual Titan V throughput
  **must be checked on the desktop** with `python scripts/check_titanv.py`.
- Confirm independent target UD quality; higher alignment agreement is not
  evidence of improved LAS by itself.

## References

- [Official PyTorch previous versions: 2.14.1 CUDA 12.6](https://pytorch.org/get-started/previous-versions/)
- [PyTorch Volta legacy binary support notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
- [SimAlign, Findings of EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.147/)
