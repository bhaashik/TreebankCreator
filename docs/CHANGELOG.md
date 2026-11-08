## v0.3.4 — Dedicated NVIDIA Titan V / Volta CUDA profile

- New isolated and opt-in `scripts/setup_titanv.sh` profile using the official
  PyTorch 2.14.1 CUDA 12.6 wheel, with a pip constraint pin retained after activation.
- Added `scripts/check_titanv.py` with GPU identity, compute capability,
  compiled `sm_70` kernels and real CUDA matmul verification.
- Added `scripts/activate_titanv.sh`, sample Titan V corpus YAML, and
  practical runtime estimates documented with their measurement limitations.
- Added CPU-only tests of installation preflight, constraints and incompatible
  CUDA/GPU rejection. Real Titan V throughput and downloads not validated in CI.
- Existing v0.3.3 installers and supplied-file priority remain unchanged.

# v0.3.3 — explicit neural-alignment and Apertium setup (2026-11-08)

- Added `scripts/install_linguistic_tools.sh`, an idempotent opt-in installer
  and `--check`/`--dry-run` preflight for SimAlign, optional legacy AWESOME-align,
  Apertium + lttoolbox, and optional language analyzer packages.
- Added optional `backend: awesome` and `backend: auto` generation through the
  AWESOME CLI with strict token/sentence-index validation and JSONL output.
- Added conditional setup documentation and tests; existing supplied-resource
  behavior remains unchanged. External actual installs are not exercised in CI.

## v0.3.2 — Conditional resource availability, import, generation and setup

- One corpus YAML can select already-available JSONL/raw files, automatically convert
  existing AWESOME/GIZA A3 and Apertium streams, or explicitly generate absent inputs.
- `mode: auto` with `available: auto|true|false` and `if_missing: generate|skip|error`;
  omitted resource default is still `skip` (fully backward compatible).
- `udtl prepare-resources --config ... [--dry-run|--force]` independently prepares
  the same resources used by `udtl refine-corpus` / `corpus_preprocess`.
- Optional setup check for installed Python aligner / lt-proc / analyzer bin;
  install command only if explicitly configured and setup is absent.
- Content signatures, idempotent reuse, audit manifest, safe managed outputs,
  and no implicit modification of supplied data. Regression tests for each path.

## v0.3.1 — Standalone import of existing alignment/Apertium files

- Added `udtl import-alignments`, `udtl import-apertium` and YAML-controlled
  `udtl import-resources`: no external aligner, analyzer or GPU needed.
- Auto-detect AWESOME-align Pharaoh vs genuine GIZA++ A3 by content, normalize
  direction and 0/1-based indices, preserve NULL and optional alignment probabilities.
- Parse native Apertium ambiguous lexical streams (sentence/token/flat/word-list),
  preserve all readings and raw tags, validate against target sentence tokens.
- Added atomic/non-overwriting JSONL outputs, example files, documentation,
  and offline regression/integration tests with corpus refinement.

## v0.3.0 — Corpus-level evidence and resource-aware preprocessing

- Added opt-in `udtl refine-corpus --config` CPU stage with explicit `supplied`,
  `skip` (default), and selected `generate` modes for initial alignments and
  Apertium. UniMorph TSV can be supplied, but is never auto-generated.
- Added sentence-global bilingual lexical co-occurrence and frequency signals,
  high-precision anchors, conservative reranking with NULL/unaligned handling,
  uncertain 1:n/n:1 alternatives, ambiguous morphological candidate lattices,
  and contrast-gated function-word/affix hypotheses.
- Added reproducible input fingerprinting, atomic output regeneration, full
  per-token correction provenance, automatic reuse of supplied resources, and
  optional integration into the existing bilingual trainer/Kaggle manifest.
- Added Bhojpuri shifted-name and cross-script tests and a CPU-only Marathi demo.
- No new mandatory ML dependencies. No claim of improved target LAS without
  independent gold evaluation; source XLM-R remains frozen in the prior trainer.

# v0.2.0 — bilingual source-guided parser and dual-GPU launcher

- New frozen-XLM-R Hindi–target neural dependency parser with target-only biaffine head.
- New trainable graph-aware source representations, cross-attention, confidence-gated soft projection supervision, alignment-row dropout.
- Resumable sharded embedding caching and step/epoch optimizer checkpoints.
- Multi-language JSONL manifest generator and dual-GPU subprocess scheduler for Kaggle T4×2 and Ubuntu desktops.
- Kaggle/Colab-friendly documentation and runnable Kaggle notebook.
- Offline model/gradient/training/resume/dual-process tests (no online GPU benchmark).

# Changelog

## v0.1.1 — 2026-11-06

- Added complete Lightning AI Pro Studio setup, billing/sleep/resume and headless OAuth guide (`docs/LIGHTNING_AI_PRO.md`).
- Added Studio-managed-environment installer (`scripts/setup_lightning.sh`) separate from the generic Linux venv installer.
- Added non-deleting rclone transfers for Google Drive and Dropbox (`scripts/cloud_io.sh`).
- Added a YAML preset for cloud-staged corpora and local model snapshots (`configs/hindi_to_marathi_lightning_cloud.yaml`).
- Added resolution of explicit local transformer/translation model directories relative to the project YAML file.
- Added automated shell-helper tests and compatibility/security notes; no changes to the UD projection algorithm.
- Full GPU training and real external cloud OAuth connections remain untested in this environment.

## v0.1.0 — 2026-11-05

- Initial research prototype with CoNLL-U projection, data preparation, checkpointable HF parser, Trankit adapter, evaluation and visualization.
