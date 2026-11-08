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
