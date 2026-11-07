# AGENTS.md

This repository follows a `src/ud_treebank_lab/` Python package layout.
Do not introduce new dependencies in the default installation unless necessary.
Keep optional ML backends lazy-loaded; the no-GPU smoke test must run offline.
Maintain UD CoNLL-U 10-column syntax; never silently train on target gold test data.
When changing projection behavior, add tests for cycles, 0/1/many-to-many alignments,
root constraints, morphology ambiguity, interruption/resume, and data leakage.
Document substantive changes in README.md and docs/.
Never describe projected annotations as validated gold data.

Keep Lightning Studio installation separate from generic Ubuntu VM venv setup.
Use scripts/cloud_io.sh with rclone COPY, never destructive sync by default,
and never package OAuth config or Hugging Face tokens with the source archive.

Bilingual extension: see docs/BILINGUAL_TRAINING.md and docs/KAGGLE_DUAL_T4.md. Cached XLM-R weights are *never optimized*; Hindi UD remains the source branch with no Hindi parser. Keep independent target gold evaluation and do not call projection agreement LAS. Do not silently treat hard word-alignment links as certain.

Codex handoff: read docs/CONVERSATION_HANDOFF.md, docs/CODEX_START_HERE.md, and docs/RELEASE_PROVENANCE.md before modifying cross-lingual components. Original release commits are historical snapshots; do not rewrite them.
