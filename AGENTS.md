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
