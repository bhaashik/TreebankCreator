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

Corpus refinement (v0.3): read `docs/CORPUS_REFINEMENT.md`. Never run SimAlign
or Apertium for supplied files. Default resource mode is skip. Never modify
supplied input files; do not claim inferred affix hypotheses are gold labels.
Keep a regression test for Hindi `ने` incorrectly paired with target
`शाहजेहन` and for repeated cross-script names. Preserve original raw
alignment provenance and soft uncertainty for training.

## External resource conversion

The separate `resource_import.py` stage converts preexisting AWESOME-align
Pharaoh/GIZA A3 alignments and native Apertium outputs to JSONL. Conversion
must never auto-run SimAlign or Apertium or overwrite supplied raw inputs.
Keep validation against source CoNLL-U and target token order. See
`docs/EXTERNAL_RESOURCE_IMPORT.md` and `tests/test_resource_import.py`.

### Resource lifecycle in v0.3.2
- Do not generate or install anything implicitly. `resources` omission means `skip`.
- Use `udtl prepare-resources --dry-run` to audit planned actions.
- Reuse user-supplied JSONL; convert user-supplied raw data into separate managed files.
- Only execute a configured `setup.install_command` if `setup.on_missing: install`,
  a generator is needed, and the executable/module/analyzer is missing.
- Never modify or overwrite an original input file. Keep resource and corpus manifests.
- Preserve old `mode: supplied|generate|skip` configurations.
