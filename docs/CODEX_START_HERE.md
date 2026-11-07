# Starting from Bhaashik Codex

**Destination:** `bhaashik/TreebankCreator` (private GitHub repository). **Current source:** UD Treebank Lab v0.2.0.

Read `AGENTS.md`, `README.md`, `docs/CONVERSATION_HANDOFF.md`, `docs/BILINGUAL_TRAINING.md`, and `docs/KAGGLE_DUAL_T4.md` before changing the trainer. The Git history contains separately committed v0.1.0, v0.1.1, v0.2.0 source snapshots; `docs/RELEASE_PROVENANCE.md` lists archive checksums.

Quick CPU verification on Ubuntu or in Codex:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
bash scripts/demo.sh
```

GPU extension (choose a CUDA/torch wheel supported by the actual hardware first):

```bash
python -m pip install -e '.[bilingual]'
udtl cache-bilingual --config configs/bilingual_marathi_demo.yaml
udtl train-bilingual --config configs/bilingual_marathi_demo.yaml
udtl predict-bilingual --config configs/bilingual_marathi_demo.yaml
```

On Kaggle T4×2, after staging target corpora and caches, use:

```bash
udtl launch-bilingual --configs configs/generated/*.yaml --gpus 0,1 --dry-run
udtl launch-bilingual --configs configs/generated/*.yaml --gpus 0,1
```

The first real GPU test must confirm model download, stable 1/2-GPU memory use, checkpoint replay, and correct output quality. Use independently verified target gold annotations; the bundled 3-sentence demo is not a benchmark.

**Conversation continuity:** This repository includes a condensed technical handoff rather than a native ChatGPT/Codex chat transfer. The original ChatGPT thread remains accessible from the user's ChatGPT history and can be referenced manually from a new Codex task.
