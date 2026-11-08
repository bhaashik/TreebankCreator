#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
python -m ud_treebank_lab refine-corpus --config configs/corpus_marathi_supplied.yaml
python - <<'PY'
from pathlib import Path
from ud_treebank_lab.utils import jsonl
out = Path('runs/marathi_corpus')
rows = list(jsonl(out/'alignments.refined.jsonl'))
assert len(rows) == 3
assert all('matrix' in r and 'pairs' in r and 'null_source' in r for r in rows)
print(f'Offline corpus refinement validated: {len(rows)} pairs; {out}')
PY
