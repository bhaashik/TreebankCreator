#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m ud_treebank_lab run examples/demo.yaml
python -m ud_treebank_lab validate runs/marathi_demo/pair_01/silver.conllu
python -m ud_treebank_lab evaluate --gold examples/marathi_gold_sample.conllu \
  --pred runs/marathi_demo/pair_01/silver.conllu --out runs/marathi_demo/metrics.json
python -m ud_treebank_lab visualize --input runs/marathi_demo/pair_01/silver.conllu \
  --metadata runs/marathi_demo/pair_01/metadata.jsonl \
  --metrics runs/marathi_demo/metrics.json --out runs/marathi_demo/preview.html
printf '\nOpen runs/marathi_demo/preview.html in a browser.\n'
