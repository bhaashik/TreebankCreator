#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-external/conll18_ud_eval.py}"
mkdir -p "$(dirname "$DEST")"
curl --fail --location --show-error --silent \
  'https://universaldependencies.org/conll18/conll18_ud_eval.py' -o "$DEST"
python3 -m py_compile "$DEST"
printf 'Downloaded CoNLL18 evaluator (MPL-2.0): %s\n' "$DEST"
