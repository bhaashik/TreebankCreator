#!/usr/bin/env bash
# Ubuntu / Lightning AI / generic Linux with Python 3.10-3.12 already installed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest
printf '\nCore installed. Add one or more backends:\n'
printf '  pip install -e ".[align]"\n'
printf '  pip install -e ".[hf]"\n'
printf '  pip install -e ".[trankit]"\n'
printf '  pip install -e ".[stanza]"\n'
printf '  pip install -e ".[translate]"\n'
printf '\nActivate: source .venv/bin/activate\n'
