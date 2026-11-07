#!/usr/bin/env bash
# Lightning AI Studio setup: install into its existing managed environment.
# Do not create a venv/conda environment inside a Lightning Studio.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'HELP'
Usage: bash scripts/setup_lightning.sh [core|align|hf|translate|trankit|stanza|align,hf|...] [--skip-tests]

Installs the package into the EXISTING Lightning AI Studio Python environment.
The default, 'core', needs no external model downloads and runs offline tests.
Specify extras separated by commas, for example 'align,hf'.

Do not combine incompatible model backends without checking package versions.
For a normal Ubuntu VM (not Lightning Studio), use scripts/setup_vm.sh.
HELP
}

EXTRAS="core"
SKIP_TESTS=0
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    core|align|hf|translate|trankit|stanza|reports|*,*)
      if [[ "$EXTRAS" != core ]]; then
        printf 'Only one extras argument is supported.\n' >&2
        exit 2
      fi
      EXTRAS="$arg" ;;
    *) printf 'Unknown argument: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v python >/dev/null 2>&1; then
  printf 'No python executable found in the active Studio environment.\n' >&2
  exit 1
fi
python - <<'PY'
import sys
print('Python:', sys.executable, sys.version.split()[0])
if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit('UD Treebank Lab needs Python >=3.10,<3.14; choose a compatible Studio base image.')
PY

printf '\nInstalling UD Treebank Lab (%s) into the current Python environment...\n' "$EXTRAS"
python -m pip install --upgrade pip
if [[ "$EXTRAS" == core ]]; then
  python -m pip install -e '.[test]'
else
  python -m pip install -e ".[test,${EXTRAS}]"
fi

if [[ "$SKIP_TESTS" == 0 ]]; then
  python -m pytest
  bash scripts/demo.sh
fi

printf '\nReady. Next: inspect docs/LIGHTNING_AI_PRO.md.\n'
printf 'Check GPU later with: python -c "import torch; print(torch.cuda.is_available())"\n'
