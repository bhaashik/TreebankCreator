#!/usr/bin/env bash
# Dedicated, idempotent Titan V setup with a pinned sm_70-compatible CUDA wheel.
# Does not change the system Python, CUDA toolkit, apt packages or other environments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv-titanv"
BASE_PYTHON=""
DRY_RUN=0
CHECK_ONLY=0
INCLUDE_TESTS=0
INCLUDE_AWESOME=0

usage() {
  cat <<'HELP'
Usage: bash scripts/setup_titanv.sh [--python PYTHON] [--venv PATH] [options]

Create/reuse an isolated environment for NVIDIA TITAN V (Volta sm_70).
Installs pinned PyTorch 2.14.1 with CUDA 12.6, SimAlign and TreebankCreator's
bilingual extras. Does NOT install NVIDIA drivers or the system CUDA toolkit.

  --python PYTHON   Python 3.10, 3.11 or 3.12 (default: detect one in that order).
  --venv PATH       Venv path (default: .venv-titanv in the project root).
  --with-awesome    Also attempt optional, legacy AWESOME-align setup.
  --with-tests      Install pytest for local development.
  --check           Verify existing environment AND execute an sm_70 GPU kernel.
  --dry-run         Print intended operations without changing files or network.
  -h, --help        Show this help.

Examples:
  bash scripts/setup_titanv.sh --dry-run
  bash scripts/setup_titanv.sh --python python3.12
  bash scripts/setup_titanv.sh --check
  source scripts/activate_titanv.sh
  python scripts/check_titanv.py

The existing resource YAML behavior is unchanged: supplied files never trigger
alignment, analyzer execution or installation. On the Titan V, use
setup.on_missing: error and run this explicit setup ONCE before training.
HELP
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[Titan V setup] %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --python) (($# >= 2)) || fail '--python requires an interpreter'; BASE_PYTHON="$2"; shift 2 ;;
    --venv) (($# >= 2)) || fail '--venv requires a path'; VENV="$2"; shift 2 ;;
    --with-awesome) INCLUDE_AWESOME=1; shift ;;
    --with-tests) INCLUDE_TESTS=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option '$1'" ;;
  esac
done
((DRY_RUN == 0 || CHECK_ONLY == 0)) || fail 'Choose only --check or --dry-run, not both'

# Keep even the default venv out of the application source and Git archive.
if [[ "$VENV" != /* ]]; then VENV="$PWD/$VENV"; fi
VENV="${VENV%/}"
if [[ -z "$BASE_PYTHON" ]]; then
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      BASE_PYTHON="$candidate"; break
    fi
  done
fi
if [[ -z "$BASE_PYTHON" ]]; then
  if ((DRY_RUN || CHECK_ONLY)); then
    BASE_PYTHON='python3.12 (not found on this host)'
  else
    fail 'Python 3.10-3.12 not found. Install/create a compatible Python first; pass --python /path/to/python.'
  fi
elif ! "$BASE_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)' >/dev/null 2>&1; then
  if ((DRY_RUN || CHECK_ONLY)); then
    log "NOTE: selected Python $BASE_PYTHON is not a supported 3.10-3.12 interpreter"
  else
    fail "Requires Python 3.10-3.12 (selected: $BASE_PYTHON)."
  fi
fi
[[ -d "$VENV" && ! -x "$VENV/bin/python" ]] && fail "Incomplete/foreign environment at $VENV; refusing to replace it"
[[ "$VENV" != "$ROOT" ]] || fail 'Refusing to install a venv over the project root'

TORCH_INDEX="https://download.pytorch.org/whl/cu126"
PIN="${ROOT}/constraints/titanv.txt"
if ((DRY_RUN)); then
  log "WOULD reuse/create venv: $VENV (via $BASE_PYTHON -m venv)"
  log "WOULD install torch==2.14.1 from $TORCH_INDEX if CUDA 12.6 build is missing"
  log "WOULD install $ROOT[align,bilingual] with constraints $PIN"
  if ((INCLUDE_TESTS)); then log "WOULD install $ROOT[test] under the same constraints"; fi
  if ((INCLUDE_AWESOME)); then log 'WOULD attempt optional AWESOME-align under the same torch constraint'; fi
  log 'WOULD test Titan V detection, sm_70 arch support and a real CUDA matrix multiplication'
  exit 0
fi

if ((CHECK_ONLY)); then
  [[ -x "$VENV/bin/python" ]] || fail "Missing environment: $VENV"
  "$VENV/bin/python" -c 'import simalign, transformers' || fail 'SimAlign or transformers missing'
  if ((INCLUDE_AWESOME)); then
    "$VENV/bin/python" -c 'import awesome_align' || fail 'AWESOME-align missing'
  fi
  exec "$VENV/bin/python" "$ROOT/scripts/check_titanv.py"
fi

# Validate the GPU before installing multiple gigabytes of dependencies.
command -v nvidia-smi >/dev/null 2>&1 || fail 'NVIDIA driver/nvidia-smi missing. Install a suitable driver first.'
nvidia-smi -L | grep -qi 'TITAN V' || fail 'No NVIDIA TITAN V listed by nvidia-smi -L'
if [[ ! -x "$VENV/bin/python" ]]; then
  log "Creating isolated environment: $VENV"
  "$BASE_PYTHON" -m venv "$VENV" || fail 'Venv creation failed; install python3-venv for your Python version'
else
  log "Reusing existing environment: $VENV"
fi
VENV_PYTHON="$VENV/bin/python"
"$VENV_PYTHON" -m pip --version >/dev/null || fail 'pip is unavailable inside the venv'

if "$VENV_PYTHON" -c \
  'import torch; assert torch.__version__.split("+",1)[0] == "2.14.1" and torch.version.cuda == "12.6"' \
  >/dev/null 2>&1; then
  log 'Pinned torch 2.14.1 CUDA 12.6 already installed; reusing'
else
  log 'Installing PyTorch 2.14.1 CUDA 12.6 (sm_70) from the official PyTorch index'
  "$VENV_PYTHON" -m pip install --upgrade --force-reinstall 'torch==2.14.1' --index-url "$TORCH_INDEX"
fi
# PIP_CONSTRAINT protects both the initial extras and subsequent installer usage.
log 'Installing project SimAlign + bilingual extras under the Titan V constraint'
PIP_CONSTRAINT="$PIN" "$VENV_PYTHON" -m pip install -e "$ROOT[align,bilingual]"
if ((INCLUDE_TESTS)); then
  PIP_CONSTRAINT="$PIN" "$VENV_PYTHON" -m pip install -e "$ROOT[test]"
fi
if ((INCLUDE_AWESOME)); then
  log 'Attempting legacy AWESOME-align; compatibility with recent Python/Transformers is not guaranteed'
  PIP_CONSTRAINT="$PIN" bash "$ROOT/scripts/install_linguistic_tools.sh" \
    --aligner awesome --python "$VENV_PYTHON"
fi
"$VENV_PYTHON" "$ROOT/scripts/check_titanv.py" || fail 'Titan V CUDA verification failed'
log 'Ready. Activate with: source scripts/activate_titanv.sh'
log 'For custom --venv paths, set TREEBANK_TITANV_VENV before sourcing the activation helper.'
