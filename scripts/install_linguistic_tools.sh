#!/usr/bin/env bash
# Explicit, idempotent Ubuntu/Debian setup for Apertium and neural aligners.
# Never run this script implicitly; pipeline auto-install requires explicit YAML opt-in.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALIGNER=none
INSTALL_APERTIUM=0
DRY_RUN=0
CHECK_ONLY=0
ASSUME_YES=0
PYTHON_BIN=""
ANALYZER_BIN=""
APERTIUM_PACKAGES=()

usage() {
  cat <<'HELP'
Usage: bash scripts/install_linguistic_tools.sh [options]

An explicit installer for Ubuntu/Debian systems (including appropriately
privileged containers). No packages are installed if run without install flags.

  --aligner simalign|awesome|auto|both|none
      Install/check SimAlign (default neural alignment backend), the optional
      AWESOME-align CLI, or both. 'auto' prefers SimAlign, falling back to
      AWESOME-align only if installing SimAlign fails. Default: none.
  --apertium
      Install/check Apertium plus lttoolbox (providing lt-proc), through apt.
  --apertium-package NAME
      Install a specific available Debian/Ubuntu language-data package. May
      be repeated; implies --apertium. Packages must be named apertium-... .
      Existence is checked in the configured apt repositories.
  --analyzer-bin FILE
      Verify that a compiled analyzer actually exists at FILE after setup.
      This does not download or compile a language-specific analyzer.
  --python PYTHON
      Select the active Python interpreter. Default: activated venv/conda,
      otherwise ROOT/.venv/bin/python if available, otherwise python3.
  --dry-run
      Report missing prerequisites / intended operations; make no changes.
  --check
      Check installation and analyzer availability; make no changes.
      Exit status is nonzero if a requested component is missing.
  --yes
      Use apt-get -y in explicit install mode (sudo may still prompt).
  -h, --help
      Show this help.

Examples:
  bash scripts/install_linguistic_tools.sh --aligner simalign --apertium
  bash scripts/install_linguistic_tools.sh --aligner awesome --check
  bash scripts/install_linguistic_tools.sh --apertium \
      --apertium-package apertium-foo --analyzer-bin /path/to/foo.automorf.bin
  bash scripts/install_linguistic_tools.sh --aligner simalign --apertium --dry-run

Note: Apertium's ENGINE and target-language ANALYZERS are separate. The
installer cannot create an analyzer for a language with no installed or
compiled data. Neural models may download weights at first alignment run.
HELP
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[linguistic setup] %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --aligner)
      (($# >= 2)) || fail '--aligner needs a value'
      ALIGNER="$2"; shift 2 ;;
    --apertium) INSTALL_APERTIUM=1; shift ;;
    --apertium-package)
      (($# >= 2)) || fail '--apertium-package needs a value'
      [[ "$2" =~ ^apertium-[a-z0-9][a-z0-9+.-]*$ ]] || fail "invalid package name: $2"
      APERTIUM_PACKAGES+=("$2"); INSTALL_APERTIUM=1; shift 2 ;;
    --analyzer-bin)
      (($# >= 2)) || fail '--analyzer-bin needs a path'
      ANALYZER_BIN="$2"; shift 2 ;;
    --python)
      (($# >= 2)) || fail '--python needs an executable'
      PYTHON_BIN="$2"; shift 2 ;;
    --yes) ASSUME_YES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option '$1'; see --help" ;;
  esac
done
case "$ALIGNER" in none|simalign|awesome|auto|both) ;; *) fail "invalid --aligner '$ALIGNER'" ;; esac
((DRY_RUN == 0 || CHECK_ONLY == 0)) || fail 'choose only one of --dry-run and --check'
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="$(command -v python || true)"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || true)"
  fi
fi

python_available() {
  [[ -n "$PYTHON_BIN" ]] && "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,10)' >/dev/null 2>&1
}
module_available() {
  "$PYTHON_BIN" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)' "$1" >/dev/null 2>&1
}
awesome_executable_available() {
  if command -v awesome-align >/dev/null 2>&1; then return 0; fi
  # The interpreter may come from .venv without having been activated.
  [[ -n "$PYTHON_BIN" ]] || return 1
  local env_bin
  env_bin="$($PYTHON_BIN -c 'import sys,sysconfig; print(sysconfig.get_path("scripts"))' 2>/dev/null)" || return 1
  [[ -x "$env_bin/awesome-align" ]]
}
apertium_present() { command -v lt-proc >/dev/null 2>&1 && command -v apertium >/dev/null 2>&1; }

if [[ "$ALIGNER" != none ]]; then
  python_available || fail "Python >=3.10 unavailable at '${PYTHON_BIN:-none}'"
fi

install_python() {
  local target="$1"
  case "$target" in
    simalign)
      if module_available simalign; then log 'SimAlign already installed; reusing'; return 0; fi
      if ((CHECK_ONLY)); then log 'MISSING: SimAlign'; return 1; fi
      if ((DRY_RUN)); then log "WOULD RUN: $PYTHON_BIN -m pip install -e $ROOT[align]"; return 0; fi
      log 'Installing SimAlign in selected Python environment'
      "$PYTHON_BIN" -m pip install -e "$ROOT[align]" || return 1
      module_available simalign || { log 'SimAlign module missing after pip installation'; return 1; } ;;
    awesome)
      if module_available awesome_align && awesome_executable_available; then
        log 'AWESOME-align already installed; reusing'; return 0
      fi
      if ((CHECK_ONLY)); then log 'MISSING: AWESOME-align Python module or CLI'; return 1; fi
      if ((DRY_RUN)); then log "WOULD RUN: $PYTHON_BIN -m pip install git+https://github.com/neulab/awesome-align.git"; return 0; fi
      command -v git >/dev/null 2>&1 || fail 'git is required to install AWESOME-align from its official repository'
      log 'Installing legacy AWESOME-align in selected Python environment'
      "$PYTHON_BIN" -m pip install 'git+https://github.com/neulab/awesome-align.git' || return 1
      module_available awesome_align && awesome_executable_available || {
        log 'AWESOME-align module/CLI missing after pip installation'; return 1;
      } ;;
  esac
}

errors=0
case "$ALIGNER" in
  simalign|awesome) install_python "$ALIGNER" || errors=1 ;;
  both)
    install_python simalign || errors=1
    install_python awesome || errors=1 ;;
  auto)
    if module_available simalign; then
      log 'SimAlign available; AWESOME-align fallback not needed'
    elif module_available awesome_align && awesome_executable_available; then
      log 'SimAlign unavailable; existing AWESOME-align is available'
    elif ((CHECK_ONLY)); then
      log 'MISSING: neither SimAlign nor AWESOME-align'; errors=1
    elif ((DRY_RUN)); then
      log 'WOULD install SimAlign; on pip installation failure would try AWESOME-align'
    elif install_python simalign; then
      log 'Selected backend: simalign'
    else
      log 'WARNING: SimAlign installation failed, trying AWESOME-align' >&2
      install_python awesome || errors=1
    fi ;;
esac

if ((INSTALL_APERTIUM)); then
  if apertium_present; then
    log 'Apertium and lt-proc are installed; reusing'
  elif ((CHECK_ONLY)); then
    log 'MISSING: Apertium/lttoolbox commands'; errors=1
  elif ((DRY_RUN)); then
    log 'WOULD RUN: apt-get update + apt-get install apertium lttoolbox'
  else
    command -v apt-get >/dev/null 2>&1 || fail 'Apertium automatic installation requires apt-get (Ubuntu/Debian)'
    apt_cmd=(apt-get)
    if ((EUID != 0)); then
      command -v sudo >/dev/null 2>&1 || fail 'Apertium installation needs root or sudo'
      apt_cmd=(sudo apt-get)
    fi
    log 'Installing Apertium and lttoolbox using apt'
    "${apt_cmd[@]}" update
    if ((ASSUME_YES)); then "${apt_cmd[@]}" install -y apertium lttoolbox
    else "${apt_cmd[@]}" install apertium lttoolbox; fi
    apertium_present || fail 'Apertium/lt-proc commands still missing after apt installation'
  fi
fi

if ((${#APERTIUM_PACKAGES[@]})); then
  if ((CHECK_ONLY)); then
    for pkg in "${APERTIUM_PACKAGES[@]}"; do
      if command -v dpkg-query >/dev/null 2>&1 && dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q 'install ok installed'; then
        log "$pkg already installed"
      else
        log "MISSING: $pkg"; errors=1
      fi
    done
  elif ((DRY_RUN)); then
    log "WOULD CHECK apt repository and install language packages: ${APERTIUM_PACKAGES[*]}"
  else
    command -v apt-cache >/dev/null 2>&1 || fail 'apt-cache is required to find Apertium language packages'
    apt_cmd=(apt-get)
    if ((EUID != 0)); then
      command -v sudo >/dev/null 2>&1 || fail 'Apertium installation needs root or sudo'
      apt_cmd=(sudo apt-get)
    fi
    missing_packages=()
    for pkg in "${APERTIUM_PACKAGES[@]}"; do
      if command -v dpkg-query >/dev/null 2>&1 && dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q 'install ok installed'; then
        log "$pkg already installed; reusing"
      else
        missing_packages+=("$pkg")
      fi
    done
    if ((${#missing_packages[@]})); then
      "${apt_cmd[@]}" update
      # Package names vary across releases/repos. Refuse unavailable names.
      for pkg in "${missing_packages[@]}"; do
        candidate="$(apt-cache policy "$pkg" | sed -n 's/^[[:space:]]*Candidate: //p' | head -1)"
        [[ -n "$candidate" && "$candidate" != '(none)' ]] ||
          fail "No apt candidate for $pkg. Configure the Apertium repository or build the language analyzer manually."
      done
      if ((ASSUME_YES)); then "${apt_cmd[@]}" install -y "${missing_packages[@]}"
      else "${apt_cmd[@]}" install "${missing_packages[@]}"; fi
    fi
  fi
fi

if [[ -n "$ANALYZER_BIN" ]]; then
  if [[ -f "$ANALYZER_BIN" ]]; then
    log "Language-specific analyzer available: $ANALYZER_BIN"
  elif ((DRY_RUN)); then
    log "MISSING: compiled analyzer $ANALYZER_BIN (must be obtained/compiled separately)"
  else
    log "MISSING: compiled analyzer $ANALYZER_BIN"; errors=1
  fi
fi

if ((errors)); then
  fail 'One or more requested prerequisites are unavailable. See messages above.'
fi
log "Complete. Python: ${PYTHON_BIN:-not requested}; aligner: $ALIGNER"
