#!/usr/bin/env bash
# Source this file; it activates the GPU venv and pins future pip installs.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Source, do not execute: source scripts/activate_titanv.sh' >&2
  exit 2
fi
_titanv_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_titanv_venv="${TREEBANK_TITANV_VENV:-${_titanv_root}/.venv-titanv}"
if [[ ! -f "$_titanv_venv/bin/activate" ]]; then
  echo "Titan V environment not found: $_titanv_venv" >&2
  return 1
fi
# venv sets VIRTUAL_ENV, PATH and the environment-local Python/pip executable.
source "$_titanv_venv/bin/activate"
export PIP_CONSTRAINT="$_titanv_root/constraints/titanv.txt"
unset _titanv_root _titanv_venv
