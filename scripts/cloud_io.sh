#!/usr/bin/env bash
# Non-destructive rclone transfers for UD Treebank Lab.
# Cloud remotes must already be configured with `rclone config`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'HELP'
Usage: bash scripts/cloud_io.sh COMMAND REMOTE:folder

Commands:       Local source/destination below the project root
  pull-data     data/             <- remote
  push-data     data/             -> remote
  pull-models   assets/models/    <- remote (downloaded MODEL SNAPSHOTS, not HF tokens)
  push-models   assets/models/    -> remote
  pull-runs     runs/             <- remote (restore checkpoint and stage state)
  push-runs     runs/             -> remote
  list          (none)           list remote directories

Examples:
  bash scripts/cloud_io.sh pull-data drive:UDTreebankLab/data
  bash scripts/cloud_io.sh pull-models dropbox:UDTreebankLab/models
  bash scripts/cloud_io.sh push-runs dropbox:UDTreebankLab/backups/run_2026-11-06
  bash scripts/cloud_io.sh pull-runs dropbox:UDTreebankLab/backups/run_2026-11-06

Use UDTL_DRY_RUN=1 before a transfer to preview changes.
Transfers use rclone COPY (does not delete destination-only files), NOT sync.
Do not back up actively-written model checkpoints. Never upload HF auth tokens.
HELP
}

if [[ $# == 1 && ( "$1" == --help || "$1" == help ) ]]; then
  usage
  exit 0
fi
if [[ $# != 2 ]]; then
  usage >&2
  exit 2
fi
ACTION="$1"
REMOTE="$2"

if ! command -v rclone >/dev/null 2>&1; then
  printf 'rclone not installed; see docs/LIGHTNING_AI_PRO.md.\n' >&2
  exit 1
fi
# Do not accidentally write to the root of a whole Google Drive / Dropbox.
if [[ "$REMOTE" != *:* || -z "${REMOTE#*:}" ]]; then
  printf 'Supply an rclone remote SUBFOLDER, for example drive:UDTreebankLab/data.\n' >&2
  exit 2
fi

case "$ACTION" in
  pull-data|push-data) TARGET="$ROOT/data" ;;
  pull-models|push-models) TARGET="$ROOT/assets/models" ;;
  pull-runs|push-runs) TARGET="$ROOT/runs" ;;
  list) rclone lsd "$REMOTE"; exit 0 ;;
  *) printf 'Unknown action: %s\n' "$ACTION" >&2; usage >&2; exit 2 ;;
esac

OPTIONS=(--progress --transfers 4 --checkers 8)
if [[ "${UDTL_DRY_RUN:-0}" == 1 ]]; then OPTIONS+=(--dry-run); fi
if [[ "$ACTION" == pull-* ]]; then
  mkdir -p "$TARGET"
  printf 'Copying cloud %s -> local %s\n' "$REMOTE" "$TARGET"
  rclone copy "$REMOTE" "$TARGET" "${OPTIONS[@]}"
else
  if [[ ! -d "$TARGET" ]]; then
    printf 'Local source not found: %s\n' "$TARGET" >&2
    exit 1
  fi
  printf 'Copying local %s -> cloud %s\n' "$TARGET" "$REMOTE"
  rclone copy "$TARGET" "$REMOTE" "${OPTIONS[@]}"
fi
