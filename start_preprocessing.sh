#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"

BASE="${BASE:-$SCRIPT_DIR}"
SEED="${SEED:-42}"

exec python -u "$BASE/start_preprocessing.py" \
  --base "$BASE" \
  --seed "$SEED" \
  "$@"
