#!/usr/bin/env bash
# Provide a small Unix shell entry point for the preprocessing workflow.
# The wrapper resolves the repository location, accepts optional BASE and SEED
# environment overrides, and forwards all additional arguments to the cross-platform
# Python launcher so validation and resume options behave identically.

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
