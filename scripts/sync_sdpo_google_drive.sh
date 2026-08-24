#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${SDPO_SYNC_PYTHON:-python3}
DATA_DIR=${SDPO_DATA_DIR:-"$ROOT/data/SDPO/verl"}
DATA_VERSION=${SDPO_DATA_VERSION:-sdpo_privileged_context_v1}

exec "$PYTHON_BIN" "$ROOT/scripts/sync_sdpo_public_bundle.py" \
  --data-dir "$DATA_DIR" \
  --expected-version "$DATA_VERSION" \
  "$@"
