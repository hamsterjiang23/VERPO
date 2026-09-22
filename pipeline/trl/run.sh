#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${VERPO_TRL_PYTHON:-${PYTHON_BIN:-python}}
if [[ -z "${VERPO_TRL_PYTHON:-}" && -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi
TRAIN_FILE=""
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/trl_verpo}"
CONFIG=""
MAX_STEPS=1
LIMIT=0

while (($#)); do
  case "$1" in
    --train-file) TRAIN_FILE=${2:?missing value for --train-file}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?missing value for --output-dir}; shift 2 ;;
    --config) CONFIG=${2:?missing value for --config}; shift 2 ;;
    --max-steps) MAX_STEPS=${2:?missing value for --max-steps}; shift 2 ;;
    --limit) LIMIT=${2:?missing value for --limit}; shift 2 ;;
    --help|-h) sed -n '1,80p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$TRAIN_FILE" ]] || { echo "--train-file is required" >&2; exit 2; }
[[ "$TRAIN_FILE" != *.parquet ]] || { echo "TRL accepts JSONL/math-text only; Parquet belongs to native veRL" >&2; exit 2; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
ARGS=(-m risk_aware_opsd.train_trl_verpo --train-file "$TRAIN_FILE" --output-dir "$OUTPUT_DIR" --max-steps "$MAX_STEPS")
[[ -n "$CONFIG" ]] && ARGS+=(--config "$CONFIG")
[[ "$LIMIT" != 0 ]] && ARGS+=(--limit "$LIMIT")
exec "$PYTHON_BIN" "${ARGS[@]}"
