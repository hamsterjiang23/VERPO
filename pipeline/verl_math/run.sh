#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
VENV_DIR=${VENV_DIR:-"$ROOT/.venvs/qwen3-1.7b-verpo-zpd"}

PROTOCOL=""
MATRIX=""
PRINT_ONLY=false
arguments=("$@")
MODEL_ARG=""
EVIDENCE_SCOPE_EXPLICIT=false
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[$index]}" in
    --model)
      if (( index + 1 < ${#arguments[@]} )); then
        MODEL_ARG=${arguments[$((index + 1))]}
      fi
      ;;
    --set)
      if (( index + 1 < ${#arguments[@]} )) && [[ "${arguments[$((index + 1))]}" == zpd.evidence_scope=* ]]; then
        EVIDENCE_SCOPE_EXPLICIT=true
      fi
      ;;
    --set=zpd.evidence_scope=*)
      EVIDENCE_SCOPE_EXPLICIT=true
      ;;
    --protocol)
      if (( index + 1 < ${#arguments[@]} )); then
        PROTOCOL=${arguments[$((index + 1))]}
      fi
      ;;
    --matrix)
      if (( index + 1 < ${#arguments[@]} )); then
        MATRIX=${arguments[$((index + 1))]}
      fi
      ;;
    --print-config|--print-command) PRINT_ONLY=true ;;
  esac
done

if [[ "$EVIDENCE_SCOPE_EXPLICIT" == false ]]; then
  case "$MODEL_ARG" in
    llama3_2_*) arguments+=(--set zpd.evidence_scope=all) ;;
    qwen3_*) arguments+=(--set zpd.evidence_scope=all) ;;
  esac
fi

if [[ -n "$PROTOCOL" && "$PROTOCOL" != sdpo_section3_* ]]; then
  echo "Only SDPO Section 3 training is active; protocol '$PROTOCOL' is deprecated or unsupported" >&2
  exit 2
fi

if [[ ( "$PROTOCOL" == sdpo_section3_* || -n "$MATRIX" ) && "$PRINT_ONLY" == false ]]; then
  VENV_DIR="$VENV_DIR" BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}" \
    bash "$SCRIPT_DIR/bootstrap_sdpo_host.sh"
fi

if [[ -n "${VERPO_CONFIG_PYTHON:-}" ]]; then
  PYTHON_BIN=$VERPO_CONFIG_PYTHON
elif [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]] && "$ROOT/.venv/bin/python" -c 'import yaml' >/dev/null 2>&1; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import yaml' >/dev/null 2>&1; then
  PYTHON_BIN=$(command -v python3)
else
  echo "No Python interpreter with PyYAML is available for the VERPO YAML resolver." >&2
  echo "Set VERPO_CONFIG_PYTHON or VENV_DIR to a verified environment." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT/scripts/launch_verpo_verl.py" "$@"
