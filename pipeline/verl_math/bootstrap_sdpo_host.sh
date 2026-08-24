#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
VENV_DIR=${VENV_DIR:-"$ROOT/.venvs/qwen3-1.7b-verpo-zpd"}
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-python3}

missing_commands=()
for command_name in "$BOOTSTRAP_PYTHON"; do
  command -v "$command_name" >/dev/null 2>&1 || missing_commands+=("$command_name")
done

run_privileged() {
  if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo -n "$@"
  else
    echo "Missing host dependencies and non-interactive root/sudo is unavailable" >&2
    return 1
  fi
}

install_host_python() {
  if command -v apt-get >/dev/null 2>&1; then
    run_privileged apt-get update
    run_privileged apt-get install -y python3 python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1; then
    run_privileged dnf install -y python3 python3-pip
  elif command -v yum >/dev/null 2>&1; then
    run_privileged yum install -y python3 python3-pip
  else
    echo "Cannot auto-install Python: supported package manager not found" >&2
    return 1
  fi
}

if [[ ${#missing_commands[@]} -gt 0 ]]; then
  install_host_python
fi

if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
  echo "BOOTSTRAP_PYTHON is unavailable after host bootstrap: $BOOTSTRAP_PYTHON" >&2
  exit 1
fi

if ! "$BOOTSTRAP_PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "SDPO training requires Python >= 3.10" >&2
  exit 1
fi

PYTHON_BIN="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  mkdir -p "$(dirname "$VENV_DIR")"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$BOOTSTRAP_PYTHON" "$VENV_DIR"
  elif ! "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"; then
    install_host_python
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
  fi
fi

if ! "$PYTHON_BIN" -c 'import yaml' >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PYTHON_BIN" 'PyYAML>=6,<7'
  else
    "$PYTHON_BIN" -m pip install 'PyYAML>=6,<7'
  fi
fi

"$PYTHON_BIN" -c 'import yaml; print("SDPO launcher bootstrap ready")'
