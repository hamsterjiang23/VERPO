#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
VENV_DIR=${VENV_DIR:-"$ROOT/.venvs/qwen3-1.7b-verpo-zpd"}
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-python3}
PYTHON_BIN="$VENV_DIR/bin/python"
export PATH="$VENV_DIR/bin:$PATH"
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-vllm}
case "$ROLLOUT_BACKEND" in
  vllm) REQUIREMENTS_FILE="$SCRIPT_DIR/../requirements.txt" ;;
  sglang) REQUIREMENTS_FILE="$SCRIPT_DIR/../requirements-sglang.txt" ;;
  *)
    echo "ROLLOUT_BACKEND must be vllm or sglang, got: $ROLLOUT_BACKEND" >&2
    exit 2
    ;;
esac
ENV_STAMP="$VENV_DIR/.qwen3_1_7b_verpo_zpd_requirements.sha256"
FLASH_ATTN_SOURCE_VERSION=${FLASH_ATTN_SOURCE_VERSION:-2.8.3.post1}
FLASH_ATTN_MIN_VERSION=${FLASH_ATTN_MIN_VERSION:-2.7.4.post1}
FLASH_ATTN_MAX_JOBS=${FLASH_ATTN_MAX_JOBS:-4}
FLASH_ATTN_PACKAGE_ROOT=${FLASH_ATTN_PACKAGE_ROOT:-/home/bingxing2/apps/package}
FLASH_ATTN_LOCAL_WHEEL=${FLASH_ATTN_LOCAL_WHEEL:-}

# Cluster SSH shells may omit an installed CUDA toolkit from PATH.  Discover
# the conventional toolkit location before deciding that FlashAttention
# cannot be compiled for the pinned PyTorch runtime.
if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
  esac
fi

if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
  echo "BOOTSTRAP_PYTHON is unavailable: $BOOTSTRAP_PYTHON" >&2
  exit 1
fi

if ! "$BOOTSTRAP_PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "VERPO-ZPD native veRL runtime requires Python >= 3.10" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  mkdir -p "$(dirname "$VENV_DIR")"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$BOOTSTRAP_PYTHON" "$VENV_DIR"
  else
    if ! "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"; then
      echo "Failed to create $VENV_DIR; install python3-venv or uv and retry" >&2
      exit 1
    fi
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment is incomplete: $PYTHON_BIN is missing" >&2
  exit 1
fi

SOURCE_FINGERPRINT=$(
  "$PYTHON_BIN" - \
    "$REQUIREMENTS_FILE" \
    "$SCRIPT_DIR/setup_env.sh" \
    "$ROOT/verl/requirements.txt" \
    "$ROOT/verl/setup.py" \
    "$ROOT/verl/pyproject.toml" <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256()
for index, raw_path in enumerate(sys.argv[1:]):
    path = pathlib.Path(raw_path).resolve()
    digest.update(str(index).encode("ascii"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
)

environment_imports_ok() {
  "$PYTHON_BIN" - "$ROLLOUT_BACKEND" <<'PY'
import importlib
import importlib.metadata
import sys

for name in (
    "huggingface_hub",
    "math_verify",
    "modelscope",
    "pyarrow",
    "pytest",
    "ray",
    "swanlab",
    "tensordict",
    "torch",
    "transfer_queue",
    "transformers",
    "verl",
    "yaml",
    "flash_attn",
):
    importlib.import_module(name)
backend = sys.argv[1]
if backend == "vllm":
    importlib.import_module("vllm")
    importlib.import_module("flashinfer")
elif backend == "sglang":
    importlib.import_module("sglang")
    try:
        importlib.import_module("sglang_kernel")
    except ImportError:
        importlib.import_module("sgl_kernel")
else:
    raise SystemExit(f"unsupported rollout backend: {backend}")
importlib.metadata.version("verl")
PY
}

flash_attention_ok() {
  "$PYTHON_BIN" - "$FLASH_ATTN_MIN_VERSION" <<'PY'
import importlib
import importlib.metadata
import sys

import torch
from packaging.version import Version

try:
    version = Version(importlib.metadata.version("flash-attn"))
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)
if not Version(sys.argv[1]) <= version < Version("3"):
    raise SystemExit(1)
flash_attn = importlib.import_module("flash_attn")
q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
output = flash_attn.flash_attn_func(q, q, q, causal=True)
output.float().sum().backward()
torch.cuda.synchronize()
if output.shape != q.shape or not torch.isfinite(output).all():
    raise SystemExit(1)
PY
}

find_local_flash_attention_wheel() {
  "$PYTHON_BIN" - "$FLASH_ATTN_LOCAL_WHEEL" "$FLASH_ATTN_PACKAGE_ROOT" "$FLASH_ATTN_MIN_VERSION" <<'PY'
import os
import pathlib
import sys

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

explicit, raw_root, raw_min_version = sys.argv[1:]
minimum_version = Version(raw_min_version)
supported_tags = set(sys_tags())


def compatible(path: pathlib.Path) -> bool:
    try:
        name, version, _, tags = parse_wheel_filename(path.name)
    except ValueError:
        return False
    return (
        canonicalize_name(name) == "flash-attn"
        and minimum_version <= version < Version("3")
        and bool(tags & supported_tags)
    )


if explicit:
    path = pathlib.Path(explicit).expanduser().resolve()
    if not path.is_file() or not compatible(path):
        raise SystemExit(f"FLASH_ATTN_LOCAL_WHEEL is not a compatible FlashAttention 2 wheel: {path}")
    print(path)
    raise SystemExit(0)

root = pathlib.Path(raw_root).expanduser()
if not root.is_dir():
    raise SystemExit(0)

candidates = []
root_depth = len(root.resolve().parts)
for directory, child_dirs, filenames in os.walk(root):
    depth = len(pathlib.Path(directory).resolve().parts) - root_depth
    if depth >= 7:
        child_dirs.clear()
    for filename in filenames:
        if not filename.lower().endswith(".whl") or "flash" not in filename.lower():
            continue
        path = pathlib.Path(directory, filename)
        if compatible(path):
            candidates.append(path.resolve())

if candidates:
    candidates.sort(key=str)
    candidates.sort(key=lambda path: parse_wheel_filename(path.name)[1], reverse=True)
    print(candidates[0])
PY
}

NEEDS_INSTALL=true
if [[ -f "$ENV_STAMP" ]] && [[ "$(<"$ENV_STAMP")" == "$SOURCE_FINGERPRINT" ]]; then
  if environment_imports_ok >/dev/null 2>&1 && flash_attention_ok >/dev/null 2>&1; then
    NEEDS_INSTALL=false
  fi
fi

if [[ "$NEEDS_INSTALL" == true ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PYTHON_BIN" --upgrade pip setuptools wheel
    uv pip install --python "$PYTHON_BIN" --requirement "$REQUIREMENTS_FILE"
    uv pip install --python "$PYTHON_BIN" --no-deps --no-build-isolation --editable "$ROOT/verl"
  else
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install --requirement "$REQUIREMENTS_FILE"
    "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation --editable "$ROOT/verl"
  fi
  if ! flash_attention_ok; then
    LOCAL_FLASH_ATTN_WHEEL=$(find_local_flash_attention_wheel)
    if [[ -n "$LOCAL_FLASH_ATTN_WHEEL" ]]; then
      echo "Installing local FlashAttention wheel: $LOCAL_FLASH_ATTN_WHEEL"
      if ! "$PYTHON_BIN" -m pip install --force-reinstall --no-deps "$LOCAL_FLASH_ATTN_WHEEL" || \
         ! flash_attention_ok; then
        echo "Local FlashAttention wheel is ABI-incompatible; rebuilding from the pinned source" >&2
      fi
    else
      echo "No compatible FlashAttention 2 wheel found under $FLASH_ATTN_PACKAGE_ROOT"
    fi
  fi
  if ! flash_attention_ok; then
    if ! command -v nvcc >/dev/null 2>&1; then
      echo "FlashAttention requires nvcc from a CUDA toolkit; load the cluster CUDA toolkit and retry" >&2
      exit 1
    fi
    MAX_JOBS="$FLASH_ATTN_MAX_JOBS" \
      "$PYTHON_BIN" -m pip install --force-reinstall --no-deps --no-build-isolation \
      "flash-attn==$FLASH_ATTN_SOURCE_VERSION"
  fi
  environment_imports_ok
  flash_attention_ok
  "$PYTHON_BIN" -m pip check
  printf '%s\n' "$SOURCE_FINGERPRINT" > "$ENV_STAMP"
fi

environment_imports_ok
flash_attention_ok
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" - "$ROLLOUT_BACKEND" <<'PY'
import importlib.metadata
import sys

names = (
    "torch",
    "transformers",
    "ray",
    "nvidia-nccl-cu12",
    "flash-attn",
    "modelscope",
    "TransferQueue",
    "verl",
)
if sys.argv[1] == "vllm":
    names += ("vllm", "flashinfer-python")
elif sys.argv[1] == "sglang":
    names += ("sglang", "sgl-kernel")
else:
    raise SystemExit(f"unsupported rollout backend: {sys.argv[1]}")
versions = ", ".join(f"{name}={importlib.metadata.version(name)}" for name in names)
print(f"Environment ready: backend={sys.argv[1]}, python={sys.version.split()[0]}, {versions}")
PY
