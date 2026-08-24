#!/usr/bin/env python3
"""Audit parameter deltas between base and trained safetensors checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def audit_parameter_delta(
    base_checkpoint: Path,
    trained_checkpoint: Path,
    *,
    optimizer_step: int,
    chunk_values: int = 1_000_000,
) -> dict[str, Any]:
    base_checkpoint = Path(base_checkpoint)
    trained_checkpoint = Path(trained_checkpoint)
    with safe_open(base_checkpoint, framework="pt", device="cpu") as base_handle:
        with safe_open(trained_checkpoint, framework="pt", device="cpu") as trained_handle:
            base_keys = set(base_handle.keys())
            trained_keys = set(trained_handle.keys())
            common_keys = sorted(base_keys & trained_keys)
            base_only = sorted(base_keys - trained_keys)
            trained_only = sorted(trained_keys - base_keys)

            changed_tensor_count = 0
            changed_value_count = 0
            total_value_count = 0
            delta_sq_sum = 0.0
            delta_max_abs = 0.0
            shape_mismatches: list[dict[str, Any]] = []

            for key in common_keys:
                base_tensor = base_handle.get_tensor(key)
                trained_tensor = trained_handle.get_tensor(key)
                if tuple(base_tensor.shape) != tuple(trained_tensor.shape):
                    shape_mismatches.append(
                        {
                            "name": key,
                            "base_shape": list(base_tensor.shape),
                            "trained_shape": list(trained_tensor.shape),
                        }
                    )
                    continue
                base_flat = base_tensor.reshape(-1)
                trained_flat = trained_tensor.reshape(-1)
                tensor_changed = False
                total_value_count += int(base_flat.numel())
                for start in range(0, base_flat.numel(), chunk_values):
                    end = min(start + chunk_values, base_flat.numel())
                    delta = trained_flat[start:end].float() - base_flat[start:end].float()
                    changed = delta != 0
                    changed_in_chunk = int(changed.sum().item())
                    if changed_in_chunk:
                        tensor_changed = True
                        changed_value_count += changed_in_chunk
                        delta_sq_sum += float(delta.double().square().sum().item())
                        delta_max_abs = max(delta_max_abs, float(delta.abs().max().item()))
                if tensor_changed:
                    changed_tensor_count += 1

    status = "pass" if changed_tensor_count > 0 and not shape_mismatches else "fail"
    return {
        "audit_scope": "full_checkpoint_common_parameter_delta",
        "base_checkpoint": str(base_checkpoint),
        "trained_checkpoint": str(trained_checkpoint),
        "optimizer_step": int(optimizer_step),
        "common_tensor_count": len(common_keys),
        "changed_tensor_count": changed_tensor_count,
        "changed_value_count": changed_value_count,
        "total_value_count": total_value_count,
        "changed_value_ratio": (
            changed_value_count / total_value_count if total_value_count else 0.0
        ),
        "parameter_delta_l2": math.sqrt(delta_sq_sum),
        "parameter_delta_max_abs": delta_max_abs,
        "base_only_tensors": base_only,
        "trained_only_tensors": trained_only,
        "shape_mismatches": shape_mismatches,
        "serialization_key_delta_expected": bool(base_only or trained_only),
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--optimizer-step", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-values", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_parameter_delta(
        Path(args.base_checkpoint),
        Path(args.trained_checkpoint),
        optimizer_step=args.optimizer_step,
        chunk_values=args.chunk_values,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
