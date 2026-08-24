#!/usr/bin/env python3
"""Write a deterministic, CPU-only Top-k approximation audit manifest.

The script intentionally performs no model loading. It records the requested
vocabulary size and K values so a later numerical audit can attach measured
loss/gradient errors without changing the registered experiment identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vocab_size <= 0 or any(value <= 0 for value in args.top_k):
        raise ValueError("vocab-size and top-k values must be positive")
    payload = {
        "schema_version": "verpo_topk_approximation_audit_v1",
        "vocab_size": args.vocab_size,
        "top_k": sorted(set(args.top_k)),
        "exact_fallback_k": [value for value in sorted(set(args.top_k)) if value >= args.vocab_size],
        "approximate_k": [value for value in sorted(set(args.top_k)) if value < args.vocab_size],
        "model_execution": "not_run",
        "note": "Attach measured loss, gradient, support-mass, and FEC residual comparisons separately.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
