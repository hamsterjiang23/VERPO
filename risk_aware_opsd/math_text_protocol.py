"""JSONL/math-text data contract for the TRL VERPO entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON objects and reject malformed records early."""

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield value


def load_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


def validate_math_text_row(row: dict[str, Any]) -> None:
    if not str(row.get("prompt", "")).strip():
        raise ValueError("math-text row requires a non-empty prompt")
    if "completion" not in row and "response" not in row:
        raise ValueError("math-text row requires completion or response")


__all__ = ["iter_jsonl", "load_jsonl", "validate_math_text_row"]
