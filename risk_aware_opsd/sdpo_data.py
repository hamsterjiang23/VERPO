"""Public five-task data contract, independent of annotation and training code."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_REVISION = "7c457fc1b1f636ae794eb0362ba37d4743b06fbc"
SOURCE_ROOT = f"https://raw.githubusercontent.com/lasgroup/SDPO/{SOURCE_REVISION}"
SCHEMA_VERSION = "sdpo_rollout_group_v1"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    expected_train_rows: int
    expected_test_rows: int


DATASET_SPECS = {
    name: DatasetSpec(name, path, train, test)
    for name, path, train, test in (
        ("biology", "datasets/sciknoweval/biology", 450, 50),
        ("chemistry", "datasets/sciknoweval/chemistry", 1890, 210),
        ("material", "datasets/sciknoweval/material", 841, 94),
        ("physics", "datasets/sciknoweval/physics", 720, 80),
        ("tooluse", "datasets/tooluse", 4046, 68),
    )
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_rows(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or any(not isinstance(row, dict) for row in parsed):
        raise ValueError("source must be an array or JSONL sequence of objects")
    return parsed


def convert_rows(
    rows: list[dict[str, Any]], dataset: str, split: str
) -> list[dict[str, Any]]:
    if dataset not in DATASET_SPECS or split not in {"train", "test"}:
        raise ValueError("unknown dataset or split")
    converted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        answer = row.get("answer")
        if not isinstance(prompt, str) or not prompt.strip() or answer is None:
            raise ValueError(
                f"{dataset}/{split}/{index}: missing prompt or verifier answer"
            )
        record_id = f"sdpo_{dataset}_{split}_{row.get('idx', index)}"
        if record_id in seen:
            raise ValueError(f"duplicate record ID: {record_id}")
        seen.add(record_id)
        messages: list[dict[str, str]] = []
        if isinstance(row.get("system"), str) and row["system"].strip():
            messages.append({"role": "system", "content": row["system"]})
        messages.append({"role": "user", "content": prompt})
        converted.append(
            {
                "record_id": record_id,
                "data_source": "tooluse" if dataset == "tooluse" else "sciknoweval",
                "prompt": messages,
                "ability": "tooluse" if dataset == "tooluse" else "mcq",
                "reward_model": {
                    "style": "tooluse" if dataset == "tooluse" else "mcq",
                    "ground_truth": str(answer),
                },
                "extra_info": {
                    "split": split,
                    "index": str(row.get("idx", index)),
                    "problem": prompt,
                    "dataset": dataset,
                    "evidence_source": "rollout_group",
                    "source_revision": SOURCE_REVISION,
                },
            }
        )
    return converted
