"""Utilities for constructing model-conditional ZPD prompt profiles.

The profile is deliberately kept separate from the source parquet.  The
source rows remain byte-for-byte equivalent at the field level; generated
responses, correctness labels, and selection diagnostics live in auditable
JSONL artifacts rather than entering the Student prompt.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable


_QWEN_CHAT_PATTERNS = (
    re.compile(r"^user\n(?P<problem>.*)\nassistant\n?$", re.DOTALL),
    re.compile(
        r"^<\|im_start\|>user\n(?P<problem>.*)<\|im_end\|>\n"
        r"<\|im_start\|>assistant\n?$",
        re.DOTALL,
    ),
)


def canonical_problem(text: str) -> str:
    """Normalize harmless whitespace without changing mathematical content."""

    return "\n".join(line.rstrip() for line in str(text).strip().splitlines())


def problem_from_source_row(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    problem = extra_info.get("problem")
    if str(problem or "").strip():
        return canonical_problem(str(problem))
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        content = prompt[0].get("content") if isinstance(prompt[0], dict) else None
        if str(content or "").strip():
            return canonical_problem(str(content))
    raise ValueError("source row has no canonical problem")


def problem_from_rollout_row(row: dict[str, Any]) -> str:
    """Recover the raw user problem from a saved veRL or profiler rollout."""

    direct = row.get("problem") or row.get("raw_user_text")
    if str(direct or "").strip():
        return canonical_problem(str(direct))
    rendered = str(row.get("input") or row.get("rendered_prompt") or "")
    if not rendered.strip():
        raise ValueError("rollout row has no non-empty prompt")
    for pattern in _QWEN_CHAT_PATTERNS:
        match = pattern.match(rendered)
        if match:
            return canonical_problem(match.group("problem"))
    raise ValueError("unsupported rendered rollout prompt format")


def problem_sha256(problem: str) -> str:
    return hashlib.sha256(canonical_problem(problem).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ZPDProfile:
    problem: str
    rollout_count: int
    correct_count: int
    valid_wrong_count: int
    invalid_count: int
    available_rollout_count: int
    first_step: int | None
    last_step: int | None

    @property
    def empirical_accuracy(self) -> float:
        return self.correct_count / self.rollout_count

    @property
    def format_valid_rate(self) -> float:
        return (self.correct_count + self.valid_wrong_count) / self.rollout_count

    @property
    def available_rollout_rate(self) -> float:
        return self.available_rollout_count / self.rollout_count

    @property
    def has_bilateral_valid_siblings(self) -> bool:
        return self.correct_count > 0 and self.valid_wrong_count > 0

    @property
    def composition(self) -> str:
        return (
            f"{self.correct_count}_correct+{self.valid_wrong_count}_valid_wrong+"
            f"{self.invalid_count}_invalid"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_sha256": problem_sha256(self.problem),
            "problem": self.problem,
            "rollout_count": self.rollout_count,
            "correct_count": self.correct_count,
            "valid_wrong_count": self.valid_wrong_count,
            "invalid_count": self.invalid_count,
            "empirical_accuracy": self.empirical_accuracy,
            "format_valid_rate": self.format_valid_rate,
            "available_rollout_count": self.available_rollout_count,
            "available_rollout_rate": self.available_rollout_rate,
            "has_bilateral_valid_siblings": self.has_bilateral_valid_siblings,
            "composition": self.composition,
            "first_step": self.first_step,
            "last_step": self.last_step,
        }


def target_excluded_available_count(
    *, correct: int, valid_wrong: int, invalid: int
) -> int:
    """Count rows that can find another valid positive and valid negative.

    This mirrors the target-excluded sibling rule.  Invalid current rows may
    still be admitted when their *other* siblings contain both valid classes.
    """

    available = 0
    if correct >= 2 and valid_wrong >= 1:
        available += correct
    if correct >= 1 and valid_wrong >= 2:
        available += valid_wrong
    if correct >= 1 and valid_wrong >= 1:
        available += invalid
    return available


def build_profiles(
    rows: Iterable[dict[str, Any]],
    *,
    format_valid: Callable[[dict[str, Any]], bool],
) -> list[ZPDProfile]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        uid = str(row.get("uid") or "")
        if uid.startswith("pad") or not str(row.get("input") or row.get("problem") or "").strip():
            continue
        grouped[problem_from_rollout_row(row)].append(row)

    profiles = []
    for problem, prompt_rows in grouped.items():
        correct = sum(float(row.get("score", 0.0)) > 0.0 for row in prompt_rows)
        valid = sum(bool(format_valid(row)) for row in prompt_rows)
        valid_wrong = valid - correct
        if valid_wrong < 0:
            raise ValueError("a correct rollout was marked format-invalid")
        invalid = len(prompt_rows) - valid
        steps = [int(row["step"]) for row in prompt_rows if row.get("step") is not None]
        profiles.append(
            ZPDProfile(
                problem=problem,
                rollout_count=len(prompt_rows),
                correct_count=correct,
                valid_wrong_count=valid_wrong,
                invalid_count=invalid,
                available_rollout_count=target_excluded_available_count(
                    correct=correct, valid_wrong=valid_wrong, invalid=invalid
                ),
                first_step=min(steps) if steps else None,
                last_step=max(steps) if steps else None,
            )
        )
    return sorted(profiles, key=lambda item: problem_sha256(item.problem))


def select_zpd_profiles(
    profiles: Iterable[ZPDProfile],
    *,
    min_rollouts: int,
    min_accuracy: float,
    max_accuracy: float,
    min_format_valid_rate: float,
) -> list[ZPDProfile]:
    if min_rollouts < 2:
        raise ValueError("min_rollouts must be at least two")
    if not 0.0 <= min_accuracy <= max_accuracy <= 1.0:
        raise ValueError("accuracy bounds must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= min_format_valid_rate <= 1.0:
        raise ValueError("min_format_valid_rate must be in [0, 1]")
    return [
        profile
        for profile in profiles
        if profile.rollout_count >= min_rollouts
        and profile.has_bilateral_valid_siblings
        and min_accuracy <= profile.empirical_accuracy <= max_accuracy
        and profile.format_valid_rate >= min_format_valid_rate
    ]
