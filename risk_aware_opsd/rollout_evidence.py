"""Deterministic, target-excluded evidence selection within one rollout batch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

EVIDENCE_SOURCE = "rollout_group"
REASONS = ("available", "no_positive_sibling", "no_negative_sibling")


@dataclass(frozen=True)
class EvidenceSelection:
    positive: tuple[int, ...]
    negative: tuple[tuple[int, ...], ...]
    reason: tuple[int, ...]

    @property
    def available(self) -> tuple[bool, ...]:
        return tuple(code == 0 for code in self.reason)


def select_rollout_evidence(
    group_ids: Sequence[str],
    rollout_ids: Sequence[str],
    correct: Sequence[bool],
    format_valid: Sequence[bool],
    responses: Sequence[str],
    *,
    require_negative: bool = True,
    num_negative: int = 1,
) -> EvidenceSelection:
    """Choose by stable rollout ID, never by transient batch position or reward."""
    size = len(group_ids)
    if any(
        len(values) != size
        for values in (rollout_ids, correct, format_valid, responses)
    ):
        raise ValueError("evidence metadata must have one entry per rollout")
    if len(set(rollout_ids)) != size or any(not value for value in rollout_ids):
        raise ValueError("rollout IDs must be nonempty and unique within the batch")
    if num_negative < 1:
        raise ValueError("num_negative must be positive")
    groups: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        groups.setdefault(group, []).append(index)

    def order(index: int) -> tuple[int, str]:
        parts = rollout_ids[index].rsplit("_", 2)
        # TransferQueue: prompt UID / integer rollout session / output turn.
        ordinal = parts[-2] if len(parts) == 3 and parts[-2].isdecimal() else parts[-1]
        return (int(ordinal) if ordinal.isdecimal() else 0, rollout_ids[index])

    positive: list[int] = []
    negative: list[tuple[int, ...]] = []
    reasons: list[int] = []
    for target, group in enumerate(group_ids):
        eligible = sorted(
            (
                i
                for i in groups[group]
                if i != target and format_valid[i] and responses[i].strip()
            ),
            key=order,
        )
        pos = next((i for i in eligible if correct[i]), -1)
        neg = [i for i in eligible if not correct[i]]
        positive.append(pos)
        negative.append(
            tuple(neg[k % len(neg)] if neg else -1 for k in range(num_negative))
        )
        reasons.append(1 if pos < 0 else 2 if require_negative and not neg else 0)
    return EvidenceSelection(tuple(positive), tuple(negative), tuple(reasons))
