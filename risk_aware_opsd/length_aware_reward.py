"""Shared SDPG/DAPO-style length-aware outcome reward utilities.

The active Qwen3-1.7B VERPO protocol uses one scalar utility everywhere:

    reward = accuracy - soft_overlong_penalty

The penalty is zero before the final buffer, increases linearly inside the
buffer, and is capped at ``penalty_factor``.  Keeping this helper independent
from either the TRL or veRL trainer prevents the two implementations from
silently drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_MAX_RESPONSE_LENGTH = 4096
DEFAULT_OVERLONG_BUFFER_LENGTH = 512
DEFAULT_OVERLONG_PENALTY_FACTOR = 1.0


@dataclass(frozen=True)
class RewardGroupState:
    """Row-level masks describing reward-ranked ZPD group states."""

    zpd_gate: torch.Tensor
    all_one: torch.Tensor
    all_zero: torch.Tensor
    all_negative_one: torch.Tensor
    zero_variance: torch.Tensor


def soft_overlong_penalty(
    response_lengths: torch.Tensor,
    *,
    max_response_length: int = DEFAULT_MAX_RESPONSE_LENGTH,
    buffer_length: int = DEFAULT_OVERLONG_BUFFER_LENGTH,
    penalty_factor: float = DEFAULT_OVERLONG_PENALTY_FACTOR,
) -> torch.Tensor:
    """Return the nonnegative DAPO soft-overlong penalty for each response."""
    max_response_length = int(max_response_length)
    buffer_length = int(buffer_length)
    penalty_factor = float(penalty_factor)
    if max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    if buffer_length <= 0 or buffer_length > max_response_length:
        raise ValueError("buffer_length must lie in [1, max_response_length]")
    if not 0.0 <= penalty_factor:
        raise ValueError("penalty_factor must be nonnegative")
    lengths = response_lengths.detach().to(dtype=torch.float32)
    free_length = float(max_response_length - buffer_length)
    return ((lengths - free_length) / float(buffer_length)).clamp(0.0, 1.0) * penalty_factor


def length_aware_outcome_rewards(
    correct_rows: torch.Tensor,
    response_lengths: torch.Tensor,
    *,
    max_response_length: int = DEFAULT_MAX_RESPONSE_LENGTH,
    buffer_length: int = DEFAULT_OVERLONG_BUFFER_LENGTH,
    penalty_factor: float = DEFAULT_OVERLONG_PENALTY_FACTOR,
) -> torch.Tensor:
    """Compute ``accuracy - soft_overlong_penalty`` as a float tensor."""
    if correct_rows.shape != response_lengths.shape:
        raise ValueError("correct_rows and response_lengths must have the same shape")
    accuracy = correct_rows.detach().to(device=response_lengths.device, dtype=torch.float32)
    penalty = soft_overlong_penalty(
        response_lengths,
        max_response_length=max_response_length,
        buffer_length=buffer_length,
        penalty_factor=penalty_factor,
    ).to(device=accuracy.device)
    return accuracy - penalty


def classify_reward_ranked_groups(
    raw_rewards: torch.Tensor,
    group_ids: list[str] | None = None,
    *,
    num_rollouts: int | None = None,
    epsilon: float = 0.0,
) -> RewardGroupState:
    """Classify groups for reward-ranked ZPD.

    All-one, all-zero, all-negative-one, and every other zero-variance group
    remain closed.  Every non-saturated group with reward variance above
    ``epsilon`` is admitted.  Returned masks have one value per input row.
    """
    rewards = raw_rewards.detach().float().flatten()
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    if group_ids is None:
        if num_rollouts is None or int(num_rollouts) <= 0:
            raise ValueError("num_rollouts must be positive when group_ids are omitted")
        group_ids = [str(index // int(num_rollouts)) for index in range(rewards.numel())]
    if len(group_ids) != rewards.numel():
        raise ValueError("group_ids and raw_rewards must have the same length")

    outputs = {
        "zpd_gate": torch.zeros_like(rewards, dtype=torch.bool),
        "all_one": torch.zeros_like(rewards, dtype=torch.bool),
        "all_zero": torch.zeros_like(rewards, dtype=torch.bool),
        "all_negative_one": torch.zeros_like(rewards, dtype=torch.bool),
        "zero_variance": torch.zeros_like(rewards, dtype=torch.bool),
    }
    groups: dict[str, list[int]] = {}
    for row, group_id in enumerate(group_ids):
        groups.setdefault(str(group_id), []).append(row)
    for indices in groups.values():
        values = rewards[indices]
        variance = ((values - values.mean()) ** 2).mean() if values.numel() > 1 else values.new_zeros(())
        all_one = bool(torch.isclose(values, torch.ones_like(values), atol=1e-6, rtol=0.0).all().item())
        all_zero = bool(torch.isclose(values, torch.zeros_like(values), atol=1e-6, rtol=0.0).all().item())
        all_negative_one = bool(
            torch.isclose(values, -torch.ones_like(values), atol=1e-6, rtol=0.0).all().item()
        )
        zero_variance = bool((variance <= float(epsilon)).item())
        outputs["all_one"][indices] = all_one
        outputs["all_zero"][indices] = all_zero
        outputs["all_negative_one"][indices] = all_negative_one
        outputs["zero_variance"][indices] = zero_variance
        outputs["zpd_gate"][indices] = not (
            all_one or all_zero or all_negative_one or zero_variance
        )
    return RewardGroupState(**outputs)
