# Copyright 2026 VERPO-ZPD contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exact full-vocabulary VERPO-ZPD loss for the native veRL FSDP actor.

The original protocol uses one fixed reference for q_ref/q0/q+/q-.  Moving
``snapshot`` and ``ema`` modes instead evaluate every Teacher branch with one
actor-side shadow refreshed after successful optimizer updates.  Only
response-token logits are retained.  Teacher tensors and controller weights
are stop-gradient; gradients flow through Student logits. The opt-in
``allow_negative_benefit`` ablation uses the raw signed controller numerator
without post-clamping and fails closed if the resulting weight is non-finite.

Under moving ``snapshot``/``ema`` modes the reference term is anchored to the
same moving shadow that supplies q0, not to a separate frozen initial policy.
This is a deliberate VERPO_CONTRASTIVE-matched deviation from the VERPO manuscript, which
specifies a third frozen q_ref; it must stay recorded wherever these runs are
reported.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from risk_aware_opsd.length_aware_reward import classify_reward_ranked_groups
from risk_aware_opsd.verpo_zpd import VERPO_EFFECTIVE_WEIGHT_THRESHOLD

from verl.trainer.distillation.verpo_topk import (
    build_truncated_topk_support,
    forward_moments,
    gather_full_softmax_log_probs,
    gather_pre_normalized_log_probs,
    reverse_geometric_correction,
    reverse_tangent_moments,
    should_use_full_vocab,
    support_mass,
    validate_verpo_vocab_config,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_torch_device
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import modulate_advantages, ppo_loss
from verl.workers.utils.padding import no_padding_2_padding


def _validate_verpo_controller(
    *,
    tau: float,
    rho: float,
    cost_floor: float,
    cost_beta: float,
    cost_alpha: float | None = None,
    cost_epsilon: float | None = None,
    temperature: float,
    allow_negative_benefit: bool = False,
) -> None:
    values = {
        "tau": float(tau),
        "rho": float(rho),
        "cost_floor": float(cost_floor),
        "cost_beta": float(cost_beta),
        "temperature": float(temperature),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("VERPO controller values must be finite")
    if values["tau"] <= 0 or values["temperature"] <= 0:
        raise ValueError("temperature and tau must be positive")
    if cost_alpha is None and cost_epsilon is None:
        if min(
            values["rho"],
            values["cost_floor"],
            values["cost_beta"],
        ) < 0:
            raise ValueError("VERPO cost parameters must be nonnegative")
        if values["rho"] + values["cost_floor"] <= 0:
            raise ValueError("rho + cost_floor must define a positive movement-cost term")
    elif cost_alpha is None or cost_epsilon is None:
        raise ValueError("cost_alpha and cost_epsilon must be set together")
    elif float(cost_alpha) < 0 or float(cost_epsilon) <= 0:
        raise ValueError("cost_alpha must be nonnegative and cost_epsilon positive")


def _resolve_cost_parameters(
    *,
    cost_alpha: float | None,
    cost_epsilon: float | None,
    tau: float,
    rho: float,
    cost_floor: float,
    cost_beta: float,
) -> tuple[float, float]:
    """Resolve the active benefit-space cost scale and positive floor."""
    if cost_alpha is None and cost_epsilon is None:
        cost_alpha = float(tau) * float(cost_beta)
        cost_epsilon = float(tau) * (float(rho) + float(cost_floor))
    elif cost_alpha is None or cost_epsilon is None:
        raise ValueError("cost_alpha and cost_epsilon must be set together")
    alpha = float(cost_alpha)
    epsilon = float(cost_epsilon)
    if not math.isfinite(alpha) or not math.isfinite(epsilon):
        raise ValueError("VERPO effective cost parameters must be finite")
    if alpha < 0 or epsilon <= 0:
        raise ValueError("cost_alpha must be nonnegative and cost_epsilon positive")
    return alpha, epsilon


def compute_group_zpd_gate_by_uid(
    raw_rewards: torch.Tensor,
    uids: list[str],
    *,
    mode: str = "binary_mixed",
    positive_reward: float = 1.0,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Return either the legacy binary-mixed or reward-ranked UID gate."""
    if mode not in {"binary_mixed", "reward_ranked"}:
        raise ValueError("group ZPD mode must be binary_mixed or reward_ranked")
    rewards = raw_rewards
    if mode == "binary_mixed":
        rewards = (raw_rewards >= float(positive_reward) - 1e-6).float()
    return classify_reward_ranked_groups(
        rewards,
        group_ids=[str(uid) for uid in uids],
        epsilon=epsilon,
    ).zpd_gate


def apply_evidence_rollout_scope(
    group_gate: torch.Tensor,
    outcome_positive_rows: torch.Tensor,
    *,
    scope: str = "all",
) -> torch.Tensor:
    """Apply the post-group-ZPD rollout scope to evidence correction only."""
    if group_gate.shape != outcome_positive_rows.shape:
        raise ValueError("group_gate and outcome_positive_rows must have the same shape")
    if scope not in {"all", "wrong_only"}:
        raise ValueError("scope must be all or wrong_only")
    gate = group_gate.detach().bool()
    if scope == "wrong_only":
        gate = gate & ~outcome_positive_rows.detach().bool()
    return gate


def _compute_flat_verpo_losses_full(
    student_logits: torch.Tensor,
    base_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    token_advantages: torch.Tensor,
    active_token_mask: torch.Tensor,
    *,
    negative_teacher_logits: torch.Tensor | None = None,
    tau: float,
    rho: float,
    cost_floor: float,
    cost_beta: float,
    cost_alpha: float | None = None,
    cost_epsilon: float | None = None,
    allow_negative_benefit: bool = False,
    temperature: float,
    vocab_chunk_size: int,
    divergence: str = "forward_kl",
    displacement_mode: str = "evidence_vs_none",
    projection_epsilon: float = 1e-8,
    vocab_mode: str = "topk_truncated",
    top_k: int = 128,
) -> dict[str, torch.Tensor]:
    """Compute Forward- or Reverse-KL VERPO losses on flattened token rows.

    Shapes are ``[N, V]`` for logits and ``[N]`` for token-level inputs.  This
    is equivalent to the TRL ``[B, T, V]`` implementation after flattening.
    Reverse KL uses the normalized geometric Teacher path and the exact local
    tangent ``xi = F(p) (log q_e - log q_b)`` for ZPD benefit and cost, where
    ``q_b=q_0`` for Fixed and ``q_b=q_-`` for CTR.
    """
    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [N, V]")
    if student_logits.shape != base_teacher_logits.shape or student_logits.shape != evidence_teacher_logits.shape:
        raise ValueError("Student, q0, and qe logits must have identical shapes")
    n_tokens = student_logits.shape[0]
    for name, value in {
        "sampled_token_ids": sampled_token_ids,
        "token_advantages": token_advantages,
        "active_token_mask": active_token_mask,
    }.items():
        if value.shape != (n_tokens,):
            raise ValueError(f"{name} must have shape [N]")
    _validate_verpo_controller(
        tau=tau,
        rho=rho,
        cost_floor=cost_floor,
        cost_beta=cost_beta,
        cost_alpha=cost_alpha,
        cost_epsilon=cost_epsilon,
        temperature=temperature,
        allow_negative_benefit=allow_negative_benefit,
    )
    effective_cost_alpha, effective_cost_epsilon = _resolve_cost_parameters(
        cost_alpha=cost_alpha,
        cost_epsilon=cost_epsilon,
        tau=tau,
        rho=rho,
        cost_floor=cost_floor,
        cost_beta=cost_beta,
    )
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    if divergence not in {"forward_kl", "reverse_kl"}:
        raise ValueError("divergence must be 'forward_kl' or 'reverse_kl'")
    contrastive_modes = {"correct_vs_incorrect", "reward_ranked", "fec"}
    if displacement_mode not in {"evidence_vs_none", *contrastive_modes}:
        raise ValueError(
            "displacement_mode must be 'evidence_vs_none', "
            "'correct_vs_incorrect', 'reward_ranked', or 'fec'"
        )
    if displacement_mode != "evidence_vs_none":
        if negative_teacher_logits is None:
            raise ValueError(f"{displacement_mode} requires negative_teacher_logits")
        if negative_teacher_logits.shape != student_logits.shape:
            raise ValueError("Student and negative Teacher logits must have identical shapes")
    if not math.isfinite(float(projection_epsilon)) or float(projection_epsilon) <= 0:
        raise ValueError("projection_epsilon must be finite and positive")
    validate_verpo_vocab_config(vocab_mode, top_k)

    scale = float(temperature)
    chunk_size = int(vocab_chunk_size)
    with torch.no_grad():
        q0_log_z = torch.logsumexp(base_teacher_logits.float() / scale, dim=-1, keepdim=True)
        qe_log_z = torch.logsumexp(evidence_teacher_logits.float() / scale, dim=-1, keepdim=True)
        qn_log_z = (
            torch.logsumexp(negative_teacher_logits.float() / scale, dim=-1, keepdim=True)
            if negative_teacher_logits is not None
            else None
        )
        reverse_base_teacher_logits = (
            negative_teacher_logits if displacement_mode in contrastive_modes else base_teacher_logits
        )
        reverse_base_log_z = qn_log_z if displacement_mode in contrastive_modes else q0_log_z
        if reverse_base_teacher_logits is None or reverse_base_log_z is None:
            raise AssertionError("Reverse-KL correction base Teacher is missing")
        p_log_z_detached = torch.logsumexp(student_logits.detach().float() / scale, dim=-1, keepdim=True)
        sampled_ids = sampled_token_ids.long().unsqueeze(-1)
        q0_sample_log_prob = (base_teacher_logits.gather(-1, sampled_ids).float() / scale - q0_log_z).squeeze(-1)

        if divergence == "forward_kl":
            mean_task = torch.zeros(n_tokens, dtype=torch.float32, device=student_logits.device)
            second_task = torch.zeros_like(mean_task)
            task_norm_sq = torch.zeros_like(mean_task)
            mean_nuisance = torch.zeros_like(mean_task)
            second_nuisance = torch.zeros_like(mean_task)
            cross_moment = torch.zeros_like(mean_task)
            nuisance_norm_sq = torch.zeros_like(mean_task)
            for start in range(0, student_logits.shape[-1], chunk_size):
                stop = min(start + chunk_size, student_logits.shape[-1])
                q0 = (base_teacher_logits[:, start:stop].float() / scale - q0_log_z).exp()
                qe = (evidence_teacher_logits[:, start:stop].float() / scale - qe_log_z).exp()
                p = (student_logits[:, start:stop].detach().float() / scale - p_log_z_detached).exp()
                if displacement_mode == "evidence_vs_none":
                    task = qe - q0
                    nuisance = torch.zeros_like(task)
                else:
                    if qn_log_z is None or negative_teacher_logits is None:
                        raise AssertionError("contrastive Teacher normalization is missing")
                    qn = (negative_teacher_logits[:, start:stop].float() / scale - qn_log_z).exp()
                    task = qe - qn
                    nuisance = 0.5 * (qe + qn) - q0 if displacement_mode == "fec" else torch.zeros_like(task)
                mean_task += (p * task).sum(dim=-1)
                second_task += (p * task.square()).sum(dim=-1)
                task_norm_sq += task.square().sum(dim=-1)
                mean_nuisance += (p * nuisance).sum(dim=-1)
                second_nuisance += (p * nuisance.square()).sum(dim=-1)
                cross_moment += (p * task * nuisance).sum(dim=-1)
                nuisance_norm_sq += nuisance.square().sum(dim=-1)

            task_variance = (second_task - mean_task.square()).clamp_min(0.0)
            nuisance_variance = (second_nuisance - mean_nuisance.square()).clamp_min(0.0)
            covariance = cross_moment - mean_task * mean_nuisance
            if displacement_mode == "fec":
                projection = covariance / (nuisance_variance + float(projection_epsilon))
                mean_delta = mean_task - projection * mean_nuisance
                fisher_cost = (
                    task_variance - 2.0 * projection * covariance + projection.square() * nuisance_variance
                ).clamp_min(0.0)
                fisher_cosine = covariance / torch.sqrt(
                    (task_variance + float(projection_epsilon)) * (nuisance_variance + float(projection_epsilon))
                )
                fisher_cosine = fisher_cosine.clamp(min=-1.0, max=1.0)
                residual_nuisance_covariance = covariance - projection * nuisance_variance
                displacement_norm_sq = torch.zeros_like(mean_task)
                for start in range(0, student_logits.shape[-1], chunk_size):
                    stop = min(start + chunk_size, student_logits.shape[-1])
                    qe = (evidence_teacher_logits[:, start:stop].float() / scale - qe_log_z).exp()
                    q0 = (base_teacher_logits[:, start:stop].float() / scale - q0_log_z).exp()
                    if qn_log_z is None or negative_teacher_logits is None:
                        raise AssertionError("contrastive Teacher normalization is missing")
                    qn = (negative_teacher_logits[:, start:stop].float() / scale - qn_log_z).exp()
                    task = qe - qn
                    nuisance = 0.5 * (qe + qn) - q0
                    delta = task - projection.unsqueeze(-1) * nuisance
                    displacement_norm_sq += delta.square().sum(dim=-1)
            else:
                projection = torch.zeros_like(mean_task)
                mean_delta = mean_task
                fisher_cost = task_variance
                fisher_cosine = torch.zeros_like(mean_task)
                displacement_norm_sq = task_norm_sq
                residual_nuisance_covariance = torch.zeros_like(mean_task)

            qe_at_sample = (
                (evidence_teacher_logits.gather(-1, sampled_ids).float() / scale - qe_log_z).exp().squeeze(-1)
            )
            q0_at_sample = (base_teacher_logits.gather(-1, sampled_ids).float() / scale - q0_log_z).exp().squeeze(-1)
            if displacement_mode == "evidence_vs_none":
                delta_at_sample = qe_at_sample - q0_at_sample
            else:
                if qn_log_z is None or negative_teacher_logits is None:
                    raise AssertionError("contrastive Teacher normalization is missing")
                qn_at_sample = (
                    (negative_teacher_logits.gather(-1, sampled_ids).float() / scale - qn_log_z).exp().squeeze(-1)
                )
                task_at_sample = qe_at_sample - qn_at_sample
                if displacement_mode == "fec":
                    nuisance_at_sample = 0.5 * (qe_at_sample + qn_at_sample) - q0_at_sample
                    delta_at_sample = task_at_sample - projection * nuisance_at_sample
                else:
                    delta_at_sample = task_at_sample
            alignment = delta_at_sample - mean_delta
            benefit = token_advantages.detach().float() * alignment
        else:
            mean_task_log_ratio = torch.zeros(n_tokens, dtype=torch.float32, device=student_logits.device)
            mean_nuisance_log_ratio = torch.zeros_like(mean_task_log_ratio)
            for start in range(0, student_logits.shape[-1], chunk_size):
                stop = min(start + chunk_size, student_logits.shape[-1])
                p = (student_logits[:, start:stop].detach().float() / scale - p_log_z_detached).exp()
                reverse_base_log_probs = reverse_base_teacher_logits[:, start:stop].float() / scale - reverse_base_log_z
                qe_log_probs = evidence_teacher_logits[:, start:stop].float() / scale - qe_log_z
                task_log_ratio = qe_log_probs - reverse_base_log_probs
                if displacement_mode == "fec":
                    q0_log_probs = base_teacher_logits[:, start:stop].float() / scale - q0_log_z
                    nuisance_log_ratio = 0.5 * (qe_log_probs + reverse_base_log_probs) - q0_log_probs
                else:
                    nuisance_log_ratio = torch.zeros_like(task_log_ratio)
                mean_task_log_ratio += (p * task_log_ratio).sum(dim=-1)
                mean_nuisance_log_ratio += (p * nuisance_log_ratio).sum(dim=-1)

            mean_task_tangent = torch.zeros_like(mean_task_log_ratio)
            mean_nuisance_tangent = torch.zeros_like(mean_task_log_ratio)
            second_task_tangent = torch.zeros_like(mean_task_log_ratio)
            second_nuisance_tangent = torch.zeros_like(mean_task_log_ratio)
            cross_tangent_moment = torch.zeros_like(mean_task_log_ratio)
            task_norm_sq = torch.zeros_like(mean_task_log_ratio)
            nuisance_norm_sq = torch.zeros_like(mean_task_log_ratio)
            task_nuisance_l2_cross = torch.zeros_like(mean_task_log_ratio)
            for start in range(0, student_logits.shape[-1], chunk_size):
                stop = min(start + chunk_size, student_logits.shape[-1])
                p = (student_logits[:, start:stop].detach().float() / scale - p_log_z_detached).exp()
                reverse_base_log_probs = reverse_base_teacher_logits[:, start:stop].float() / scale - reverse_base_log_z
                qe_log_probs = evidence_teacher_logits[:, start:stop].float() / scale - qe_log_z
                task_log_ratio = qe_log_probs - reverse_base_log_probs
                if displacement_mode == "fec":
                    q0_log_probs = base_teacher_logits[:, start:stop].float() / scale - q0_log_z
                    nuisance_log_ratio = 0.5 * (qe_log_probs + reverse_base_log_probs) - q0_log_probs
                else:
                    nuisance_log_ratio = torch.zeros_like(task_log_ratio)
                task_tangent = p * (task_log_ratio - mean_task_log_ratio.unsqueeze(-1))
                nuisance_tangent = p * (nuisance_log_ratio - mean_nuisance_log_ratio.unsqueeze(-1))
                mean_task_tangent += (p * task_tangent).sum(dim=-1)
                mean_nuisance_tangent += (p * nuisance_tangent).sum(dim=-1)
                second_task_tangent += (p * task_tangent.square()).sum(dim=-1)
                second_nuisance_tangent += (p * nuisance_tangent.square()).sum(dim=-1)
                cross_tangent_moment += (p * task_tangent * nuisance_tangent).sum(dim=-1)
                task_norm_sq += task_tangent.square().sum(dim=-1)
                nuisance_norm_sq += nuisance_tangent.square().sum(dim=-1)
                task_nuisance_l2_cross += (task_tangent * nuisance_tangent).sum(dim=-1)

            task_fisher = (second_task_tangent - mean_task_tangent.square()).clamp_min(0.0)
            nuisance_fisher = (second_nuisance_tangent - mean_nuisance_tangent.square()).clamp_min(0.0)
            covariance = cross_tangent_moment - mean_task_tangent * mean_nuisance_tangent
            if displacement_mode == "fec":
                projection = covariance / (nuisance_fisher + float(projection_epsilon))
                mean_tangent = mean_task_tangent - projection * mean_nuisance_tangent
                fisher_cost = (
                    task_fisher - 2.0 * projection * covariance + projection.square() * nuisance_fisher
                ).clamp_min(0.0)
                displacement_norm_sq = (
                    task_norm_sq - 2.0 * projection * task_nuisance_l2_cross + projection.square() * nuisance_norm_sq
                ).clamp_min(0.0)
                fisher_cosine = covariance / torch.sqrt(
                    (task_fisher + float(projection_epsilon)) * (nuisance_fisher + float(projection_epsilon))
                )
                fisher_cosine = fisher_cosine.clamp(min=-1.0, max=1.0)
                residual_nuisance_covariance = covariance - projection * nuisance_fisher
            else:
                projection = torch.zeros_like(mean_task_log_ratio)
                mean_tangent = mean_task_tangent
                fisher_cost = task_fisher
                displacement_norm_sq = task_norm_sq
                fisher_cosine = torch.zeros_like(mean_task_log_ratio)
                residual_nuisance_covariance = torch.zeros_like(mean_task_log_ratio)

            p_at_sample = (
                (student_logits.gather(-1, sampled_ids).detach().float() / scale - p_log_z_detached).exp().squeeze(-1)
            )
            qe_sample_log_prob = (evidence_teacher_logits.gather(-1, sampled_ids).float() / scale - qe_log_z).squeeze(
                -1
            )
            reverse_base_sample_log_prob = (
                reverse_base_teacher_logits.gather(-1, sampled_ids).float() / scale - reverse_base_log_z
            ).squeeze(-1)
            task_log_ratio_at_sample = qe_sample_log_prob - reverse_base_sample_log_prob
            task_tangent_at_sample = p_at_sample * (task_log_ratio_at_sample - mean_task_log_ratio)
            if displacement_mode == "fec":
                nuisance_log_ratio_at_sample = (
                    0.5 * (qe_sample_log_prob + reverse_base_sample_log_prob) - q0_sample_log_prob
                )
                nuisance_tangent_at_sample = p_at_sample * (nuisance_log_ratio_at_sample - mean_nuisance_log_ratio)
                tangent_at_sample = task_tangent_at_sample - projection * nuisance_tangent_at_sample
            else:
                tangent_at_sample = task_tangent_at_sample
            alignment = tangent_at_sample - mean_tangent
            benefit = token_advantages.detach().float() * alignment

        h = benefit if allow_negative_benefit else benefit.clamp_min(0.0)
        cost = effective_cost_alpha * fisher_cost + effective_cost_epsilon
        weights = torch.where(
            active_token_mask.detach().bool(),
            h / (h + cost),
            torch.zeros_like(h),
        )
        if not bool(torch.isfinite(weights).all().item()):
            raise ValueError("VERPO token weights must be finite")

    student_log_z = torch.logsumexp(student_logits.float() / scale, dim=-1, keepdim=True)
    reference_loss = torch.zeros(n_tokens, dtype=torch.float32, device=student_logits.device)
    evidence_loss = torch.zeros_like(reference_loss)
    if divergence == "forward_kl":
        for start in range(0, student_logits.shape[-1], chunk_size):
            stop = min(start + chunk_size, student_logits.shape[-1])
            student_log_probs = student_logits[:, start:stop].float() / scale - student_log_z
            with torch.no_grad():
                q0_log_probs = base_teacher_logits[:, start:stop].float() / scale - q0_log_z
                qe_log_probs = evidence_teacher_logits[:, start:stop].float() / scale - qe_log_z
                q0 = q0_log_probs.exp()
                qe = qe_log_probs.exp()
                if displacement_mode == "evidence_vs_none":
                    delta = qe - q0
                else:
                    if qn_log_z is None or negative_teacher_logits is None:
                        raise AssertionError("contrastive Teacher normalization is missing")
                    qn = (negative_teacher_logits[:, start:stop].float() / scale - qn_log_z).exp()
                    delta = qe - qn
                    if displacement_mode == "fec":
                        nuisance = 0.5 * (qe + qn) - q0
                        delta = delta - projection.unsqueeze(-1) * nuisance
            reference_loss += scale * (q0 * (q0_log_probs - student_log_probs)).sum(dim=-1)
            evidence_loss -= scale * weights * (delta * student_log_probs).sum(dim=-1)
    else:
        with torch.no_grad():
            geometric_log_z = torch.full_like(weights, -torch.inf)
            for start in range(0, student_logits.shape[-1], chunk_size):
                stop = min(start + chunk_size, student_logits.shape[-1])
                reverse_base_log_probs = reverse_base_teacher_logits[:, start:stop].float() / scale - reverse_base_log_z
                qe_log_probs = evidence_teacher_logits[:, start:stop].float() / scale - qe_log_z
                log_ratio = qe_log_probs - reverse_base_log_probs
                if displacement_mode == "fec":
                    q0_log_probs = base_teacher_logits[:, start:stop].float() / scale - q0_log_z
                    nuisance_log_ratio = 0.5 * (qe_log_probs + reverse_base_log_probs) - q0_log_probs
                    log_ratio = log_ratio - projection.unsqueeze(-1) * (nuisance_log_ratio)
                geometric_log_density = reverse_base_log_probs + weights.unsqueeze(-1) * log_ratio
                geometric_log_z = torch.logaddexp(
                    geometric_log_z,
                    torch.logsumexp(geometric_log_density, dim=-1),
                )

        expected_log_ratio = torch.zeros_like(reference_loss)
        for start in range(0, student_logits.shape[-1], chunk_size):
            stop = min(start + chunk_size, student_logits.shape[-1])
            student_log_probs = student_logits[:, start:stop].float() / scale - student_log_z
            student_probs = student_log_probs.exp()
            with torch.no_grad():
                q0_log_probs = base_teacher_logits[:, start:stop].float() / scale - q0_log_z
                qe_log_probs = evidence_teacher_logits[:, start:stop].float() / scale - qe_log_z
                reverse_base_log_probs = reverse_base_teacher_logits[:, start:stop].float() / scale - reverse_base_log_z
                log_ratio = qe_log_probs - reverse_base_log_probs
                if displacement_mode == "fec":
                    nuisance_log_ratio = 0.5 * (qe_log_probs + reverse_base_log_probs) - q0_log_probs
                    log_ratio = log_ratio - projection.unsqueeze(-1) * (nuisance_log_ratio)
            reference_loss += scale * (student_probs * (student_log_probs - q0_log_probs)).sum(dim=-1)
            expected_log_ratio += (student_probs * log_ratio).sum(dim=-1)
        evidence_loss = scale * (geometric_log_z - weights * expected_log_ratio)

    result = {
        "reference_loss": reference_loss,
        "evidence_loss": evidence_loss,
        "weights": weights,
        "benefit": benefit,
        "fisher_cost": fisher_cost,
        "displacement_norm": displacement_norm_sq.sqrt(),
        "task_displacement_norm": task_norm_sq.sqrt(),
        "nuisance_displacement_norm": nuisance_norm_sq.sqrt(),
        "task_nuisance_fisher_cosine": fisher_cosine,
        "nuisance_projection_coefficient": projection,
        "residual_nuisance_fisher_covariance": (residual_nuisance_covariance),
        "alignment": alignment,
        "q0_sample_log_prob": q0_sample_log_prob,
    }
    if should_use_full_vocab(vocab_mode, top_k, student_logits.shape[-1]):
        result.update(
            reference_support_size=torch.full_like(
                reference_loss, float(student_logits.shape[-1])
            ),
            support_size=torch.full_like(reference_loss, float(student_logits.shape[-1])),
            reference_support_mass=torch.ones_like(reference_loss),
            evidence_support_mass=torch.ones_like(reference_loss),
            student_support_mass=torch.ones_like(reference_loss),
        )
        if displacement_mode != "evidence_vs_none":
            result["negative_support_mass"] = torch.ones_like(reference_loss)
    else:
        support = build_truncated_topk_support(
            teacher_logits=[base_teacher_logits, evidence_teacher_logits]
            + ([negative_teacher_logits] if negative_teacher_logits is not None else []),
            student_logits=student_logits if divergence == "reverse_kl" else None,
            sampled_token_ids=sampled_token_ids,
            top_k=top_k,
        )
        result.update(
            support_size=support.valid_mask.sum(dim=-1).float(),
            reference_support_mass=support_mass(
                gather_full_softmax_log_probs(base_teacher_logits, support, temperature=temperature), support
            ),
            evidence_support_mass=support_mass(
                gather_full_softmax_log_probs(evidence_teacher_logits, support, temperature=temperature), support
            ),
            student_support_mass=support_mass(
                gather_full_softmax_log_probs(student_logits, support, temperature=temperature), support
            ),
        )
        if negative_teacher_logits is not None:
            result["negative_support_mass"] = support_mass(
                gather_full_softmax_log_probs(negative_teacher_logits, support, temperature=temperature), support
            )
    return result


def _compute_flat_truncated_verpo_losses(
    student_logits: torch.Tensor,
    base_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    token_advantages: torch.Tensor,
    active_token_mask: torch.Tensor,
    *,
    negative_teacher_logits: torch.Tensor | None,
    negative_teacher_probabilities: torch.Tensor | None,
    tau: float,
    rho: float,
    cost_floor: float,
    cost_beta: float,
    cost_alpha: float | None = None,
    cost_epsilon: float | None = None,
    allow_negative_benefit: bool = False,
    temperature: float,
    divergence: str,
    displacement_mode: str,
    projection_epsilon: float,
    top_k: int,
    vocab_chunk_size: int,
) -> dict[str, torch.Tensor]:
    """Compute the explicit-support VERPO approximation on flattened rows."""
    _validate_verpo_controller(
        tau=tau,
        rho=rho,
        cost_floor=cost_floor,
        cost_beta=cost_beta,
        cost_alpha=cost_alpha,
        cost_epsilon=cost_epsilon,
        temperature=temperature,
        allow_negative_benefit=allow_negative_benefit,
    )
    effective_cost_alpha, effective_cost_epsilon = _resolve_cost_parameters(
        cost_alpha=cost_alpha,
        cost_epsilon=cost_epsilon,
        tau=tau,
        rho=rho,
        cost_floor=cost_floor,
        cost_beta=cost_beta,
    )
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    if divergence not in {"forward_kl", "reverse_kl"}:
        raise ValueError("divergence must be 'forward_kl' or 'reverse_kl'")
    if (
        negative_teacher_logits is not None
        and negative_teacher_probabilities is not None
    ):
        raise ValueError(
            "negative Teacher logits and probabilities are mutually exclusive"
        )
    if (
        negative_teacher_probabilities is not None
        and negative_teacher_probabilities.shape != student_logits.shape
    ):
        raise ValueError(
            "Student and negative Teacher probabilities must have identical shapes"
        )
    negative_teacher_source = (
        negative_teacher_probabilities
        if negative_teacher_probabilities is not None
        else negative_teacher_logits
    )
    reference_support = build_truncated_topk_support(
        teacher_logits=[base_teacher_logits],
        student_logits=student_logits if divergence == "reverse_kl" else None,
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    if displacement_mode == "evidence_vs_none":
        correction_teachers = [base_teacher_logits, evidence_teacher_logits]
    elif displacement_mode in {"correct_vs_incorrect", "reward_ranked"}:
        if negative_teacher_source is None:
            raise ValueError(f"{displacement_mode} requires a negative Teacher")
        correction_teachers = [evidence_teacher_logits, negative_teacher_source]
    else:
        if negative_teacher_source is None:
            raise ValueError("fec requires a negative Teacher")
        correction_teachers = [
            base_teacher_logits,
            evidence_teacher_logits,
            negative_teacher_source,
        ]
    correction_support = build_truncated_topk_support(
        teacher_logits=correction_teachers,
        student_logits=student_logits if divergence == "reverse_kl" else None,
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    lp_ref = gather_full_softmax_log_probs(
        student_logits,
        reference_support,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk_size,
    )
    l0_ref = gather_full_softmax_log_probs(
        base_teacher_logits,
        reference_support,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk_size,
    ).detach()
    lp = gather_full_softmax_log_probs(
        student_logits,
        correction_support,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk_size,
    )
    l0 = gather_full_softmax_log_probs(
        base_teacher_logits,
        correction_support,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk_size,
    ).detach()
    le = gather_full_softmax_log_probs(
        evidence_teacher_logits,
        correction_support,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk_size,
    ).detach()
    reference_valid = reference_support.valid_mask
    correction_valid = correction_support.valid_mask
    if negative_teacher_probabilities is not None:
        qn_log_probs = gather_pre_normalized_log_probs(
            negative_teacher_probabilities, correction_support
        ).detach()
    elif negative_teacher_logits is not None:
        qn_log_probs = gather_full_softmax_log_probs(
            negative_teacher_logits,
            correction_support,
            temperature=temperature,
            vocab_chunk_size=vocab_chunk_size,
        ).detach()
    else:
        qn_log_probs = l0
    with torch.no_grad():
        p_controller_log = lp.detach()
        q0 = torch.where(correction_valid, l0.exp(), torch.zeros_like(l0))
        qe = torch.where(correction_valid, le.exp(), torch.zeros_like(le))
        qn = torch.where(
            correction_valid, qn_log_probs.exp(), torch.zeros_like(qn_log_probs)
        )
        if divergence == "forward_kl":
            task = qe - (q0 if displacement_mode == "evidence_vs_none" else qn)
            nuisance = 0.5 * (qe + qn) - q0
            moments = forward_moments(
                p_controller_log,
                task,
                correction_support,
                nuisance if displacement_mode == "fec" else None,
                projection_epsilon=projection_epsilon,
            )
            controller_direction = moments.residual
        else:
            reverse_base_log_probs = l0 if displacement_mode == "evidence_vs_none" else qn_log_probs
            task_log_ratio = torch.where(
                correction_valid,
                le - reverse_base_log_probs,
                torch.zeros_like(le),
            )
            nuisance_log_ratio = torch.where(
                correction_valid,
                0.5 * (le + qn_log_probs) - l0,
                torch.zeros_like(le),
            )
            moments = reverse_tangent_moments(
                p_controller_log,
                task_log_ratio,
                correction_support,
                nuisance_log_ratio if displacement_mode == "fec" else None,
                projection_epsilon=projection_epsilon,
            )
            controller_direction = moments.residual
        alpha = moments.projection
    sampled_match = correction_support.valid_mask & correction_support.token_ids.eq(
        sampled_token_ids.long().unsqueeze(-1)
    )
    with torch.no_grad():
        sampled_direction = torch.where(
            sampled_match, controller_direction, torch.zeros_like(controller_direction)
        ).sum(-1)
        alignment = sampled_direction - moments.residual_mean
        benefit = token_advantages.detach().float() * alignment
        fisher_cost = moments.fisher_cost
        h = benefit if allow_negative_benefit else benefit.clamp_min(0.0)
        cost = effective_cost_alpha * fisher_cost + effective_cost_epsilon
        weights = torch.where(
            active_token_mask.detach().bool(),
            h / (h + cost),
            torch.zeros_like(h),
        )
        if not bool(torch.isfinite(weights).all().item()):
            raise ValueError("VERPO token weights must be finite")
        weights = weights.detach()
    if divergence == "forward_kl":
        p_ref_teacher = torch.where(
            reference_valid, l0_ref.exp(), torch.zeros_like(l0_ref)
        )
        reference_loss = float(temperature) * (
            p_ref_teacher * (l0_ref - lp_ref)
        ).sum(-1)
        evidence_loss = -float(temperature) * weights * (
            moments.residual.detach() * lp
        ).sum(-1)
    else:
        reverse_base_log_probs = l0 if displacement_mode == "evidence_vs_none" else qn_log_probs
        log_ratio = torch.where(
            correction_valid,
            le - reverse_base_log_probs,
            torch.zeros_like(le),
        ).detach()
        if displacement_mode == "fec":
            nuisance_log_ratio = torch.where(
                correction_valid,
                0.5 * (le + qn_log_probs) - l0,
                torch.zeros_like(le),
            ).detach()
            log_ratio = log_ratio - alpha.unsqueeze(-1) * nuisance_log_ratio
        p_ref = torch.where(reference_valid, lp_ref.exp(), torch.zeros_like(lp_ref))
        reference_loss = float(temperature) * (
            p_ref * (lp_ref - l0_ref)
        ).sum(-1)
        evidence_loss = reverse_geometric_correction(
            lp,
            reverse_base_log_probs,
            log_ratio,
            weights,
            correction_support,
            temperature=temperature,
        )
    result = {
        "reference_loss": reference_loss,
        "evidence_loss": evidence_loss,
        "weights": weights.detach(),
        "benefit": benefit.detach(),
        "fisher_cost": fisher_cost.detach(),
        "displacement_norm": moments.displacement_norm.detach(),
        "task_displacement_norm": moments.task_displacement_norm.detach(),
        "nuisance_displacement_norm": moments.nuisance_displacement_norm.detach(),
        "task_nuisance_fisher_cosine": moments.task_nuisance_fisher_cosine.detach(),
        "nuisance_projection_coefficient": alpha.detach(),
        "residual_nuisance_fisher_covariance": (
            moments.residual_nuisance_fisher_covariance.detach()
        ),
        "alignment": alignment.detach(),
        "q0_sample_log_prob": (
            l0_ref
            * reference_support.valid_mask
            * reference_support.token_ids.eq(
                sampled_token_ids.long().unsqueeze(-1)
            )
        ).sum(-1),
        "reference_support_size": reference_support.valid_mask.sum(-1).float(),
        "support_size": correction_support.valid_mask.sum(-1).float(),
        "reference_support_mass": support_mass(l0_ref, reference_support),
        "evidence_support_mass": support_mass(le, correction_support),
        "student_support_mass": support_mass(lp, correction_support),
    }
    if negative_teacher_source is not None:
        result["negative_support_mass"] = support_mass(
            qn_log_probs,
            correction_support,
        )
    return result


def compute_flat_verpo_losses(
    student_logits: torch.Tensor,
    base_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    token_advantages: torch.Tensor,
    active_token_mask: torch.Tensor,
    *,
    negative_teacher_logits: torch.Tensor | None = None,
    negative_teacher_probabilities: torch.Tensor | None = None,
    tau: float,
    rho: float,
    cost_floor: float,
    cost_beta: float,
    cost_alpha: float | None = None,
    cost_epsilon: float | None = None,
    allow_negative_benefit: bool = False,
    temperature: float,
    vocab_chunk_size: int,
    divergence: str = "forward_kl",
    displacement_mode: str = "evidence_vs_none",
    projection_epsilon: float = 1e-8,
    vocab_mode: str = "topk_truncated",
    top_k: int = 128,
) -> dict[str, torch.Tensor]:
    """Dispatch exact or configured truncated VERPO vocabulary accounting.

    The native full loss remains the compatibility implementation.  Top-k
    support diagnostics are attached by the same path, while K>=V retains the
    exact full-vocabulary behavior.
    """
    validate_verpo_vocab_config(vocab_mode, top_k)
    if not should_use_full_vocab(vocab_mode, top_k, student_logits.shape[-1]):
        return _compute_flat_truncated_verpo_losses(
            student_logits,
            base_teacher_logits,
            evidence_teacher_logits,
            sampled_token_ids,
            token_advantages,
            active_token_mask,
            negative_teacher_logits=negative_teacher_logits,
            negative_teacher_probabilities=negative_teacher_probabilities,
            tau=tau,
            rho=rho,
            cost_floor=cost_floor,
            cost_beta=cost_beta,
            cost_alpha=cost_alpha,
            cost_epsilon=cost_epsilon,
            allow_negative_benefit=allow_negative_benefit,
            temperature=temperature,
            divergence=divergence,
            displacement_mode=displacement_mode,
            projection_epsilon=projection_epsilon,
            top_k=top_k,
            vocab_chunk_size=vocab_chunk_size,
        )
    if negative_teacher_probabilities is not None:
        raise ValueError(
            "pre-normalized negative Teacher probabilities require topk_truncated"
        )
    return _compute_flat_verpo_losses_full(
        student_logits,
        base_teacher_logits,
        evidence_teacher_logits,
        sampled_token_ids,
        token_advantages,
        active_token_mask,
        negative_teacher_logits=negative_teacher_logits,
        tau=tau,
        rho=rho,
        cost_floor=cost_floor,
        cost_beta=cost_beta,
        cost_alpha=cost_alpha,
        cost_epsilon=cost_epsilon,
        allow_negative_benefit=allow_negative_benefit,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk_size,
        divergence=divergence,
        displacement_mode=displacement_mode,
        projection_epsilon=projection_epsilon,
        vocab_mode=vocab_mode,
        top_k=top_k,
    )


def _flatten_logits(raw_output: Any) -> torch.Tensor:
    logits = raw_output.logits if hasattr(raw_output, "logits") else raw_output["logits"]
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            return logits.reshape(-1, logits.shape[-1])
        logits = logits.squeeze(0)
    if logits.ndim != 2:
        raise ValueError(f"Expected flattened logits [N, V], got {tuple(logits.shape)}")
    return logits


def _teacher_forward(
    reference_engine, data: TensorDict, actor_teacher=None
) -> torch.Tensor:
    teacher_engine = (
        actor_teacher.engine
        if actor_teacher is not None
        else reference_engine
    )
    model_inputs, _ = teacher_engine.prepare_model_inputs(micro_batch=data)
    teacher_context = (
        actor_teacher.forward_context()
        if actor_teacher is not None
        else torch.no_grad()
    )
    with teacher_context, torch.no_grad():
        raw_output = teacher_engine.module(**model_inputs, use_cache=False)
    return _flatten_logits(raw_output)


def _teacher_data(data: TensorDict, prefix: str) -> TensorDict:
    teacher = TensorDict(
        {
            "input_ids": data[f"{prefix}_input_ids"],
            "position_ids": data[f"{prefix}_position_ids"],
        },
        batch_size=data.batch_size,
    )
    temperature = tu.get_non_tensor_data(data=data, key="temperature", default=None)
    if temperature is None:
        raise KeyError("temperature")
    if isinstance(temperature, torch.Tensor) and temperature.ndim > 0:
        teacher["temperature"] = temperature
    else:
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.item()
        tu.assign_non_tensor_data(teacher, "temperature", temperature)
    for key in ("use_remove_padding", "use_dynamic_bsz", "use_fused_kernels", "pad_mode"):
        value = tu.get_non_tensor_data(data=data, key=key, default=None)
        if value is not None:
            tu.assign_non_tensor_data(teacher, key, value)
    return teacher


def marginalize_teacher_logits(teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Represent the uniform probability mixture over the leading Teacher axis."""
    if teacher_logits.ndim != 3 or teacher_logits.shape[0] <= 0:
        raise ValueError("teacher_logits must have shape [K, N, V] with K > 0")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    scale = float(temperature)
    with torch.no_grad():
        component_log_probs = torch.log_softmax(teacher_logits.float() / scale, dim=-1)
        mixture_log_probs = torch.logsumexp(component_log_probs, dim=0) - math.log(teacher_logits.shape[0])
    return mixture_log_probs * scale


def _accumulate_indexed_teacher_probabilities_(
    accumulator: torch.Tensor,
    teacher_logits: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    temperature: float,
    vocab_chunk_size: int,
    weight: float,
) -> None:
    """Accumulate selected-row full-Softmax probabilities with bounded FP32 workspace."""
    if accumulator.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("accumulator and teacher_logits must have shape [N, V]")
    if accumulator.shape != (row_indices.numel(), teacher_logits.shape[-1]):
        raise ValueError("accumulator shape must match indexed Teacher response logits")
    if accumulator.device != teacher_logits.device or row_indices.device != teacher_logits.device:
        raise ValueError("accumulator, Teacher logits, and indices must share a device")
    scale = float(temperature)
    chunk_size = int(vocab_chunk_size)
    if scale <= 0 or chunk_size <= 0 or float(weight) <= 0:
        raise ValueError("temperature, vocab_chunk_size, and weight must be positive")
    with torch.no_grad():
        log_z = None
        for start in range(0, teacher_logits.shape[-1], chunk_size):
            stop = min(start + chunk_size, teacher_logits.shape[-1])
            chunk = teacher_logits[row_indices, start:stop].float() / scale
            chunk_log_z = torch.logsumexp(chunk, dim=-1, keepdim=True)
            log_z = (
                chunk_log_z
                if log_z is None
                else torch.logaddexp(log_z, chunk_log_z)
            )
        if log_z is None:
            raise ValueError("Teacher logits must have a non-empty vocabulary")
        for start in range(0, teacher_logits.shape[-1], chunk_size):
            stop = min(start + chunk_size, teacher_logits.shape[-1])
            probabilities = (
                teacher_logits[row_indices, start:stop].float() / scale - log_z
            ).exp()
            accumulator[:, start:stop].add_(
                probabilities.to(accumulator.dtype),
                alpha=float(weight),
            )


def _response_predictor_indices(
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
) -> torch.Tensor:
    if not input_ids.is_nested:
        raise ValueError("VERPO currently requires use_remove_padding=True nested inputs")
    offsets = input_ids.offsets()[:-1].to(prompt_lens.device)
    pieces = []
    for offset, prompt_len, response_len in zip(offsets, prompt_lens, response_lens, strict=True):
        if int(prompt_len.item()) <= 0:
            raise ValueError("Every VERPO prompt must contain at least one token")
        pieces.append(offset + prompt_len - 1 + torch.arange(response_len, device=prompt_lens.device))
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long, device=prompt_lens.device)


def _repeat_rows_over_responses(values: torch.Tensor, response_lens: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(values, response_lens.to(values.device))


def _indexed_rows_view_when_contiguous(
    values: torch.Tensor, row_indices: torch.Tensor
) -> torch.Tensor:
    """Select rows without materializing an ``N x vocab`` copy when possible.

    Native veRL uses one nested sequence per actor micro-batch for the A800
    VERPO_CONTRASTIVE profile.  Its response predictor rows are therefore contiguous.  A
    regular advanced-indexing expression copies every selected vocabulary row;
    for a 16K response and the Qwen vocabulary that temporary is several GiB.
    Keep a narrow view for this common case and retain advanced indexing only
    for genuinely non-contiguous batches.
    """
    if row_indices.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")
    if row_indices.numel() == 0:
        return values.narrow(0, 0, 0)
    start = int(row_indices[0].item())
    if row_indices.numel() == 1 or bool(
        row_indices.diff().eq(1).all().item()
    ):
        return values.narrow(0, start, row_indices.numel())
    return values[row_indices]


def _first_nested_value_per_row(values: torch.Tensor) -> torch.Tensor:
    if not values.is_nested:
        raise ValueError("Expected a nested response tensor")
    offsets = values.offsets()
    if bool((offsets.diff() <= 0).any().item()):
        raise ValueError("VERPO does not support empty sampled responses")
    return values.values()[offsets[:-1]]


def _verpo_logits_processor(
    *,
    config: ActorConfig,
    reference_engine,
    actor_teacher,
    student_logits: torch.Tensor,
    data: TensorDict,
) -> dict[str, torch.Tensor]:
    student_flat = student_logits.squeeze(0)
    # FSDPEngine normalizes eager logits by rollout temperature before invoking
    # the processor. Recover raw Student logits so all three distributions are
    # tempered exactly once inside compute_flat_verpo_losses.
    if float(config.verpo.temperature) != 1.0:
        student_distribution_logits = student_flat * float(config.verpo.temperature)
    else:
        student_distribution_logits = student_flat
    q0_started = time.perf_counter()
    q0_flat = _teacher_forward(reference_engine, data, actor_teacher)
    q0_elapsed = time.perf_counter() - q0_started
    displacement_mode = config.verpo.displacement_mode

    prompt_lens = data["prompts"].offsets().diff().to(student_flat.device)
    response_lens = data["responses"].offsets().diff().to(student_flat.device)
    actor_indices = _response_predictor_indices(data["input_ids"], prompt_lens, response_lens)
    q0_response_logits = _indexed_rows_view_when_contiguous(q0_flat, actor_indices)
    del q0_flat
    evidence_teacher_elapsed = 0.0
    if displacement_mode == "evidence_vs_none":
        evidence_prefix = "verpo_evidence"
        evidence_started = time.perf_counter()
        qe_flat = _teacher_forward(
            reference_engine, _teacher_data(data, evidence_prefix), actor_teacher
        )
        evidence_teacher_elapsed += time.perf_counter() - evidence_started
        evidence_indices = _response_predictor_indices(
            data[f"{evidence_prefix}_input_ids"],
            data[f"{evidence_prefix}_prompt_lengths"].to(student_flat.device),
            response_lens,
        )
        evidence_response_logits = _indexed_rows_view_when_contiguous(
            qe_flat, evidence_indices
        )
        del qe_flat
        negative_response_logits = None
        negative_teacher_probabilities = None
    else:
        evidence_prefix = "verpo_positive"
        evidence_started = time.perf_counter()
        qe_flat = _teacher_forward(
            reference_engine, _teacher_data(data, evidence_prefix), actor_teacher
        )
        evidence_teacher_elapsed += time.perf_counter() - evidence_started
        evidence_indices = _response_predictor_indices(
            data[f"{evidence_prefix}_input_ids"],
            data[f"{evidence_prefix}_prompt_lengths"].to(student_flat.device),
            response_lens,
        )
        evidence_response_logits = _indexed_rows_view_when_contiguous(
            qe_flat, evidence_indices
        )
        del qe_flat
        negative_logsumexp = None
        negative_teacher_probabilities = None
        scale = float(config.verpo.temperature)
        negative_count = int(config.verpo.contrastive_num_negative_hints)
        use_truncated_vocab = not should_use_full_vocab(
            config.verpo.vocab_mode,
            config.verpo.top_k,
            student_distribution_logits.shape[-1],
        )
        for negative_index in range(negative_count):
            prefix = f"verpo_negative_{negative_index}"
            negative_flat = _teacher_forward(
                reference_engine, _teacher_data(data, prefix), actor_teacher
            )
            negative_indices = _response_predictor_indices(
                data[f"{prefix}_input_ids"],
                data[f"{prefix}_prompt_lengths"].to(student_flat.device),
                response_lens,
            )
            if actor_indices.numel() != negative_indices.numel():
                raise AssertionError("Student and negative Teacher response suffixes are not aligned")
            if use_truncated_vocab:
                # A single negative hint is already the required probability
                # mixture.  Passing its logits directly preserves the exact
                # Top-K semantics and avoids the old full-vocabulary
                # ``zeros([response_tokens, vocab])`` accumulator, which was
                # 3.81 GiB for the formal Qwen3-1.7B step-1 batch.
                if negative_count == 1:
                    negative_response_logits = _indexed_rows_view_when_contiguous(
                        negative_flat, negative_indices
                    )
                    negative_teacher_probabilities = None
                    break
                if negative_teacher_probabilities is None:
                    negative_teacher_probabilities = torch.zeros(
                        (negative_indices.numel(), negative_flat.shape[-1]),
                        dtype=negative_flat.dtype,
                        device=negative_flat.device,
                    )
                _accumulate_indexed_teacher_probabilities_(
                    negative_teacher_probabilities,
                    negative_flat,
                    negative_indices,
                    temperature=scale,
                    vocab_chunk_size=config.verpo.vocab_chunk_size,
                    weight=1.0 / negative_count,
                )
            else:
                with torch.no_grad():
                    component_log_probs = torch.log_softmax(
                        negative_flat[negative_indices].float() / scale,
                        dim=-1,
                    )
                    negative_logsumexp = (
                        component_log_probs
                        if negative_logsumexp is None
                        else torch.logaddexp(negative_logsumexp, component_log_probs)
                    )
            del negative_flat
        if use_truncated_vocab and negative_count == 1:
            if negative_response_logits is None:
                raise AssertionError("The single negative Teacher is required")
        elif use_truncated_vocab:
            if negative_teacher_probabilities is None:
                raise AssertionError("At least one negative Teacher is required")
            negative_response_logits = None
        else:
            if negative_logsumexp is None:
                raise AssertionError("At least one negative Teacher is required")
            negative_response_logits = scale * (
                negative_logsumexp - math.log(negative_count)
            )
    if actor_indices.numel() != evidence_indices.numel():
        raise AssertionError("Student and evidence Teacher response suffixes are not aligned")

    sampled_ids = data["responses"].values().to(student_flat.device)
    response_mask = data["response_mask"].values().bool().to(student_flat.device)
    row_advantages = _first_nested_value_per_row(data["advantages"]).to(student_flat.device)
    row_gate = data["verpo_evidence_rollout_gate"].bool().to(student_flat.device)
    token_advantages = _repeat_rows_over_responses(row_advantages, response_lens)
    token_gate = _repeat_rows_over_responses(row_gate, response_lens) & response_mask

    outputs = compute_flat_verpo_losses(
        _indexed_rows_view_when_contiguous(student_distribution_logits, actor_indices),
        q0_response_logits,
        evidence_response_logits,
        sampled_ids,
        token_advantages,
        token_gate,
        negative_teacher_logits=negative_response_logits,
        negative_teacher_probabilities=negative_teacher_probabilities,
        cost_alpha=config.verpo.cost_alpha,
        cost_epsilon=config.verpo.cost_epsilon,
        # Legacy aliases are passed for compatibility with old checkpoints.
        tau=config.verpo.tau,
        rho=config.verpo.rho,
        cost_floor=config.verpo.cost_floor,
        cost_beta=config.verpo.cost_beta,
        allow_negative_benefit=config.verpo.allow_negative_benefit,
        temperature=config.verpo.temperature,
        vocab_chunk_size=config.verpo.vocab_chunk_size,
        divergence=config.verpo.divergence,
        displacement_mode=displacement_mode,
        projection_epsilon=config.verpo.projection_epsilon,
        vocab_mode=config.verpo.vocab_mode,
        top_k=config.verpo.top_k,
    )
    tu.assign_non_tensor_data(
        data,
        "verpo_qref_forward_time_seconds",
        q0_elapsed,
    )
    tu.assign_non_tensor_data(data, "verpo_q0_forward_time_seconds", q0_elapsed)
    tu.assign_non_tensor_data(
        data,
        "verpo_qe_forward_time_seconds",
        evidence_teacher_elapsed,
    )
    full_outputs: dict[str, torch.Tensor] = {}
    for name, response_values in outputs.items():
        full = torch.zeros(student_flat.shape[0], dtype=response_values.dtype, device=response_values.device)
        full[actor_indices] = response_values
        full_outputs[f"verpo_{name}"] = full.unsqueeze(0)
    return full_outputs


def _record_verpo_weight_audit(
    data: TensorDict,
    *,
    response_mask: torch.Tensor,
    available_mask: torch.Tensor,
    evidence_rollout_gate: torch.Tensor,
    outcome_positive: torch.Tensor,
    weights: torch.Tensor,
    benefit: torch.Tensor,
    fisher_cost: torch.Tensor,
    alignment: torch.Tensor,
) -> None:
    """Persist top weighted response tokens when explicitly enabled.

    The normal training path keeps only scalar monitoring aggregates.  When
    ``VERPO_WEIGHT_AUDIT_DIR`` is set, this sidecar records token ids and the
    controller values needed to decode which response tokens actually carried
    evidence weight.  Each data-parallel rank writes its own JSONL shard.
    """
    audit_dir = os.environ.get("VERPO_WEIGHT_AUDIT_DIR", "").strip()
    if not audit_dir:
        return
    try:
        response_ids = data["responses"]
        if getattr(response_ids, "is_nested", False):
            response_ids = response_ids.to_padded_tensor(0)
        response_ids = response_ids.detach().to(device=weights.device).long()
        if response_ids.shape != weights.shape:
            return
        top_k = int(os.environ.get("VERPO_WEIGHT_AUDIT_TOPK", "64"))
        if top_k < 0:
            top_k = 0
        step_tensor = data.get("verpo_audit_step", None)
        if step_tensor is not None:
            step = int(step_tensor.reshape(-1)[0].item())
        else:
            step = tu.get_non_tensor_data(data=data, key="global_steps", default=-1)
        try:
            step = int(step)
        except (TypeError, ValueError):
            step = -1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(torch.distributed.get_rank())
        else:
            rank = 0
        records = []
        for row in range(int(weights.shape[0])):
            valid = response_mask[row] & available_mask[row] & weights[row].gt(0)
            positions = torch.nonzero(valid, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            values = weights[row, positions].detach().float()
            if top_k > 0 and positions.numel() > top_k:
                order = torch.argsort(values, descending=True)[:top_k]
                positions = positions[order]
            rows = []
            for position in positions.tolist():
                rows.append(
                    {
                        "position": int(position),
                        "token_id": int(response_ids[row, position].item()),
                        "weight": float(weights[row, position].detach().float().item()),
                        "benefit": float(benefit[row, position].detach().float().item()),
                        "fisher_cost": float(fisher_cost[row, position].detach().float().item()),
                        "alignment": float(alignment[row, position].detach().float().item()),
                    }
                )
            records.append(
                {
                    "step": step,
                    "rank": rank,
                    "row": row,
                    "outcome_positive": bool(outcome_positive[row].item()),
                    "evidence_rollout_gate": bool(evidence_rollout_gate[row].item()),
                    "topk": top_k,
                    "tokens": rows,
                }
            )
        if not records:
            return
        target_dir = Path(audit_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"weight_audit.rank{rank}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Auditing must never change the training result.  The caller still
        # retains the scalar monitoring metrics if a sidecar write fails.
        return


def verpo_zpd_ppo_loss(
    config: ActorConfig,
    reference_engine,
    actor_teacher=None,
    gradient_auditor=None,
    model_output: dict | None = None,
    data: TensorDict | None = None,
    dp_group=None,
    student_logits: torch.Tensor | None = None,
    data_format: str = "thd",
):
    """Serve as both the FSDP logits processor and the final PPO loss."""
    del data_format
    if data is None:
        raise ValueError("VERPO loss requires a TensorDict micro-batch")
    if student_logits is not None:
        return _verpo_logits_processor(
            config=config,
            reference_engine=reference_engine,
            actor_teacher=actor_teacher,
            student_logits=student_logits,
            data=data,
        )
    if model_output is None:
        raise ValueError("VERPO final loss requires model_output")

    response_mask = data["response_mask"].bool()
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    advantages_override = None
    advantage_mod_scale = None
    if config.verpo.advantage_modulation == "multiplicative_w":
        weights_for_advantage = no_padding_2_padding(
            model_output["verpo_weights"], data
        ).detach()
        padded_advantages = data.select("advantages").to_padded_tensor()[
            "advantages"
        ].to(device=weights_for_advantage.device)
        if padded_advantages.shape != weights_for_advantage.shape:
            raise ValueError(
                "VERPO advantage modulation requires weights and advantages to have "
                "the same padded shape"
            )
        advantage_mod_scale = (
            1.0
            + float(config.verpo.advantage_modulation_lambda)
            * weights_for_advantage
        ).detach()
        advantages_override = modulate_advantages(
            padded_advantages,
            weights_for_advantage,
            float(config.verpo.advantage_modulation_lambda),
        )

    policy_loss, metrics = ppo_loss(
        config,
        model_output,
        data,
        dp_group,
        reference_log_probs=model_output["verpo_q0_sample_log_prob"],
        advantages_override=advantages_override,
    )
    if advantage_mod_scale is not None:
        active_scale = advantage_mod_scale.masked_select(response_mask)
        active_weights = weights_for_advantage.masked_select(response_mask)
        metric_aggregation = (
            AggregationType.SUM if config.global_batch_info else AggregationType.MEAN
        )
        if active_scale.numel() > 0:
            metrics["verpo/advantage_modulation_scale"] = Metric(
                aggregation=metric_aggregation, value=active_scale.mean()
            )
            metrics["verpo/advantage_modulation_scale_min"] = Metric(
                aggregation=metric_aggregation, value=active_scale.min()
            )
            metrics["verpo/advantage_modulation_scale_max"] = Metric(
                aggregation=metric_aggregation, value=active_scale.max()
            )
            metrics["verpo/advantage_modulation_weight_zero_rate"] = Metric(
                aggregation=metric_aggregation,
                value=active_weights.eq(0).float().mean(),
            )
            outcome_positive = data["verpo_outcome_positive"].bool().to(
                response_mask.device
            )
            for label, outcome_mask in (
                ("success", outcome_positive.unsqueeze(-1)),
                ("failure", ~outcome_positive.unsqueeze(-1)),
            ):
                selected_scale = advantage_mod_scale.masked_select(
                    response_mask & outcome_mask
                )
                # Emit both outcome keys on every micro-batch.  A batch can
                # contain no success (or no failure) tokens; omitting that
                # key makes DP metric aggregation see different list lengths.
                selected_value = (
                    selected_scale.mean()
                    if selected_scale.numel() > 0
                    else advantage_mod_scale.new_zeros(())
                )
                metrics[f"verpo/advantage_modulation_{label}_scale"] = Metric(
                    aggregation=metric_aggregation, value=selected_value
                )
    evidence_rollout_gate = data["verpo_evidence_rollout_gate"].bool().to(
        response_mask.device
    )
    evidence_response_mask = response_mask & evidence_rollout_gate.unsqueeze(-1)
    reference_losses = no_padding_2_padding(model_output["verpo_reference_loss"], data)
    evidence_losses = no_padding_2_padding(model_output["verpo_evidence_loss"], data)
    reference_loss = agg_loss(
        loss_mat=reference_losses,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )

    def repeated_global_count(key: str, fallback: int) -> int:
        if key in data:
            values = data[key].reshape(-1)
            if values.numel():
                return int(values[0].item())
        if int(config.global_batch_info.get("dp_size", 1)) > 1:
            raise ValueError(f"Missing global VERPO routing count: {key}")
        return int(fallback)

    evidence_token_count = repeated_global_count(
        "verpo_evidence_batch_num_tokens", int(evidence_response_mask.sum().item())
    )
    evidence_row_count = repeated_global_count(
        "verpo_evidence_global_batch_size",
        int(evidence_response_mask.any(dim=-1).sum().item()),
    )
    evidence_global_info = dict(config.global_batch_info)
    evidence_global_info["batch_num_tokens"] = evidence_token_count
    evidence_global_info["global_batch_size"] = evidence_row_count
    required_evidence_count = (
        evidence_token_count
        if config.loss_agg_mode == "token-mean"
        else evidence_row_count
    )
    if required_evidence_count > 0:
        evidence_loss = agg_loss(
            loss_mat=evidence_losses,
            loss_mask=evidence_response_mask,
            loss_agg_mode=config.loss_agg_mode,
            **evidence_global_info,
        )
    else:
        evidence_loss = evidence_losses.sum() * 0.0
    grpo_loss = policy_loss
    scaled_reference_loss = config.verpo.lambda_ref * reference_loss
    scaled_evidence_loss = config.verpo.lambda_evi * evidence_loss
    if gradient_auditor is not None:
        audit_metrics = gradient_auditor.audit(
            grpo_loss=grpo_loss,
            reference_scaled_loss=scaled_reference_loss,
            evidence_scaled_loss=scaled_evidence_loss,
        )
        for name, value in audit_metrics.items():
            metrics[name] = Metric(aggregation=AggregationType.MEAN, value=value)
    policy_loss = grpo_loss + scaled_reference_loss + scaled_evidence_loss

    metric_aggregation = AggregationType.SUM if config.global_batch_info else AggregationType.MEAN
    metrics["verpo/reference_loss"] = Metric(aggregation=metric_aggregation, value=reference_loss)
    metrics["verpo/evidence_loss"] = Metric(aggregation=metric_aggregation, value=evidence_loss)

    global_row_count = config.global_batch_info.get("global_batch_size")
    if global_row_count is None:
        global_row_count = int(response_mask.any(dim=-1).sum().item())
    available_rows = data.get(
        "verpo_contrastive_available",
        torch.ones_like(evidence_rollout_gate),
    ).bool().to(response_mask.device)
    available_mask = response_mask & available_rows.unsqueeze(-1)
    outcome_positive = data["verpo_outcome_positive"].bool().to(response_mask.device)
    raw_group_gate = data["verpo_group_gate"].bool().to(response_mask.device)
    mixed_success_mask = (
        available_mask
        & raw_group_gate.unsqueeze(-1)
        & outcome_positive.unsqueeze(-1)
    )
    mixed_failure_mask = (
        available_mask
        & raw_group_gate.unsqueeze(-1)
        & ~outcome_positive.unsqueeze(-1)
    )
    fec_enabled = config.verpo.displacement_mode == "fec"
    fec_mask = available_mask if fec_enabled else torch.zeros_like(available_mask)
    fec_correct_mask = (
        evidence_response_mask & outcome_positive.unsqueeze(-1)
        if fec_enabled
        else torch.zeros_like(evidence_response_mask)
    )
    fec_wrong_mask = (
        evidence_response_mask & ~outcome_positive.unsqueeze(-1)
        if fec_enabled
        else torch.zeros_like(evidence_response_mask)
    )

    tensor_metrics = {
        "weights": no_padding_2_padding(model_output["verpo_weights"], data),
        "benefit": no_padding_2_padding(model_output["verpo_benefit"], data),
        "fisher_cost": no_padding_2_padding(model_output["verpo_fisher_cost"], data),
        "displacement_norm": no_padding_2_padding(
            model_output["verpo_displacement_norm"], data
        ),
        "task_norm": no_padding_2_padding(
            model_output["verpo_task_displacement_norm"], data
        ),
        "nuisance_norm": no_padding_2_padding(
            model_output["verpo_nuisance_displacement_norm"], data
        ),
        "fec_cosine": no_padding_2_padding(
            model_output["verpo_task_nuisance_fisher_cosine"], data
        ),
        "fec_projection": no_padding_2_padding(
            model_output["verpo_nuisance_projection_coefficient"], data
        ),
        "fec_residual_covariance": no_padding_2_padding(
            model_output["verpo_residual_nuisance_fisher_covariance"], data
        ),
        "alignment": no_padding_2_padding(model_output["verpo_alignment"], data),
    }
    movement_cost = (
        float(config.verpo.cost_alpha) * tensor_metrics["fisher_cost"]
        + float(config.verpo.cost_epsilon)
    )
    controller_benefit = (
        tensor_metrics["benefit"]
        if config.verpo.allow_negative_benefit
        else tensor_metrics["benefit"].clamp_min(0.0)
    )
    benefit_cost_ratio = controller_benefit / movement_cost.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    _record_verpo_weight_audit(
        data,
        response_mask=response_mask,
        available_mask=available_mask,
        evidence_rollout_gate=evidence_rollout_gate,
        outcome_positive=outcome_positive,
        weights=tensor_metrics["weights"],
        benefit=tensor_metrics["benefit"],
        fisher_cost=tensor_metrics["fisher_cost"],
        alignment=tensor_metrics["alignment"],
    )

    internal_prefix = "_verpo_internal/"

    def raw_sum(name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        local_sum = values.detach().float().masked_select(mask).sum()
        # Keep VERPO monitoring totals out of Metric.aggregate_dp: that helper
        # cannot distinguish DP ranks from microbatches after object gather.
        # The worker finalizer recursively combines these singleton payloads.
        metrics[f"{internal_prefix}sum/{name}"] = [float(local_sum.item())]

    def raw_count(name: str, mask: torch.Tensor) -> None:
        local_count = mask.detach().sum().float()
        metrics[f"{internal_prefix}count/{name}"] = [float(local_count.item())]

    raw_count("reference", response_mask)
    raw_count("evidence", evidence_response_mask)
    raw_count("available", available_mask)
    raw_count("weight", available_mask)
    raw_count("fec", fec_mask)
    raw_count("fec_correct", fec_correct_mask)
    raw_count("fec_wrong", fec_wrong_mask)
    raw_count("mixed_success", mixed_success_mask)
    raw_count("mixed_failure", mixed_failure_mask)
    raw_count("weight_zero", available_mask & tensor_metrics["weights"].eq(0))
    raw_count("weight_nonzero", available_mask & tensor_metrics["weights"].gt(0))
    raw_count(
        "weight_effective",
        available_mask
        & tensor_metrics["weights"].gt(VERPO_EFFECTIVE_WEIGHT_THRESHOLD),
    )
    raw_count("weight_gt_05", available_mask & tensor_metrics["weights"].gt(0.5))
    raw_count("weight_negative", available_mask & tensor_metrics["weights"].lt(0))
    raw_count("weight_ge_one", available_mask & tensor_metrics["weights"].ge(1))
    negative_benefit_mask = available_mask & tensor_metrics["benefit"].lt(0)
    positive_benefit_mask = available_mask & tensor_metrics["benefit"].gt(0)
    raw_count("benefit_negative", negative_benefit_mask)
    raw_count("benefit_positive", positive_benefit_mask)
    raw_count(
        "negative_benefit_effective",
        negative_benefit_mask
        & tensor_metrics["weights"].gt(VERPO_EFFECTIVE_WEIGHT_THRESHOLD),
    )
    raw_count(
        "fec_sign_pass",
        (fec_correct_mask & tensor_metrics["alignment"].gt(0))
        | (fec_wrong_mask & tensor_metrics["alignment"].lt(0)),
    )

    raw_sum("reference_loss", reference_losses, response_mask)
    raw_sum(
        "scaled_reference_loss",
        float(config.verpo.lambda_ref) * reference_losses,
        response_mask,
    )
    raw_sum("evidence_loss", evidence_losses, evidence_response_mask)
    raw_sum(
        "scaled_evidence_loss",
        float(config.verpo.lambda_evi) * evidence_losses,
        evidence_response_mask,
    )
    raw_sum("evidence_displacement", tensor_metrics["displacement_norm"], available_mask)
    raw_sum(
        "accepted_movement",
        tensor_metrics["weights"] * tensor_metrics["displacement_norm"],
        available_mask,
    )
    raw_sum("weight", tensor_metrics["weights"], available_mask)
    raw_sum("benefit", tensor_metrics["benefit"], available_mask)
    raw_sum("negative_benefit", tensor_metrics["benefit"], negative_benefit_mask)
    raw_sum(
        "negative_benefit_weight",
        tensor_metrics["weights"],
        negative_benefit_mask,
    )
    raw_sum("fisher_cost", tensor_metrics["fisher_cost"], available_mask)
    raw_sum("fec_task_norm", tensor_metrics["task_norm"], fec_mask)
    raw_sum("fec_nuisance_norm", tensor_metrics["nuisance_norm"], fec_mask)
    raw_sum("fec_cosine", tensor_metrics["fec_cosine"], fec_mask)
    raw_sum("fec_projection", tensor_metrics["fec_projection"], fec_mask)
    raw_sum(
        "fec_residual_covariance",
        tensor_metrics["fec_residual_covariance"],
        fec_mask,
    )
    raw_sum("fec_correct_alignment", tensor_metrics["alignment"], fec_correct_mask)
    raw_sum("fec_wrong_alignment", tensor_metrics["alignment"], fec_wrong_mask)
    raw_sum("benefit_cost_ratio", benefit_cost_ratio, available_mask)
    raw_sum("mixed_success_weight", tensor_metrics["weights"], mixed_success_mask)
    raw_sum("mixed_failure_weight", tensor_metrics["weights"], mixed_failure_mask)
    raw_sum("selected_benefit", tensor_metrics["benefit"], evidence_response_mask)
    raw_sum("selected_cost", tensor_metrics["fisher_cost"], evidence_response_mask)
    raw_sum(
        "selected_displacement",
        tensor_metrics["displacement_norm"],
        evidence_response_mask,
    )
    raw_sum("selected_weight", tensor_metrics["weights"], evidence_response_mask)

    topk_metric_names = {
        "support_size": "topk_support_size",
        "reference_support_mass": "topk_reference_support_mass",
        "evidence_support_mass": "topk_evidence_support_mass",
        "negative_support_mass": "topk_negative_support_mass",
        "student_support_mass": "topk_student_support_mass",
    }
    for output_name, metric_name in topk_metric_names.items():
        key = f"verpo_{output_name}"
        if key in model_output:
            raw_sum(
                metric_name,
                no_padding_2_padding(model_output[key], data),
                evidence_response_mask,
            )

    available_weights = tensor_metrics["weights"].detach().float().masked_select(
        available_mask
    )
    metrics[f"{internal_prefix}weight_values"] = available_weights.cpu().tolist()

    for name, key in {
        "student": "verpo_student_forward_time_seconds",
        "qref": "verpo_qref_forward_time_seconds",
        "q0": "verpo_q0_forward_time_seconds",
        "qe": "verpo_qe_forward_time_seconds",
    }.items():
        metrics[f"{internal_prefix}time/{name}"] = [
            float(tu.get_non_tensor_data(data=data, key=key, default=0.0))
        ]
    device = get_torch_device()
    memory_allocated = (
        device.max_memory_allocated() / (1024**3)
        if hasattr(device, "max_memory_allocated")
        else 0.0
    )
    memory_reserved = (
        device.max_memory_reserved() / (1024**3)
        if hasattr(device, "max_memory_reserved")
        else 0.0
    )
    metrics[f"{internal_prefix}max/memory_allocated"] = [memory_allocated]
    metrics[f"{internal_prefix}max/memory_reserved"] = [memory_reserved]

    def globally_weighted_row_mean(values: torch.Tensor) -> Metric:
        if int(global_row_count) <= 0:
            contribution = values.sum() * 0.0
        else:
            contribution = (
                values.float().sum()
                / float(global_row_count)
                * float(config.global_batch_info.get("dp_size", 1))
            )
        return Metric(aggregation=AggregationType.SUM, value=contribution)

    metrics["verpo/group_gate_rate"] = globally_weighted_row_mean(
        data["verpo_group_gate"]
    )
    metrics["verpo/available_group_gate_rate"] = globally_weighted_row_mean(
        data["verpo_available_group_gate"]
    )
    metrics["verpo/evidence_rollout_gate_rate"] = globally_weighted_row_mean(
        data["verpo_evidence_rollout_gate"]
    )
    return policy_loss, metrics
