"""Package-local truncated Top-k support for native veRL VERPO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

VALID_VERPO_VOCAB_MODES = frozenset({"full", "topk_truncated"})
TOPK_TRUNCATED_FINGERPRINT = "topk_truncated_v1"


@dataclass(frozen=True)
class TruncatedTopKSupport:
    token_ids: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.token_ids.shape != self.valid_mask.shape:
            raise ValueError("token_ids and valid_mask must match")
        if self.token_ids.dtype != torch.long:
            raise ValueError("token_ids must use torch.long")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must use torch.bool")


@dataclass(frozen=True)
class ForwardMoments:
    projection: torch.Tensor
    residual_mean: torch.Tensor
    fisher_cost: torch.Tensor
    displacement_norm: torch.Tensor
    task_displacement_norm: torch.Tensor
    nuisance_displacement_norm: torch.Tensor
    task_nuisance_fisher_cosine: torch.Tensor
    residual_nuisance_fisher_covariance: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class ReverseTangentMoments:
    projection: torch.Tensor
    task_log_ratio_mean: torch.Tensor
    nuisance_log_ratio_mean: torch.Tensor
    task_tangent_mean: torch.Tensor
    nuisance_tangent_mean: torch.Tensor
    residual_mean: torch.Tensor
    fisher_cost: torch.Tensor
    displacement_norm: torch.Tensor
    task_displacement_norm: torch.Tensor
    nuisance_displacement_norm: torch.Tensor
    task_nuisance_fisher_cosine: torch.Tensor
    residual_nuisance_fisher_covariance: torch.Tensor
    task_tangent: torch.Tensor
    nuisance_tangent: torch.Tensor
    residual: torch.Tensor


def _chunked_logsumexp_forward(
    logits: torch.Tensor,
    *,
    temperature: float,
    vocab_chunk_size: int,
) -> torch.Tensor:
    log_z = None
    for start in range(0, logits.shape[-1], vocab_chunk_size):
        stop = min(start + vocab_chunk_size, logits.shape[-1])
        chunk_log_z = torch.logsumexp(
            logits[..., start:stop].float() / temperature,
            dim=-1,
            keepdim=True,
        )
        log_z = chunk_log_z if log_z is None else torch.logaddexp(log_z, chunk_log_z)
    if log_z is None:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    return log_z


class _ChunkedLogsumexp(torch.autograd.Function):
    """Exact full-vocabulary logsumexp without retaining full FP32 logits."""

    @staticmethod
    def forward(ctx, logits, temperature, vocab_chunk_size):
        log_z = _chunked_logsumexp_forward(
            logits,
            temperature=temperature,
            vocab_chunk_size=vocab_chunk_size,
        )
        ctx.save_for_backward(logits, log_z)
        ctx.temperature = temperature
        ctx.vocab_chunk_size = vocab_chunk_size
        return log_z

    @staticmethod
    def backward(ctx, grad_output):
        logits, log_z = ctx.saved_tensors
        scale = ctx.temperature
        chunk_size = ctx.vocab_chunk_size
        grad_logits = torch.empty_like(logits)
        for start in range(0, logits.shape[-1], chunk_size):
            stop = min(start + chunk_size, logits.shape[-1])
            probabilities = (
                logits[..., start:stop].float() / scale - log_z
            ).exp()
            grad_chunk = grad_output.float() * probabilities / scale
            grad_logits[..., start:stop] = grad_chunk.to(logits.dtype)
        return grad_logits, None, None


def chunked_logsumexp(
    logits: torch.Tensor,
    *,
    temperature: float,
    vocab_chunk_size: int,
) -> torch.Tensor:
    """Return the FP32 full-vocabulary log normalizer with bounded workspace."""
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("temperature must be positive and finite")
    chunk_size = int(vocab_chunk_size)
    if chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    if logits.ndim < 1 or logits.shape[-1] <= 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    if logits.requires_grad:
        return _ChunkedLogsumexp.apply(logits, scale, chunk_size)
    return _chunked_logsumexp_forward(
        logits,
        temperature=scale,
        vocab_chunk_size=chunk_size,
    )


def validate_verpo_vocab_config(vocab_mode: str, top_k: int) -> None:
    if vocab_mode not in VALID_VERPO_VOCAB_MODES:
        raise ValueError("VERPO vocab_mode must be 'full' or 'topk_truncated'")
    if int(top_k) <= 0:
        raise ValueError("VERPO top_k must be positive")


def should_use_full_vocab(vocab_mode: str, top_k: int, vocab_size: int) -> bool:
    validate_verpo_vocab_config(vocab_mode, top_k)
    return vocab_mode == "full" or int(top_k) >= int(vocab_size)


def build_truncated_topk_support(
    *,
    teacher_logits: Sequence[torch.Tensor],
    sampled_token_ids: torch.Tensor,
    top_k: int,
    student_logits: torch.Tensor | None = None,
) -> TruncatedTopKSupport:
    if not teacher_logits:
        raise ValueError("teacher_logits must not be empty")
    shape = teacher_logits[0].shape
    if any(value.shape != shape for value in teacher_logits):
        raise ValueError("Teacher logits must have identical shapes")
    if student_logits is not None and student_logits.shape != shape:
        raise ValueError("Student and Teacher logits must match")
    if sampled_token_ids.shape != shape[:-1]:
        raise ValueError("sampled_token_ids must match token dimensions")
    k = min(int(top_k), int(shape[-1]))
    candidates = [torch.topk(value.detach(), k=k, dim=-1).indices for value in teacher_logits]
    if student_logits is not None:
        candidates.append(torch.topk(student_logits.detach(), k=k, dim=-1).indices)
    candidates.append(sampled_token_ids.long().unsqueeze(-1))
    sorted_ids = torch.sort(torch.cat(candidates, dim=-1), dim=-1).values
    valid = torch.ones_like(sorted_ids, dtype=torch.bool)
    valid[..., 1:] = sorted_ids[..., 1:] != sorted_ids[..., :-1]
    return TruncatedTopKSupport(
        token_ids=torch.where(valid, sorted_ids, torch.zeros_like(sorted_ids)),
        valid_mask=valid,
    )


def gather_full_softmax_log_probs(
    logits: torch.Tensor,
    support: TruncatedTopKSupport,
    *,
    temperature: float,
    vocab_chunk_size: int = 4096,
) -> torch.Tensor:
    if support.token_ids.shape[:-1] != logits.shape[:-1]:
        raise ValueError("support must match the logits token dimensions")
    if bool(
        (
            support.valid_mask
            & ((support.token_ids < 0) | (support.token_ids >= logits.shape[-1]))
        ).any().item()
    ):
        raise ValueError("support token_ids must be valid vocabulary IDs")
    safe_ids = torch.where(
        support.valid_mask, support.token_ids, torch.zeros_like(support.token_ids)
    )
    selected = logits.gather(-1, safe_ids).float() / float(temperature)
    selected = selected - chunked_logsumexp(
        logits,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk_size,
    )
    return torch.where(support.valid_mask, selected, torch.zeros_like(selected))


def gather_pre_normalized_log_probs(
    probabilities: torch.Tensor,
    support: TruncatedTopKSupport,
) -> torch.Tensor:
    """Gather a full-vocabulary probability tensor without renormalizing it."""
    if support.token_ids.shape[:-1] != probabilities.shape[:-1]:
        raise ValueError("support must match the probability token dimensions")
    if bool(
        (
            support.valid_mask
            & ((support.token_ids < 0) | (support.token_ids >= probabilities.shape[-1]))
        ).any().item()
    ):
        raise ValueError("support token_ids must be valid vocabulary IDs")
    safe_ids = torch.where(
        support.valid_mask, support.token_ids, torch.zeros_like(support.token_ids)
    )
    selected = probabilities.gather(-1, safe_ids).float().clamp_min(
        torch.finfo(torch.float32).tiny
    ).log()
    return torch.where(support.valid_mask, selected, torch.zeros_like(selected))


def masked_sum(values: torch.Tensor, support: TruncatedTopKSupport) -> torch.Tensor:
    return torch.where(support.valid_mask, values, torch.zeros_like(values)).sum(dim=-1)


def support_mass(
    selected_log_probs: torch.Tensor, support: TruncatedTopKSupport
) -> torch.Tensor:
    return masked_sum(selected_log_probs.exp(), support)


def forward_moments(
    student_log_probs: torch.Tensor,
    task: torch.Tensor,
    support: TruncatedTopKSupport,
    nuisance: torch.Tensor | None = None,
    *,
    projection_epsilon: float = 1e-8,
) -> ForwardMoments:
    """Compute selected-support Forward-KL Fisher moments without renormalizing."""
    for name, values in {"student_log_probs": student_log_probs, "task": task}.items():
        if values.shape != support.token_ids.shape:
            raise ValueError(f"{name} must match the support shape")
    if nuisance is not None and nuisance.shape != support.token_ids.shape:
        raise ValueError("nuisance must match the support shape")
    epsilon = float(projection_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("projection_epsilon must be finite and positive")

    valid = support.valid_mask
    probabilities = torch.where(
        valid, student_log_probs.detach().exp(), torch.zeros_like(student_log_probs)
    )
    task = torch.where(valid, task.detach(), torch.zeros_like(task))
    mean_task = (probabilities * task).sum(dim=-1)
    task_second = (probabilities * task.square()).sum(dim=-1)
    task_variance = (task_second - mean_task.square()).clamp_min(0.0)
    task_norm = task.square().sum(dim=-1).sqrt()
    zeros = torch.zeros_like(mean_task)
    if nuisance is None:
        return ForwardMoments(
            projection=zeros,
            residual_mean=mean_task,
            fisher_cost=task_variance,
            displacement_norm=task_norm,
            task_displacement_norm=task_norm,
            nuisance_displacement_norm=zeros,
            task_nuisance_fisher_cosine=zeros,
            residual_nuisance_fisher_covariance=zeros,
            residual=task,
        )

    nuisance = torch.where(valid, nuisance.detach(), torch.zeros_like(nuisance))
    mean_nuisance = (probabilities * nuisance).sum(dim=-1)
    nuisance_second = (probabilities * nuisance.square()).sum(dim=-1)
    nuisance_variance = (nuisance_second - mean_nuisance.square()).clamp_min(0.0)
    covariance = (probabilities * task * nuisance).sum(dim=-1) - mean_task * mean_nuisance
    projection = covariance / (nuisance_variance + epsilon)
    residual = task - projection.unsqueeze(-1) * nuisance
    residual_variance = (
        task_variance
        - 2.0 * projection * covariance
        + projection.square() * nuisance_variance
    ).clamp_min(0.0)
    fisher_cosine = covariance / torch.sqrt(
        (task_variance + epsilon) * (nuisance_variance + epsilon)
    )
    return ForwardMoments(
        projection=projection,
        residual_mean=mean_task - projection * mean_nuisance,
        fisher_cost=residual_variance,
        displacement_norm=residual.square().sum(dim=-1).sqrt(),
        task_displacement_norm=task_norm,
        nuisance_displacement_norm=nuisance.square().sum(dim=-1).sqrt(),
        task_nuisance_fisher_cosine=fisher_cosine.clamp(min=-1.0, max=1.0),
        residual_nuisance_fisher_covariance=covariance - projection * nuisance_variance,
        residual=residual,
    )


def reverse_tangent_moments(
    student_log_probs: torch.Tensor,
    task_log_ratio: torch.Tensor,
    support: TruncatedTopKSupport,
    nuisance_log_ratio: torch.Tensor | None = None,
    *,
    projection_epsilon: float = 1e-8,
) -> ReverseTangentMoments:
    """Compute the selected-support ``xi = F(p) r`` controller geometry."""
    for name, values in {
        "student_log_probs": student_log_probs,
        "task_log_ratio": task_log_ratio,
    }.items():
        if values.shape != support.token_ids.shape:
            raise ValueError(f"{name} must match the support shape")
    if nuisance_log_ratio is not None and nuisance_log_ratio.shape != support.token_ids.shape:
        raise ValueError("nuisance_log_ratio must match the support shape")
    epsilon = float(projection_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("projection_epsilon must be finite and positive")

    valid = support.valid_mask
    probabilities = torch.where(
        valid, student_log_probs.detach().exp(), torch.zeros_like(student_log_probs)
    )
    task_ratio = torch.where(valid, task_log_ratio.detach(), torch.zeros_like(task_log_ratio))
    task_ratio_mean = (probabilities * task_ratio).sum(dim=-1)
    task_tangent = probabilities * (task_ratio - task_ratio_mean.unsqueeze(-1))
    task_tangent = torch.where(valid, task_tangent, torch.zeros_like(task_tangent))
    task_tangent_mean = (probabilities * task_tangent).sum(dim=-1)
    task_fisher = (
        (probabilities * task_tangent.square()).sum(dim=-1) - task_tangent_mean.square()
    ).clamp_min(0.0)
    task_norm = task_tangent.square().sum(dim=-1).sqrt()
    zeros = torch.zeros_like(task_ratio_mean)
    zero_tangent = torch.zeros_like(task_tangent)
    if nuisance_log_ratio is None:
        return ReverseTangentMoments(
            projection=zeros,
            task_log_ratio_mean=task_ratio_mean,
            nuisance_log_ratio_mean=zeros,
            task_tangent_mean=task_tangent_mean,
            nuisance_tangent_mean=zeros,
            residual_mean=task_tangent_mean,
            fisher_cost=task_fisher,
            displacement_norm=task_norm,
            task_displacement_norm=task_norm,
            nuisance_displacement_norm=zeros,
            task_nuisance_fisher_cosine=zeros,
            residual_nuisance_fisher_covariance=zeros,
            task_tangent=task_tangent,
            nuisance_tangent=zero_tangent,
            residual=task_tangent,
        )

    nuisance_ratio = torch.where(
        valid, nuisance_log_ratio.detach(), torch.zeros_like(nuisance_log_ratio)
    )
    nuisance_ratio_mean = (probabilities * nuisance_ratio).sum(dim=-1)
    nuisance_tangent = probabilities * (nuisance_ratio - nuisance_ratio_mean.unsqueeze(-1))
    nuisance_tangent = torch.where(valid, nuisance_tangent, torch.zeros_like(nuisance_tangent))
    nuisance_tangent_mean = (probabilities * nuisance_tangent).sum(dim=-1)
    nuisance_fisher = (
        (probabilities * nuisance_tangent.square()).sum(dim=-1)
        - nuisance_tangent_mean.square()
    ).clamp_min(0.0)
    covariance = (
        (probabilities * task_tangent * nuisance_tangent).sum(dim=-1)
        - task_tangent_mean * nuisance_tangent_mean
    )
    projection = covariance / (nuisance_fisher + epsilon)
    residual = task_tangent - projection.unsqueeze(-1) * nuisance_tangent
    residual_fisher = (
        task_fisher
        - 2.0 * projection * covariance
        + projection.square() * nuisance_fisher
    ).clamp_min(0.0)
    fisher_cosine = covariance / torch.sqrt(
        (task_fisher + epsilon) * (nuisance_fisher + epsilon)
    )
    return ReverseTangentMoments(
        projection=projection,
        task_log_ratio_mean=task_ratio_mean,
        nuisance_log_ratio_mean=nuisance_ratio_mean,
        task_tangent_mean=task_tangent_mean,
        nuisance_tangent_mean=nuisance_tangent_mean,
        residual_mean=task_tangent_mean - projection * nuisance_tangent_mean,
        fisher_cost=residual_fisher,
        displacement_norm=residual.square().sum(dim=-1).sqrt(),
        task_displacement_norm=task_norm,
        nuisance_displacement_norm=nuisance_tangent.square().sum(dim=-1).sqrt(),
        task_nuisance_fisher_cosine=fisher_cosine.clamp(min=-1.0, max=1.0),
        residual_nuisance_fisher_covariance=covariance - projection * nuisance_fisher,
        task_tangent=task_tangent,
        nuisance_tangent=nuisance_tangent,
        residual=residual,
    )


def reverse_geometric_correction(
    student_log_probs: torch.Tensor,
    base_log_probs: torch.Tensor,
    log_ratio: torch.Tensor,
    weights: torch.Tensor,
    support: TruncatedTopKSupport,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return the restricted normalized-geometric Reverse-KL correction."""
    valid = support.valid_mask
    detached_weights = weights.detach().float().clamp(min=0.0, max=1.0)
    detached_ratio = torch.where(valid, log_ratio.detach(), torch.zeros_like(log_ratio))
    base_density = base_log_probs.detach().masked_fill(~valid, -torch.inf)
    geometric_density = (
        base_log_probs.detach() + detached_weights.unsqueeze(-1) * detached_ratio
    ).masked_fill(~valid, -torch.inf)
    relative_log_z = torch.logsumexp(geometric_density, dim=-1) - torch.logsumexp(
        base_density, dim=-1
    )
    student_probabilities = torch.where(
        valid, student_log_probs.exp(), torch.zeros_like(student_log_probs)
    )
    retained_mass = student_probabilities.sum(dim=-1)
    expected_ratio = (student_probabilities * detached_ratio).sum(dim=-1)
    correction = float(temperature) * (
        retained_mass * relative_log_z - detached_weights * expected_ratio
    )
    if not bool(torch.isfinite(correction).all().item()):
        raise ValueError("Reverse-KL correction must be finite")
    return correction
