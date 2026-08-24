"""Fixed-width support helpers for truncated VERPO vocabulary calculations.

The support only limits which coordinates are retained.  Log-probabilities are
always normalized over the full vocabulary, so the omitted tail is never
silently renormalized or represented by a synthetic bucket.  When ``K < V``,
loss directions, Fisher moments, and displacement norms are selected-support
approximations even though every selected log-probability uses the exact full
Softmax normalizer.  Truncated Reverse-KL geometric corrections are measured
relative to the base Teacher partition on the same support, so the correction
is exactly zero at zero path weight without renormalizing the support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch


VALID_VERPO_VOCAB_MODES = frozenset({"full", "topk_truncated"})
TOPK_TRUNCATED_FINGERPRINT = "topk_truncated_v1"
_TORCH_INTEGRAL_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


@dataclass(frozen=True)
class TruncatedTopKSupport:
    """Fixed-width token support with invalid duplicate coordinates masked."""

    token_ids: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.token_ids.shape != self.valid_mask.shape:
            raise ValueError("token_ids and valid_mask must have identical shapes")
        if self.token_ids.dtype != torch.long:
            raise ValueError("token_ids must have dtype torch.long")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must have dtype torch.bool")


@dataclass(frozen=True)
class ForwardMoments:
    """Named selected-support Forward-KL displacement statistics."""

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
    """Named selected-support Reverse-KL tangent statistics."""

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


@dataclass(frozen=True)
class _FullSoftmaxNormalization:
    """Live scaled logits and their full-vocabulary FP32 log normalizer."""

    scaled_logits: torch.Tensor
    log_z: torch.Tensor


def _validated_top_k(top_k: object) -> int:
    """Return ``top_k`` as a positive, exact host integer.

    Configuration values may be Python or NumPy integral scalars.  A CPU
    zero-dimensional integral Tensor is also accepted for callers that retain
    numeric configuration in Torch containers.  Booleans and fractional
    values are rejected rather than silently coerced.
    """
    if isinstance(top_k, bool):
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    if isinstance(top_k, Integral):
        top_k_value = int(top_k)
    elif (
        isinstance(top_k, torch.Tensor)
        and top_k.ndim == 0
        and top_k.device.type == "cpu"
        and top_k.dtype in _TORCH_INTEGRAL_DTYPES
    ):
        top_k_value = int(top_k.item())
    else:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    if top_k_value <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    return top_k_value


def validate_verpo_vocab_config(vocab_mode: str, top_k: object) -> None:
    """Validate the configured VERPO vocabulary mode and support size."""
    if not isinstance(vocab_mode, str) or vocab_mode not in VALID_VERPO_VOCAB_MODES:
        raise ValueError(
            "vocab_mode must be one of "
            f"{sorted(VALID_VERPO_VOCAB_MODES)}, got {vocab_mode!r}"
        )
    _validated_top_k(top_k)


def should_use_full_vocab(vocab_mode: str, top_k: object, vocab_size: int) -> bool:
    """Return whether this configuration retains every vocabulary coordinate."""
    validate_verpo_vocab_config(vocab_mode, top_k)
    if int(vocab_size) <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size!r}")
    return vocab_mode == "full" or _validated_top_k(top_k) >= int(vocab_size)


def build_truncated_topk_support(
    *,
    teacher_logits: Sequence[torch.Tensor],
    sampled_token_ids: torch.Tensor,
    top_k: object,
    student_logits: torch.Tensor | None = None,
) -> TruncatedTopKSupport:
    """Build a sorted union of source Top-K IDs and the sampled token ID.

    Every Teacher (and optionally the Student) contributes ``min(top_k, V)``
    candidates.  The realized sampled ID is appended once, then duplicate IDs
    are masked in place so the output width is stable across token positions.
    """
    if not teacher_logits:
        raise ValueError("teacher_logits must contain at least one tensor")
    top_k_value = _validated_top_k(top_k)

    reference_shape = teacher_logits[0].shape
    if len(reference_shape) < 1 or reference_shape[-1] <= 0:
        raise ValueError("teacher_logits must have a non-empty vocabulary dimension")
    for index, logits in enumerate(teacher_logits):
        if logits.shape != reference_shape:
            raise ValueError(
                "teacher_logits tensors must have identical shapes; "
                f"teacher_logits[{index}] differs"
            )
    if student_logits is not None and student_logits.shape != reference_shape:
        raise ValueError("student_logits must match teacher_logits shape")

    token_shape = reference_shape[:-1]
    if sampled_token_ids.shape != token_shape:
        raise ValueError("sampled_token_ids must match the logits token dimensions")
    if sampled_token_ids.dtype != torch.long:
        raise ValueError("sampled_token_ids must have dtype torch.long")
    vocab_size = reference_shape[-1]
    if bool(
        ((sampled_token_ids < 0) | (sampled_token_ids >= vocab_size)).any().item()
    ):
        raise ValueError("sampled_token_ids must contain vocabulary IDs")

    sources = [*teacher_logits]
    if student_logits is not None:
        sources.append(student_logits)
    source_top_k = min(top_k_value, vocab_size)
    candidates = [
        torch.topk(logits, k=source_top_k, dim=-1).indices for logits in sources
    ]
    candidates.append(sampled_token_ids.unsqueeze(-1))
    sorted_ids = torch.sort(torch.cat(candidates, dim=-1), dim=-1).values
    valid_mask = torch.ones_like(sorted_ids, dtype=torch.bool)
    valid_mask[..., 1:] = sorted_ids[..., 1:] != sorted_ids[..., :-1]
    token_ids = torch.where(valid_mask, sorted_ids, torch.zeros_like(sorted_ids))
    return TruncatedTopKSupport(token_ids=token_ids, valid_mask=valid_mask)


def _full_softmax_normalization(
    logits: torch.Tensor,
    *,
    temperature: float,
) -> _FullSoftmaxNormalization:
    """Compute one live full-vocabulary FP32 normalization for reused gathers."""
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("temperature must be positive and finite")
    if logits.ndim < 1 or logits.shape[-1] <= 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    scaled_logits = logits.float() / scale
    return _FullSoftmaxNormalization(
        scaled_logits=scaled_logits,
        log_z=torch.logsumexp(scaled_logits, dim=-1, keepdim=True),
    )


def _gather_full_softmax_log_probs(
    normalization: _FullSoftmaxNormalization,
    support: TruncatedTopKSupport,
) -> torch.Tensor:
    """Gather one support while preserving a shared live Student normalizer."""
    scaled_logits = normalization.scaled_logits
    if (
        support.token_ids.ndim < 1
        or support.token_ids.shape[:-1] != scaled_logits.shape[:-1]
    ):
        raise ValueError("support must match the logits token dimensions")
    if bool(
        (
            support.valid_mask
            & ((support.token_ids < 0) | (support.token_ids >= scaled_logits.shape[-1]))
        ).any().item()
    ):
        raise ValueError("support token_ids must be valid vocabulary IDs")
    safe_token_ids = torch.where(
        support.valid_mask, support.token_ids, torch.zeros_like(support.token_ids)
    )
    selected = (
        scaled_logits.gather(dim=-1, index=safe_token_ids) - normalization.log_z
    )
    return torch.where(support.valid_mask, selected, torch.zeros_like(selected))


def gather_full_softmax_log_probs(
    logits: torch.Tensor,
    support: TruncatedTopKSupport,
    *,
    temperature: float,
) -> torch.Tensor:
    """Gather support log-probabilities under full-vocabulary FP32 Softmax."""
    normalization = _full_softmax_normalization(logits, temperature=temperature)
    return _gather_full_softmax_log_probs(normalization, support)


def masked_sum(values: torch.Tensor, support: TruncatedTopKSupport) -> torch.Tensor:
    """Sum support values across valid coordinates only."""
    if values.shape != support.token_ids.shape:
        raise ValueError("values must match the support shape")
    return torch.where(support.valid_mask, values, torch.zeros_like(values)).sum(dim=-1)


def support_mass(
    selected_log_probs: torch.Tensor,
    support: TruncatedTopKSupport,
) -> torch.Tensor:
    """Return retained full-Softmax probability mass for each token position."""
    return masked_sum(selected_log_probs.exp(), support)


def compute_verpo_topk_support_diagnostics(
    *,
    reference_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    vocab_mode: str,
    top_k: object,
    temperature: float,
    reverse_kl: bool,
    negative_teacher_logits: torch.Tensor | None = None,
    additional_teacher_logits: Sequence[torch.Tensor] = (),
) -> dict[str, torch.Tensor]:
    """Measure retained support mass without renormalizing selected values.

    Support is the union of every Teacher Top-K set used by the evidence
    correction branch and the sampled token.  Reverse-KL additionally includes
    the Student Top-K set because its expectation is restricted to the selected
    coordinates.  The independent reference mass is measured on its own
    ``q_ref`` plus sampled-token support.  All returned tensors are detached
    diagnostics and therefore cannot add a live full-vocabulary Student
    autograd graph to the training loss.
    """
    teacher_logits = [evidence_teacher_logits]
    if negative_teacher_logits is not None:
        teacher_logits.append(negative_teacher_logits)
    teacher_logits.extend(additional_teacher_logits)
    for name, logits in {
        "evidence": evidence_teacher_logits,
        "student": student_logits,
        "negative": negative_teacher_logits,
    }.items():
        if logits is not None and logits.shape != reference_teacher_logits.shape:
            raise ValueError(
                f"reference and {name} logits must have identical shapes"
            )
    for index, logits in enumerate(additional_teacher_logits):
        if logits.shape != reference_teacher_logits.shape:
            raise ValueError(
                "additional Teacher logits must match reference logits; "
                f"item {index} differs"
            )
    token_shape = reference_teacher_logits.shape[:-1]
    vocab_size = reference_teacher_logits.shape[-1]
    if sampled_token_ids.shape != token_shape:
        raise ValueError("sampled_token_ids must match the logits token dimensions")

    with torch.no_grad():
        if should_use_full_vocab(vocab_mode, top_k, vocab_size):
            outputs = {
                "support_size": student_logits.new_full(
                    token_shape, float(vocab_size), dtype=torch.float32
                ),
                "reference_support_mass": student_logits.new_ones(
                    token_shape, dtype=torch.float32
                ),
                "evidence_support_mass": student_logits.new_ones(
                    token_shape, dtype=torch.float32
                ),
                "student_support_mass": student_logits.new_ones(
                    token_shape, dtype=torch.float32
                ),
            }
            if negative_teacher_logits is not None:
                outputs["negative_support_mass"] = student_logits.new_ones(
                    token_shape, dtype=torch.float32
                )
            return outputs

        support = build_truncated_topk_support(
            teacher_logits=teacher_logits,
            student_logits=student_logits if reverse_kl else None,
            sampled_token_ids=sampled_token_ids,
            top_k=top_k,
        )
        reference_support = build_truncated_topk_support(
            teacher_logits=[reference_teacher_logits],
            sampled_token_ids=sampled_token_ids,
            top_k=top_k,
        )
        outputs = {
            "support_size": support.valid_mask.sum(dim=-1).to(torch.float32),
            "reference_support_mass": support_mass(
                _selected_log_probs(
                    reference_teacher_logits,
                    reference_support,
                    temperature=temperature,
                    detach=True,
                ),
                reference_support,
            ),
            "evidence_support_mass": support_mass(
                _selected_log_probs(
                    evidence_teacher_logits,
                    support,
                    temperature=temperature,
                    detach=True,
                ),
                support,
            ),
            "student_support_mass": support_mass(
                _selected_log_probs(
                    student_logits,
                    support,
                    temperature=temperature,
                    detach=True,
                ),
                support,
            ),
        }
        if negative_teacher_logits is not None:
            outputs["negative_support_mass"] = support_mass(
                _selected_log_probs(
                    negative_teacher_logits,
                    support,
                    temperature=temperature,
                    detach=True,
                ),
                support,
            )
        return outputs


def forward_moments(
    student_log_probs: torch.Tensor,
    task: torch.Tensor,
    support: TruncatedTopKSupport,
    nuisance: torch.Tensor | None = None,
    *,
    projection_epsilon: float = 1e-8,
) -> ForwardMoments:
    """Return selected-support Forward-KL Fisher moments.

    Student probabilities retain their full-vocabulary normalization. Omitted
    coordinates contribute zero to every explicit sum, so when ``K < V`` the
    Fisher geometry and every returned norm are selected-support approximations.
    """
    if student_log_probs.shape != support.token_ids.shape:
        raise ValueError("student_log_probs must match the support shape")
    if task.shape != support.token_ids.shape:
        raise ValueError("task must match the support shape")
    if nuisance is not None and nuisance.shape != support.token_ids.shape:
        raise ValueError("nuisance must match the support shape")
    epsilon = float(projection_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("projection_epsilon must be finite and positive")

    valid = support.valid_mask
    probabilities = torch.where(valid, student_log_probs.exp(), torch.zeros_like(task))
    task = torch.where(valid, task, torch.zeros_like(task))
    mean_task = (probabilities * task).sum(dim=-1)
    task_second = (probabilities * task.square()).sum(dim=-1)
    task_variance = (task_second - mean_task.square()).clamp(min=0.0)
    task_norm = task.square().sum(dim=-1).sqrt()

    if nuisance is None:
        zeros = torch.zeros_like(mean_task)
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

    nuisance = torch.where(valid, nuisance, torch.zeros_like(nuisance))
    mean_nuisance = (probabilities * nuisance).sum(dim=-1)
    nuisance_second = (probabilities * nuisance.square()).sum(dim=-1)
    nuisance_variance = (
        nuisance_second - mean_nuisance.square()
    ).clamp(min=0.0)
    covariance = (
        (probabilities * task * nuisance).sum(dim=-1)
        - mean_task * mean_nuisance
    )
    projection = covariance / (nuisance_variance + epsilon)
    residual = task - projection.unsqueeze(-1) * nuisance
    residual_mean = mean_task - projection * mean_nuisance
    residual_variance = (
        task_variance
        - 2.0 * projection * covariance
        + projection.square() * nuisance_variance
    ).clamp(min=0.0)
    nuisance_norm = nuisance.square().sum(dim=-1).sqrt()
    displacement_norm = residual.square().sum(dim=-1).sqrt()
    fisher_cosine = covariance / torch.sqrt(
        (task_variance + epsilon) * (nuisance_variance + epsilon)
    )
    fisher_cosine = fisher_cosine.clamp(min=-1.0, max=1.0)
    residual_nuisance_covariance = covariance - projection * nuisance_variance
    return ForwardMoments(
        projection=projection,
        residual_mean=residual_mean,
        fisher_cost=residual_variance,
        displacement_norm=displacement_norm,
        task_displacement_norm=task_norm,
        nuisance_displacement_norm=nuisance_norm,
        task_nuisance_fisher_cosine=fisher_cosine,
        residual_nuisance_fisher_covariance=residual_nuisance_covariance,
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
    """Return selected-support Reverse-KL tangent and Fisher diagnostics.

    The selected Student probabilities retain their full-vocabulary Softmax
    normalization.  Omitted coordinates are explicitly zero, so for ``K < V``
    both ``xi = p * (r - sum_S p r)`` and the subsequent Fisher projection are
    selected-support approximations without retained-mass renormalization.
    """
    if student_log_probs.shape != support.token_ids.shape:
        raise ValueError("student_log_probs must match the support shape")
    if task_log_ratio.shape != support.token_ids.shape:
        raise ValueError("task_log_ratio must match the support shape")
    if (
        nuisance_log_ratio is not None
        and nuisance_log_ratio.shape != support.token_ids.shape
    ):
        raise ValueError("nuisance_log_ratio must match the support shape")
    epsilon = float(projection_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("projection_epsilon must be finite and positive")

    valid = support.valid_mask
    probabilities = torch.where(
        valid, student_log_probs.exp(), torch.zeros_like(student_log_probs)
    )
    task_ratio = torch.where(
        valid, task_log_ratio, torch.zeros_like(task_log_ratio)
    )
    task_ratio_mean = (probabilities * task_ratio).sum(dim=-1)
    task_tangent = probabilities * (
        task_ratio - task_ratio_mean.unsqueeze(-1)
    )
    task_tangent = torch.where(
        valid, task_tangent, torch.zeros_like(task_tangent)
    )
    task_tangent_mean = (probabilities * task_tangent).sum(dim=-1)
    task_fisher = (
        (probabilities * task_tangent.square()).sum(dim=-1)
        - task_tangent_mean.square()
    ).clamp(min=0.0)
    task_norm = task_tangent.square().sum(dim=-1).sqrt()

    if nuisance_log_ratio is None:
        zeros = torch.zeros_like(task_ratio_mean)
        zero_tangent = torch.zeros_like(task_tangent)
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
        valid, nuisance_log_ratio, torch.zeros_like(nuisance_log_ratio)
    )
    nuisance_ratio_mean = (probabilities * nuisance_ratio).sum(dim=-1)
    nuisance_tangent = probabilities * (
        nuisance_ratio - nuisance_ratio_mean.unsqueeze(-1)
    )
    nuisance_tangent = torch.where(
        valid, nuisance_tangent, torch.zeros_like(nuisance_tangent)
    )
    nuisance_tangent_mean = (probabilities * nuisance_tangent).sum(dim=-1)
    nuisance_fisher = (
        (probabilities * nuisance_tangent.square()).sum(dim=-1)
        - nuisance_tangent_mean.square()
    ).clamp(min=0.0)
    covariance = (
        (probabilities * task_tangent * nuisance_tangent).sum(dim=-1)
        - task_tangent_mean * nuisance_tangent_mean
    )
    projection = covariance / (nuisance_fisher + epsilon)
    residual = task_tangent - projection.unsqueeze(-1) * nuisance_tangent
    residual_mean = task_tangent_mean - projection * nuisance_tangent_mean
    residual_fisher = (
        task_fisher
        - 2.0 * projection * covariance
        + projection.square() * nuisance_fisher
    ).clamp(min=0.0)
    nuisance_norm = nuisance_tangent.square().sum(dim=-1).sqrt()
    residual_norm = residual.square().sum(dim=-1).sqrt()
    fisher_cosine = covariance / torch.sqrt(
        (task_fisher + epsilon) * (nuisance_fisher + epsilon)
    )
    fisher_cosine = fisher_cosine.clamp(min=-1.0, max=1.0)
    residual_nuisance_covariance = covariance - projection * nuisance_fisher
    return ReverseTangentMoments(
        projection=projection,
        task_log_ratio_mean=task_ratio_mean,
        nuisance_log_ratio_mean=nuisance_ratio_mean,
        task_tangent_mean=task_tangent_mean,
        nuisance_tangent_mean=nuisance_tangent_mean,
        residual_mean=residual_mean,
        fisher_cost=residual_fisher,
        displacement_norm=residual_norm,
        task_displacement_norm=task_norm,
        nuisance_displacement_norm=nuisance_norm,
        task_nuisance_fisher_cosine=fisher_cosine,
        residual_nuisance_fisher_covariance=residual_nuisance_covariance,
        task_tangent=task_tangent,
        nuisance_tangent=nuisance_tangent,
        residual=residual,
    )


def _selected_log_probs(
    logits: torch.Tensor,
    support: TruncatedTopKSupport,
    *,
    temperature: float,
    detach: bool,
) -> torch.Tensor:
    values = gather_full_softmax_log_probs(logits, support, temperature=temperature)
    return values.detach() if detach else values


def _selected_log_probs_from_normalization(
    normalization: _FullSoftmaxNormalization,
    support: TruncatedTopKSupport,
    *,
    detach: bool,
) -> torch.Tensor:
    values = _gather_full_softmax_log_probs(normalization, support)
    return values.detach() if detach else values


def _restricted_forward_kl(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    support: TruncatedTopKSupport,
    *,
    temperature: float,
) -> torch.Tensor:
    teacher_probs = teacher_log_probs.exp()
    return float(temperature) * masked_sum(
        teacher_probs * (teacher_log_probs - student_log_probs), support
    )


def _sampled_support_index(
    sampled_token_ids: torch.Tensor,
    support: TruncatedTopKSupport,
) -> torch.Tensor:
    matches = support.valid_mask & sampled_token_ids.unsqueeze(-1).eq(
        support.token_ids
    )
    if not bool(matches.any(dim=-1).all().item()):
        raise ValueError("sampled token must be present in truncated support")
    return matches.to(dtype=torch.long).argmax(dim=-1, keepdim=True)


def reverse_geometric_correction(
    student_log_probs: torch.Tensor,
    base_log_probs: torch.Tensor,
    log_ratio: torch.Tensor,
    weights: torch.Tensor,
    support: TruncatedTopKSupport,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return the selected-support geometric Reverse-KL correction.

    The restricted coordinate contribution is

        T [mass_p(S) (log Z_S(w) - log Z_S(0)) - w sum_S p r],

    where ``Z_S(w) = sum_S q_base exp(w r)`` and
    ``Z_S(0) = sum_S q_base``.  Both partition terms are detached Teacher
    quantities.  The retained Student mass is mandatory because neither
    Student nor Teacher probabilities are renormalized on the selected
    support.  At full support ``log Z_S(0) = 0``, recovering the exact
    full-vocabulary formula.
    """
    for name, values in {
        "student_log_probs": student_log_probs,
        "base_log_probs": base_log_probs,
        "log_ratio": log_ratio,
    }.items():
        if values.shape != support.token_ids.shape:
            raise ValueError(f"{name} must match the support shape")
    if weights.shape != support.token_ids.shape[:-1]:
        raise ValueError("weights must match the support token dimensions")
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("temperature must be positive and finite")

    with torch.no_grad():
        detached_weights = weights.detach().float()
        detached_base = base_log_probs.detach()
        detached_ratio = torch.where(
            support.valid_mask,
            log_ratio.detach(),
            torch.zeros_like(log_ratio),
        )
        base_density = detached_base.masked_fill(
            ~support.valid_mask, -torch.inf
        )
        geometric_density = (
            detached_base
            + detached_weights.unsqueeze(-1) * detached_ratio
        ).masked_fill(~support.valid_mask, -torch.inf)
        base_log_z = torch.logsumexp(base_density, dim=-1)
        log_z = torch.logsumexp(geometric_density, dim=-1)
        relative_log_z = log_z - base_log_z
    retained_mass = support_mass(student_log_probs, support)
    expected_ratio = masked_sum(
        student_log_probs.exp() * detached_ratio, support
    )
    correction = scale * (
        retained_mass * relative_log_z - detached_weights * expected_ratio
    )
    if not bool(torch.isfinite(correction).all().item()):
        raise ValueError("Reverse-KL correction must be finite")
    return correction


def truncated_reverse_displacement_stats(
    evidence_teacher_logits: torch.Tensor,
    base_teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected-support Reverse-KL benefit, cost, and tangent norm."""
    if evidence_teacher_logits.shape != base_teacher_logits.shape:
        raise ValueError("evidence and base Teacher logits must have identical shapes")
    if evidence_teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher and Student logits must have identical shapes")
    if advantages.shape != student_logits.shape[:1]:
        raise ValueError("advantages must have shape [n_rows]")
    support = build_truncated_topk_support(
        teacher_logits=[base_teacher_logits, evidence_teacher_logits],
        student_logits=student_logits,
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    with torch.no_grad():
        student_log_probs = _selected_log_probs(
            student_logits, support, temperature=temperature, detach=True
        )
        base_log_probs = _selected_log_probs(
            base_teacher_logits, support, temperature=temperature, detach=True
        )
        evidence_log_probs = _selected_log_probs(
            evidence_teacher_logits, support, temperature=temperature, detach=True
        )
        moments = reverse_tangent_moments(
            student_log_probs,
            evidence_log_probs - base_log_probs,
            support,
        )
        sampled_tangent = moments.residual.gather(
            -1, _sampled_support_index(sampled_token_ids, support)
        ).squeeze(-1)
        centered_alignment = sampled_tangent - moments.residual_mean
        benefit = advantages.detach().to(centered_alignment.dtype).unsqueeze(-1) * (
            centered_alignment
        )
    return benefit, moments.fisher_cost, moments.displacement_norm


def truncated_reverse_fec_stats(
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
    projection_epsilon: float,
) -> dict[str, torch.Tensor]:
    """Return selected-support Reverse-KL FEC tangent statistics."""
    for name, logits in {
        "negative": negative_teacher_logits,
        "no-evidence": no_evidence_teacher_logits,
        "student": student_logits,
    }.items():
        if positive_teacher_logits.shape != logits.shape:
            raise ValueError(f"positive and {name} logits must have identical shapes")
    if advantages.shape != student_logits.shape[:1]:
        raise ValueError("advantages must have shape [n_rows]")
    support = build_truncated_topk_support(
        teacher_logits=[
            positive_teacher_logits,
            negative_teacher_logits,
            no_evidence_teacher_logits,
        ],
        student_logits=student_logits,
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    with torch.no_grad():
        student_log_probs = _selected_log_probs(
            student_logits, support, temperature=temperature, detach=True
        )
        positive_log_probs = _selected_log_probs(
            positive_teacher_logits, support, temperature=temperature, detach=True
        )
        negative_log_probs = _selected_log_probs(
            negative_teacher_logits, support, temperature=temperature, detach=True
        )
        no_evidence_log_probs = _selected_log_probs(
            no_evidence_teacher_logits, support, temperature=temperature, detach=True
        )
        task_log_ratio = positive_log_probs - negative_log_probs
        nuisance_log_ratio = (
            0.5 * (positive_log_probs + negative_log_probs)
            - no_evidence_log_probs
        )
        moments = reverse_tangent_moments(
            student_log_probs,
            task_log_ratio,
            support,
            nuisance_log_ratio,
            projection_epsilon=projection_epsilon,
        )
        sampled_index = _sampled_support_index(sampled_token_ids, support)
        fec_alignment = (
            moments.residual.gather(-1, sampled_index).squeeze(-1)
            - moments.residual_mean
        )
        benefit = advantages.detach().to(fec_alignment.dtype).unsqueeze(-1) * (
            fec_alignment
        )
        outputs = {
            "benefit": benefit,
            "fisher_cost": moments.fisher_cost,
            "displacement_norm": moments.displacement_norm,
            "task_displacement_norm": moments.task_displacement_norm,
            "nuisance_displacement_norm": moments.nuisance_displacement_norm,
            "task_nuisance_fisher_cosine": (
                moments.task_nuisance_fisher_cosine
            ),
            "nuisance_projection_coefficient": moments.projection,
            "residual_nuisance_fisher_covariance": (
                moments.residual_nuisance_fisher_covariance
            ),
            "fec_alignment": fec_alignment,
        }
    if any(not bool(torch.isfinite(value).all().item()) for value in outputs.values()):
        raise ValueError("Reverse-KL FEC statistics must be finite")
    return outputs


def _restricted_reverse_kl(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    support: TruncatedTopKSupport,
    *,
    temperature: float,
) -> torch.Tensor:
    return float(temperature) * masked_sum(
        student_log_probs.exp() * (student_log_probs - teacher_log_probs),
        support,
    )


def _truncated_reverse_teacher_losses(
    student_logits: torch.Tensor,
    reference_teacher_logits: torch.Tensor,
    base_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independent reference and geometric Reverse-KL corrections.

    Fixed uses ``q_ref/q_0`` for the independent anchor and ``q_0 -> q_e``
    for the path.  CTR uses the same helper with ``q_ref=q_0`` and the path
    ``q_- -> q_+``.  Both supports include the relevant Teacher Top-K sets,
    Student Top-K, and sampled token while sharing one live Student normalizer.
    """
    for name, logits in {
        "reference": reference_teacher_logits,
        "base": base_teacher_logits,
        "evidence": evidence_teacher_logits,
    }.items():
        if student_logits.shape != logits.shape:
            raise ValueError(
                f"student and {name} Teacher logits must have identical shapes"
            )
    if evidence_weights.shape != student_logits.shape[:-1]:
        raise ValueError("evidence_weights must match the token dimensions")
    reference_support = build_truncated_topk_support(
        teacher_logits=[reference_teacher_logits],
        student_logits=student_logits,
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    correction_support = build_truncated_topk_support(
        teacher_logits=[base_teacher_logits, evidence_teacher_logits],
        student_logits=student_logits,
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    student_normalization = _full_softmax_normalization(
        student_logits, temperature=temperature
    )
    student_reference = _selected_log_probs_from_normalization(
        student_normalization, reference_support, detach=False
    )
    student_correction = _selected_log_probs_from_normalization(
        student_normalization, correction_support, detach=False
    )
    with torch.no_grad():
        reference_log_probs = _selected_log_probs(
            reference_teacher_logits,
            reference_support,
            temperature=temperature,
            detach=True,
        )
        base_log_probs = _selected_log_probs(
            base_teacher_logits,
            correction_support,
            temperature=temperature,
            detach=True,
        )
        evidence_log_probs = _selected_log_probs(
            evidence_teacher_logits,
            correction_support,
            temperature=temperature,
            detach=True,
        )
    reference_loss = _restricted_reverse_kl(
        student_reference,
        reference_log_probs,
        reference_support,
        temperature=temperature,
    )
    correction = reverse_geometric_correction(
        student_correction,
        base_log_probs,
        evidence_log_probs - base_log_probs,
        evidence_weights,
        correction_support,
        temperature=temperature,
    )
    return reference_loss, correction


def truncated_reverse_fixed_losses(
    student_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return truncated Reverse-KL Fixed anchor and ``q_0 -> q_e`` correction.

    Fixed VERPO uses the evidence-free Teacher ``q_0`` for both the
    independent reference anchor and the base of the geometric correction
    path.  Keeping that equality in this public signature avoids treating the
    Fixed and CTR Teacher roles as interchangeable at call sites.
    """
    return _truncated_reverse_teacher_losses(
        student_logits,
        no_evidence_teacher_logits,
        no_evidence_teacher_logits,
        evidence_teacher_logits,
        sampled_token_ids,
        evidence_weights,
        top_k=top_k,
        temperature=temperature,
    )


def truncated_reverse_ctr_losses(
    student_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return truncated Reverse-KL CTR ``q_0`` anchor and ``q_- -> q_+`` path.

    The independent anchor remains the no-evidence Teacher ``q_0``.  The
    signed geometric correction instead starts at the contrastive incorrect
    sibling distribution ``q_-`` and moves toward ``q_+``.
    """
    return _truncated_reverse_teacher_losses(
        student_logits,
        no_evidence_teacher_logits,
        negative_teacher_logits,
        positive_teacher_logits,
        sampled_token_ids,
        evidence_weights,
        top_k=top_k,
        temperature=temperature,
    )


def _truncated_reverse_correction_loss(
    student_logits: torch.Tensor,
    base_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
    vocab_mode: str = "topk_truncated",
) -> torch.Tensor:
    """Return only a selected-support Reverse-KL geometric correction."""
    validate_verpo_vocab_config(vocab_mode, top_k)
    if vocab_mode != "topk_truncated":
        raise ValueError("truncated correction requires vocab_mode=topk_truncated")
    if student_logits.shape != base_teacher_logits.shape:
        raise ValueError("student and base Teacher logits must have identical shapes")
    if student_logits.shape != evidence_teacher_logits.shape:
        raise ValueError(
            "student and evidence Teacher logits must have identical shapes"
        )
    if evidence_weights.shape != student_logits.shape[:-1]:
        raise ValueError("evidence_weights must match the token dimensions")
    support = build_truncated_topk_support(
        teacher_logits=[base_teacher_logits, evidence_teacher_logits],
        student_logits=student_logits,
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    student_log_probs = _selected_log_probs(
        student_logits, support, temperature=temperature, detach=False
    )
    with torch.no_grad():
        base_log_probs = _selected_log_probs(
            base_teacher_logits, support, temperature=temperature, detach=True
        )
        evidence_log_probs = _selected_log_probs(
            evidence_teacher_logits, support, temperature=temperature, detach=True
        )
    return reverse_geometric_correction(
        student_log_probs,
        base_log_probs,
        evidence_log_probs - base_log_probs,
        evidence_weights,
        support,
        temperature=temperature,
    )


def truncated_reverse_fixed_loss(
    student_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
    vocab_mode: str = "topk_truncated",
) -> torch.Tensor:
    """Return only the truncated Fixed ``q_0 -> q_e`` correction."""
    return _truncated_reverse_correction_loss(
        student_logits,
        no_evidence_teacher_logits,
        evidence_teacher_logits,
        sampled_token_ids,
        evidence_weights,
        top_k=top_k,
        temperature=temperature,
        vocab_mode=vocab_mode,
    )


def truncated_reverse_ctr_loss(
    student_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
    vocab_mode: str = "topk_truncated",
) -> torch.Tensor:
    """Return only the truncated CTR ``q_- -> q_+`` correction."""
    return _truncated_reverse_correction_loss(
        student_logits,
        negative_teacher_logits,
        positive_teacher_logits,
        sampled_token_ids,
        evidence_weights,
        top_k=top_k,
        temperature=temperature,
        vocab_mode=vocab_mode,
    )


def truncated_reverse_fec_loss(
    student_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    nuisance_projection_coefficient: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
) -> torch.Tensor:
    """Return correction-only selected-support Reverse-KL FEC loss."""
    for name, logits in {
        "positive": positive_teacher_logits,
        "negative": negative_teacher_logits,
        "no-evidence": no_evidence_teacher_logits,
    }.items():
        if student_logits.shape != logits.shape:
            raise ValueError(
                f"student and {name} Teacher logits must have identical shapes"
            )
    token_shape = student_logits.shape[:-1]
    if evidence_weights.shape != token_shape:
        raise ValueError("evidence_weights must match the token dimensions")
    if nuisance_projection_coefficient.shape != token_shape:
        raise ValueError(
            "nuisance_projection_coefficient must match the token dimensions"
        )
    support = build_truncated_topk_support(
        teacher_logits=[
            positive_teacher_logits,
            negative_teacher_logits,
            no_evidence_teacher_logits,
        ],
        student_logits=student_logits,
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    student_normalization = _full_softmax_normalization(
        student_logits, temperature=temperature
    )
    student_log_probs = _selected_log_probs_from_normalization(
        student_normalization, support, detach=False
    )
    with torch.no_grad():
        projection = nuisance_projection_coefficient.detach().float()
        if not bool(torch.isfinite(projection).all().item()):
            raise ValueError("nuisance_projection_coefficient must be finite")
        positive_log_probs = _selected_log_probs(
            positive_teacher_logits, support, temperature=temperature, detach=True
        )
        negative_log_probs = _selected_log_probs(
            negative_teacher_logits, support, temperature=temperature, detach=True
        )
        no_evidence_log_probs = _selected_log_probs(
            no_evidence_teacher_logits, support, temperature=temperature, detach=True
        )
        task_log_ratio = positive_log_probs - negative_log_probs
        nuisance_log_ratio = (
            0.5 * (positive_log_probs + negative_log_probs)
            - no_evidence_log_probs
        )
        fec_log_ratio = task_log_ratio - projection.unsqueeze(-1) * (
            nuisance_log_ratio
        )
    return reverse_geometric_correction(
        student_log_probs,
        negative_log_probs,
        fec_log_ratio,
        evidence_weights,
        support,
        temperature=temperature,
    )


def truncated_forward_displacement_stats(
    evidence_teacher_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected-support Fixed/CTR benefit, Fisher cost, and norm.

    Full-Softmax normalization is retained, but for ``K < V`` the direction,
    Fisher geometry, and displacement norm are selected-support approximations.
    """
    if evidence_teacher_logits.shape != no_evidence_teacher_logits.shape:
        raise ValueError("evidence and base Teacher logits must have identical shapes")
    if evidence_teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher and Student logits must have identical shapes")
    if advantages.shape != student_logits.shape[:1]:
        raise ValueError("advantages must have shape [n_rows]")
    support = build_truncated_topk_support(
        teacher_logits=[evidence_teacher_logits, no_evidence_teacher_logits],
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    with torch.no_grad():
        student_log_probs = _selected_log_probs(
            student_logits, support, temperature=temperature, detach=True
        )
        evidence_probs = _selected_log_probs(
            evidence_teacher_logits, support, temperature=temperature, detach=True
        ).exp()
        no_evidence_probs = _selected_log_probs(
            no_evidence_teacher_logits, support, temperature=temperature, detach=True
        ).exp()
        task = evidence_probs - no_evidence_probs
        moments = forward_moments(student_log_probs, task, support)
        sampled_task = task.gather(
            -1, _sampled_support_index(sampled_token_ids, support)
        ).squeeze(-1)
        centered = sampled_task - moments.residual_mean
        benefit = advantages.detach().to(centered.dtype).unsqueeze(-1) * centered
    return benefit, moments.fisher_cost, moments.displacement_norm


def truncated_forward_fec_stats(
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
    projection_epsilon: float,
) -> dict[str, torch.Tensor]:
    """Return approximate Forward-FEC statistics on the Teacher Top-K union.

    Selected probabilities retain full-Softmax normalization.  For ``K < V``,
    the direction, Fisher projection geometry, and reported norms omit the tail.
    """
    for name, logits in {
        "negative": negative_teacher_logits,
        "no-evidence": no_evidence_teacher_logits,
        "student": student_logits,
    }.items():
        if positive_teacher_logits.shape != logits.shape:
            raise ValueError(f"positive and {name} logits must have identical shapes")
    if advantages.shape != student_logits.shape[:1]:
        raise ValueError("advantages must have shape [n_rows]")
    support = build_truncated_topk_support(
        teacher_logits=[
            positive_teacher_logits,
            negative_teacher_logits,
            no_evidence_teacher_logits,
        ],
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    with torch.no_grad():
        student_log_probs = _selected_log_probs(
            student_logits, support, temperature=temperature, detach=True
        )
        positive_probs = _selected_log_probs(
            positive_teacher_logits, support, temperature=temperature, detach=True
        ).exp()
        negative_probs = _selected_log_probs(
            negative_teacher_logits, support, temperature=temperature, detach=True
        ).exp()
        no_evidence_probs = _selected_log_probs(
            no_evidence_teacher_logits, support, temperature=temperature, detach=True
        ).exp()
        task = positive_probs - negative_probs
        nuisance = 0.5 * (positive_probs + negative_probs) - no_evidence_probs
        moments = forward_moments(
            student_log_probs,
            task,
            support,
            nuisance,
            projection_epsilon=projection_epsilon,
        )
        sampled_index = _sampled_support_index(sampled_token_ids, support)
        fec_alignment = (
            moments.residual.gather(-1, sampled_index).squeeze(-1)
            - moments.residual_mean
        )
        benefit = advantages.detach().to(fec_alignment.dtype).unsqueeze(-1) * fec_alignment
        outputs = {
            "benefit": benefit,
            "fisher_cost": moments.fisher_cost,
            "displacement_norm": moments.displacement_norm,
            "task_displacement_norm": moments.task_displacement_norm,
            "nuisance_displacement_norm": moments.nuisance_displacement_norm,
            "task_nuisance_fisher_cosine": moments.task_nuisance_fisher_cosine,
            "nuisance_projection_coefficient": moments.projection,
            "residual_nuisance_fisher_covariance": (
                moments.residual_nuisance_fisher_covariance
            ),
            "fec_alignment": fec_alignment,
        }
    if any(not bool(torch.isfinite(value).all().item()) for value in outputs.values()):
        raise ValueError("FEC statistics must be finite")
    return outputs


def truncated_forward_fixed_losses(
    student_logits: torch.Tensor,
    reference_teacher_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected-support reference FKL and signed Fixed correction.

    Both supports share one live full-vocabulary Student normalization.  For
    ``K < V``, both losses and their gradients are selected-support
    approximations despite using full-Softmax-normalized log-probabilities.
    """
    for name, logits in {
        "reference": reference_teacher_logits,
        "base": no_evidence_teacher_logits,
        "evidence": evidence_teacher_logits,
    }.items():
        if student_logits.shape != logits.shape:
            raise ValueError(f"student and {name} Teacher logits must have identical shapes")
    if evidence_weights.shape != student_logits.shape[:-1]:
        raise ValueError("evidence_weights must match the token dimensions")
    reference_support = build_truncated_topk_support(
        teacher_logits=[reference_teacher_logits],
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    correction_support = build_truncated_topk_support(
        teacher_logits=[no_evidence_teacher_logits, evidence_teacher_logits],
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    student_normalization = _full_softmax_normalization(
        student_logits, temperature=temperature
    )
    student_reference = _selected_log_probs_from_normalization(
        student_normalization, reference_support, detach=False
    )
    student_correction = _selected_log_probs_from_normalization(
        student_normalization, correction_support, detach=False
    )
    with torch.no_grad():
        weights = evidence_weights.detach().float()
        reference_log_probs = _selected_log_probs(
            reference_teacher_logits, reference_support, temperature=temperature, detach=True
        )
        base_log_probs = _selected_log_probs(
            no_evidence_teacher_logits, correction_support, temperature=temperature, detach=True
        )
        evidence_log_probs = _selected_log_probs(
            evidence_teacher_logits, correction_support, temperature=temperature, detach=True
        )
        displacement = evidence_log_probs.exp() - base_log_probs.exp()
    reference_loss = _restricted_forward_kl(
        reference_log_probs, student_reference, reference_support, temperature=temperature
    )
    correction = -float(temperature) * weights * masked_sum(
        displacement * student_correction, correction_support
    )
    return reference_loss, correction


def truncated_forward_fixed_loss(
    student_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
    vocab_mode: str = "topk_truncated",
) -> torch.Tensor:
    """Return only the truncated signed Fixed Forward-KL correction."""
    validate_verpo_vocab_config(vocab_mode, top_k)
    if vocab_mode != "topk_truncated":
        raise ValueError("truncated correction requires vocab_mode=topk_truncated")
    if student_logits.shape != no_evidence_teacher_logits.shape:
        raise ValueError("student and base Teacher logits must have identical shapes")
    if student_logits.shape != evidence_teacher_logits.shape:
        raise ValueError(
            "student and evidence Teacher logits must have identical shapes"
        )
    if evidence_weights.shape != student_logits.shape[:-1]:
        raise ValueError("evidence_weights must match the token dimensions")
    support = build_truncated_topk_support(
        teacher_logits=[no_evidence_teacher_logits, evidence_teacher_logits],
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    student_log_probs = _selected_log_probs(
        student_logits, support, temperature=temperature, detach=False
    )
    with torch.no_grad():
        weights = evidence_weights.detach().float()
        base_log_probs = _selected_log_probs(
            no_evidence_teacher_logits,
            support,
            temperature=temperature,
            detach=True,
        )
        evidence_log_probs = _selected_log_probs(
            evidence_teacher_logits,
            support,
            temperature=temperature,
            detach=True,
        )
        displacement = evidence_log_probs.exp() - base_log_probs.exp()
    return -float(temperature) * weights * masked_sum(
        displacement * student_log_probs, support
    )


def _truncated_forward_reference_loss(
    student_logits: torch.Tensor,
    reference_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
) -> torch.Tensor:
    """Return the reference Forward-KL on its independent Top-K support."""
    if student_logits.shape != reference_teacher_logits.shape:
        raise ValueError("student and reference Teacher logits must have identical shapes")
    reference_support = build_truncated_topk_support(
        teacher_logits=[reference_teacher_logits],
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    student_normalization = _full_softmax_normalization(
        student_logits, temperature=temperature
    )
    student_reference = _selected_log_probs_from_normalization(
        student_normalization, reference_support, detach=False
    )
    with torch.no_grad():
        reference_log_probs = _selected_log_probs(
            reference_teacher_logits, reference_support, temperature=temperature, detach=True
        )
    return _restricted_forward_kl(
        reference_log_probs, student_reference, reference_support, temperature=temperature
    )


def _truncated_forward_ctr_losses(
    student_logits: torch.Tensor,
    reference_teacher_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return composed truncated CTR losses with one live Student normalizer."""
    for name, logits in {
        "reference": reference_teacher_logits,
        "positive": positive_teacher_logits,
        "negative": negative_teacher_logits,
    }.items():
        if student_logits.shape != logits.shape:
            raise ValueError(f"student and {name} Teacher logits must have identical shapes")
    if evidence_weights.shape != student_logits.shape[:-1]:
        raise ValueError("evidence_weights must match the token dimensions")
    reference_support = build_truncated_topk_support(
        teacher_logits=[reference_teacher_logits],
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    correction_support = build_truncated_topk_support(
        teacher_logits=[positive_teacher_logits, negative_teacher_logits],
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    student_normalization = _full_softmax_normalization(
        student_logits, temperature=temperature
    )
    student_reference = _selected_log_probs_from_normalization(
        student_normalization, reference_support, detach=False
    )
    student_correction = _selected_log_probs_from_normalization(
        student_normalization, correction_support, detach=False
    )
    with torch.no_grad():
        weights = evidence_weights.detach().float()
        reference_log_probs = _selected_log_probs(
            reference_teacher_logits,
            reference_support,
            temperature=temperature,
            detach=True,
        )
        positive_probs = _selected_log_probs(
            positive_teacher_logits,
            correction_support,
            temperature=temperature,
            detach=True,
        ).exp()
        negative_probs = _selected_log_probs(
            negative_teacher_logits,
            correction_support,
            temperature=temperature,
            detach=True,
        ).exp()
        displacement = positive_probs - negative_probs
    reference_loss = _restricted_forward_kl(
        reference_log_probs,
        student_reference,
        reference_support,
        temperature=temperature,
    )
    correction = -float(temperature) * weights * masked_sum(
        displacement * student_correction, correction_support
    )
    return reference_loss, correction


def truncated_forward_ctr_loss(
    student_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
    vocab_mode: str = "topk_truncated",
) -> torch.Tensor:
    """Return the signed CTR correction on the positive/negative Top-K support.

    This correction-only entry point computes no reference anchor.  For
    ``K < V``, its direction is a selected-support approximation under the
    full-vocabulary Student Softmax normalization.
    """
    validate_verpo_vocab_config(vocab_mode, top_k)
    if vocab_mode != "topk_truncated":
        raise ValueError("truncated correction requires vocab_mode=topk_truncated")
    for name, logits in {
        "positive": positive_teacher_logits,
        "negative": negative_teacher_logits,
    }.items():
        if student_logits.shape != logits.shape:
            raise ValueError(f"student and {name} Teacher logits must have identical shapes")
    if evidence_weights.shape != student_logits.shape[:-1]:
        raise ValueError("evidence_weights must match the token dimensions")
    correction_support = build_truncated_topk_support(
        teacher_logits=[positive_teacher_logits, negative_teacher_logits],
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    student_normalization = _full_softmax_normalization(
        student_logits, temperature=temperature
    )
    student_correction = _selected_log_probs_from_normalization(
        student_normalization, correction_support, detach=False
    )
    with torch.no_grad():
        weights = evidence_weights.detach().float()
        positive_probs = _selected_log_probs(
            positive_teacher_logits, correction_support, temperature=temperature, detach=True
        ).exp()
        negative_probs = _selected_log_probs(
            negative_teacher_logits, correction_support, temperature=temperature, detach=True
        ).exp()
        displacement = positive_probs - negative_probs
    correction = -float(temperature) * weights * masked_sum(
        displacement * student_correction, correction_support
    )
    return correction


def truncated_forward_fec_loss(
    student_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    evidence_weights: torch.Tensor,
    nuisance_projection_coefficient: torch.Tensor,
    *,
    top_k: object,
    temperature: float,
) -> torch.Tensor:
    """Return the signed Forward-FEC correction on selected support.

    For ``K < V``, the residual direction is a selected-support approximation;
    selected Student log-probabilities still use full-Softmax normalization.
    """
    for name, logits in {
        "positive": positive_teacher_logits,
        "negative": negative_teacher_logits,
        "no-evidence": no_evidence_teacher_logits,
    }.items():
        if student_logits.shape != logits.shape:
            raise ValueError(f"student and {name} Teacher logits must have identical shapes")
    token_shape = student_logits.shape[:-1]
    if evidence_weights.shape != token_shape:
        raise ValueError("evidence_weights must match the token dimensions")
    if nuisance_projection_coefficient.shape != token_shape:
        raise ValueError("nuisance_projection_coefficient must match the token dimensions")
    support = build_truncated_topk_support(
        teacher_logits=[
            positive_teacher_logits,
            negative_teacher_logits,
            no_evidence_teacher_logits,
        ],
        sampled_token_ids=sampled_token_ids,
        top_k=top_k,
    )
    student_log_probs = _selected_log_probs(
        student_logits, support, temperature=temperature, detach=False
    )
    with torch.no_grad():
        weights = evidence_weights.detach().float()
        projection = nuisance_projection_coefficient.detach().float()
        if not bool(torch.isfinite(projection).all().item()):
            raise ValueError("nuisance_projection_coefficient must be finite")
        positive_probs = _selected_log_probs(
            positive_teacher_logits, support, temperature=temperature, detach=True
        ).exp()
        negative_probs = _selected_log_probs(
            negative_teacher_logits, support, temperature=temperature, detach=True
        ).exp()
        no_evidence_probs = _selected_log_probs(
            no_evidence_teacher_logits, support, temperature=temperature, detach=True
        ).exp()
        task = positive_probs - negative_probs
        nuisance = 0.5 * (positive_probs + negative_probs) - no_evidence_probs
        residual = task - projection.unsqueeze(-1) * nuisance
    return -float(temperature) * weights * masked_sum(
        residual * student_log_probs, support
    )
