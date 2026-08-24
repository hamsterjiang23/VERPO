"""Two-level ZPD control for VERPO, following the decoupled derivation.

Group level: prompt groups are ranked by the shared length-aware outcome
reward.  Saturated all-one, all-zero, all-negative-one, and other
zero-variance groups remain closed; every group with a genuine reward spread
is admitted.

Token level: the local weight is the RL-aligned evidence benefit divided by the
local Fisher/KL policy movement cost.  Forward KL uses the probability-space
Teacher displacement ``q_e - q_0``.  The Reverse-KL ablation instead uses the
geometric Teacher path and its Student-logit tangent

    r_t = log q_e,t - log q_0,t,
    xi_t = F(p_t) r_t,

with benefit ``a_t (e_y - p_t)^T xi_t`` and cost ``xi_t^T F(p_t) xi_t``.

For the default Forward-KL objective,

    w_t = h_t / (h_t + tau * (c_t + rho)),
    h_t = [a_t * (delta_t(y_t) - E_{v~p_t}[delta_t(v)])]_+,
    c_t = c_0 + beta * Var_{v~p_t}[delta_t(v)],

where delta_t = q_e,t - q_0,t is the pure evidence displacement between the
evidence-conditioned and evidence-free Teacher, not a Teacher-Student gap.
The contrastive VERPO_CONTRASTIVE mode replaces this with delta_t = q_+,t - q_-,t, where
q_- is the probability-space mixture of target-excluded incorrect hints.  The
FEC mode additionally measures the evidence-presence nuisance

    n_t = (q_+,t + q_-,t) / 2 - q_0,t

and removes its Student-local Fisher projection from the contrastive task
direction before computing benefit, cost, and the signed correction loss.
The opt-in ``allow_negative_benefit`` ablation removes only the positive-part
operator on ``h_t`` and keeps the same rational formula without post-clamping.

All exact direction, Fisher-geometry, and norm identities in this module refer
to full-vocabulary mode.  Explicit ``topk_truncated`` mode keeps full-Softmax
normalization for selected coordinates, but when ``K < V`` its directions,
Fisher moments, and norms are selected-support approximations.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from risk_aware_opsd.length_aware_reward import classify_reward_ranked_groups

from risk_aware_opsd.verpo_topk import (
    _truncated_reverse_teacher_losses,
    _truncated_forward_ctr_losses,
    should_use_full_vocab,
    truncated_forward_displacement_stats,
    truncated_forward_fec_loss,
    truncated_forward_fec_stats,
    truncated_forward_fixed_losses,
    truncated_reverse_displacement_stats,
    truncated_reverse_ctr_losses,
    truncated_reverse_fec_loss,
    truncated_reverse_fec_stats,
    truncated_reverse_fixed_losses,
)


# A weight above this threshold is treated as materially active in coverage
# diagnostics; strict ``w > 0`` remains reported separately.
VERPO_EFFECTIVE_WEIGHT_THRESHOLD = 1.0e-3


def _should_use_full_vocab_compat(
    vocab_mode: str,
    top_k: object,
    vocab_size: int,
) -> bool:
    """Dispatch while preserving legacy ``top_k=None`` full-mode calls.

    The package-level Top-K helpers deliberately require a positive integral
    support width.  Before explicit truncated vocabulary mode existed,
    :func:`compute_evidence_displacement_stats` accepted ``None`` as a
    compatibility-only argument and still evaluated the exact full vocabulary.
    Keep that wrapper behavior without weakening the strict helper API or
    admitting ``None`` for ``topk_truncated``.
    """
    if top_k is None and vocab_mode == "full":
        return should_use_full_vocab(vocab_mode, 1, vocab_size)
    return should_use_full_vocab(vocab_mode, top_k, vocab_size)


def _chunked_vocab_logsumexp(
    logits: torch.Tensor,
    *,
    scale: float,
    vocab_chunk_size: int,
) -> torch.Tensor:
    """Compute FP32 vocabulary log-normalization without a full FP32 copy."""
    chunk_size = int(vocab_chunk_size)
    if chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    log_norm: torch.Tensor | None = None
    for start in range(0, logits.shape[-1], chunk_size):
        stop = min(start + chunk_size, logits.shape[-1])
        chunk_log_norm = torch.logsumexp(
            logits[..., start:stop].float() / float(scale),
            dim=-1,
            keepdim=True,
        )
        log_norm = (
            chunk_log_norm
            if log_norm is None
            else torch.logaddexp(log_norm, chunk_log_norm)
        )
    if log_norm is None:
        raise ValueError("vocabulary dimension must be nonempty")
    return log_norm


def marginalize_teacher_logits(
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return logits representing the probability mixture over K Teachers.

    ``teacher_logits`` has shape ``[K, n_rows, seq_len, vocab]``.  The returned
    tensor has shape ``[n_rows, seq_len, vocab]`` and satisfies

        softmax(result / T) = mean_k softmax(teacher_logits[k] / T).

    Averaging probabilities (rather than raw logits) is required for the
    contrastive negative marginal used by VERPO_CONTRASTIVE-style supervision.
    """
    if teacher_logits.ndim < 2:
        raise ValueError("teacher_logits must include a non-empty K dimension")
    if teacher_logits.shape[0] <= 0:
        raise ValueError("teacher_logits must include at least one Teacher")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    scale = float(temperature)
    with torch.no_grad():
        component_log_probs = torch.log_softmax(
            teacher_logits.float() / scale,
            dim=-1,
        )
        mixture_log_probs = torch.logsumexp(component_log_probs, dim=0) - torch.log(
            component_log_probs.new_tensor(float(teacher_logits.shape[0]))
        )
    return mixture_log_probs * scale


def compute_contrastive_teacher_token_losses(
    student_logits: torch.Tensor,
    reference_teacher_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    vocab_mode: str = "topk_truncated",
    top_k: int | None = 128,
    sampled_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reference KL and the signed positive-minus-negative correction.

    The correction is

        L_ctr,t = -T w_t sum_v (q_+,t(v) - q_-,t(v)) log p_t(v),

    In full mode, Student-logit gradient descent follows exactly
    ``w_t (q_+ - q_-)``.  When ``topk_truncated`` retains ``K < V``, the
    selected coordinates keep full-Softmax normalization but the direction is
    a selected-support approximation.  This avoids constructing
    ``q_ref + w(q_+ - q_-)``, which need not be a valid probability
    distribution.
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        if sampled_token_ids is None:
            raise ValueError(
                "sampled_token_ids is required for topk_truncated Forward-KL"
            )
        return _truncated_forward_ctr_losses(
            student_logits,
            reference_teacher_logits,
            positive_teacher_logits,
            negative_teacher_logits,
            sampled_token_ids,
            evidence_weights,
            top_k=top_k,
            temperature=temperature,
        )

    for name, logits in {
        "reference": reference_teacher_logits,
        "positive": positive_teacher_logits,
        "negative": negative_teacher_logits,
    }.items():
        if student_logits.shape != logits.shape:
            raise ValueError(
                f"student and {name} Teacher logits must have identical shapes"
            )
    if evidence_weights.shape != student_logits.shape[:-1]:
        raise ValueError("evidence_weights must match the token dimensions")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if int(vocab_chunk_size) <= 0:
        raise ValueError("vocab_chunk_size must be positive")

    scale = float(temperature)
    with torch.no_grad():
        weights = evidence_weights.detach().float()
        reference_log_norm = torch.logsumexp(
            reference_teacher_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        positive_log_norm = torch.logsumexp(
            positive_teacher_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        negative_log_norm = torch.logsumexp(
            negative_teacher_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
    student_log_norm = torch.logsumexp(
        student_logits.float() / scale,
        dim=-1,
        keepdim=True,
    )
    reference_loss = student_logits.new_zeros(
        student_logits.shape[:-1], dtype=torch.float32
    )
    contrastive_loss = reference_loss.clone()
    for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
        stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
        student_log_probs = (
            student_logits[..., start:stop].float() / scale - student_log_norm
        )
        with torch.no_grad():
            reference_log_probs = (
                reference_teacher_logits[..., start:stop].float() / scale
                - reference_log_norm
            )
            reference_probs = reference_log_probs.exp()
            positive_probs = (
                positive_teacher_logits[..., start:stop].float() / scale
                - positive_log_norm
            ).exp()
            negative_probs = (
                negative_teacher_logits[..., start:stop].float() / scale
                - negative_log_norm
            ).exp()
            displacement = positive_probs - negative_probs
        reference_loss = reference_loss + scale * (
            reference_probs * (reference_log_probs - student_log_probs)
        ).sum(dim=-1)
        contrastive_loss = contrastive_loss - scale * weights * (
            displacement * student_log_probs
        ).sum(dim=-1)
    return reference_loss, contrastive_loss


def compute_fec_teacher_token_losses(
    student_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    evidence_weights: torch.Tensor,
    nuisance_projection_coefficient: torch.Tensor,
    *,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    vocab_mode: str = "topk_truncated",
    top_k: int | None = 128,
    sampled_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the FEC signed correction using a nuisance-residual direction.

    With ``d_task = q_+ - q_-`` and
    ``d_nuis = (q_+ + q_-)/2 - q_0``, the correction direction is

        d_fec = d_task - alpha_t d_nuis,

    where ``alpha_t`` is computed by :func:`compute_fec_evidence_stats` using
    the Student-local Fisher inner product.  In full mode, gradient descent
    therefore follows exactly ``w_t d_fec`` at each token.  When
    ``topk_truncated`` retains ``K < V``, the direction and Fisher projection
    geometry are selected-support approximations under full-Softmax
    normalization.
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        if sampled_token_ids is None:
            raise ValueError(
                "sampled_token_ids is required for topk_truncated Forward-KL"
            )
        return truncated_forward_fec_loss(
            student_logits,
            positive_teacher_logits,
            negative_teacher_logits,
            no_evidence_teacher_logits,
            sampled_token_ids,
            evidence_weights,
            nuisance_projection_coefficient,
            top_k=top_k,
            temperature=temperature,
        )

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
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if int(vocab_chunk_size) <= 0:
        raise ValueError("vocab_chunk_size must be positive")

    scale = float(temperature)
    with torch.no_grad():
        weights = evidence_weights.detach().float()
        projection = nuisance_projection_coefficient.detach().float()
        if not bool(torch.isfinite(projection).all().item()):
            raise ValueError("nuisance_projection_coefficient must be finite")
        positive_log_norm = _chunked_vocab_logsumexp(
            positive_teacher_logits,
            scale=scale,
            vocab_chunk_size=vocab_chunk_size,
        )
        negative_log_norm = _chunked_vocab_logsumexp(
            negative_teacher_logits,
            scale=scale,
            vocab_chunk_size=vocab_chunk_size,
        )
        no_evidence_log_norm = _chunked_vocab_logsumexp(
            no_evidence_teacher_logits,
            scale=scale,
            vocab_chunk_size=vocab_chunk_size,
        )
    student_log_norm = _chunked_vocab_logsumexp(
        student_logits,
        scale=scale,
        vocab_chunk_size=vocab_chunk_size,
    )
    fec_loss = student_logits.new_zeros(token_shape, dtype=torch.float32)
    for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
        stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
        student_log_probs = (
            student_logits[..., start:stop].float() / scale - student_log_norm
        )
        with torch.no_grad():
            positive_probs = (
                positive_teacher_logits[..., start:stop].float() / scale
                - positive_log_norm
            ).exp()
            negative_probs = (
                negative_teacher_logits[..., start:stop].float() / scale
                - negative_log_norm
            ).exp()
            no_evidence_probs = (
                no_evidence_teacher_logits[..., start:stop].float() / scale
                - no_evidence_log_norm
            ).exp()
            task_displacement = positive_probs - negative_probs
            nuisance_displacement = (
                0.5 * (positive_probs + negative_probs) - no_evidence_probs
            )
            fec_displacement = task_displacement - projection.unsqueeze(-1) * (
                nuisance_displacement
            )
        fec_loss = fec_loss - scale * weights * (
            fec_displacement * student_log_probs
        ).sum(dim=-1)
    return fec_loss


def compute_fixed_teacher_token_losses(
    student_logits: torch.Tensor,
    reference_teacher_logits: torch.Tensor,
    base_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    return_interpolated_probs: bool = True,
    vocab_mode: str = "topk_truncated",
    top_k: int | None = 128,
    sampled_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return the Fixed Forward-KL anchor and signed displacement loss.

    The immutable reference Teacher supplies the global anchoring term, while
    the base/evidence Teacher pair supplies the signed evidence correction:

        L_ref,t = KL(q_ref,t || p_t)
        L_evi,t = -T sg[w_t] sum_v (q_e,t(v) - q_0,t(v)) log p_t(v).

    All Teacher distributions and ``evidence_weights`` are stop-gradient. A
    negative weight reverses the Fixed evidence correction without constructing
    an extrapolated probability distribution. In full mode, the resulting
    Student-logit descent direction is exactly

        (q_ref - p) + w (q_e - q_0),

    before the two independent outer coefficients are applied by the trainer.
    When ``topk_truncated`` retains ``K < V``, each selected log-probability is
    still normalized by the full Student Softmax, but both loss channels and
    their gradient directions are selected-support approximations.
    ``L_evi`` is intentionally signed; it is a correction, not a standalone
    non-negative regularizer. ``return_interpolated_probs`` is a compatibility
    diagnostic available only when every weight lies in [0, 1].
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        if sampled_token_ids is None:
            raise ValueError(
                "sampled_token_ids is required for topk_truncated Forward-KL"
            )
        if return_interpolated_probs:
            raise ValueError(
                "topk_truncated Forward-KL cannot return a full-vocabulary "
                "interpolated distribution; set return_interpolated_probs=False"
            )
        reference_loss, evidence_correction_loss = truncated_forward_fixed_losses(
            student_logits,
            reference_teacher_logits,
            base_teacher_logits,
            evidence_teacher_logits,
            sampled_token_ids,
            evidence_weights,
            top_k=top_k,
            temperature=temperature,
        )
        return reference_loss, evidence_correction_loss, None

    if student_logits.shape != reference_teacher_logits.shape:
        raise ValueError(
            "student and reference Teacher logits must have identical shapes"
        )
    if student_logits.shape != base_teacher_logits.shape:
        raise ValueError("student and base Teacher logits must have identical shapes")
    if student_logits.shape != evidence_teacher_logits.shape:
        raise ValueError(
            "student and evidence Teacher logits must have identical shapes"
        )
    if evidence_weights.shape != student_logits.shape[:-1]:
        raise ValueError("evidence_weights must match the token dimensions")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if int(vocab_chunk_size) <= 0:
        raise ValueError("vocab_chunk_size must be positive")

    scale = float(temperature)
    with torch.no_grad():
        weights = evidence_weights.detach().float()
        if not bool(torch.isfinite(weights).all().item()):
            raise ValueError("evidence_weights must be finite")
        if return_interpolated_probs and bool(
            ((weights < 0) | (weights > 1)).any().item()
        ):
            raise ValueError(
                "interpolated probability diagnostics require weights in [0, 1]"
            )
        reference_log_norm = torch.logsumexp(
            reference_teacher_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        base_log_norm = torch.logsumexp(
            base_teacher_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        evidence_log_norm = torch.logsumexp(
            evidence_teacher_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
    student_log_norm = torch.logsumexp(
        student_logits.float() / scale,
        dim=-1,
        keepdim=True,
    )

    reference_loss = student_logits.new_zeros(
        student_logits.shape[:-1],
        dtype=torch.float32,
    )
    evidence_correction_loss = reference_loss.clone()
    interpolated_chunks: list[torch.Tensor] = []
    for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
        stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
        student_log_probs = (
            student_logits[..., start:stop].float() / scale - student_log_norm
        )
        with torch.no_grad():
            reference_log_probs = (
                reference_teacher_logits[..., start:stop].float() / scale
                - reference_log_norm
            )
            base_log_probs = (
                base_teacher_logits[..., start:stop].float() / scale - base_log_norm
            )
            evidence_log_probs = (
                evidence_teacher_logits[..., start:stop].float() / scale
                - evidence_log_norm
            )
            reference_probs = reference_log_probs.exp()
            base_probs = base_log_probs.exp()
            evidence_probs = evidence_log_probs.exp()
            if return_interpolated_probs:
                interpolated_probs = (
                    1.0 - weights.unsqueeze(-1)
                ) * base_probs + weights.unsqueeze(-1) * evidence_probs
                interpolated_chunks.append(interpolated_probs)

        # In this full-vocabulary branch, multiplying by temperature cancels
        # the 1/T from the Student-logit derivative and preserves the exact
        # p-q direction for tempered targets.
        reference_loss = reference_loss + scale * (
            reference_probs * (reference_log_probs - student_log_probs)
        ).sum(dim=-1)
        evidence_correction_loss = evidence_correction_loss - scale * weights * (
            (evidence_probs - base_probs) * student_log_probs
        ).sum(dim=-1)
    full_interpolated_probs = (
        torch.cat(interpolated_chunks, dim=-1) if return_interpolated_probs else None
    )
    return reference_loss, evidence_correction_loss, full_interpolated_probs


def compute_interpolated_teacher_token_losses(
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compatibility alias for the Fixed signed-displacement implementation."""
    return compute_fixed_teacher_token_losses(*args, **kwargs)


def compute_reverse_kl_fixed_teacher_token_losses(
    student_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    return_interpolated_probs: bool = True,
    vocab_mode: str = "topk_truncated",
    top_k: int | None = 128,
    sampled_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return Reverse-KL Fixed losses with a ``q_0`` anchor and path.

    This preserves the three-return loss-wrapper contract while making the
    Fixed Teacher roles explicit: the evidence-free ``q_0`` independently
    anchors the Student and is also the base of the ``q_0 -> q_e`` correction.
    Truncated mode measures that correction relative to the ``q_0`` partition
    on the selected support.
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        if sampled_token_ids is None:
            raise ValueError(
                "sampled_token_ids is required for topk_truncated Reverse-KL Fixed"
            )
        if return_interpolated_probs:
            raise ValueError(
                "return_interpolated_probs=True is unsupported for "
                "topk_truncated Reverse-KL because the selected support is not "
                "a full normalized distribution"
            )
        reference_loss, correction = truncated_reverse_fixed_losses(
            student_logits,
            no_evidence_teacher_logits,
            evidence_teacher_logits,
            sampled_token_ids,
            evidence_weights,
            top_k=top_k,
            temperature=temperature,
        )
        return reference_loss, correction, None
    return compute_reverse_kl_teacher_token_losses(
        student_logits,
        no_evidence_teacher_logits,
        no_evidence_teacher_logits,
        evidence_teacher_logits,
        evidence_weights,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk_size,
        return_interpolated_probs=return_interpolated_probs,
        vocab_mode=vocab_mode,
        top_k=top_k,
        sampled_token_ids=sampled_token_ids,
    )


def compute_reverse_kl_ctr_teacher_token_losses(
    student_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    return_interpolated_probs: bool = True,
    vocab_mode: str = "topk_truncated",
    top_k: int | None = 128,
    sampled_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return Reverse-KL CTR losses with a ``q_0`` anchor and ``q_- -> q_+`` path.

    This preserves the three-return loss-wrapper contract.  Unlike Fixed, CTR
    keeps the no-evidence ``q_0`` reference independent from its contrastive
    geometric path, which begins at ``q_-`` and moves toward ``q_+``.
    Truncated mode measures that correction relative to the ``q_-`` partition
    on the selected support.
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        if sampled_token_ids is None:
            raise ValueError(
                "sampled_token_ids is required for topk_truncated Reverse-KL CTR"
            )
        if return_interpolated_probs:
            raise ValueError(
                "return_interpolated_probs=True is unsupported for "
                "topk_truncated Reverse-KL because the selected support is not "
                "a full normalized distribution"
            )
        reference_loss, correction = truncated_reverse_ctr_losses(
            student_logits,
            no_evidence_teacher_logits,
            positive_teacher_logits,
            negative_teacher_logits,
            sampled_token_ids,
            evidence_weights,
            top_k=top_k,
            temperature=temperature,
        )
        return reference_loss, correction, None
    return compute_reverse_kl_teacher_token_losses(
        student_logits,
        no_evidence_teacher_logits,
        negative_teacher_logits,
        positive_teacher_logits,
        evidence_weights,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk_size,
        return_interpolated_probs=return_interpolated_probs,
        vocab_mode=vocab_mode,
        top_k=top_k,
        sampled_token_ids=sampled_token_ids,
    )


def compute_reverse_kl_teacher_token_losses(
    student_logits: torch.Tensor,
    reference_teacher_logits: torch.Tensor,
    base_teacher_logits: torch.Tensor,
    evidence_teacher_logits: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    return_interpolated_probs: bool = True,
    vocab_mode: str = "topk_truncated",
    top_k: int | None = 128,
    sampled_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return Reverse-KL reference and geometric-path evidence losses.

    The immutable reference Teacher supplies ``T KL(p || q_ref)``.  Evidence
    follows the normalized geometric Teacher path

        q_w(v) = q_0(v) exp(w r(v)) / Z(w),
        r(v) = log q_e(v) - log q_0(v),

    so in full mode the signed correction is exactly

        T [KL(p || q_w) - KL(p || q_0)]
        = T [log Z(w) - w E_p[r]].

    Teacher quantities and ``evidence_weights`` are stop-gradient.  The
    vocabulary is accumulated in bounded chunks; the optional returned
    distribution is the normalized geometric interpolation ``q_w``.  Explicit
    truncated mode returns no full-distribution diagnostic and uses the
    retained-mass coordinate formula relative to the base Teacher partition
    on the selected support, as documented in :mod:`verpo_topk`.
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        if sampled_token_ids is None:
            raise ValueError(
                "sampled_token_ids is required for topk_truncated Reverse-KL"
            )
        if return_interpolated_probs:
            raise ValueError(
                "return_interpolated_probs=True is unsupported for "
                "topk_truncated Reverse-KL because the selected support is not "
                "a full normalized distribution"
            )
        reference_loss, correction = _truncated_reverse_teacher_losses(
            student_logits,
            reference_teacher_logits,
            base_teacher_logits,
            evidence_teacher_logits,
            sampled_token_ids,
            evidence_weights,
            top_k=top_k,
            temperature=temperature,
        )
        return reference_loss, correction, None

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
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if int(vocab_chunk_size) <= 0:
        raise ValueError("vocab_chunk_size must be positive")

    scale = float(temperature)
    with torch.no_grad():
        weights = evidence_weights.detach().float()
        reference_log_norm = torch.logsumexp(
            reference_teacher_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        base_log_norm = torch.logsumexp(
            base_teacher_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        evidence_log_norm = torch.logsumexp(
            evidence_teacher_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        geometric_log_norm = torch.full_like(weights, -torch.inf)
        for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
            stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
            base_log_probs = (
                base_teacher_logits[..., start:stop].float() / scale - base_log_norm
            )
            evidence_log_probs = (
                evidence_teacher_logits[..., start:stop].float() / scale
                - evidence_log_norm
            )
            geometric_log_density = base_log_probs + weights.unsqueeze(-1) * (
                evidence_log_probs - base_log_probs
            )
            geometric_log_norm = torch.logaddexp(
                geometric_log_norm,
                torch.logsumexp(geometric_log_density, dim=-1),
            )

    student_log_norm = torch.logsumexp(
        student_logits.float() / scale,
        dim=-1,
        keepdim=True,
    )
    reference_loss = student_logits.new_zeros(
        student_logits.shape[:-1], dtype=torch.float32
    )
    expected_log_ratio = reference_loss.clone()
    geometric_chunks: list[torch.Tensor] = []
    for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
        stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
        student_log_probs = (
            student_logits[..., start:stop].float() / scale - student_log_norm
        )
        student_probs = student_log_probs.exp()
        with torch.no_grad():
            reference_log_probs = (
                reference_teacher_logits[..., start:stop].float() / scale
                - reference_log_norm
            )
            base_log_probs = (
                base_teacher_logits[..., start:stop].float() / scale - base_log_norm
            )
            evidence_log_probs = (
                evidence_teacher_logits[..., start:stop].float() / scale
                - evidence_log_norm
            )
            log_ratio = evidence_log_probs - base_log_probs
            if return_interpolated_probs:
                geometric_chunks.append(
                    (
                        base_log_probs
                        + weights.unsqueeze(-1) * log_ratio
                        - geometric_log_norm.unsqueeze(-1)
                    ).exp()
                )
        reference_loss = reference_loss + scale * (
            student_probs * (student_log_probs - reference_log_probs)
        ).sum(dim=-1)
        expected_log_ratio = expected_log_ratio + (student_probs * log_ratio).sum(
            dim=-1
        )

    evidence_correction_loss = scale * (
        geometric_log_norm - weights * expected_log_ratio
    )
    full_geometric_probs = (
        torch.cat(geometric_chunks, dim=-1) if return_interpolated_probs else None
    )
    return reference_loss, evidence_correction_loss, full_geometric_probs


def compute_reverse_kl_fec_teacher_token_losses(
    student_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    evidence_weights: torch.Tensor,
    nuisance_projection_coefficient: torch.Tensor,
    *,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    vocab_mode: str = "topk_truncated",
    top_k: int | None = 128,
    sampled_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the exact Reverse-KL FEC geometric-path correction.

    RKL-FEC is defined in Teacher log-density space:

        r_task = log q_+ - log q_-
        r_nuis = (log q_+ + log q_-)/2 - log q_0
        r_fec  = r_task - alpha r_nuis.

    The correction path starts at ``q_-`` and follows
    ``q_w proportional to q_- exp(w r_fec)``.  In full mode this function
    returns the exact signed difference ``T[KL(p || q_w) - KL(p || q_-)]``.
    Explicit truncated mode is a selected-coordinate approximation with full
    Softmax normalization, the mandatory retained Student mass factor, and a
    subtraction of the base Teacher partition on the same selected support.
    Teacher quantities, ``alpha``, and ``w`` are stop-gradient; the Student
    expectation remains live.
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        if sampled_token_ids is None:
            raise ValueError(
                "sampled_token_ids is required for topk_truncated Reverse-KL FEC"
            )
        return truncated_reverse_fec_loss(
            student_logits,
            positive_teacher_logits,
            negative_teacher_logits,
            no_evidence_teacher_logits,
            sampled_token_ids,
            evidence_weights,
            nuisance_projection_coefficient,
            top_k=top_k,
            temperature=temperature,
        )

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
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if int(vocab_chunk_size) <= 0:
        raise ValueError("vocab_chunk_size must be positive")

    scale = float(temperature)
    with torch.no_grad():
        weights = evidence_weights.detach().float()
        projection = nuisance_projection_coefficient.detach().float()
        if not bool(torch.isfinite(projection).all().item()):
            raise ValueError("nuisance_projection_coefficient must be finite")
        positive_log_norm = torch.logsumexp(
            positive_teacher_logits.float() / scale, dim=-1, keepdim=True
        )
        negative_log_norm = torch.logsumexp(
            negative_teacher_logits.float() / scale, dim=-1, keepdim=True
        )
        no_evidence_log_norm = torch.logsumexp(
            no_evidence_teacher_logits.float() / scale, dim=-1, keepdim=True
        )
        geometric_log_norm = torch.full_like(weights, -torch.inf)
        for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
            stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
            positive_log_probs = (
                positive_teacher_logits[..., start:stop].float() / scale
                - positive_log_norm
            )
            negative_log_probs = (
                negative_teacher_logits[..., start:stop].float() / scale
                - negative_log_norm
            )
            no_evidence_log_probs = (
                no_evidence_teacher_logits[..., start:stop].float() / scale
                - no_evidence_log_norm
            )
            task_log_ratio = positive_log_probs - negative_log_probs
            nuisance_log_ratio = (
                0.5 * (positive_log_probs + negative_log_probs) - no_evidence_log_probs
            )
            fec_log_ratio = task_log_ratio - projection.unsqueeze(-1) * (
                nuisance_log_ratio
            )
            geometric_log_density = negative_log_probs + weights.unsqueeze(-1) * (
                fec_log_ratio
            )
            geometric_log_norm = torch.logaddexp(
                geometric_log_norm,
                torch.logsumexp(geometric_log_density, dim=-1),
            )

    student_log_norm = torch.logsumexp(
        student_logits.float() / scale, dim=-1, keepdim=True
    )
    expected_fec_log_ratio = student_logits.new_zeros(token_shape, dtype=torch.float32)
    for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
        stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
        student_probs = (
            student_logits[..., start:stop].float() / scale - student_log_norm
        ).exp()
        with torch.no_grad():
            positive_log_probs = (
                positive_teacher_logits[..., start:stop].float() / scale
                - positive_log_norm
            )
            negative_log_probs = (
                negative_teacher_logits[..., start:stop].float() / scale
                - negative_log_norm
            )
            no_evidence_log_probs = (
                no_evidence_teacher_logits[..., start:stop].float() / scale
                - no_evidence_log_norm
            )
            task_log_ratio = positive_log_probs - negative_log_probs
            nuisance_log_ratio = (
                0.5 * (positive_log_probs + negative_log_probs) - no_evidence_log_probs
            )
            fec_log_ratio = task_log_ratio - projection.unsqueeze(-1) * (
                nuisance_log_ratio
            )
        expected_fec_log_ratio = expected_fec_log_ratio + (
            student_probs * fec_log_ratio
        ).sum(dim=-1)

    correction = scale * (geometric_log_norm - weights * expected_fec_log_ratio)
    if not bool(torch.isfinite(correction).all().item()):
        raise ValueError("Reverse-KL FEC correction must be finite")
    return correction


def compose_dual_decoupled_verpo_loss(
    grpo_loss: torch.Tensor,
    reference_loss: torch.Tensor,
    evidence_correction_loss: torch.Tensor,
    *,
    lambda_ref: float,
    lambda_evi: float,
) -> torch.Tensor:
    """Compose GRPO, reference anchoring, and evidence correction independently."""
    if float(lambda_ref) < 0 or float(lambda_evi) < 0:
        raise ValueError("VERPO coefficients must be nonnegative")
    return (
        grpo_loss
        + float(lambda_ref) * reference_loss
        + float(lambda_evi) * evidence_correction_loss
    )


def compute_group_zpd_gate(
    raw_rewards: torch.Tensor,
    *,
    num_rollouts: int,
    mode: str = "binary_mixed",
    positive_reward: float = 1.0,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Gate rows whose prompt group has reward-ranked ZPD signal.

    All-one, all-zero, all-negative-one, and every other zero-variance group
    remain closed.  Intermediate length-aware rewards therefore admit more
    groups than the legacy binary correctness-mixed gate.

    Args:
        raw_rewards: [n_rows] pre-advantage scalar rewards
        num_rollouts: group size (rollouts per prompt)
        epsilon: numerical tolerance on the within-group variance

    Returns:
        gate: [n_rows] bool, True where the row's group is in the group-level ZPD
    """
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative")
    if mode not in {"binary_mixed", "reward_ranked"}:
        raise ValueError("group ZPD mode must be binary_mixed or reward_ranked")
    rewards = raw_rewards.detach().to(dtype=torch.float32)
    if num_rollouts <= 1 or rewards.numel() == 0:
        return torch.zeros_like(rewards, dtype=torch.bool)
    if mode == "binary_mixed":
        rewards = (rewards >= float(positive_reward) - 1e-6).float()
    return classify_reward_ranked_groups(
        rewards,
        num_rollouts=num_rollouts,
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


def compute_evidence_displacement_stats(
    teacher_evidence_logits: torch.Tensor,
    teacher_no_evidence_logits: torch.Tensor,
    student_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    top_k: int | None = 128,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    vocab_mode: str = "topk_truncated",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Benefit and cost proxies for the pure evidence displacement direction.

    Full mode accumulates the exact Fisher quadratic over the whole vocabulary
    in bounded chunks. ``top_k`` controls support only when
    ``vocab_mode='topk_truncated'``; with ``K < V``, the returned direction,
    Fisher cost, and norm are selected-support approximations under full
    Student/Teacher Softmax normalization.

    Args:
        teacher_evidence_logits: [n_rows, seq_len, vocab] evidence-conditioned Teacher
        teacher_no_evidence_logits: [n_rows, seq_len, vocab] evidence-free Teacher
        student_logits: [n_rows, seq_len, vocab] current Student
        sampled_token_ids: [n_rows, seq_len] realized tokens
        advantages: [n_rows] signed GRPO advantage, broadcast across the trajectory
        top_k: Teacher support width in explicit truncated mode; ``None`` is
            retained as a compatibility-only no-op in full mode
        temperature: Softmax temperature shared with the VERPO loss
        vocab_chunk_size: exact vocabulary accumulation chunk size

    Returns:
        benefit: [n_rows, seq_len] signed alignment before the positive part
        fisher_cost: [n_rows, seq_len] Var_{v~p_t}[delta_t(v)]
        displacement_norm: [n_rows, seq_len] full-vocabulary L2 norm in full
            mode, or selected-support L2 norm when truncated with ``K < V``
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        return truncated_forward_displacement_stats(
            teacher_evidence_logits,
            teacher_no_evidence_logits,
            student_logits,
            sampled_token_ids,
            advantages,
            top_k=top_k,
            temperature=temperature,
        )

    if teacher_evidence_logits.shape != teacher_no_evidence_logits.shape:
        raise ValueError("evidence and base Teacher logits must have identical shapes")
    if teacher_evidence_logits.shape != student_logits.shape:
        raise ValueError("Teacher and Student logits must have identical shapes")
    if sampled_token_ids.shape != student_logits.shape[:-1]:
        raise ValueError("sampled_token_ids must match the token dimensions")
    if advantages.shape != student_logits.shape[:1]:
        raise ValueError("advantages must have shape [n_rows]")
    with torch.no_grad():
        if float(temperature) <= 0:
            raise ValueError("temperature must be positive")
        if int(vocab_chunk_size) <= 0:
            raise ValueError("vocab_chunk_size must be positive")
        scale = float(temperature)
        token_ids = sampled_token_ids.unsqueeze(-1)

        evidence_log_norm = torch.logsumexp(
            teacher_evidence_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        base_log_norm = torch.logsumexp(
            teacher_no_evidence_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        student_log_norm = torch.logsumexp(
            student_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        mean_delta = student_logits.new_zeros(
            student_logits.shape[:-1],
            dtype=torch.float32,
        )
        second_moment = mean_delta.clone()
        displacement_norm_sq = mean_delta.clone()
        for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
            stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
            q_evidence = (
                teacher_evidence_logits[..., start:stop].float() / scale
                - evidence_log_norm
            ).exp()
            q_base = (
                teacher_no_evidence_logits[..., start:stop].float() / scale
                - base_log_norm
            ).exp()
            p_student = (
                student_logits[..., start:stop].float() / scale - student_log_norm
            ).exp()
            delta = q_evidence - q_base
            mean_delta = mean_delta + (p_student * delta).sum(dim=-1)
            second_moment = second_moment + (p_student * delta * delta).sum(dim=-1)
            displacement_norm_sq = displacement_norm_sq + (delta * delta).sum(dim=-1)

        # Centering removes the common logit shift the Softmax is blind to.
        fisher_cost = (second_moment - mean_delta * mean_delta).clamp(min=0.0)

        delta_at_token = (
            (
                teacher_evidence_logits.gather(-1, token_ids).float() / scale
                - evidence_log_norm
            ).exp()
            - (
                teacher_no_evidence_logits.gather(-1, token_ids).float() / scale
                - base_log_norm
            ).exp()
        ).squeeze(-1)
        centered = delta_at_token - mean_delta
        benefit = advantages.detach().to(dtype=centered.dtype).unsqueeze(-1) * centered

        displacement_norm = displacement_norm_sq.sqrt()
        return benefit, fisher_cost, displacement_norm


def compute_fec_evidence_stats(
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    projection_epsilon: float = 1e-8,
    vocab_mode: str = "topk_truncated",
    top_k: int | None = 128,
) -> dict[str, torch.Tensor]:
    """Return FEC benefit/cost and task-versus-nuisance diagnostics.

    FEC starts from the contrastive task direction and removes the component
    explained by merely adding evidence:

        d_task = q_+ - q_-
        d_nuis = (q_+ + q_-)/2 - q_0
        alpha = Cov_p(d_task, d_nuis) / (Var_p(d_nuis) + epsilon)
        d_fec = d_task - alpha d_nuis.

    In full mode, all inner products are centered under the complete current
    Student distribution and use the exact local Fisher geometry.  When
    truncated with ``K < V``, selected probabilities retain full-Softmax
    normalization, but the direction, Fisher geometry, and norms are
    selected-support approximations.  The returned ``benefit``,
    ``fisher_cost``, and ``displacement_norm`` are computed from ``d_fec`` and
    can directly replace the existing delta branch in the token controller.
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        return truncated_forward_fec_stats(
            positive_teacher_logits,
            negative_teacher_logits,
            no_evidence_teacher_logits,
            student_logits,
            sampled_token_ids,
            advantages,
            top_k=top_k,
            temperature=temperature,
            projection_epsilon=projection_epsilon,
        )

    teacher_shapes = {
        "negative": negative_teacher_logits.shape,
        "no-evidence": no_evidence_teacher_logits.shape,
    }
    for name, shape in teacher_shapes.items():
        if positive_teacher_logits.shape != shape:
            raise ValueError(
                f"positive and {name} Teacher logits must have identical shapes"
            )
    if positive_teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher and Student logits must have identical shapes")
    if sampled_token_ids.shape != student_logits.shape[:-1]:
        raise ValueError("sampled_token_ids must match the token dimensions")
    if advantages.shape != student_logits.shape[:1]:
        raise ValueError("advantages must have shape [n_rows]")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if int(vocab_chunk_size) <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    if not math.isfinite(float(projection_epsilon)) or float(projection_epsilon) <= 0:
        raise ValueError("projection_epsilon must be finite and positive")

    with torch.no_grad():
        scale = float(temperature)
        token_shape = student_logits.shape[:-1]
        positive_log_norm = _chunked_vocab_logsumexp(
            positive_teacher_logits,
            scale=scale,
            vocab_chunk_size=vocab_chunk_size,
        )
        negative_log_norm = _chunked_vocab_logsumexp(
            negative_teacher_logits,
            scale=scale,
            vocab_chunk_size=vocab_chunk_size,
        )
        no_evidence_log_norm = _chunked_vocab_logsumexp(
            no_evidence_teacher_logits,
            scale=scale,
            vocab_chunk_size=vocab_chunk_size,
        )
        student_log_norm = _chunked_vocab_logsumexp(
            student_logits,
            scale=scale,
            vocab_chunk_size=vocab_chunk_size,
        )
        mean_task = student_logits.new_zeros(token_shape, dtype=torch.float32)
        mean_nuisance = mean_task.clone()
        second_task = mean_task.clone()
        second_nuisance = mean_task.clone()
        cross_moment = mean_task.clone()
        task_norm_sq = mean_task.clone()
        nuisance_norm_sq = mean_task.clone()

        for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
            stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
            positive_probs = (
                positive_teacher_logits[..., start:stop].float() / scale
                - positive_log_norm
            ).exp()
            negative_probs = (
                negative_teacher_logits[..., start:stop].float() / scale
                - negative_log_norm
            ).exp()
            no_evidence_probs = (
                no_evidence_teacher_logits[..., start:stop].float() / scale
                - no_evidence_log_norm
            ).exp()
            student_probs = (
                student_logits[..., start:stop].float() / scale - student_log_norm
            ).exp()
            task = positive_probs - negative_probs
            nuisance = 0.5 * (positive_probs + negative_probs) - no_evidence_probs
            mean_task += (student_probs * task).sum(dim=-1)
            mean_nuisance += (student_probs * nuisance).sum(dim=-1)
            second_task += (student_probs * task.square()).sum(dim=-1)
            second_nuisance += (student_probs * nuisance.square()).sum(dim=-1)
            cross_moment += (student_probs * task * nuisance).sum(dim=-1)
            task_norm_sq += task.square().sum(dim=-1)
            nuisance_norm_sq += nuisance.square().sum(dim=-1)

        task_variance = (second_task - mean_task.square()).clamp(min=0.0)
        nuisance_variance = (second_nuisance - mean_nuisance.square()).clamp(min=0.0)
        covariance = cross_moment - mean_task * mean_nuisance
        epsilon = float(projection_epsilon)
        projection = covariance / (nuisance_variance + epsilon)
        residual_nuisance_fisher_covariance = (
            covariance - projection * nuisance_variance
        )
        fec_mean = mean_task - projection * mean_nuisance
        fec_fisher_cost = (
            task_variance
            - 2.0 * projection * covariance
            + projection.square() * nuisance_variance
        ).clamp(min=0.0)
        fisher_cosine = covariance / torch.sqrt(
            (task_variance + epsilon) * (nuisance_variance + epsilon)
        )
        fisher_cosine = fisher_cosine.clamp(min=-1.0, max=1.0)
        residual_nuisance_covariance = covariance - projection * nuisance_variance

        fec_norm_sq = student_logits.new_zeros(token_shape, dtype=torch.float32)
        for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
            stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
            positive_probs = (
                positive_teacher_logits[..., start:stop].float() / scale
                - positive_log_norm
            ).exp()
            negative_probs = (
                negative_teacher_logits[..., start:stop].float() / scale
                - negative_log_norm
            ).exp()
            no_evidence_probs = (
                no_evidence_teacher_logits[..., start:stop].float() / scale
                - no_evidence_log_norm
            ).exp()
            task = positive_probs - negative_probs
            nuisance = 0.5 * (positive_probs + negative_probs) - no_evidence_probs
            fec = task - projection.unsqueeze(-1) * nuisance
            fec_norm_sq += fec.square().sum(dim=-1)

        token_ids = sampled_token_ids.unsqueeze(-1)
        positive_at_token = (
            (
                positive_teacher_logits.gather(-1, token_ids).float() / scale
                - positive_log_norm
            )
            .exp()
            .squeeze(-1)
        )
        negative_at_token = (
            (
                negative_teacher_logits.gather(-1, token_ids).float() / scale
                - negative_log_norm
            )
            .exp()
            .squeeze(-1)
        )
        no_evidence_at_token = (
            (
                no_evidence_teacher_logits.gather(-1, token_ids).float() / scale
                - no_evidence_log_norm
            )
            .exp()
            .squeeze(-1)
        )
        task_at_token = positive_at_token - negative_at_token
        nuisance_at_token = (
            0.5 * (positive_at_token + negative_at_token) - no_evidence_at_token
        )
        fec_at_token = task_at_token - projection * nuisance_at_token
        centered_alignment = fec_at_token - fec_mean
        benefit = (
            advantages.detach().to(dtype=centered_alignment.dtype).unsqueeze(-1)
            * centered_alignment
        )

        outputs = {
            "benefit": benefit,
            "fisher_cost": fec_fisher_cost,
            "displacement_norm": fec_norm_sq.sqrt(),
            "task_displacement_norm": task_norm_sq.sqrt(),
            "nuisance_displacement_norm": nuisance_norm_sq.sqrt(),
            "task_nuisance_fisher_cosine": fisher_cosine,
            "nuisance_projection_coefficient": projection,
            "residual_nuisance_fisher_covariance": residual_nuisance_covariance,
            "fec_alignment": centered_alignment,
        }
        if any(
            not bool(torch.isfinite(value).all().item()) for value in outputs.values()
        ):
            raise ValueError("FEC statistics must be finite")
        return outputs


def compute_reverse_kl_evidence_stats(
    teacher_evidence_logits: torch.Tensor,
    teacher_no_evidence_logits: torch.Tensor,
    student_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    top_k: int | None = 128,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    vocab_mode: str = "topk_truncated",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the paper-exact local geometry for the Reverse-KL ablation.

    With ``r = log q_e - log q_0`` and ``xi = F(p) r``, this computes

        benefit = a (e_y - p)^T xi,
        fisher_cost = xi^T F(p) xi,
        displacement_norm = ||xi||_2.

    The full vocabulary is accumulated exactly in two bounded-memory passes.
    Explicit ``topk_truncated`` mode with ``K < V`` instead evaluates the same
    construction on the Teacher/Student/sample union while retaining full
    Softmax normalization; exact direction identities apply only in full mode.
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        return truncated_reverse_displacement_stats(
            teacher_evidence_logits,
            teacher_no_evidence_logits,
            student_logits,
            sampled_token_ids,
            advantages,
            top_k=top_k,
            temperature=temperature,
        )

    if teacher_evidence_logits.shape != teacher_no_evidence_logits.shape:
        raise ValueError("evidence and base Teacher logits must have identical shapes")
    if teacher_evidence_logits.shape != student_logits.shape:
        raise ValueError("Teacher and Student logits must have identical shapes")
    if sampled_token_ids.shape != student_logits.shape[:-1]:
        raise ValueError("sampled_token_ids must match the token dimensions")
    if advantages.shape != student_logits.shape[:1]:
        raise ValueError("advantages must have shape [n_rows]")
    with torch.no_grad():
        if float(temperature) <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and int(top_k) <= 0:
            raise ValueError("top_k must be positive when provided")
        if int(vocab_chunk_size) <= 0:
            raise ValueError("vocab_chunk_size must be positive")
        scale = float(temperature)
        token_ids = sampled_token_ids.unsqueeze(-1)
        evidence_log_norm = torch.logsumexp(
            teacher_evidence_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        base_log_norm = torch.logsumexp(
            teacher_no_evidence_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        student_log_norm = torch.logsumexp(
            student_logits.float() / scale,
            dim=-1,
            keepdim=True,
        )
        mean_log_ratio = student_logits.new_zeros(
            student_logits.shape[:-1], dtype=torch.float32
        )
        for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
            stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
            student_probs = (
                student_logits[..., start:stop].float() / scale - student_log_norm
            ).exp()
            base_log_probs = (
                teacher_no_evidence_logits[..., start:stop].float() / scale
                - base_log_norm
            )
            evidence_log_probs = (
                teacher_evidence_logits[..., start:stop].float() / scale
                - evidence_log_norm
            )
            mean_log_ratio = mean_log_ratio + (
                student_probs * (evidence_log_probs - base_log_probs)
            ).sum(dim=-1)

        mean_xi = mean_log_ratio.new_zeros(mean_log_ratio.shape)
        second_moment_xi = mean_xi.clone()
        tangent_norm_sq = mean_xi.clone()
        for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
            stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
            student_probs = (
                student_logits[..., start:stop].float() / scale - student_log_norm
            ).exp()
            base_log_probs = (
                teacher_no_evidence_logits[..., start:stop].float() / scale
                - base_log_norm
            )
            evidence_log_probs = (
                teacher_evidence_logits[..., start:stop].float() / scale
                - evidence_log_norm
            )
            log_ratio = evidence_log_probs - base_log_probs
            tangent = student_probs * (log_ratio - mean_log_ratio.unsqueeze(-1))
            mean_xi = mean_xi + (student_probs * tangent).sum(dim=-1)
            second_moment_xi = second_moment_xi + (
                student_probs * tangent * tangent
            ).sum(dim=-1)
            tangent_norm_sq = tangent_norm_sq + (tangent * tangent).sum(dim=-1)

        base_log_prob_at_token = (
            teacher_no_evidence_logits.gather(-1, token_ids).float() / scale
            - base_log_norm
        ).squeeze(-1)
        evidence_log_prob_at_token = (
            teacher_evidence_logits.gather(-1, token_ids).float() / scale
            - evidence_log_norm
        ).squeeze(-1)
        student_prob_at_token = (
            (student_logits.gather(-1, token_ids).float() / scale - student_log_norm)
            .exp()
            .squeeze(-1)
        )
        tangent_at_token = student_prob_at_token * (
            evidence_log_prob_at_token - base_log_prob_at_token - mean_log_ratio
        )
        centered_alignment = tangent_at_token - mean_xi
        benefit = (
            advantages.detach().to(dtype=centered_alignment.dtype).unsqueeze(-1)
            * centered_alignment
        )
        fisher_cost = (second_moment_xi - mean_xi * mean_xi).clamp(min=0.0)
        return benefit, fisher_cost, tangent_norm_sq.sqrt()


def compute_reverse_kl_fec_evidence_stats(
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    no_evidence_teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    projection_epsilon: float = 1e-8,
    vocab_mode: str = "topk_truncated",
    top_k: int | None = 128,
) -> dict[str, torch.Tensor]:
    """Return complete Reverse-KL FEC tangent statistics.

    Teacher task and nuisance effects are first separated in log-density space,
    mapped through ``F(p)``, and only then projected in tangent Fisher geometry.
    The returned residual tangent drives benefit, movement cost, token weight,
    and the matching geometric-path correction loss.
    """
    if not _should_use_full_vocab_compat(
        vocab_mode, top_k, student_logits.shape[-1]
    ):
        return truncated_reverse_fec_stats(
            positive_teacher_logits,
            negative_teacher_logits,
            no_evidence_teacher_logits,
            student_logits,
            sampled_token_ids,
            advantages,
            top_k=top_k,
            temperature=temperature,
            projection_epsilon=projection_epsilon,
        )

    for name, logits in {
        "negative": negative_teacher_logits,
        "no-evidence": no_evidence_teacher_logits,
    }.items():
        if positive_teacher_logits.shape != logits.shape:
            raise ValueError(
                f"positive and {name} Teacher logits must have identical shapes"
            )
    if positive_teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher and Student logits must have identical shapes")
    if sampled_token_ids.shape != student_logits.shape[:-1]:
        raise ValueError("sampled_token_ids must match the token dimensions")
    if advantages.shape != student_logits.shape[:1]:
        raise ValueError("advantages must have shape [n_rows]")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if int(vocab_chunk_size) <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    if not math.isfinite(float(projection_epsilon)) or float(projection_epsilon) <= 0:
        raise ValueError("projection_epsilon must be finite and positive")

    with torch.no_grad():
        scale = float(temperature)
        epsilon = float(projection_epsilon)
        token_shape = student_logits.shape[:-1]
        positive_log_norm = torch.logsumexp(
            positive_teacher_logits.float() / scale, dim=-1, keepdim=True
        )
        negative_log_norm = torch.logsumexp(
            negative_teacher_logits.float() / scale, dim=-1, keepdim=True
        )
        no_evidence_log_norm = torch.logsumexp(
            no_evidence_teacher_logits.float() / scale, dim=-1, keepdim=True
        )
        student_log_norm = torch.logsumexp(
            student_logits.float() / scale, dim=-1, keepdim=True
        )

        mean_task_log_ratio = student_logits.new_zeros(token_shape, dtype=torch.float32)
        mean_nuisance_log_ratio = mean_task_log_ratio.clone()
        for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
            stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
            student_probs = (
                student_logits[..., start:stop].float() / scale - student_log_norm
            ).exp()
            positive_log_probs = (
                positive_teacher_logits[..., start:stop].float() / scale
                - positive_log_norm
            )
            negative_log_probs = (
                negative_teacher_logits[..., start:stop].float() / scale
                - negative_log_norm
            )
            no_evidence_log_probs = (
                no_evidence_teacher_logits[..., start:stop].float() / scale
                - no_evidence_log_norm
            )
            task_log_ratio = positive_log_probs - negative_log_probs
            nuisance_log_ratio = (
                0.5 * (positive_log_probs + negative_log_probs) - no_evidence_log_probs
            )
            mean_task_log_ratio += (student_probs * task_log_ratio).sum(dim=-1)
            mean_nuisance_log_ratio += (student_probs * nuisance_log_ratio).sum(dim=-1)

        mean_task_tangent = mean_task_log_ratio.new_zeros(token_shape)
        mean_nuisance_tangent = mean_task_tangent.clone()
        second_task_tangent = mean_task_tangent.clone()
        second_nuisance_tangent = mean_task_tangent.clone()
        cross_tangent_moment = mean_task_tangent.clone()
        task_tangent_norm_sq = mean_task_tangent.clone()
        nuisance_tangent_norm_sq = mean_task_tangent.clone()
        task_nuisance_l2_cross = mean_task_tangent.clone()
        for start in range(0, student_logits.shape[-1], int(vocab_chunk_size)):
            stop = min(start + int(vocab_chunk_size), student_logits.shape[-1])
            student_probs = (
                student_logits[..., start:stop].float() / scale - student_log_norm
            ).exp()
            positive_log_probs = (
                positive_teacher_logits[..., start:stop].float() / scale
                - positive_log_norm
            )
            negative_log_probs = (
                negative_teacher_logits[..., start:stop].float() / scale
                - negative_log_norm
            )
            no_evidence_log_probs = (
                no_evidence_teacher_logits[..., start:stop].float() / scale
                - no_evidence_log_norm
            )
            task_log_ratio = positive_log_probs - negative_log_probs
            nuisance_log_ratio = (
                0.5 * (positive_log_probs + negative_log_probs) - no_evidence_log_probs
            )
            task_tangent = student_probs * (
                task_log_ratio - mean_task_log_ratio.unsqueeze(-1)
            )
            nuisance_tangent = student_probs * (
                nuisance_log_ratio - mean_nuisance_log_ratio.unsqueeze(-1)
            )
            mean_task_tangent += (student_probs * task_tangent).sum(dim=-1)
            mean_nuisance_tangent += (student_probs * nuisance_tangent).sum(dim=-1)
            second_task_tangent += (student_probs * task_tangent.square()).sum(dim=-1)
            second_nuisance_tangent += (student_probs * nuisance_tangent.square()).sum(
                dim=-1
            )
            cross_tangent_moment += (
                student_probs * task_tangent * nuisance_tangent
            ).sum(dim=-1)
            task_tangent_norm_sq += task_tangent.square().sum(dim=-1)
            nuisance_tangent_norm_sq += nuisance_tangent.square().sum(dim=-1)
            task_nuisance_l2_cross += (task_tangent * nuisance_tangent).sum(dim=-1)

        task_fisher = (second_task_tangent - mean_task_tangent.square()).clamp(min=0.0)
        nuisance_fisher = (
            second_nuisance_tangent - mean_nuisance_tangent.square()
        ).clamp(min=0.0)
        task_nuisance_covariance = (
            cross_tangent_moment - mean_task_tangent * mean_nuisance_tangent
        )
        projection = task_nuisance_covariance / (nuisance_fisher + epsilon)
        residual_mean = mean_task_tangent - projection * mean_nuisance_tangent
        residual_fisher = (
            task_fisher
            - 2.0 * projection * task_nuisance_covariance
            + projection.square() * nuisance_fisher
        ).clamp(min=0.0)
        residual_nuisance_covariance = (
            task_nuisance_covariance - projection * nuisance_fisher
        )
        fisher_cosine = task_nuisance_covariance / torch.sqrt(
            (task_fisher + epsilon) * (nuisance_fisher + epsilon)
        )
        fisher_cosine = fisher_cosine.clamp(min=-1.0, max=1.0)
        residual_tangent_norm_sq = (
            task_tangent_norm_sq
            - 2.0 * projection * task_nuisance_l2_cross
            + projection.square() * nuisance_tangent_norm_sq
        ).clamp(min=0.0)

        token_ids = sampled_token_ids.unsqueeze(-1)
        positive_at_token = (
            positive_teacher_logits.gather(-1, token_ids).float() / scale
            - positive_log_norm
        ).squeeze(-1)
        negative_at_token = (
            negative_teacher_logits.gather(-1, token_ids).float() / scale
            - negative_log_norm
        ).squeeze(-1)
        no_evidence_at_token = (
            no_evidence_teacher_logits.gather(-1, token_ids).float() / scale
            - no_evidence_log_norm
        ).squeeze(-1)
        student_prob_at_token = (
            (student_logits.gather(-1, token_ids).float() / scale - student_log_norm)
            .exp()
            .squeeze(-1)
        )
        task_log_ratio_at_token = positive_at_token - negative_at_token
        nuisance_log_ratio_at_token = (
            0.5 * (positive_at_token + negative_at_token) - no_evidence_at_token
        )
        task_tangent_at_token = student_prob_at_token * (
            task_log_ratio_at_token - mean_task_log_ratio
        )
        nuisance_tangent_at_token = student_prob_at_token * (
            nuisance_log_ratio_at_token - mean_nuisance_log_ratio
        )
        residual_tangent_at_token = (
            task_tangent_at_token - projection * nuisance_tangent_at_token
        )
        centered_alignment = residual_tangent_at_token - residual_mean
        benefit = (
            advantages.detach().to(dtype=centered_alignment.dtype).unsqueeze(-1)
            * centered_alignment
        )

        outputs = {
            "benefit": benefit,
            "fisher_cost": residual_fisher,
            "displacement_norm": residual_tangent_norm_sq.sqrt(),
            "task_displacement_norm": task_tangent_norm_sq.sqrt(),
            "nuisance_displacement_norm": nuisance_tangent_norm_sq.sqrt(),
            "task_nuisance_fisher_cosine": fisher_cosine,
            "nuisance_projection_coefficient": projection,
            "residual_nuisance_fisher_covariance": residual_nuisance_covariance,
            "fec_alignment": centered_alignment,
        }
        if any(
            not bool(torch.isfinite(value).all().item()) for value in outputs.values()
        ):
            raise ValueError("Reverse-KL FEC statistics must be finite")
        return outputs


def compute_verpo_token_weights(
    benefit: torch.Tensor,
    fisher_cost: torch.Tensor,
    *,
    group_gate: torch.Tensor,
    token_mask: torch.Tensor,
    cost_alpha: float | None = None,
    cost_epsilon: float | None = None,
    # Legacy controls are retained for checkpoint/test compatibility.  New
    # callers should pass cost_alpha and cost_epsilon instead; they satisfy
    # cost_alpha=tau*cost_beta and cost_epsilon=tau*(rho+cost_floor).
    tau: float | None = None,
    rho: float | None = None,
    cost_floor: float | None = None,
    cost_beta: float | None = None,
    token_weight_mode: str = "paper",
    allow_negative_benefit: bool = False,
) -> torch.Tensor:
    """Token weight from the benefit/cost ratio.

    Args:
        benefit: [n_rows, seq_len] signed alignment from the displacement stats
        fisher_cost: [n_rows, seq_len] local policy movement cost
        group_gate: [n_rows] bool group-level ZPD gate
        token_mask: [n_rows, seq_len] bool valid response tokens
        cost_alpha: effective Fisher-cost scale in the benefit units
        cost_epsilon: effective positive denominator floor in benefit units
        tau/rho/cost_floor/cost_beta: deprecated legacy parameterization,
            used only when the two effective parameters are omitted
        token_weight_mode: ``paper`` for the derived continuous controller or
            ``uniform`` for the registered T1 ablation
        allow_negative_benefit: use the raw signed benefit as ``h`` instead of
            applying ``max(benefit, 0)``. This experimental mode can produce
            negative weights or weights greater than or equal to one.

    Returns:
        weights: [n_rows, seq_len]. Default paper mode lies in [0, 1).
    """
    if benefit.shape != fisher_cost.shape:
        raise ValueError("benefit and fisher_cost must have identical shapes")
    if token_mask.shape != benefit.shape:
        raise ValueError("token_mask must match the token dimensions")
    if group_gate.shape != benefit.shape[:1]:
        raise ValueError("group_gate must have shape [n_rows]")
    if token_weight_mode not in {"paper", "uniform"}:
        raise ValueError("token_weight_mode must be paper or uniform")
    if cost_alpha is None or cost_epsilon is None:
        if None in (tau, rho, cost_floor, cost_beta):
            raise ValueError("cost_alpha and cost_epsilon are required")
        cost_alpha = float(tau) * float(cost_beta)
        cost_epsilon = float(tau) * (float(rho) + float(cost_floor))
    scalar_values = {
        "cost_alpha": float(cost_alpha),
        "cost_epsilon": float(cost_epsilon),
    }
    if any(not math.isfinite(value) for value in scalar_values.values()):
        raise ValueError("VERPO token-weight hyperparameters must be finite")
    if float(cost_alpha) < 0 or float(cost_epsilon) <= 0:
        raise ValueError("cost_alpha must be nonnegative and cost_epsilon positive")
    with torch.no_grad():
        active_mask = group_gate.unsqueeze(-1) & token_mask
        if token_weight_mode == "uniform":
            return active_mask.to(dtype=benefit.dtype)
        h = benefit if allow_negative_benefit else benefit.clamp(min=0.0)
        c = float(cost_alpha) * fisher_cost + float(cost_epsilon)
        weights = torch.where(
            active_mask,
            h / (h + c),
            torch.zeros_like(h),
        )
        if not torch.isfinite(weights).all().item():
            raise ValueError("VERPO paper token weights must be finite")
        if not allow_negative_benefit and (
            (weights < 0).any().item() or (weights >= 1).any().item()
        ):
            raise ValueError("VERPO paper token weights must lie in [0, 1)")
        return weights
