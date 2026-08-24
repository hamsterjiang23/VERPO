"""Independent paper-faithful SDPO and SRPO actor objectives.

This module intentionally does not import or call VERPO-ZPD code.  SDPO and
SRPO share only their own top-K-plus-tail JSD primitive and EMA self-Teacher.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.padding import no_padding_2_padding


def add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    """Append ``log(1-sum(top-K probabilities))`` stably."""

    log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_mass = torch.clamp(log_mass, max=-1e-7)
    tail_log_prob = torch.log(-torch.expm1(log_mass))
    return torch.cat((log_probs, tail_log_prob), dim=-1)


def topk_tail_jsd_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    top_k: int = 100,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Official SDPO JSD on Student top-K IDs plus one tail bucket."""

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must have shape [tokens, vocab]")
    if not 0 < int(top_k) < student_logits.shape[-1]:
        raise ValueError("top_k must be positive and smaller than vocabulary size")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("JSD alpha must be in (0, 1)")
    student = student_logits.float()
    teacher = teacher_logits.detach().float()
    student_log_z = torch.logsumexp(student, dim=-1, keepdim=True)
    teacher_log_z = torch.logsumexp(teacher, dim=-1, keepdim=True)
    student_top_values, student_top_ids = torch.topk(student, k=int(top_k), dim=-1)
    student_top_log_probs = student_top_values - student_log_z
    teacher_top_log_probs = (
        teacher.gather(dim=-1, index=student_top_ids) - teacher_log_z
    )
    student_distribution = add_tail_bucket(student_top_log_probs)
    teacher_distribution = add_tail_bucket(teacher_top_log_probs)
    alpha_tensor = torch.as_tensor(
        float(alpha), device=student.device, dtype=student.dtype
    )
    mixture = torch.logsumexp(
        torch.stack(
            (
                student_distribution + torch.log1p(-alpha_tensor),
                teacher_distribution + torch.log(alpha_tensor),
            )
        ),
        dim=0,
    )
    student_kl = (
        student_distribution.exp() * (student_distribution - mixture)
    ).sum(dim=-1)
    teacher_kl = (
        teacher_distribution.exp() * (teacher_distribution - mixture)
    ).sum(dim=-1)
    jsd = torch.lerp(student_kl, teacher_kl, alpha_tensor).clamp_min(0.0)
    diagnostics = {
        "student_topk_mass": student_top_log_probs.exp().sum(dim=-1),
        "teacher_selected_mass": teacher_top_log_probs.exp().sum(dim=-1),
        "student_sample_log_z": student_log_z.squeeze(-1),
        "teacher_log_z": teacher_log_z.squeeze(-1),
        "student_top_ids": student_top_ids,
    }
    return jsd, diagnostics


def exact_entropy_from_logits(
    logits: torch.Tensor, *, vocab_chunk_size: int = 4096
) -> torch.Tensor:
    """Compute full-vocabulary entropy without materializing full probabilities."""

    if logits.ndim != 2 or int(vocab_chunk_size) <= 0:
        raise ValueError("entropy logits must be [tokens, vocab] with positive chunk size")
    values = logits.detach().float()
    log_z = torch.logsumexp(values, dim=-1)
    expected_logit = torch.zeros_like(log_z)
    for start in range(0, values.shape[-1], int(vocab_chunk_size)):
        chunk = values[:, start : start + int(vocab_chunk_size)]
        probabilities = torch.exp(chunk - log_z.unsqueeze(-1))
        expected_logit.add_((probabilities * chunk).sum(dim=-1))
    return (log_z - expected_logit).clamp_min(0.0)


def entropy_weights(
    entropy: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta: float,
    dp_group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize ``exp(-beta*H)`` over valid routed tokens and DP ranks."""

    if entropy.shape != valid_mask.shape:
        raise ValueError("entropy and valid_mask must have the same shape")
    raw = torch.exp(-float(beta) * entropy.float())
    valid = valid_mask.bool()
    local = torch.stack(
        (
            raw[valid].sum() if bool(valid.any().item()) else raw.new_zeros(()),
            raw.new_tensor(float(valid.sum().item())),
        )
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            local, op=torch.distributed.ReduceOp.SUM, group=dp_group
        )
    mean = local[0] / local[1].clamp_min(1.0)
    normalized = torch.where(valid, raw / mean.clamp_min(1e-30), torch.zeros_like(raw))
    return normalized, mean


def _flatten_logits(raw_output: Any) -> torch.Tensor:
    logits = raw_output.logits if hasattr(raw_output, "logits") else raw_output["logits"]
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            return logits.reshape(-1, logits.shape[-1])
        logits = logits.squeeze(0)
    if logits.ndim != 2:
        raise ValueError(f"Expected flattened logits [N, V], got {tuple(logits.shape)}")
    return logits


def _predictor_indices(
    prompt_lengths: torch.Tensor, response_lengths: torch.Tensor
) -> torch.Tensor:
    if prompt_lengths.numel() != response_lengths.numel():
        raise ValueError("prompt and response lengths must have the same row count")
    indices: list[torch.Tensor] = []
    sequence_offset = 0
    device = prompt_lengths.device
    for prompt_length, response_length in zip(
        prompt_lengths.tolist(), response_lengths.tolist(), strict=True
    ):
        if int(prompt_length) <= 0 or int(response_length) <= 0:
            raise ValueError("paper baselines require non-empty prompts and responses")
        start = sequence_offset + int(prompt_length) - 1
        indices.append(
            torch.arange(start, start + int(response_length), device=device)
        )
        sequence_offset += int(prompt_length) + int(response_length)
    return torch.cat(indices) if indices else torch.empty(0, dtype=torch.long, device=device)


def _row_token_indices(lengths: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(dim=0)))
    pieces = [
        torch.arange(
            int(offsets[row].item()),
            int(offsets[row + 1].item()),
            device=lengths.device,
        )
        for row in row_indices.tolist()
    ]
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long, device=lengths.device)


def _nested_subset(values: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    rows = [values[int(index)] for index in row_indices.tolist()]
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _teacher_micro_batch(
    data: TensorDict,
    row_indices: torch.Tensor,
) -> TensorDict:
    teacher = TensorDict(
        {
            "input_ids": _nested_subset(data["paper_teacher_input_ids"], row_indices),
            "position_ids": _nested_subset(
                data["paper_teacher_position_ids"], row_indices
            ),
        },
        batch_size=int(row_indices.numel()),
    )
    temperature = tu.get_non_tensor_data(data=data, key="temperature", default=None)
    if temperature is None:
        raise KeyError("temperature")
    if isinstance(temperature, torch.Tensor) and temperature.ndim > 0:
        teacher["temperature"] = temperature[row_indices]
    else:
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.item()
        tu.assign_non_tensor_data(teacher, "temperature", temperature)
    for key in ("use_remove_padding", "use_dynamic_bsz", "use_fused_kernels", "pad_mode"):
        value = tu.get_non_tensor_data(data=data, key=key, default=None)
        if value is not None:
            tu.assign_non_tensor_data(teacher, key, value)
    return teacher


def _flatten_response_field(
    value: torch.Tensor,
    response_lengths: torch.Tensor,
) -> torch.Tensor:
    if value.is_nested:
        flattened = value.values()
    elif value.ndim >= 2:
        flattened = torch.cat(
            [value[index, : int(length.item())] for index, length in enumerate(response_lengths)]
        )
    else:
        flattened = value.reshape(-1)
    if flattened.shape[0] != int(response_lengths.sum().item()):
        raise ValueError("response field does not align with sampled response lengths")
    return flattened


def _paper_logits_processor(
    *,
    config: ActorConfig,
    baseline_teacher,
    student_logits: torch.Tensor,
    data: TensorDict,
) -> dict[str, torch.Tensor]:
    baseline = config.paper_baseline
    student_flat = student_logits.squeeze(0)
    prompt_lengths = data["prompts"].offsets().diff().to(student_flat.device)
    response_lengths = data["responses"].offsets().diff().to(student_flat.device)
    actor_indices = _predictor_indices(prompt_lengths, response_lengths)
    route_rows = data["paper_sdpo_route"].bool().to(student_flat.device)
    active_rows = route_rows.nonzero(as_tuple=False).reshape(-1)
    response_mask_flat = _flatten_response_field(
        data["response_mask"], response_lengths
    ).bool().to(student_flat.device)
    active_response_indices = _row_token_indices(response_lengths, active_rows)
    active_valid_mask = response_mask_flat[active_response_indices]

    output_values = {
        "paper_baseline_jsd": student_flat.new_zeros(student_flat.shape[0], dtype=torch.float32),
        "paper_baseline_raw_jsd": student_flat.new_zeros(student_flat.shape[0], dtype=torch.float32),
        "paper_baseline_teacher_entropy": student_flat.new_zeros(student_flat.shape[0], dtype=torch.float32),
        "paper_baseline_entropy_weight": student_flat.new_zeros(student_flat.shape[0], dtype=torch.float32),
    }
    dp_group = baseline_teacher.engine.get_data_parallel_group()
    # Every DP rank must enter the colocated FSDP Teacher forward once per
    # micro-batch, even when that rank has no routed rows. Forward all local
    # rows, then select only routed response tokens for the objective.
    all_rows = torch.arange(response_lengths.numel(), device=student_flat.device)
    teacher_data = _teacher_micro_batch(data, all_rows)
    teacher_inputs, _ = baseline_teacher.engine.prepare_model_inputs(
        micro_batch=teacher_data
    )
    with baseline_teacher.forward_context(), torch.no_grad():
        teacher_output = baseline_teacher.engine.module(
            **teacher_inputs, use_cache=False
        )
    teacher_flat = _flatten_logits(teacher_output)
    teacher_prompt_lengths = data["paper_teacher_prompt_lengths"].to(
        student_flat.device
    )
    teacher_indices = _predictor_indices(
        teacher_prompt_lengths, response_lengths
    )
    teacher_response_logits = teacher_flat[teacher_indices][
        active_response_indices
    ] / float(baseline.temperature)

    if active_rows.numel() == 0:
        if baseline.objective == "srpo":
            entropy_weights(
                student_flat.new_zeros(0, dtype=torch.float32),
                torch.zeros(0, dtype=torch.bool, device=student_flat.device),
                beta=baseline.entropy_beta,
                dp_group=dp_group,
            )
        return {key: value.unsqueeze(0) for key, value in output_values.items()}

    active_actor_indices = actor_indices[active_response_indices]
    student_response_logits = student_flat[active_actor_indices]
    if student_response_logits.shape != teacher_response_logits.shape:
        raise AssertionError("Student and paper self-Teacher response logits are misaligned")

    raw_jsd, diagnostics = topk_tail_jsd_from_logits(
        student_response_logits,
        teacher_response_logits,
        top_k=baseline.top_k,
        alpha=baseline.jsd_alpha,
    )
    sampled_ids = data["responses"].values().to(student_flat.device)[
        active_response_indices
    ]
    student_sample_log_probs = (
        student_response_logits.float().gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
        - diagnostics["student_sample_log_z"]
    )
    old_log_probs = _flatten_response_field(
        data["old_log_probs"], response_lengths
    ).to(student_flat.device)[active_response_indices]
    is_ratio = torch.exp(
        (student_sample_log_probs - old_log_probs).detach().clamp(-20.0, 20.0)
    ).clamp(max=float(baseline.is_clip))
    corrected_jsd = raw_jsd * is_ratio
    if "rollout_is_weights" in data:
        rollout_weights = _flatten_response_field(
            data["rollout_is_weights"], response_lengths
        ).to(student_flat.device)[active_response_indices]
        corrected_jsd = corrected_jsd * rollout_weights

    teacher_entropy = exact_entropy_from_logits(
        teacher_response_logits,
        vocab_chunk_size=4096,
    )
    if baseline.objective == "srpo":
        normalized_weights, _ = entropy_weights(
            teacher_entropy,
            active_valid_mask,
            beta=baseline.entropy_beta,
            dp_group=dp_group,
        )
    else:
        normalized_weights = torch.ones_like(teacher_entropy)
    weighted_jsd = corrected_jsd * normalized_weights

    output_values["paper_baseline_raw_jsd"][active_actor_indices] = raw_jsd
    output_values["paper_baseline_jsd"][active_actor_indices] = weighted_jsd
    output_values["paper_baseline_teacher_entropy"][active_actor_indices] = teacher_entropy
    output_values["paper_baseline_entropy_weight"][active_actor_indices] = normalized_weights
    return {key: value.unsqueeze(0) for key, value in output_values.items()}


def _padded_response(value: torch.Tensor, data: TensorDict, *, fill: float) -> torch.Tensor:
    if value.is_nested:
        return value.to_padded_tensor(fill)
    return value


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1)


def _grpo_token_losses(
    config: ActorConfig,
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    negative_approx_kl = (log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio,
        1.0 - float(config.clip_ratio_low),
        1.0 + float(config.clip_ratio_high),
    )
    clipped = torch.maximum(pg_losses1, pg_losses2)
    dual_clipped = torch.min(-advantages * float(config.clip_ratio_c), clipped)
    losses = torch.where(advantages < 0, dual_clipped, clipped)
    if rollout_is_weights is not None:
        losses = losses * rollout_is_weights
    metrics = {
        "pg_clipfrac": _masked_mean((pg_losses2 > pg_losses1).float(), response_mask),
        "ppo_kl": _masked_mean(-negative_approx_kl, response_mask),
        "pg_clipfrac_lower": _masked_mean(
            ((clipped > -advantages * float(config.clip_ratio_c)) & (advantages < 0)).float(),
            response_mask,
        ),
    }
    return losses, metrics


def paper_sdpo_srpo_loss(
    config: ActorConfig,
    baseline_teacher,
    model_output: dict | None = None,
    data: TensorDict | None = None,
    dp_group=None,
    student_logits: torch.Tensor | None = None,
    data_format: str = "thd",
):
    """Serve as both FSDP logits processor and final routed token loss."""

    del data_format
    if data is None:
        raise ValueError("paper baseline loss requires a TensorDict micro-batch")
    if student_logits is not None:
        return _paper_logits_processor(
            config=config,
            baseline_teacher=baseline_teacher,
            student_logits=student_logits,
            data=data,
        )
    if model_output is None:
        raise ValueError("paper baseline final loss requires model_output")

    response_mask = _padded_response(data["response_mask"], data, fill=0).bool()
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    old_log_prob = _padded_response(data["old_log_probs"], data, fill=0.0)
    advantages = _padded_response(data["advantages"], data, fill=0.0)
    rollout_weights = (
        _padded_response(data["rollout_is_weights"], data, fill=0.0)
        if "rollout_is_weights" in data
        else None
    )
    jsd = no_padding_2_padding(model_output["paper_baseline_jsd"], data)
    raw_jsd = no_padding_2_padding(model_output["paper_baseline_raw_jsd"], data)
    teacher_entropy = no_padding_2_padding(
        model_output["paper_baseline_teacher_entropy"], data
    )
    entropy_weight = no_padding_2_padding(
        model_output["paper_baseline_entropy_weight"], data
    )
    sdpo_rows = data["paper_sdpo_route"].bool().to(response_mask.device)
    grpo_rows = data["paper_grpo_route"].bool().to(response_mask.device)
    sdpo_mask = response_mask & sdpo_rows.unsqueeze(-1)
    grpo_mask = response_mask & grpo_rows.unsqueeze(-1)
    baseline = config.paper_baseline

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if baseline.objective == "sdpo":
        count_values = data.get("paper_sdpo_batch_num_tokens")
        if count_values is None:
            if int(data["dp_size"]) > 1:
                raise ValueError("missing global SDPO routed-token count")
            routed_tokens = int(sdpo_mask.sum().item())
        else:
            routed_tokens = int(count_values.reshape(-1)[0].item())
        if routed_tokens > 0:
            info = dict(config.global_batch_info)
            info["batch_num_tokens"] = routed_tokens
            loss = agg_loss(
                loss_mat=jsd,
                loss_mask=sdpo_mask,
                loss_agg_mode="token-mean",
                **info,
            )
        else:
            loss = jsd.sum() * 0.0
        grpo_metrics = {
            "pg_clipfrac": loss.detach() * 0.0,
            "ppo_kl": loss.detach() * 0.0,
            "pg_clipfrac_lower": loss.detach() * 0.0,
        }
    else:
        grpo_losses, grpo_metrics = _grpo_token_losses(
            config,
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            advantages=advantages,
            response_mask=grpo_mask,
            rollout_is_weights=rollout_weights,
        )
        combined = torch.where(sdpo_rows.unsqueeze(-1), jsd, grpo_losses)
        loss = agg_loss(
            loss_mat=combined,
            loss_mask=response_mask,
            loss_agg_mode="token-mean",
            **config.global_batch_info,
        )

    metric_aggregation = AggregationType.SUM
    metrics: dict[str, Any] = {
        "actor/pg_loss": Metric(value=loss, aggregation=metric_aggregation),
        "paper_baseline/sdpo_route_fraction": float(sdpo_rows.float().mean().item()),
        "paper_baseline/grpo_route_fraction": float(grpo_rows.float().mean().item()),
        "paper_baseline/jsd": float(_masked_mean(raw_jsd, sdpo_mask).detach().item()),
        "paper_baseline/teacher_entropy": float(
            _masked_mean(teacher_entropy, sdpo_mask).detach().item()
        ),
        "paper_baseline/entropy_weight": float(
            _masked_mean(entropy_weight, sdpo_mask).detach().item()
        ),
        "actor/pg_clipfrac": float(grpo_metrics["pg_clipfrac"].detach().item()),
        "actor/ppo_kl": float(grpo_metrics["ppo_kl"].detach().item()),
        "actor/pg_clipfrac_lower": float(
            grpo_metrics["pg_clipfrac_lower"].detach().item()
        ),
    }
    return loss, metrics


__all__ = [
    "add_tail_bucket",
    "entropy_weights",
    "exact_entropy_from_logits",
    "paper_sdpo_srpo_loss",
    "topk_tail_jsd_from_logits",
]
