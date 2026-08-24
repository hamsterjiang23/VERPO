"""EMA Teacher state and diagnostics for the V24 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class EMATeacherMetadata:
    ema_decay: float
    ema_update_count: int
    student_initial_fingerprint: str
    ema_teacher_enabled: bool = True
    ema_update_after_optimizer_step: bool = True
    ema_initialization: str = "student_initial_weights"
    format_version: str = "ema_teacher_checkpoint_v2"


def update_ema_shadow(
    shadow: dict[str, torch.Tensor],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    decay: float,
) -> None:
    if not 0.0 <= float(decay) <= 1.0:
        raise ValueError("EMA decay must be in [0, 1]")
    for name, parameter in named_parameters:
        if name not in shadow:
            continue
        ema_value = shadow[name]
        student_value = parameter.detach().to(
            device=ema_value.device,
            dtype=ema_value.dtype,
        )
        ema_value.mul_(float(decay)).add_(student_value, alpha=1.0 - float(decay))


def save_ema_checkpoint(
    checkpoint_dir: Path,
    parameters: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    metadata: EMATeacherMetadata,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in parameters.items()},
        checkpoint_dir / "ema_teacher.pt",
    )
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in buffers.items()},
        checkpoint_dir / "ema_teacher_buffers.pt",
    )
    (checkpoint_dir / "ema_teacher_metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_ema_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_decay: float | None = None,
    expected_update_count: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], EMATeacherMetadata]:
    parameter_path = checkpoint_dir / "ema_teacher.pt"
    buffer_path = checkpoint_dir / "ema_teacher_buffers.pt"
    metadata_path = checkpoint_dir / "ema_teacher_metadata.json"
    missing = [
        str(path.name)
        for path in (parameter_path, buffer_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing EMA teacher checkpoint state in {checkpoint_dir}: {', '.join(missing)}"
        )
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = EMATeacherMetadata(
        ema_decay=float(raw_metadata["ema_decay"]),
        ema_update_count=int(raw_metadata["ema_update_count"]),
        student_initial_fingerprint=str(raw_metadata["student_initial_fingerprint"]),
        ema_teacher_enabled=bool(raw_metadata.get("ema_teacher_enabled", True)),
        ema_update_after_optimizer_step=bool(
            raw_metadata.get("ema_update_after_optimizer_step", True)
        ),
        ema_initialization=str(
            raw_metadata.get("ema_initialization", "student_initial_weights")
        ),
        format_version=str(
            raw_metadata.get("format_version", "ema_teacher_checkpoint_v2")
        ),
    )
    if expected_decay is not None and not math.isclose(
        metadata.ema_decay,
        float(expected_decay),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "EMA teacher decay mismatch: "
            f"checkpoint={metadata.ema_decay}, expected={expected_decay}"
        )
    if (
        expected_update_count is not None
        and metadata.ema_update_count != int(expected_update_count)
    ):
        raise ValueError(
            "EMA teacher update count mismatch: "
            f"checkpoint={metadata.ema_update_count}, expected={expected_update_count}"
        )
    parameters = torch.load(parameter_path, map_location="cpu", weights_only=True)
    buffers = torch.load(buffer_path, map_location="cpu", weights_only=True)
    if not isinstance(parameters, dict) or not isinstance(buffers, dict):
        raise ValueError(f"invalid EMA teacher tensor state in {checkpoint_dir}")
    return (
        {str(name): tensor.detach().float().clone() for name, tensor in parameters.items()},
        {str(name): tensor.detach().clone() for name, tensor in buffers.items()},
        metadata,
    )


def compute_ema_token_diagnostics(
    *,
    student_logits: torch.Tensor,
    teacher_no_evidence_logits: torch.Tensor,
    teacher_evidence_logits: torch.Tensor,
    top_k: int,
) -> dict[str, torch.Tensor]:
    if not (
        student_logits.shape
        == teacher_no_evidence_logits.shape
        == teacher_evidence_logits.shape
    ):
        raise ValueError("Student, q0, and qe logits must have identical shapes")
    if student_logits.ndim != 3:
        raise ValueError("EMA token diagnostics expect [batch, tokens, vocabulary] logits")
    selected_k = min(max(int(top_k), 1), int(student_logits.shape[-1]))
    topk_ids = torch.topk(
        teacher_evidence_logits.detach(),
        k=selected_k,
        dim=-1,
    ).indices

    def projected(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = torch.gather(logits.detach(), dim=-1, index=topk_ids).float()
        log_probs = torch.log_softmax(gathered, dim=-1)
        return log_probs, log_probs.exp()

    student_log_probs, student_probs = projected(student_logits)
    q0_log_probs, q0_probs = projected(teacher_no_evidence_logits)
    qe_log_probs, qe_probs = projected(teacher_evidence_logits)

    anchor = q0_probs - student_probs
    evidence = qe_probs - q0_probs
    total = qe_probs - student_probs
    decomposition_error = (total - anchor - evidence).abs()

    def forward_kl(q_probs: torch.Tensor, q_log_probs: torch.Tensor, p_log_probs: torch.Tensor) -> torch.Tensor:
        return (q_probs * (q_log_probs - p_log_probs)).sum(dim=-1).clamp_min(0.0)

    def jsd(
        left_probs: torch.Tensor,
        left_log_probs: torch.Tensor,
        right_probs: torch.Tensor,
        right_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        mixture = 0.5 * (left_probs + right_probs)
        mixture_log = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
        left_kl = (left_probs * (left_log_probs - mixture_log)).sum(dim=-1)
        right_kl = (right_probs * (right_log_probs - mixture_log)).sum(dim=-1)
        return (0.5 * (left_kl + right_kl)).clamp_min(0.0)

    def vector_norm(vector: torch.Tensor) -> torch.Tensor:
        return vector.square().sum(dim=-1).sqrt()

    def cosine(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left_norm = vector_norm(left)
        right_norm = vector_norm(right)
        valid = (left_norm > 0) & (right_norm > 0)
        denominator = (left_norm * right_norm).clamp_min(
            torch.finfo(left.dtype).tiny
        )
        value = (left * right).sum(dim=-1) / denominator
        value = torch.where(valid, value, torch.zeros_like(value))
        return value, valid

    cosine_anchor_evidence, cosine_anchor_evidence_valid = cosine(anchor, evidence)
    cosine_anchor_total, cosine_anchor_total_valid = cosine(anchor, total)
    cosine_evidence_total, cosine_evidence_total_valid = cosine(evidence, total)

    return {
        "qe_student_fkl": forward_kl(qe_probs, qe_log_probs, student_log_probs),
        "qe_student_l1": anchor.new_tensor(0.0) + total.abs().sum(dim=-1),
        "qe_student_l2": vector_norm(total),
        "q0_student_fkl": forward_kl(q0_probs, q0_log_probs, student_log_probs),
        "q0_student_l1": anchor.abs().sum(dim=-1),
        "q0_student_l2": vector_norm(anchor),
        "qe_q0_jsd": jsd(qe_probs, qe_log_probs, q0_probs, q0_log_probs),
        "qe_q0_l1": evidence.abs().sum(dim=-1),
        "qe_q0_l2": vector_norm(evidence),
        "cosine_anchor_evidence": cosine_anchor_evidence,
        "cosine_anchor_evidence_valid": cosine_anchor_evidence_valid,
        "cosine_anchor_total": cosine_anchor_total,
        "cosine_anchor_total_valid": cosine_anchor_total_valid,
        "cosine_evidence_total": cosine_evidence_total,
        "cosine_evidence_total_valid": cosine_evidence_total_valid,
        "decomposition_max_abs_error": decomposition_error.max(dim=-1).values,
        "decomposition_mean_abs_error": decomposition_error.mean(dim=-1),
    }


def aggregate_ema_token_diagnostics(
    diagnostics: dict[str, torch.Tensor],
    *,
    all_valid_mask: torch.Tensor,
    routed_mask: torch.Tensor,
    selected_mask: torch.Tensor,
) -> dict[str, float]:
    all_valid_mask = all_valid_mask.to(dtype=torch.bool)
    routed_mask = routed_mask.to(dtype=torch.bool) & all_valid_mask
    selected_mask = selected_mask.to(dtype=torch.bool) & routed_mask
    outside_mask = routed_mask & ~selected_mask

    result: dict[str, float] = {}

    def rate(numerator_mask: torch.Tensor, denominator_mask: torch.Tensor) -> float:
        denominator = float(denominator_mask.sum().item())
        return float(numerator_mask.sum().item()) / denominator if denominator else 0.0

    result["route_token_rate"] = rate(routed_mask, all_valid_mask)
    result["selected_token_coverage_within_routed"] = rate(selected_mask, routed_mask)
    result["total_selected_token_coverage"] = rate(selected_mask, all_valid_mask)
    result["all_valid_token_count"] = float(all_valid_mask.sum().item())
    result["routed_token_count"] = float(routed_mask.sum().item())
    result["selected_token_count"] = float(selected_mask.sum().item())

    metric_names = (
        "qe_student_fkl",
        "qe_student_l1",
        "qe_student_l2",
        "q0_student_fkl",
        "q0_student_l1",
        "q0_student_l2",
        "qe_q0_jsd",
        "qe_q0_l1",
        "qe_q0_l2",
        "cosine_anchor_evidence",
        "cosine_anchor_total",
        "cosine_evidence_total",
        "decomposition_mean_abs_error",
    )
    for name in metric_names:
        values = diagnostics[name].detach().float()
        valid_mask = selected_mask & torch.isfinite(values)
        validity_key = f"{name}_valid"
        if validity_key in diagnostics:
            valid_mask &= diagnostics[validity_key].to(dtype=torch.bool)
        numerator = float(values.masked_select(valid_mask).sum().item())
        denominator = float(valid_mask.sum().item())
        prefix = "selected_jsd" if name == "qe_q0_jsd" else f"selected_{name}"
        result[f"{prefix}_numerator"] = numerator
        result[f"{prefix}_denominator"] = denominator
        result[prefix if name != "qe_q0_jsd" else "selected_jsd_mean"] = (
            numerator / denominator if denominator else 0.0
        )
        if name.startswith("cosine_"):
            result[f"{name}_numerator"] = numerator
            result[f"{name}_denominator"] = denominator
            result[name] = numerator / denominator if denominator else 0.0

    selected_jsd = diagnostics["qe_q0_jsd"].detach().float().masked_select(selected_mask)
    result["selected_jsd_p50"] = (
        float(torch.quantile(selected_jsd, 0.5).item()) if selected_jsd.numel() else 0.0
    )
    result["selected_jsd_p90"] = (
        float(torch.quantile(selected_jsd, 0.9).item()) if selected_jsd.numel() else 0.0
    )
    outside_jsd = diagnostics["qe_q0_jsd"].detach().float()
    outside_valid = outside_mask & torch.isfinite(outside_jsd)
    outside_numerator = float(outside_jsd.masked_select(outside_valid).sum().item())
    outside_denominator = float(outside_valid.sum().item())
    result["outside_jsd_numerator"] = outside_numerator
    result["outside_jsd_denominator"] = outside_denominator
    result["outside_jsd_mean"] = (
        outside_numerator / outside_denominator if outside_denominator else 0.0
    )
    result["jsd_concentration_ratio"] = (
        result["selected_jsd_mean"] / result["outside_jsd_mean"]
        if result["outside_jsd_mean"] > 0
        else 0.0
    )

    decomposition_max = diagnostics["decomposition_max_abs_error"].detach().float()
    result["decomposition_max_abs_error"] = (
        float(decomposition_max.masked_select(selected_mask).max().item())
        if bool(selected_mask.any().item())
        else 0.0
    )
    result["anchor_mass"] = (
        result["route_token_rate"]
        * result["selected_token_coverage_within_routed"]
        * result["selected_q0_student_l1"]
    )
    result["evidence_mass"] = (
        result["route_token_rate"]
        * result["selected_token_coverage_within_routed"]
        * result["selected_qe_q0_l1"]
    )
    result["total_gradient_proxy_mass"] = (
        result["route_token_rate"]
        * result["selected_token_coverage_within_routed"]
        * result["selected_qe_student_l1"]
    )
    return result


def compute_ema_parameter_drift(
    shadow: dict[str, torch.Tensor],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, float]:
    difference_squared = 0.0
    student_squared = 0.0
    ema_squared = 0.0
    dot = 0.0
    matched = 0
    for name, parameter in named_parameters:
        if name not in shadow:
            continue
        ema_value = shadow[name].detach().float()
        student_value = parameter.detach().to(
            device=ema_value.device,
            dtype=torch.float32,
        )
        difference_squared += float((ema_value - student_value).square().sum().item())
        student_squared += float(student_value.square().sum().item())
        ema_squared += float(ema_value.square().sum().item())
        dot += float((ema_value * student_value).sum().item())
        matched += 1
    l2 = math.sqrt(max(difference_squared, 0.0))
    student_l2 = math.sqrt(max(student_squared, 0.0))
    ema_l2 = math.sqrt(max(ema_squared, 0.0))
    cosine = dot / (student_l2 * ema_l2) if student_l2 > 0 and ema_l2 > 0 else 0.0
    return {
        "ema_student_parameter_l2": l2,
        "ema_student_parameter_relative_l2": l2 / student_l2 if student_l2 > 0 else 0.0,
        "ema_student_parameter_cosine": cosine,
        "ema_student_parameter_matched_count": float(matched),
    }
