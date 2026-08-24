"""Small pure-VERPO trainer shared by the JSONL TRL entry point.

This module intentionally keeps the data contract narrow: a row contains a
prompt, optional evidence/negative hints, a scalar correctness label, and a
completion.  Model-specific generation remains in the caller.  The trainer
only turns aligned Student/Teacher logits into GRPO and VERPO losses.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from .length_aware_reward import length_aware_outcome_rewards
from .verpo_launch_config import VERPOConfig
from .verpo_zpd import (
    compute_contrastive_teacher_token_losses,
    compute_fec_teacher_token_losses,
    compute_fixed_teacher_token_losses,
    compute_reverse_kl_ctr_teacher_token_losses,
    compute_reverse_kl_fec_teacher_token_losses,
    compute_reverse_kl_fixed_teacher_token_losses,
    compute_verpo_token_weights,
)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a finite mean over ``mask`` or a differentiable zero."""

    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    mask_f = mask.to(dtype=values.dtype)
    denominator = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denominator


def grpo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> torch.Tensor:
    """Token-level clipped GRPO/PPO objective."""

    if not (log_probs.shape == old_log_probs.shape == advantages.shape == response_mask.shape):
        raise ValueError("GRPO tensors must have identical shapes")
    ratio = (log_probs - old_log_probs.detach()).exp()
    clipped = ratio.clamp(1.0 - float(clip_low), 1.0 + float(clip_high))
    objective = torch.minimum(ratio * advantages.detach(), clipped * advantages.detach())
    return -masked_mean(objective, response_mask)


class VERPOTrainer:
    """Backend-neutral loss façade with optional Hugging Face integration.

    ``model`` and ``optimizer`` are optional so the mathematical API can be
    exercised on CPU fixtures without installing Transformers.
    """

    def __init__(self, model: Any | None = None, *, config: VERPOConfig | None = None, optimizer: Any | None = None):
        self.model = model
        self.config = config or VERPOConfig()
        self.optimizer = optimizer
        self.global_step = 0

    def compute_verpo_loss(
        self,
        student_logits: torch.Tensor,
        *,
        reference_logits: torch.Tensor,
        base_teacher_logits: torch.Tensor | None = None,
        evidence_teacher_logits: torch.Tensor | None = None,
        positive_teacher_logits: torch.Tensor | None = None,
        negative_teacher_logits: torch.Tensor | None = None,
        no_evidence_teacher_logits: torch.Tensor | None = None,
        sampled_token_ids: torch.Tensor | None = None,
        evidence_weights: torch.Tensor | None = None,
        nuisance_projection: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reference and signed VERPO terms for one aligned batch."""

        cfg = self.config
        token_shape = student_logits.shape[:-1]
        weights = evidence_weights
        if weights is None:
            weights = torch.ones(token_shape, device=student_logits.device, dtype=student_logits.dtype)
        if weights.shape != token_shape:
            raise ValueError("evidence_weights must match student token dimensions")
        mode = cfg.displacement_mode
        if mode == "evidence_vs_none":
            if base_teacher_logits is None or evidence_teacher_logits is None:
                raise ValueError("Fixed VERPO requires base and evidence Teacher logits")
            if cfg.divergence == "forward_kl":
                ref, correction, _ = compute_fixed_teacher_token_losses(
                    student_logits, reference_logits, base_teacher_logits,
                    evidence_teacher_logits, weights,
                    temperature=cfg.temperature, vocab_chunk_size=cfg.vocab_chunk_size,
                    return_interpolated_probs=False, vocab_mode=cfg.vocab_mode,
                    top_k=cfg.top_k, sampled_token_ids=sampled_token_ids,
                )
            else:
                ref, correction, _ = compute_reverse_kl_fixed_teacher_token_losses(
                    student_logits, base_teacher_logits, evidence_teacher_logits,
                    weights,
                    temperature=cfg.temperature, vocab_chunk_size=cfg.vocab_chunk_size,
                    vocab_mode=cfg.vocab_mode, top_k=cfg.top_k,
                    sampled_token_ids=sampled_token_ids,
                )
        elif mode in {"correct_vs_incorrect", "reward_ranked"}:
            if positive_teacher_logits is None or negative_teacher_logits is None:
                raise ValueError("Contrastive VERPO requires positive and negative Teacher logits")
            if cfg.divergence == "forward_kl":
                ref, correction = compute_contrastive_teacher_token_losses(
                    student_logits, reference_logits, positive_teacher_logits,
                    negative_teacher_logits, weights,
                    temperature=cfg.temperature, vocab_chunk_size=cfg.vocab_chunk_size,
                    vocab_mode=cfg.vocab_mode, top_k=cfg.top_k,
                    sampled_token_ids=sampled_token_ids,
                )
            else:
                ref, correction, _ = compute_reverse_kl_ctr_teacher_token_losses(
                    student_logits, reference_logits, positive_teacher_logits,
                    negative_teacher_logits, weights,
                    temperature=cfg.temperature, vocab_chunk_size=cfg.vocab_chunk_size,
                    vocab_mode=cfg.vocab_mode, top_k=cfg.top_k,
                    sampled_token_ids=sampled_token_ids,
                )
        elif mode == "fec":
            if positive_teacher_logits is None or negative_teacher_logits is None or no_evidence_teacher_logits is None:
                raise ValueError("FEC VERPO requires positive, negative, and no-evidence Teacher logits")
            projection = nuisance_projection
            if projection is None:
                projection = torch.zeros(token_shape, device=student_logits.device, dtype=student_logits.dtype)
            if cfg.divergence == "forward_kl":
                correction = compute_fec_teacher_token_losses(
                    student_logits, positive_teacher_logits, negative_teacher_logits,
                    no_evidence_teacher_logits, weights, projection,
                    temperature=cfg.temperature, vocab_chunk_size=cfg.vocab_chunk_size,
                    vocab_mode=cfg.vocab_mode, top_k=cfg.top_k,
                    sampled_token_ids=sampled_token_ids,
                )
                ref = torch.zeros_like(correction)
            else:
                correction = compute_reverse_kl_fec_teacher_token_losses(
                    student_logits, positive_teacher_logits,
                    negative_teacher_logits, no_evidence_teacher_logits, weights,
                    projection, temperature=cfg.temperature,
                    vocab_chunk_size=cfg.vocab_chunk_size, vocab_mode=cfg.vocab_mode,
                    top_k=cfg.top_k, sampled_token_ids=sampled_token_ids,
                )
        else:
            raise ValueError(f"unsupported displacement mode: {mode}")
        reference = masked_mean(ref, torch.ones_like(ref, dtype=torch.bool))
        evidence = masked_mean(correction, torch.ones_like(correction, dtype=torch.bool))
        return cfg.lambda_ref * reference + cfg.lambda_evi * evidence, {
            "reference_loss": reference.detach(),
            "evidence_loss": evidence.detach(),
            "total_loss": (cfg.lambda_ref * reference + cfg.lambda_evi * evidence).detach(),
        }

    def compute_loss(self, model: Any, batch: Mapping[str, Any], *, return_outputs: bool = False):
        """Transformers-compatible hook for precomputed Teacher logits."""

        outputs = model(**{k: v for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}})
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        loss, metrics = self.compute_verpo_loss(
            logits,
            reference_logits=batch["reference_logits"],
            base_teacher_logits=batch.get("base_teacher_logits"),
            evidence_teacher_logits=batch.get("evidence_teacher_logits"),
            positive_teacher_logits=batch.get("positive_teacher_logits"),
            negative_teacher_logits=batch.get("negative_teacher_logits"),
            no_evidence_teacher_logits=batch.get("no_evidence_teacher_logits"),
            sampled_token_ids=batch.get("sampled_token_ids"),
            evidence_weights=batch.get("evidence_weights"),
            nuisance_projection=batch.get("nuisance_projection"),
        )
        return (loss, {**metrics, "outputs": outputs}) if return_outputs else loss

    def save_checkpoint(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {"global_step": self.global_step, "config": asdict(self.config)}
        if self.model is not None and hasattr(self.model, "state_dict"):
            payload["model"] = self.model.state_dict()
        if self.optimizer is not None and hasattr(self.optimizer, "state_dict"):
            payload["optimizer"] = self.optimizer.state_dict()
        torch.save(payload, path / "verpo_checkpoint.pt")

    def load_checkpoint(self, directory: str | Path) -> None:
        payload = torch.load(Path(directory) / "verpo_checkpoint.pt", map_location="cpu", weights_only=False)
        if self.model is not None and "model" in payload:
            self.model.load_state_dict(payload["model"])
        if self.optimizer is not None and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))


__all__ = ["VERPOTrainer", "grpo_policy_loss", "masked_mean"]
