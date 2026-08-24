"""Bounded parameter-gradient diagnostics for native VERPO training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def select_bounded_tail_parameters(
    module: torch.nn.Module,
    *,
    max_parameter_elements: int,
    max_parameter_tensors: int,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Select a deterministic bounded trainable scope nearest the model output."""
    if max_parameter_elements <= 0 or max_parameter_tensors <= 0:
        raise ValueError("VERPO gradient-audit limits must be positive")
    trainable = [
        (name, parameter)
        for name, parameter in module.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    ]
    selected: list[tuple[str, torch.nn.Parameter]] = []
    selected_ids: set[int] = set()
    element_count = 0
    for name, parameter in reversed(trainable):
        if id(parameter) in selected_ids:
            continue
        local_elements = int(_local_tensor(parameter).numel())
        if local_elements <= 0 or element_count + local_elements > max_parameter_elements:
            continue
        selected.append((name, parameter))
        selected_ids.add(id(parameter))
        element_count += local_elements
        if len(selected) >= max_parameter_tensors or element_count >= max_parameter_elements:
            break
    if not selected:
        raise ValueError(
            "VERPO gradient audit could not select a trainable tensor within "
            f"max_parameter_elements={max_parameter_elements}"
        )
    return list(reversed(selected))


@dataclass
class _GradientSnapshot:
    tensors: list[torch.Tensor]


class VerpoGradientAuditor:
    """Measure scaled branch gradients on a bounded FSDP-compatible parameter scope.

    The audit performs three retained diagnostic backwards on the first microbatch
    of each armed train batch, restores a clean gradient state, and leaves the
    ordinary combined-loss backward unchanged.
    """

    def __init__(
        self,
        *,
        max_train_batches: int,
        max_parameter_elements: int,
        max_parameter_tensors: int,
        epsilon: float = 1e-12,
    ) -> None:
        if max_train_batches <= 0:
            raise ValueError("VERPO gradient_audit_max_steps must be positive when enabled")
        self.max_train_batches = int(max_train_batches)
        self.max_parameter_elements = int(max_parameter_elements)
        self.max_parameter_tensors = int(max_parameter_tensors)
        self.epsilon = float(epsilon)
        self._module: torch.nn.Module | None = None
        self._optimizer = None
        self._parameters: list[tuple[str, torch.nn.Parameter]] | None = None
        self._train_batch_index = 0
        self._armed = False

    def bind(self, module: torch.nn.Module, optimizer) -> None:
        self._module = module
        self._optimizer = optimizer

    def begin_train_batch(self) -> None:
        self._train_batch_index += 1
        self._armed = self._train_batch_index <= self.max_train_batches

    def _selected_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        if self._module is None:
            raise RuntimeError("VERPO gradient auditor is not bound to an actor module")
        if self._parameters is None:
            self._parameters = select_bounded_tail_parameters(
                self._module,
                max_parameter_elements=self.max_parameter_elements,
                max_parameter_tensors=self.max_parameter_tensors,
            )
        return self._parameters

    def _capture(self, loss: torch.Tensor) -> _GradientSnapshot:
        assert self._optimizer is not None
        self._optimizer.zero_grad(set_to_none=True)
        loss.float().backward(retain_graph=True)
        tensors = []
        for _name, parameter in self._selected_parameters():
            local_parameter = _local_tensor(parameter)
            gradient = parameter.grad
            if gradient is None:
                tensors.append(torch.zeros_like(local_parameter, dtype=torch.float32))
            else:
                tensors.append(_local_tensor(gradient).detach().float().clone())
        return _GradientSnapshot(tensors=tensors)

    @staticmethod
    def _dot(left: _GradientSnapshot, right: _GradientSnapshot) -> torch.Tensor:
        if len(left.tensors) != len(right.tensors):
            raise ValueError("VERPO gradient snapshots have different parameter scopes")
        result = left.tensors[0].new_zeros(())
        for left_tensor, right_tensor in zip(left.tensors, right.tensors, strict=True):
            result = result + (left_tensor * right_tensor).sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        return result

    def audit(
        self,
        *,
        grpo_loss: torch.Tensor,
        reference_scaled_loss: torch.Tensor,
        evidence_scaled_loss: torch.Tensor,
    ) -> dict[str, float]:
        if not self._armed:
            return {}
        if self._optimizer is None:
            raise RuntimeError("VERPO gradient auditor is not bound to an optimizer")
        self._armed = False
        try:
            grpo = self._capture(grpo_loss)
            reference = self._capture(reference_scaled_loss)
            evidence = self._capture(evidence_scaled_loss)
        finally:
            self._optimizer.zero_grad(set_to_none=True)

        gg = self._dot(grpo, grpo)
        rr = self._dot(reference, reference)
        ee = self._dot(evidence, evidence)
        gr = self._dot(grpo, reference)
        ge = self._dot(grpo, evidence)
        re = self._dot(reference, evidence)

        def norm(square: torch.Tensor) -> float:
            return float(square.clamp_min(0.0).sqrt().item())

        def cosine(dot: torch.Tensor, left_square: torch.Tensor, right_square: torch.Tensor) -> float:
            denominator = (left_square.clamp_min(0.0) * right_square.clamp_min(0.0)).sqrt()
            if float(denominator.item()) <= self.epsilon:
                return 0.0
            return float((dot / denominator).clamp(min=-1.0, max=1.0).item())

        grpo_norm = norm(gg)
        reference_norm = norm(rr)
        evidence_norm = norm(ee)
        values = {
            "verpo_grad/grpo_norm": grpo_norm,
            "verpo_grad/reference_scaled_norm": reference_norm,
            "verpo_grad/evidence_scaled_norm": evidence_norm,
            "verpo_grad/grpo_reference_cosine": cosine(gr, gg, rr),
            "verpo_grad/grpo_evidence_cosine": cosine(ge, gg, ee),
            "verpo_grad/reference_evidence_cosine": cosine(re, rr, ee),
            "verpo_grad/reference_scaled_to_grpo_ratio": reference_norm / max(grpo_norm, self.epsilon),
            "verpo_grad/evidence_scaled_to_grpo_ratio": evidence_norm / max(grpo_norm, self.epsilon),
            "verpo_grad/audit_train_batch_index": float(self._train_batch_index),
            "verpo_grad/parameter_tensor_count": float(len(self._selected_parameters())),
            "verpo_grad/parameter_element_count": float(
                sum(_local_tensor(parameter).numel() for _name, parameter in self._selected_parameters())
            ),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("VERPO gradient audit produced a non-finite metric")
        return values
