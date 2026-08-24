"""Independent EMA self-Teacher state for the paper SDPO/SRPO baselines."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


class PaperBaselineEMATeacher:
    """Maintain the paper's ``teacher=0.95*teacher+0.05*student`` state."""

    _STATE_VERSION = 1

    def __init__(self, engine, *, objective: str, ema_decay: float = 0.95) -> None:
        objective = str(objective).strip().lower()
        if objective not in {"sdpo", "srpo"}:
            raise ValueError("paper baseline Teacher objective must be sdpo or srpo")
        if not 0.0 < float(ema_decay) < 1.0:
            raise ValueError("paper baseline EMA decay must be in (0, 1)")
        self.engine = engine
        self.objective = objective
        self.ema_decay = float(ema_decay)
        self.optimizer_update_count = 0
        self.last_sync_update = 0
        self.shadow = self._clone_trainable_parameters()
        if not self.shadow:
            raise ValueError("paper baseline EMA Teacher requires trainable parameters")

    def _named_trainable_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.engine.module.named_parameters()
            if parameter.requires_grad
        ]

    def _clone_trainable_parameters(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.data.detach().clone()
            for name, parameter in self._named_trainable_parameters()
        }

    @torch.no_grad()
    def after_optimizer_step(self, *, update_applied: bool) -> bool:
        if not update_applied:
            return False
        self.optimizer_update_count += 1
        for name, parameter in self._named_trainable_parameters():
            if name not in self.shadow:
                raise KeyError(f"paper baseline EMA Teacher is missing {name!r}")
            self.shadow[name].mul_(self.ema_decay).add_(
                parameter.data.detach(), alpha=1.0 - self.ema_decay
            )
        self.last_sync_update = self.optimizer_update_count
        return True

    @contextmanager
    def forward_context(self) -> Iterator[None]:
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            try:
                for name, parameter in self._named_trainable_parameters():
                    if name not in self.shadow:
                        raise KeyError(
                            f"paper baseline EMA Teacher is missing {name!r}"
                        )
                    backup[name] = parameter.data.detach().clone()
                    value = self.shadow[name]
                    if value.device != parameter.device or value.dtype != parameter.dtype:
                        value = value.to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    parameter.data.copy_(value)
                yield
            finally:
                for name, parameter in self._named_trainable_parameters():
                    if name in backup:
                        parameter.data.copy_(backup[name])

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name in sorted(self.shadow):
            tensor = self.shadow[name].detach()
            if hasattr(tensor, "to_local"):
                tensor = tensor.to_local()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            flat = tensor.detach().float().cpu().reshape(-1)
            if flat.numel():
                sample_count = min(flat.numel(), 16)
                indices = (
                    torch.zeros(1, dtype=torch.long)
                    if sample_count == 1
                    else torch.arange(sample_count, dtype=torch.long)
                    * (flat.numel() - 1)
                    // (sample_count - 1)
                )
                digest.update(flat[indices].numpy().tobytes())
        return digest.hexdigest()

    def _state_path(self, checkpoint_dir: str | Path) -> Path:
        path = Path(checkpoint_dir)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        return path / (
            f"paper_{self.objective}_ema_teacher_world_size_{world_size}_rank_{rank}.pt"
        )

    def save(self, checkpoint_dir: str | Path) -> Path:
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        state_path = self._state_path(path)
        torch.save(
            {
                "version": self._STATE_VERSION,
                "objective": self.objective,
                "ema_decay": self.ema_decay,
                "optimizer_update_count": self.optimizer_update_count,
                "last_sync_update": self.last_sync_update,
                "shadow": self.shadow,
                "fingerprint": self.fingerprint(),
            },
            state_path,
        )
        return state_path

    def load(self, checkpoint_dir: str | Path) -> Path:
        state_path = self._state_path(checkpoint_dir)
        if not state_path.exists():
            raise FileNotFoundError(
                f"paper baseline EMA Teacher checkpoint is required: {state_path}"
            )
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        if int(payload.get("version", -1)) != self._STATE_VERSION:
            raise ValueError(f"unsupported paper EMA Teacher state: {state_path}")
        if str(payload.get("objective", "")) != self.objective:
            raise ValueError("paper EMA Teacher objective does not match the run")
        if float(payload.get("ema_decay", float("nan"))) != self.ema_decay:
            raise ValueError("paper EMA Teacher decay does not match the run")
        loaded = payload.get("shadow")
        if not isinstance(loaded, dict) or set(loaded) != set(self.shadow):
            raise ValueError("paper EMA Teacher parameter names do not match actor")
        for name, current in self.shadow.items():
            value = loaded[name]
            if tuple(value.shape) != tuple(current.shape):
                raise ValueError(f"paper EMA Teacher shape mismatch for {name!r}")
            self.shadow[name] = value.to(device=current.device, dtype=current.dtype)
        self.optimizer_update_count = int(payload["optimizer_update_count"])
        self.last_sync_update = int(payload["last_sync_update"])
        if payload.get("fingerprint") != self.fingerprint():
            raise ValueError(f"paper EMA Teacher fingerprint mismatch: {state_path}")
        return state_path


__all__ = ["PaperBaselineEMATeacher"]
