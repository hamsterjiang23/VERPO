"""Actor-side snapshot/EMA Teacher state for VERPO training."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


class ActorSideTeacher:
    """Maintain a shard-local actor Teacher refreshed by snapshot or EMA."""

    _STATE_VERSION = 2

    def __init__(
        self,
        engine,
        *,
        mode: str = "snapshot",
        sync_interval: int = 10,
        ema_decay: float = 0.95,
    ) -> None:
        mode = str(mode).strip().lower()
        if mode not in {"snapshot", "ema"}:
            raise ValueError("actor-side Teacher mode must be 'snapshot' or 'ema'")
        if int(sync_interval) <= 0:
            raise ValueError("actor-side Teacher sync_interval must be positive")
        if not 0.0 < float(ema_decay) < 1.0:
            raise ValueError("EMA Teacher decay must be in (0, 1)")
        self.engine = engine
        self.mode = mode
        self.sync_interval = int(sync_interval)
        self.ema_decay = float(ema_decay)
        self.optimizer_update_count = 0
        self.last_sync_update = 0
        self.shadow = self._clone_trainable_parameters()
        if not self.shadow:
            raise ValueError("actor-side Teacher requires at least one trainable actor parameter")

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
        """Advance after a successful optimizer update."""
        if not update_applied:
            return False
        self.optimizer_update_count += 1
        if self.mode == "ema":
            for name, parameter in self._named_trainable_parameters():
                if name not in self.shadow:
                    raise KeyError(f"EMA Teacher is missing actor parameter {name!r}")
                self.shadow[name].mul_(self.ema_decay).add_(
                    parameter.data.detach(), alpha=1.0 - self.ema_decay
                )
            self.last_sync_update = self.optimizer_update_count
            return True
        if self.optimizer_update_count % self.sync_interval != 0:
            return False
        for name, parameter in self._named_trainable_parameters():
            if name not in self.shadow:
                    raise KeyError(f"actor-side Teacher is missing actor parameter {name!r}")
            self.shadow[name].copy_(parameter.data.detach())
        self.last_sync_update = self.optimizer_update_count
        return True

    @contextmanager
    def forward_context(self) -> Iterator[None]:
        """Temporarily replace actor shards with the current Teacher snapshot."""
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            try:
                for name, parameter in self._named_trainable_parameters():
                    if name not in self.shadow:
                        raise KeyError(f"actor-side Teacher is missing actor parameter {name!r}")
                    backup[name] = parameter.data.detach().clone()
                    teacher_value = self.shadow[name]
                    if teacher_value.device != parameter.device or teacher_value.dtype != parameter.dtype:
                        teacher_value = teacher_value.to(device=parameter.device, dtype=parameter.dtype)
                    parameter.data.copy_(teacher_value)
                yield
            finally:
                for name, parameter in self._named_trainable_parameters():
                    if name in backup:
                        parameter.data.copy_(backup[name])

    def fingerprint(self) -> str:
        """Return a deterministic sampled fingerprint without gathering full shards."""
        digest = hashlib.sha256()
        for name in sorted(self.shadow):
            tensor = self.shadow[name].detach()
            if hasattr(tensor, "to_local"):
                tensor = tensor.to_local()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            # Materialize the small sampled view on CPU before indexing.  For
            # FSDP2/DTensor shards, indexing a CUDA/DTensor flat view with a
            # generated index tensor can dispatch to the CUDA IndexKernel and
            # hit invalid local-storage bounds even when the logical shape is
            # valid.  Fingerprinting is diagnostic only, so CPU sampling is
            # both safe and negligible relative to training.
            flat = tensor.detach().float().cpu().reshape(-1)
            if flat.numel():
                sample_count = min(flat.numel(), 16)
                if sample_count == 1:
                    indices = torch.zeros(1, dtype=torch.long)
                else:
                    indices = (
                        torch.arange(sample_count, dtype=torch.long)
                        * (flat.numel() - 1)
                        // (sample_count - 1)
                    )
                digest.update(flat[indices].numpy().tobytes())
        return digest.hexdigest()

    def save(self, checkpoint_dir: str | Path) -> Path:
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        state_path = path / f"verpo_actor_teacher_world_size_{world_size}_rank_{rank}.pt"
        torch.save(
            {
                "version": self._STATE_VERSION,
                "mode": self.mode,
                "sync_interval": self.sync_interval,
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
        path = Path(checkpoint_dir)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        state_path = path / f"verpo_actor_teacher_world_size_{world_size}_rank_{rank}.pt"
        if not state_path.exists():
            raise FileNotFoundError(
                f"actor-side Teacher checkpoint is required for resume: {state_path}"
            )
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        if int(payload.get("version", -1)) != self._STATE_VERSION:
            raise ValueError(f"unsupported actor-side Teacher checkpoint version: {state_path}")
        if str(payload.get("mode", "")) != self.mode:
            raise ValueError(
                "actor-side Teacher mode mismatch: "
                f"checkpoint={payload.get('mode')}, config={self.mode}"
            )
        if int(payload.get("sync_interval", -1)) != self.sync_interval:
            raise ValueError(
                "actor-side Teacher sync interval mismatch: "
                f"checkpoint={payload.get('sync_interval')}, config={self.sync_interval}"
            )
        if float(payload.get("ema_decay", float("nan"))) != self.ema_decay:
            raise ValueError(
                "EMA Teacher decay mismatch: "
                f"checkpoint={payload.get('ema_decay')}, config={self.ema_decay}"
            )
        loaded = payload.get("shadow")
        if not isinstance(loaded, dict) or set(loaded) != set(self.shadow):
            raise ValueError("actor-side Teacher parameter names do not match the current actor")
        for name, current in self.shadow.items():
            value = loaded[name]
            if tuple(value.shape) != tuple(current.shape):
                raise ValueError(f"actor-side Teacher shape mismatch for {name!r}")
            self.shadow[name] = value.to(device=current.device, dtype=current.dtype)
        self.optimizer_update_count = int(payload["optimizer_update_count"])
        self.last_sync_update = int(payload["last_sync_update"])
        if payload.get("fingerprint") != self.fingerprint():
            raise ValueError(f"actor-side Teacher fingerprint mismatch: {state_path}")
        return state_path


# Backward-compatible import for the initial snapshot-only implementation.
SnapshotTeacher = ActorSideTeacher
