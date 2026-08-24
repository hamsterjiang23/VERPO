"""Resolve portable VERPO semantic YAML into backend projections."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "verpo"
ALLOWED_SECTIONS = frozenset({"identity", "execution", "model", "finetuning", "hardware", "protocol", "teacher", "method", "zpd", "reward", "audit", "runtime"})


class VerpoLaunchConfigError(ValueError):
    """Raised when a semantic configuration cannot be launched safely."""


@dataclass(frozen=True)
class VERPOConfig:
    """Pure VERPO controls shared by TRL and native veRL."""

    divergence: str = "forward_kl"
    displacement_mode: str = "evidence_vs_none"
    lambda_ref: float = 0.1
    lambda_evi: float = 0.5
    teacher_mode: str = "fixed_initial"
    teacher_sync_interval: int = 10
    teacher_ema_decay: float = 0.95
    temperature: float = 1.0
    vocab_mode: str = "full"
    top_k: int = 128
    vocab_chunk_size: int = 4096
    cost_alpha: float = 0.0025
    cost_epsilon: float = 2.5e-5
    group_zpd_enabled: bool = False
    group_zpd_mode: str = "reward_ranked"
    sibling_selection_mode: str = "correctness"
    evidence_rollout_scope: str = "all"
    allow_negative_benefit: bool = False

    def __post_init__(self) -> None:
        if self.divergence not in {"forward_kl", "reverse_kl"}:
            raise ValueError("divergence must be forward_kl or reverse_kl")
        if self.displacement_mode not in {"evidence_vs_none", "correct_vs_incorrect", "reward_ranked", "fec"}:
            raise ValueError("unsupported displacement_mode")
        if self.teacher_mode not in {"fixed_initial", "snapshot", "ema"}:
            raise ValueError("teacher_mode must be fixed_initial, snapshot, or ema")
        if int(self.teacher_sync_interval) <= 0 or not 0 < float(self.teacher_ema_decay) < 1:
            raise ValueError("invalid Teacher synchronization settings")
        if self.vocab_mode not in {"full", "topk_truncated"} or int(self.top_k) <= 0:
            raise ValueError("invalid vocabulary settings")
        if float(self.lambda_ref) < 0 or float(self.lambda_evi) < 0:
            raise ValueError("VERPO coefficients must be nonnegative")
        if float(self.cost_alpha) < 0 or float(self.cost_epsilon) <= 0:
            raise ValueError("cost_alpha must be nonnegative and cost_epsilon positive")


@dataclass(frozen=True)
class ConfigSource:
    kind: str
    identifier: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ResolvedVerpoLaunch:
    config: dict[str, Any]
    config_hash: str
    sources: tuple[ConfigSource, ...] = ()
    cli_overrides: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": 2, "config_hash": self.config_hash, "config": self.config, "sources": [asdict(source) for source in self.sources], "cli_overrides": list(self.cli_overrides)}


@dataclass(frozen=True)
class LaunchMatrixCell:
    identifier: str
    protocol: str | None
    teacher: str | None
    arm: str
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchMatrixSpec:
    identifier: str
    cells: tuple[LaunchMatrixCell, ...]
    overrides: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _set_dotted(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise VerpoLaunchConfigError(f"override must be PATH=VALUE: {expression}")
    path, raw = expression.split("=", 1)
    if not path or path.lower().startswith(("p" + "gr.", "op" + "sd.")):
        raise VerpoLaunchConfigError("removed legacy fields are not accepted")
    value = yaml.safe_load(raw)
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _load_overlay(kind: str, identifier: str, *, config_root: Path) -> tuple[dict[str, Any], ConfigSource]:
    directory = {"finetuning": "finetuning", "hardware": "hardware"}.get(kind, f"{kind}s")
    path = config_root / directory / f"{identifier}.yaml"
    if not path.is_file():
        raise VerpoLaunchConfigError(f"unknown {kind} config: {identifier}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if document.get("kind") != kind or document.get("id") != identifier:
        raise VerpoLaunchConfigError(f"{path} has inconsistent kind/id metadata")
    overlay = document.get("config", {})
    if not isinstance(overlay, dict):
        raise VerpoLaunchConfigError(f"{path}: config must be a mapping")
    if any(str(key).lower() in {"p" + "gr", "op" + "sd"} for key in overlay):
        raise VerpoLaunchConfigError(f"removed legacy section present in {path}")
    return overlay, ConfigSource(kind, identifier, str(path.relative_to(ROOT)), _sha256(path))


def _defaults(config_root: Path) -> tuple[dict[str, Any], ConfigSource]:
    path = config_root / "base.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = document.get("config", {})
    if not isinstance(config, dict):
        raise VerpoLaunchConfigError("base config must contain a mapping")
    return copy.deepcopy(config), ConfigSource("base", str(document.get("id", "base")), str(path.relative_to(ROOT)), _sha256(path))


def _validate(config: dict[str, Any]) -> None:
    unknown = set(config) - ALLOWED_SECTIONS
    if unknown:
        raise VerpoLaunchConfigError(f"unknown semantic sections: {sorted(unknown)}")
    for name in ("identity", "model", "finetuning", "hardware", "protocol", "teacher", "method", "zpd", "reward", "audit"):
        if not isinstance(config.get(name), dict):
            raise VerpoLaunchConfigError(f"missing semantic section: {name}")
    identity = config["identity"]
    for key in ("model", "finetuning", "hardware", "protocol", "teacher", "arm"):
        if not identity.get(key):
            raise VerpoLaunchConfigError(f"identity.{key} is required")
    method = config["method"]
    objective = str(method.get("objective", "verpo"))
    if objective not in {"verpo", "grpo", "sdpo", "srpo"}:
        raise VerpoLaunchConfigError(f"unsupported method.objective: {objective}")
    if objective == "verpo":
        VERPOConfig(
            divergence=str(method.get("divergence", "forward_kl")),
            displacement_mode=str(method.get("displacement", "fec")),
            lambda_ref=float(method.get("lambda_ref", 0.1)),
            lambda_evi=float(method.get("lambda_evi", 0.5)),
            teacher_mode=str(config["teacher"].get("mode", "fixed_initial")),
            teacher_sync_interval=int(config["teacher"].get("sync_interval", 10)),
            teacher_ema_decay=float(config["teacher"].get("ema_decay", 0.95)),
            temperature=float(method.get("temperature", 1.0)),
            vocab_mode=str(method.get("vocab_mode", "full")),
            top_k=int(method.get("top_k", 128)),
            vocab_chunk_size=int(method.get("vocab_chunk_size", 4096)),
            cost_alpha=float(method.get("cost_alpha", 0.0025)),
            cost_epsilon=float(method.get("cost_epsilon", 2.5e-5)),
            group_zpd_enabled=bool(config["zpd"].get("group_enabled", False)),
            group_zpd_mode=str(config["zpd"].get("group_mode", "reward_ranked")),
            sibling_selection_mode=str(config["zpd"].get("sibling_selection", "correctness")),
            evidence_rollout_scope=str(config["zpd"].get("evidence_scope", "all")),
        )
    protocol = config["protocol"]
    hardware = config["hardware"]
    if int(protocol.get("global_prompt_batch", 1)) <= 0 or int(protocol.get("rollout_n", 1)) <= 0:
        raise VerpoLaunchConfigError("protocol batch and rollout_n must be positive")
    if int(hardware.get("gpu_count", 1)) <= 0 or int(protocol["global_prompt_batch"]) % int(hardware["gpu_count"]):
        raise VerpoLaunchConfigError("global_prompt_batch must divide evenly across GPUs")
    if str(identity["protocol"]).startswith("rlcsd"):
        raise VerpoLaunchConfigError("RLCSD is archival and cannot be launched")


def resolve_verpo_launch(*, model: str, finetuning: str = "full", hardware: str, protocol: str | None, teacher: str | None, arm: str, cli_overrides: tuple[str, ...] = (), config_root: Path = CONFIG_ROOT) -> ResolvedVerpoLaunch:
    config, base_source = _defaults(config_root)
    sources = [base_source]
    for kind, identifier in (("model", model), ("finetuning", finetuning), ("hardware", hardware), ("protocol", protocol), ("teacher", teacher), ("arm", arm)):
        if identifier is None:
            raise VerpoLaunchConfigError(f"{kind} is required")
        overlay, source = _load_overlay(kind, identifier, config_root=config_root)
        _deep_merge(config, overlay)
        sources.append(source)
    config.setdefault("identity", {}).update({"model": model, "finetuning": finetuning, "hardware": hardware, "protocol": protocol, "teacher": teacher, "arm": arm})
    for expression in cli_overrides:
        _set_dotted(config, expression)
    _validate(config)
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return ResolvedVerpoLaunch(config, hashlib.sha256(canonical.encode()).hexdigest(), tuple(sources), tuple(cli_overrides))


def load_launch_matrix_spec(matrix: str, *, config_root: Path = CONFIG_ROOT) -> LaunchMatrixSpec:
    path = config_root / "matrices" / f"{matrix}.yaml"
    if not path.is_file():
        raise VerpoLaunchConfigError(f"unknown matrix: {matrix}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if document.get("kind") != "matrix":
        raise VerpoLaunchConfigError(f"{path} is not a matrix")
    protocols = [str(value) for value in document.get("protocols", [])]
    teachers = [str(value) for value in document.get("teachers", [])]
    arms = [str(value) for value in document.get("arms", [])]
    cells = [LaunchMatrixCell(f"{protocol}__{teacher}__{arm}", protocol, teacher, arm) for protocol in protocols for teacher in teachers for arm in arms]
    if not cells:
        cells = [LaunchMatrixCell(arm, None, None, arm) for arm in arms]
    if not cells:
        raise VerpoLaunchConfigError(f"matrix {matrix} has no cells")
    return LaunchMatrixSpec(str(document.get("id", matrix)), tuple(cells), tuple(str(value) for value in document.get("overrides", [])))


def load_launch_matrix(matrix: str, *, config_root: Path = CONFIG_ROOT) -> tuple[LaunchMatrixCell, ...]:
    return load_launch_matrix_spec(matrix, config_root=config_root).cells


def native_verl_projection(resolved: ResolvedVerpoLaunch) -> dict[str, Any]:
    config = resolved.config
    method, teacher, zpd = config["method"], config["teacher"], config["zpd"]
    return {"algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False}, "actor_rollout_ref": {"actor": {"loss_mode": "verpo_zpd" if method.get("objective") == "verpo" else "ppo", "verpo": {"enabled": method.get("objective") == "verpo", "teacher_mode": teacher.get("mode"), "teacher_sync_interval": teacher.get("sync_interval", 10), "teacher_ema_decay": teacher.get("ema_decay", 0.95), "divergence": method.get("divergence"), "displacement_mode": method.get("displacement"), "lambda_ref": method.get("lambda_ref", 0.0), "lambda_evi": method.get("lambda_evi", 0.0), "vocab_mode": method.get("vocab_mode"), "top_k": method.get("top_k"), "group_zpd_enabled": zpd.get("group_enabled", False), "group_zpd_mode": zpd.get("group_mode", "reward_ranked"), "sibling_selection_mode": zpd.get("sibling_selection", "correctness"), "evidence_rollout_scope": zpd.get("evidence_scope", "all")}}}, "data": {"train_files": config["protocol"].get("train_file"), "val_files": config["protocol"].get("validation_file")}}


def trl_verpo_projection(resolved: ResolvedVerpoLaunch) -> dict[str, Any]:
    config = resolved.config
    return {"model": config["model"], "protocol": config["protocol"], "teacher": config["teacher"], "method": config["method"], "zpd": config["zpd"], "reward": config["reward"]}


def beta_engine_environment(resolved: ResolvedVerpoLaunch) -> dict[str, str]:
    config = resolved.config
    method, teacher, zpd = config["method"], config["teacher"], config["zpd"]
    values: dict[str, Any] = {"QWEN3_TRAINING_OBJECTIVE": method.get("objective", "verpo"), "VERPO_MODEL_REPO": config["model"].get("hf_id", ""), "VERPO_MODEL_REVISION": config["model"].get("revision", "main"), "VERPO_EXPERIMENT_ID": config["identity"].get("arm", "verpo"), "VERPO_DIVERGENCE": method.get("divergence", "forward_kl"), "VERPO_DISPLACEMENT_MODE": method.get("displacement", "fec"), "VERPO_LAMBDA_REF": method.get("lambda_ref", 0.0), "VERPO_LAMBDA_EVI": method.get("lambda_evi", 0.0), "VERPO_VOCAB_MODE": method.get("vocab_mode", "topk_truncated"), "VERPO_TOP_K": method.get("top_k", 128), "VERPO_TEMPERATURE": method.get("temperature", 1.0), "VERPO_TEACHER_MODE": teacher.get("mode", "fixed_initial"), "VERPO_TEACHER_SYNC_INTERVAL": teacher.get("sync_interval", 10), "VERPO_TEACHER_EMA_DECAY": teacher.get("ema_decay", 0.95), "VERPO_GROUP_ZPD_ENABLED": str(bool(zpd.get("group_enabled", False))).lower(), "VERPO_GROUP_ZPD_MODE": zpd.get("group_mode", "reward_ranked"), "VERPO_SIBLING_SELECTION_MODE": zpd.get("sibling_selection", "correctness"), "VERPO_EVIDENCE_ROLLOUT_SCOPE": zpd.get("evidence_scope", "all"), "SWANLAB_PROJECT": config["protocol"].get("swanlab_project", "verpo-zpd")}
    return {str(key): str(value) for key, value in values.items()}


__all__ = ["VERPOConfig", "VerpoLaunchConfigError", "ResolvedVerpoLaunch", "LaunchMatrixCell", "LaunchMatrixSpec", "resolve_verpo_launch", "load_launch_matrix", "load_launch_matrix_spec", "native_verl_projection", "trl_verpo_projection", "beta_engine_environment"]
