"""Resolve portable VERPO semantic YAML into backend projections."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "verpo"
ALLOWED_SECTIONS = frozenset(
    {
        "identity",
        "execution",
        "model",
        "finetuning",
        "hardware",
        "protocol",
        "teacher",
        "method",
        "zpd",
        "reward",
        "audit",
        "runtime",
    }
)


class VerpoLaunchConfigError(ValueError):
    """Raised when a semantic configuration cannot be launched safely."""


@dataclass(frozen=True)
class VERPOConfig:
    """Pure VERPO controls shared by TRL and native veRL."""

    divergence: str = "forward_kl"
    displacement_mode: str = "evidence_vs_none"
    lambda_ref: float = 0.1
    lambda_evi: float = 0.5
    evidence_source: str = "rollout_group"
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
        if self.evidence_source != "rollout_group":
            raise ValueError("public VERPO evidence_source must be rollout_group")
        if self.divergence not in {"forward_kl", "reverse_kl"}:
            raise ValueError("divergence must be forward_kl or reverse_kl")
        if self.displacement_mode not in {
            "evidence_vs_none",
            "correct_vs_incorrect",
            "reward_ranked",
            "fec",
        }:
            raise ValueError("unsupported displacement_mode")
        if self.teacher_mode not in {"fixed_initial", "snapshot", "ema"}:
            raise ValueError("teacher_mode must be fixed_initial, snapshot, or ema")
        if (
            int(self.teacher_sync_interval) <= 0
            or not 0 < float(self.teacher_ema_decay) < 1
        ):
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
        return {
            "schema_version": 2,
            "config_hash": self.config_hash,
            "config": self.config,
            "sources": [asdict(source) for source in self.sources],
            "cli_overrides": list(self.cli_overrides),
        }


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


def _load_overlay(
    kind: str, identifier: str, *, config_root: Path
) -> tuple[dict[str, Any], ConfigSource]:
    directory = {"finetuning": "finetuning", "hardware": "hardware"}.get(
        kind, f"{kind}s"
    )
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
    return overlay, ConfigSource(
        kind, identifier, str(path.relative_to(ROOT)), _sha256(path)
    )


def _defaults(config_root: Path) -> tuple[dict[str, Any], ConfigSource]:
    path = config_root / "base.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = document.get("config", {})
    if not isinstance(config, dict):
        raise VerpoLaunchConfigError("base config must contain a mapping")
    return copy.deepcopy(config), ConfigSource(
        "base",
        str(document.get("id", "base")),
        str(path.relative_to(ROOT)),
        _sha256(path),
    )


def _validate(config: dict[str, Any]) -> None:
    unknown = set(config) - ALLOWED_SECTIONS
    if unknown:
        raise VerpoLaunchConfigError(f"unknown semantic sections: {sorted(unknown)}")
    for name in (
        "identity",
        "model",
        "finetuning",
        "hardware",
        "protocol",
        "teacher",
        "method",
        "zpd",
        "reward",
        "audit",
    ):
        if not isinstance(config.get(name), dict):
            raise VerpoLaunchConfigError(f"missing semantic section: {name}")
    identity = config["identity"]
    for key in ("model", "finetuning", "hardware", "protocol", "teacher", "arm"):
        if not identity.get(key):
            raise VerpoLaunchConfigError(f"identity.{key} is required")
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise VerpoLaunchConfigError(f"{section}.{key} must be finite")
    method = config["method"]
    objective = str(method.get("objective", "verpo"))
    if objective not in {"verpo", "pure_grpo", "sdpo", "srpo"}:
        raise VerpoLaunchConfigError(f"unsupported method.objective: {objective}")
    if objective == "verpo":
        VERPOConfig(
            evidence_source=str(method.get("evidence_source", "rollout_group")),
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
            sibling_selection_mode=str(
                config["zpd"].get("sibling_selection", "correctness")
            ),
            evidence_rollout_scope=str(config["zpd"].get("evidence_scope", "all")),
        )
    if config["finetuning"]["mode"] not in {"full", "lora"}:
        raise VerpoLaunchConfigError("finetuning.mode must be full or lora")
    if method.get("evidence_source") != "rollout_group":
        raise VerpoLaunchConfigError("public launches require rollout_group evidence")
    if config["zpd"].get("sibling_selection") != "correctness":
        raise VerpoLaunchConfigError(
            "rollout_group evidence requires verifier correctness"
        )
    if config["zpd"].get("evidence_scope") not in {"all", "wrong_only"}:
        raise VerpoLaunchConfigError("invalid evidence scope")
    if (
        config["identity"]["arm"] == "fec_fkl_wrong_only"
        and config["zpd"]["evidence_scope"] != "wrong_only"
    ):
        raise VerpoLaunchConfigError("wrong-only arm must retain wrong_only scope")
    if method.get("advantage_modulation", "none") not in {"none", "multiplicative_w"}:
        raise VerpoLaunchConfigError("invalid advantage modulation")
    if (
        method.get("advantage_modulation") == "multiplicative_w"
        and method.get("displacement") != "fec"
    ):
        raise VerpoLaunchConfigError("advantage modulation requires FEC")
    if bool(config["audit"].get("gradient_enabled")):
        raise VerpoLaunchConfigError(
            "gradient audit is not a supported public launcher option"
        )
    expected_reward = {
        "correct_reward": 1.0,
        "valid_wrong_reward": 0.0,
        "format_invalid_reward": 0.0,
        "length_aware_enabled": False,
    }
    if any(
        config["reward"].get(key) != value for key, value in expected_reward.items()
    ):
        raise VerpoLaunchConfigError(
            "SDPO uses fixed format-gated binary rewards; reward overrides are unsupported"
        )
    protocol = config["protocol"]
    if float(method["temperature"]) != float(protocol["rollout_temperature"]):
        raise VerpoLaunchConfigError(
            "method.temperature must equal protocol.rollout_temperature"
        )
    for key in (
        "total_training_steps",
        "rollout_n",
        "mini_batch_trajectories",
        "max_prompt_length",
        "max_response_length",
    ):
        if int(protocol[key]) <= 0:
            raise VerpoLaunchConfigError(f"protocol.{key} must be positive")
    if int(protocol["mini_batch_trajectories"]) % int(protocol["rollout_n"]):
        raise VerpoLaunchConfigError(
            "mini_batch_trajectories must be divisible by rollout_n"
        )
    if (int(protocol["global_prompt_batch"]) * int(protocol["rollout_n"])) % int(
        protocol["mini_batch_trajectories"]
    ):
        raise VerpoLaunchConfigError(
            "trajectory batch must be divisible by mini_batch_trajectories"
        )
    if float(method.get("advantage_modulation_lambda", 0)) < 0:
        raise VerpoLaunchConfigError(
            "advantage modulation coefficient must be nonnegative"
        )
    hardware = config["hardware"]
    if (
        int(protocol.get("global_prompt_batch", 1)) <= 0
        or int(protocol.get("rollout_n", 1)) <= 0
    ):
        raise VerpoLaunchConfigError("protocol batch and rollout_n must be positive")
    if not 0 < float(hardware["vllm_gpu_memory_utilization"]) <= 0.60:
        raise VerpoLaunchConfigError(
            "colocated rollout utilization must be in (0, 0.60]"
        )
    if int(hardware.get("gpu_count", 1)) <= 0 or int(
        protocol["global_prompt_batch"]
    ) % int(hardware["gpu_count"]):
        raise VerpoLaunchConfigError(
            "global_prompt_batch must divide evenly across GPUs"
        )
    if str(identity["protocol"]).startswith("rlcsd"):
        raise VerpoLaunchConfigError("RLCSD is archival and cannot be launched")


def resolve_verpo_launch(
    *,
    model: str,
    finetuning: str = "full",
    hardware: str,
    protocol: str | None,
    teacher: str | None,
    arm: str,
    cli_overrides: tuple[str, ...] = (),
    config_root: Path = CONFIG_ROOT,
) -> ResolvedVerpoLaunch:
    config, base_source = _defaults(config_root)
    sources = [base_source]
    for kind, identifier in (
        ("model", model),
        ("finetuning", finetuning),
        ("hardware", hardware),
        ("protocol", protocol),
        ("teacher", teacher),
        ("arm", arm),
    ):
        if identifier is None:
            raise VerpoLaunchConfigError(f"{kind} is required")
        overlay, source = _load_overlay(kind, identifier, config_root=config_root)
        _deep_merge(config, overlay)
        sources.append(source)
    config.setdefault("identity", {}).update(
        {
            "model": model,
            "finetuning": finetuning,
            "hardware": hardware,
            "protocol": protocol,
            "teacher": teacher,
            "arm": arm,
        }
    )
    for expression in cli_overrides:
        path = expression.split("=", 1)[0].split(".")
        current = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                raise VerpoLaunchConfigError(
                    f"unknown semantic field: {'.'.join(path)}"
                )
            current = current[key]
        _set_dotted(config, expression)
    _validate(config)
    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return ResolvedVerpoLaunch(
        config,
        hashlib.sha256(canonical.encode()).hexdigest(),
        tuple(sources),
        tuple(cli_overrides),
    )


def load_launch_matrix_spec(
    matrix: str, *, config_root: Path = CONFIG_ROOT
) -> LaunchMatrixSpec:
    path = config_root / "matrices" / f"{matrix}.yaml"
    if not path.is_file():
        raise VerpoLaunchConfigError(f"unknown matrix: {matrix}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if document.get("kind") != "matrix":
        raise VerpoLaunchConfigError(f"{path} is not a matrix")
    protocols = [str(value) for value in document.get("protocols", [])]
    teachers = [str(value) for value in document.get("teachers", [])]
    arms = [str(value) for value in document.get("arms", [])]
    cells = [
        LaunchMatrixCell(
            str(cell["identifier"]),
            cell.get("protocol"),
            cell.get("teacher"),
            str(cell["arm"]),
            tuple(cell.get("overrides", [])),
        )
        for cell in document.get("cells", [])
    ]
    if not cells:
        cells = [
            LaunchMatrixCell(f"{protocol}__{teacher}__{arm}", protocol, teacher, arm)
            for protocol in protocols
            for teacher in teachers
            for arm in arms
        ]
    if not cells:
        cells = [LaunchMatrixCell(arm, None, None, arm) for arm in arms]
    if not cells:
        raise VerpoLaunchConfigError(f"matrix {matrix} has no cells")
    return LaunchMatrixSpec(
        str(document.get("id", matrix)),
        tuple(cells),
        tuple(str(value) for value in document.get("overrides", [])),
    )


def load_launch_matrix(
    matrix: str, *, config_root: Path = CONFIG_ROOT
) -> tuple[LaunchMatrixCell, ...]:
    return load_launch_matrix_spec(matrix, config_root=config_root).cells


def engine_command(resolved: ResolvedVerpoLaunch) -> list[str]:
    config = resolved.config
    hardware, protocol = config["hardware"], config["protocol"]
    bash = os.environ.get("VERPO_BASH")
    if not bash:
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = (
            str(git_bash)
            if os.name == "nt" and git_bash.exists()
            else shutil.which("bash")
        )
    if not bash:
        raise VerpoLaunchConfigError(
            "Bash is required; set VERPO_BASH to its executable"
        )
    return [
        bash,
        str(ROOT / "pipeline/verl_math/engines/sdpo_section3.sh"),
        str(hardware["gpu_count"]),
        str(protocol["global_prompt_batch"]),
        str(int(protocol["global_prompt_batch"]) // int(hardware["gpu_count"])),
        str(protocol["total_epochs"]),
        str(hardware["gpu_name_substring"]),
        str(config["method"]["divergence"]),
        str(config["method"]["displacement"]),
    ]


def render_engine(
    resolved: ResolvedVerpoLaunch, environment: Mapping[str, str] | None = None
) -> str:
    # Do not inherit method overrides from a previous shell run.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VERPO_", "PAPER_BASELINE_", "QWEN3_"))
    }
    env.update(beta_engine_environment(resolved))
    if environment is not None:
        env.update(environment)
    env["QWEN3_PRINT_RESOLVED_CONFIG"] = "1"
    result = subprocess.run(
        engine_command(resolved),
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise VerpoLaunchConfigError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def native_verl_projection(
    resolved: ResolvedVerpoLaunch, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Parse the actual engine dry-run arguments; no separate shadow projection."""
    projection: dict[str, Any] = {}
    rendered = render_engine(resolved, environment)
    for argument in rendered.split("# hydra_arguments\n", 1)[1].splitlines():
        if "=" in argument:
            _set_dotted(projection, argument.lstrip("+"))
    return projection


def trl_verpo_projection(resolved: ResolvedVerpoLaunch) -> dict[str, Any]:
    config = resolved.config
    return {
        "model": config["model"],
        "protocol": config["protocol"],
        "teacher": config["teacher"],
        "method": config["method"],
        "zpd": config["zpd"],
        "reward": config["reward"],
    }


ENGINE_FIELDS: dict[str, dict[str, str]] = {
    "identity": {"model": "MODEL_ID", "status": "CONFIG_STATUS"},
    "model": {
        "hf_id": "MODEL_REPO",
        "revision": "MODEL_REVISION",
        "trust_remote_code": "MODEL_TRUST_REMOTE_CODE",
        "prompt_template_version": "PROMPT_TEMPLATE_VERSION",
        "chat_template_thinking_control": "CHAT_TEMPLATE_THINKING_CONTROL",
    },
    "hardware": {
        "gpu_count": "EXPECTED_GPUS",
        "gpu_name_substring": "GPU_NAME_SUBSTRING",
        "gpu_memory_mib_min": "GPU_MEMORY_MIN_MIB",
        "gpu_memory_mib_max": "GPU_MEMORY_MAX_MIB",
        "rollout_backend": "ROLLOUT_BACKEND",
        "actor_micro_batch_per_gpu": "ACTOR_MICRO_BATCH_PER_GPU",
        "vllm_gpu_memory_utilization": "VLLM_GPU_MEMORY_UTILIZATION",
        "rollout_enforce_eager": "ROLLOUT_ENFORCE_EAGER",
        "free_cache_engine": "FREE_CACHE_ENGINE",
        "tensor_model_parallel_size": "TENSOR_MODEL_PARALLEL_SIZE",
    },
    "protocol": {
        "rollout_n": "ROLLOUT_N",
        "mini_batch_trajectories": "MINI_BATCH_TRAJECTORIES",
        "max_prompt_length": "MAX_PROMPT_LENGTH",
        "max_response_length": "MAX_RESPONSE_LENGTH",
        "validation_response_length": "VAL_RESPONSE_LENGTH",
        "teacher_max_token_len_per_gpu": "TEACHER_MAX_TOKEN_LEN_PER_GPU",
        "max_model_len": "MAX_MODEL_LEN",
        "validation_batch": "VAL_BATCH",
        "validation_n": "VAL_N",
        "rollout_temperature": "ROLLOUT_TEMPERATURE",
        "rollout_top_p": "ROLLOUT_TOP_P",
        "rollout_top_k": "ROLLOUT_TOP_K",
        "validation_temperature": "VAL_TEMPERATURE",
        "validation_top_p": "VAL_TOP_P",
        "validation_top_k": "VAL_TOP_K",
        "rollout_importance_clip": "ROLLOUT_IMPORTANCE_CLIP",
        "max_reprompt_length": "MAX_REPROMPT_LENGTH",
        "learning_rate": "LEARNING_RATE",
        "weight_decay": "WEIGHT_DECAY",
        "warmup_steps": "WARMUP_STEPS",
        "scheduler": "LR_SCHEDULER",
        "grad_clip": "ACTOR_GRAD_CLIP",
        "ppo_epochs": "PPO_EPOCHS",
        "save_frequency": "SAVE_FREQUENCY",
        "validation_frequency": "VALIDATION_FREQUENCY",
        "total_training_steps": "TOTAL_TRAINING_STEPS",
        "logging_frequency": "LOGGING_FREQUENCY",
        "validation_before_training": "VALIDATION_BEFORE_TRAINING",
        "train_shuffle": "TRAIN_SHUFFLE",
        "actor_shuffle": "ACTOR_SHUFFLE",
        "data_seed": "DATA_SEED",
        "seed": "SEED",
        "normalize_advantage_by_std": "NORMALIZE_ADVANTAGE_BY_STD",
        "clip_ratio_high": "CLIP_RATIO_HIGH",
        "filter_groups_enabled": "FILTER_GROUPS_ENABLED",
        "train_max_samples": "TRAIN_MAX_SAMPLES",
        "student_enable_thinking": "STUDENT_ENABLE_THINKING",
        "teacher_enable_thinking": "TEACHER_ENABLE_THINKING",
        "validation_enable_thinking": "VALIDATION_ENABLE_THINKING",
        "reward_mode": "REWARD_MODE",
    },
    "method": {
        "divergence": "DIVERGENCE",
        "displacement": "DISPLACEMENT_MODE",
        "lambda_ref": "LAMBDA_REF",
        "lambda_evi": "LAMBDA_EVI",
        "advantage_modulation": "ADVANTAGE_MODULATION",
        "advantage_modulation_lambda": "ADVANTAGE_MODULATION_LAMBDA",
        "temperature": "TEMPERATURE",
        "vocab_mode": "VOCAB_MODE",
        "top_k": "TOP_K",
        "vocab_chunk_size": "VOCAB_CHUNK_SIZE",
        "cost_alpha": "COST_ALPHA",
        "cost_epsilon": "COST_EPSILON",
        "evidence_source": "EVIDENCE_SOURCE",
    },
    "zpd": {
        "group_enabled": "GROUP_ZPD_ENABLED",
        "group_epsilon": "GROUP_ZPD_EPSILON",
        "group_mode": "GROUP_ZPD_MODE",
        "sibling_selection": "SIBLING_SELECTION_MODE",
        "evidence_scope": "EVIDENCE_ROLLOUT_SCOPE",
        "contrastive_num_negative_hints": "CONTRASTIVE_NUM_NEGATIVE_HINTS",
        "projection_epsilon": "PROJECTION_EPSILON",
        "allow_negative_benefit": "ALLOW_NEGATIVE_BENEFIT",
    },
    "teacher": {
        "mode": "TEACHER_MODE",
        "sync_interval": "TEACHER_SYNC_INTERVAL",
        "ema_decay": "TEACHER_EMA_DECAY",
    },
    "runtime": {"save_training_rollouts": "SAVE_TRAINING_ROLLOUTS"},
    "finetuning": {
        "lora_rank": "LORA_RANK",
        "lora_alpha": "LORA_ALPHA",
        "target_modules": "TARGET_MODULES",
    },
}


def beta_engine_environment(resolved: ResolvedVerpoLaunch) -> dict[str, str]:
    config = resolved.config
    method = config["method"]
    values: dict[str, Any] = {}
    for section, fields in ENGINE_FIELDS.items():
        for key, suffix in fields.items():
            if key in config[section]:
                values[f"VERPO_{suffix}"] = config[section][key]
    objective = method["objective"]
    values.update(
        {
            "QWEN3_TRAINING_OBJECTIVE": objective,
            "VERPO_EXPERIMENT_ID": config["identity"]["arm"],
            "VERPO_PROJECT_NAME": config["protocol"]["project_name"],
            "PAPER_BASELINE_ENABLED": objective in {"sdpo", "srpo"},
            "PAPER_BASELINE_OBJECTIVE": objective
            if objective in {"sdpo", "srpo"}
            else "sdpo",
            "PAPER_BASELINE_PROMPT_PROFILE": config["model"]["prompt_template_version"],
            "PAPER_BASELINE_EMA_DECAY": config["teacher"]["ema_decay"],
            "VERPO_LORA_RANK": config["finetuning"]["lora_rank"]
            if config["finetuning"]["mode"] == "lora"
            else 0,
        }
    )
    return {
        key: str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in values.items()
    }


__all__ = [
    "LaunchMatrixCell",
    "LaunchMatrixSpec",
    "ResolvedVerpoLaunch",
    "VERPOConfig",
    "VerpoLaunchConfigError",
    "beta_engine_environment",
    "load_launch_matrix",
    "load_launch_matrix_spec",
    "native_verl_projection",
    "resolve_verpo_launch",
    "trl_verpo_projection",
]
