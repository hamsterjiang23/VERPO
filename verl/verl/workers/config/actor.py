# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig, RolloutCorrectionConfig
from verl.utils.profiler.config import ProfilerConfig
from verl.utils.qat import QATConfig

from .checkpoint import McoreCheckpointConfig, MindSpeedCheckpointConfig
from .engine import (
    FSDPEngineConfig,
    McoreEngineConfig,
    MindSpeedEngineConfig,
    TorchtitanEngineConfig,
    VeOmniEngineConfig,
)
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "PolicyLossConfig",
    "PaperBaselineConfig",
    "VerpoZPDConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
    "VeOmniActorConfig",
    "QATConfig",
    "TorchTitanActorConfig",
    "MindSpeedActorConfig",
]


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
        rollout_correction (RolloutCorrectionConfig): Configuration for rollout correction.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)


@dataclass
class PaperBaselineConfig(BaseConfig):
    """Independent paper-faithful SDPO/SRPO objective configuration."""

    enabled: bool = False
    objective: str = "sdpo"
    divergence: str = "jsd"
    jsd_alpha: float = 0.5
    top_k: int = 100
    add_tail_bucket: bool = True
    success_reward_threshold: float = 0.5
    teacher_ema_decay: float = 0.95
    temperature: float = 1.0
    is_clip: float = 2.0
    entropy_beta: float = 1.0
    entropy_support: str = "full_vocab"
    sibling_selection: str = "first_correct"
    remove_thinking_from_demonstration: bool = True
    teacher_max_token_len_per_gpu: int = 18944
    max_reprompt_len: int = 10240
    prompt_profile: str = "sdpo_official_v2"

    def __post_init__(self):
        objective = str(self.objective).strip().lower()
        if objective not in {"sdpo", "srpo"}:
            raise ValueError("paper_baseline.objective must be sdpo or srpo")
        object.__setattr__(self, "objective", objective)
        if self.divergence != "jsd":
            raise ValueError("paper SDPO/SRPO baselines require JSD")
        finite_values = {
            "jsd_alpha": self.jsd_alpha,
            "success_reward_threshold": self.success_reward_threshold,
            "teacher_ema_decay": self.teacher_ema_decay,
            "temperature": self.temperature,
            "is_clip": self.is_clip,
            "entropy_beta": self.entropy_beta,
        }
        if any(not math.isfinite(float(value)) for value in finite_values.values()):
            raise ValueError("paper baseline floating-point values must be finite")
        if not 0.0 < float(self.jsd_alpha) < 1.0:
            raise ValueError("paper_baseline.jsd_alpha must be in (0, 1)")
        if not 0.0 < float(self.teacher_ema_decay) < 1.0:
            raise ValueError("paper_baseline.teacher_ema_decay must be in (0, 1)")
        if self.top_k <= 0:
            raise ValueError("paper_baseline.top_k must be positive")
        if not self.add_tail_bucket:
            raise ValueError("paper SDPO/SRPO requires the top-K tail bucket")
        if self.temperature <= 0 or self.is_clip <= 0 or self.entropy_beta <= 0:
            raise ValueError("paper baseline temperature, IS clip, and entropy beta must be positive")
        if self.entropy_support != "full_vocab":
            raise ValueError("paper SRPO entropy must use the full Teacher vocabulary")
        if self.sibling_selection != "first_correct":
            raise ValueError("paper baseline sibling_selection must be first_correct")
        if self.teacher_max_token_len_per_gpu <= 0 or self.max_reprompt_len <= 0:
            raise ValueError("paper baseline Teacher token limits must be positive")
        if self.prompt_profile not in {"sdpo_official_v2", "srpo_v1"}:
            raise ValueError("paper_baseline.prompt_profile is unsupported")


@dataclass
class VerpoZPDConfig(BaseConfig):
    """Configuration for exact dual-Teacher VERPO-ZPD supervision."""

    enabled: bool = False
    teacher_mode: str = "fixed_initial"
    teacher_sync_interval: int = 10
    teacher_ema_decay: float = 0.95
    divergence: str = "forward_kl"
    displacement_mode: str = "evidence_vs_none"
    group_zpd_enabled: bool = False
    group_zpd_epsilon: float = 0.0
    group_zpd_mode: str = "binary_mixed"
    sibling_selection_mode: str = "correctness"
    evidence_rollout_scope: str = "all"
    evidence_source: str = "rollout_group"
    contrastive_num_negative_hints: int = 1
    smoke_allow_unboxed_contrastive: bool = False
    projection_epsilon: float = 1e-8
    lambda_ref: float = 0.1
    lambda_evi: float = 0.5
    # Optional FEC controller reuse in the PPO/GRPO advantage magnitude.
    advantage_modulation: str = "none"
    advantage_modulation_lambda: float = 0.0
    # Active controller parameterization in benefit units.  Legacy tau/rho/
    # cost_floor/cost_beta fields below remain for old manifests/checkpoints.
    cost_alpha: float | None = None
    cost_epsilon: float | None = None
    tau: float = 1.0
    rho: float = 1e-4
    cost_floor: float = 0.001
    cost_beta: float = 1.0
    allow_negative_benefit: bool = False
    temperature: float = 1.0
    vocab_chunk_size: int = 4096
    vocab_mode: str = "topk_truncated"
    top_k: int = 128
    teacher_max_token_len_per_gpu: int = 40960
    # Zero preserves the historical behavior (only the combined token budget).
    teacher_max_reprompt_len: int = 0
    gradient_audit_enabled: bool = False
    gradient_audit_max_steps: int = 0
    gradient_audit_max_parameter_elements: int = 1_000_000
    gradient_audit_max_parameter_tensors: int = 16

    def __post_init__(self):
        teacher_mode = str(self.teacher_mode).strip().lower()
        if teacher_mode == "snapshot_anchor":
            teacher_mode = "snapshot"
        if teacher_mode not in {"fixed_initial", "snapshot", "ema"}:
            raise ValueError(
                "VERPO teacher_mode must be 'fixed_initial', 'snapshot', or 'ema'"
            )
        object.__setattr__(self, "teacher_mode", teacher_mode)
        if int(self.teacher_sync_interval) <= 0:
            raise ValueError("VERPO teacher_sync_interval must be positive")
        if not 0.0 < float(self.teacher_ema_decay) < 1.0:
            raise ValueError("VERPO teacher_ema_decay must be in (0, 1)")
        float_values = {
            "lambda_ref": self.lambda_ref,
            "lambda_evi": self.lambda_evi,
            "advantage_modulation_lambda": self.advantage_modulation_lambda,
            "tau": self.tau,
            "rho": self.rho,
            "cost_floor": self.cost_floor,
            "cost_beta": self.cost_beta,
            "temperature": self.temperature,
            "group_zpd_epsilon": self.group_zpd_epsilon,
        }
        if self.cost_alpha is None and self.cost_epsilon is None:
            effective_cost_alpha = float(self.tau) * float(self.cost_beta)
            effective_cost_epsilon = float(self.tau) * (
                float(self.rho) + float(self.cost_floor)
            )
            object.__setattr__(self, "cost_alpha", effective_cost_alpha)
            object.__setattr__(self, "cost_epsilon", effective_cost_epsilon)
        elif self.cost_alpha is None or self.cost_epsilon is None:
            raise ValueError("VERPO cost_alpha and cost_epsilon must be set together")
        float_values.update(
            cost_alpha=self.cost_alpha,
            cost_epsilon=self.cost_epsilon,
        )
        if any(not math.isfinite(float(value)) for value in float_values.values()):
            raise ValueError("VERPO floating-point configuration values must be finite")
        if float(self.cost_alpha) < 0 or float(self.cost_epsilon) <= 0:
            raise ValueError("VERPO cost_alpha must be nonnegative and cost_epsilon positive")
        if self.divergence not in {"forward_kl", "reverse_kl"}:
            raise ValueError("VERPO divergence must be 'forward_kl' or 'reverse_kl'")
        if self.advantage_modulation not in {"none", "multiplicative_w"}:
            raise ValueError(
                "VERPO advantage_modulation must be 'none' or 'multiplicative_w'"
            )
        valid_displacement_modes = {
            "evidence_vs_none",
            "correct_vs_incorrect",
            "reward_ranked",
            "fec",
        }
        if self.displacement_mode not in valid_displacement_modes:
            raise ValueError(
                "VERPO displacement_mode must be 'evidence_vs_none', "
                "'correct_vs_incorrect', 'reward_ranked', or 'fec'"
            )
        if self.group_zpd_mode not in {"binary_mixed", "reward_ranked"}:
            raise ValueError("VERPO group_zpd_mode must be binary_mixed or reward_ranked")
        if float(self.group_zpd_epsilon) < 0:
            raise ValueError("VERPO group_zpd_epsilon must be nonnegative")
        if self.sibling_selection_mode not in {"correctness", "reward_ranked"}:
            raise ValueError(
                "VERPO sibling_selection_mode must be correctness or reward_ranked"
            )
        if self.evidence_rollout_scope not in {"all", "wrong_only"}:
            raise ValueError("VERPO evidence_rollout_scope must be 'all' or 'wrong_only'")
        if self.contrastive_num_negative_hints <= 0:
            raise ValueError("VERPO contrastive_num_negative_hints must be positive")
        reward_ranked_siblings = (
            self.sibling_selection_mode == "reward_ranked"
            or self.displacement_mode == "reward_ranked"
        )
        if reward_ranked_siblings:
            required = {
                "group_zpd_mode": (self.group_zpd_mode, "reward_ranked"),
                "sibling_selection_mode": (
                    self.sibling_selection_mode,
                    "reward_ranked",
                ),
                "evidence_rollout_scope": (self.evidence_rollout_scope, "all"),
                "contrastive_num_negative_hints": (
                    self.contrastive_num_negative_hints,
                    1,
                ),
            }
            mismatches = [
                f"{name}={actual} (required {expected})"
                for name, (actual, expected) in required.items()
                if actual != expected
            ]
            if mismatches:
                raise ValueError(
                    "VERPO reward-ranked protocol mismatch: " + ", ".join(mismatches)
                )
        if not math.isfinite(float(self.projection_epsilon)) or self.projection_epsilon <= 0:
            raise ValueError("VERPO projection_epsilon must be finite and positive")
        if min(
            self.lambda_ref,
            self.lambda_evi,
            self.advantage_modulation_lambda,
            self.rho,
            self.cost_floor,
            self.cost_beta,
        ) < 0:
            raise ValueError("VERPO coefficients and cost parameters must be nonnegative")
        if self.advantage_modulation == "none" and self.advantage_modulation_lambda != 0:
            raise ValueError(
                "VERPO advantage_modulation_lambda must be zero when modulation is none"
            )
        if self.advantage_modulation == "multiplicative_w" and self.displacement_mode != "fec":
            raise ValueError(
                "multiplicative_w advantage modulation requires displacement_mode=fec"
            )
        if self.tau <= 0 or self.temperature <= 0:
            raise ValueError("VERPO tau and temperature must be positive")
        if self.rho + self.cost_floor <= 0:
            raise ValueError("VERPO rho + cost_floor must be positive")
        if self.vocab_chunk_size <= 0 or self.teacher_max_token_len_per_gpu <= 0:
            raise ValueError("VERPO token limits must be positive")
        if self.teacher_max_reprompt_len < 0:
            raise ValueError("VERPO teacher_max_reprompt_len must be nonnegative")
        if self.vocab_mode not in {"full", "topk_truncated"}:
            raise ValueError("VERPO vocab_mode must be 'full' or 'topk_truncated'")
        if self.top_k <= 0:
            raise ValueError("VERPO top_k must be positive")
        if self.gradient_audit_max_steps < 0:
            raise ValueError("VERPO gradient_audit_max_steps must be nonnegative")
        if self.gradient_audit_enabled and self.gradient_audit_max_steps <= 0:
            raise ValueError(
                "VERPO gradient_audit_max_steps must be positive when gradient audit is enabled"
            )
        if self.gradient_audit_max_parameter_elements <= 0:
            raise ValueError("VERPO gradient_audit_max_parameter_elements must be positive")
        if self.gradient_audit_max_parameter_tensors <= 0:
            raise ValueError("VERPO gradient_audit_max_parameter_tensors must be positive")


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    loss_mode: str = "ppo"
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    calculate_sum_pi_squared: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 42
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)
    qat: QATConfig = field(default_factory=QATConfig)
    verpo: VerpoZPDConfig = field(default_factory=VerpoZPDConfig)
    paper_baseline: PaperBaselineConfig = field(default_factory=PaperBaselineConfig)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if self.loss_mode not in {"ppo", "verpo_zpd", "sdpo_jsd", "srpo_jsd"}:
            raise ValueError(
                "actor.loss_mode must be ppo, verpo_zpd, sdpo_jsd, or srpo_jsd"
            )
        if (self.loss_mode == "verpo_zpd") != bool(self.verpo.enabled):
            raise ValueError("actor.loss_mode=verpo_zpd and actor.verpo.enabled=true must be set together")
        paper_loss_modes = {"sdpo_jsd": "sdpo", "srpo_jsd": "srpo"}
        paper_enabled = bool(self.paper_baseline.enabled)
        if (self.loss_mode in paper_loss_modes) != paper_enabled:
            raise ValueError(
                "actor.loss_mode=sdpo_jsd/srpo_jsd and actor.paper_baseline.enabled=true "
                "must be set together"
            )
        if paper_enabled and self.paper_baseline.objective != paper_loss_modes[self.loss_mode]:
            raise ValueError("paper baseline objective does not match actor.loss_mode")
        if paper_enabled and self.verpo.enabled:
            raise ValueError("paper SDPO/SRPO baselines cannot enable VERPO-ZPD")
        if paper_enabled and self.loss_agg_mode != "token-mean":
            raise ValueError("paper SDPO/SRPO baselines require token-mean aggregation")
        if paper_enabled and self.use_kl_loss:
            raise ValueError("paper SDPO/SRPO baselines require actor.use_kl_loss=false")
        if self.verpo.enabled and float(self.kl_loss_coef) != 0.0:
            raise ValueError(
                "VERPO-ZPD owns reference anchoring through actor.verpo.lambda_ref; "
                "set actor.kl_loss_coef=0 (actor.use_kl_loss may remain enabled "
                "as an VERPO_CONTRASTIVE-compatible zero-coefficient switch)"
            )
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
        checkpoint (McoreCheckpointConfig): Megatron-specific checkpoint config
            that adds ``mbridge_config`` on top of the base checkpoint fields.
    """

    strategy: str = "megatron"
    entropy_from_logits_with_chunking: bool = False
    entropy_from_logits_chunk_size: int = 2048
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False
    checkpoint: McoreCheckpointConfig = field(default_factory=McoreCheckpointConfig)

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_from_logits_chunk_size: int = 2048
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config
        # Sync strategy to engine config so engine_workers can pick the right FSDP version.
        # EngineConfig.strategy defaults to None, so without this, engine_workers.py always
        # falls back to FSDP1 even when actor.strategy="fsdp2".
        object.__setattr__(self.engine, "strategy", self.strategy)

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)
        if (
            self.ulysses_sequence_parallel_size > 1
            and model_config
            and not model_config.get("use_remove_padding", False)
        ):
            raise ValueError(
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )


@dataclass
class VeOmniActorConfig(ActorConfig):
    """Configuration for VeOmni actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'veomni' for VeOmni parallelism.
        veomni (dict[str, Any]): Configuration for VeOmni settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "veomni"
    veomni: VeOmniEngineConfig = field(default_factory=VeOmniEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate VeOmni actor configuration parameters."""
        super().__post_init__()
        self.engine = self.veomni
        if self.veomni.router_replay.mode != "disabled" and not self.use_remove_padding:
            raise RuntimeError(
                "router_replay requires use_remove_padding=True. In VeOmni engine, "
                "the non-remove-padding path also disables Ulysses SP slicing and "
                "the fused-kernel log_probs path, and is not a tested production "
                "configuration for MoE routing replay. Set "
                "actor.use_remove_padding=True or router_replay.mode='disabled'."
            )


@dataclass
class TorchTitanActorConfig(ActorConfig):
    """Configuration for TorchTitan actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'torchtitan' for TorchTitan parallelism.
        torchtitan (TorchtitanEngineConfig): Configuration for TorchTitan engine settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
        use_rollout_log_probs (bool): Whether to use log probabilities from rollout engine
    """

    strategy: str = "torchtitan"
    torchtitan: TorchtitanEngineConfig = field(default_factory=TorchtitanEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate TorchTitan actor configuration parameters."""
        super().__post_init__()
        self.engine = self.torchtitan


@dataclass
class MindSpeedActorConfig(ActorConfig):
    """Configuration for mindspeed actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'mindspeed' for mindspeed parallelism.
        mindspeed (dict[str, Any]): Configuration for mindspeed parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
        use_rollout_log_probs (bool): Whether to use log probabilities from rollout engine.
        checkpoint (MindSpeedCheckpointConfig): MindSpeed-specific checkpoint config
            (inherits ``mbridge_config`` from :class:`McoreCheckpointConfig`).
    """

    strategy: str = "mindspeed"
    mindspeed: MindSpeedEngineConfig = field(default_factory=MindSpeedEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False
    checkpoint: MindSpeedCheckpointConfig = field(default_factory=MindSpeedCheckpointConfig)

    def __post_init__(self):
        """Validate MindSpeed actor configuration parameters."""
        super().__post_init__()
        self.engine = self.mindspeed
