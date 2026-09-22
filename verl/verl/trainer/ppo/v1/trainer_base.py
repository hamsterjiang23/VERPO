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

import hashlib
import json
import logging
import math
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transfer_queue import KVBatchMeta

from verl.checkpoint_engine import CheckpointEngineManager
from risk_aware_opsd.length_aware_reward import classify_reward_ranked_groups
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.protocol import DataProto, DataProtoFuture
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.baselines.paper_batch import (
    build_paper_routing,
    build_paper_teacher_fields,
    rollout_group_ids,
)
from verl.trainer.distillation.verpo_protocol import (
    build_contrastive_evidence_teacher_fields,
)
from verl.trainer.distillation.verpo_zpd import (
    apply_evidence_rollout_scope,
    compute_group_zpd_gate_by_uid,
)
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_spec_decode_metrics
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.utils import (
    Role,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
    need_teacher_policy,
)
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from verl.utils import hf_processor, hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.best_checkpoint_retention import retain_best_verpo_contrastive_checkpoints
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.checkpoint.modelscope_upload import (
    AsyncModelScopeCheckpointUploader,
    upload_checkpoint_to_modelscope,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_extern_type
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import Tracking, ValidationGenerationsLogger
from verl.workers.config import CriticConfig, DistillationConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import response_from_nested, response_to_nested


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


def _resolve_verpo_outcome_positive(
    rewards: torch.Tensor,
    extra_fields: Any,
    *,
    require_independent_acc: bool,
) -> torch.Tensor:
    """Resolve binary outcomes without silently treating shaped reward as accuracy."""
    if rewards.ndim != 1:
        raise ValueError("VERPO sequence rewards must have shape [n_rows]")
    if extra_fields is None:
        if require_independent_acc:
            raise ValueError(
                "Correctness-dependent VERPO routing requires "
                "reward_extra_info.acc in extra_fields"
            )
        return rewards >= 1.0 - 1e-6

    rows = extra_fields.tolist() if hasattr(extra_fields, "tolist") else list(extra_fields)
    if len(rows) != rewards.shape[0]:
        raise ValueError("VERPO extra_fields must contain one entry per rollout")
    outcome_values = []
    for row_index, (extra_field, reward) in enumerate(
        zip(rows, rewards.tolist(), strict=True)
    ):
        reward_info = (
            extra_field.get("reward_extra_info", {})
            if isinstance(extra_field, dict)
            else {}
        )
        if require_independent_acc and "acc" not in reward_info:
            raise ValueError(
                "Correctness-dependent VERPO routing requires "
                f"reward_extra_info.acc for rollout {row_index}"
            )
        outcome_values.append(
            bool(float(reward_info.get("acc", reward)) >= 1.0 - 1e-6)
        )
    return torch.tensor(outcome_values, dtype=torch.bool)


def select_reward_variable_groups(
    keys: list[str],
    sequence_rewards: list[float],
    *,
    min_reward_range: float,
    max_groups: int,
    expected_group_size: int | None = None,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Select complete GRPO groups whose final reward has real variation.

    The final scalar is shared by GRPO, reward-ranked ZPD admission, and this
    dynamic sampler.  Correctness labels remain independent Teacher-sibling
    metadata and are deliberately not consulted here.
    """
    if len(keys) != len(sequence_rewards):
        raise ValueError("keys and sequence_rewards must have the same length")
    if min_reward_range < 0:
        raise ValueError("min_reward_range must be non-negative")
    if max_groups <= 0:
        raise ValueError("max_groups must be positive")
    if expected_group_size is not None and expected_group_size <= 0:
        raise ValueError("expected_group_size must be positive when provided")

    grouped: dict[str, list[tuple[str, float]]] = {}
    for key, reward in zip(keys, sequence_rewards, strict=True):
        uid = key.split("_", 1)[0]
        grouped.setdefault(uid, []).append((key, float(reward)))

    accepted_uids = []
    constant_uids = []
    for uid, rows in grouped.items():
        rewards = [reward for _, reward in rows]
        complete = expected_group_size is None or len(rows) == expected_group_size
        if complete and all(math.isfinite(reward) for reward in rewards) and max(rewards) - min(rewards) > min_reward_range:
            accepted_uids.append(uid)
        else:
            constant_uids.append(uid)

    selected_uids = set(accepted_uids[:max_groups])
    selected_keys = [key for key in keys if key.split("_", 1)[0] in selected_uids]
    dropped_keys = [key for key in keys if key.split("_", 1)[0] not in selected_uids]
    return selected_keys, dropped_keys, {
        "generated_groups": len(grouped),
        "accepted_groups": len(selected_uids),
        "constant_groups": len(constant_uids),
        "excess_variable_groups": max(0, len(accepted_uids) - len(selected_uids)),
    }


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class PPOTrainer(ABC):
    """Base class for PPO trainer.

    Args:
        config: DictConfig from yaml config file.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self._modelscope_upload_queue: Optional[AsyncModelScopeCheckpointUploader] = None
        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.trainer_mode = self.config.trainer.v1.trainer_mode
        self.parameter_sync_step = self.config.trainer.v1.get(self.trainer_mode, {}).get("parameter_sync_step", 1)
        self.replay_buffer = self._build_replay_buffer()

    def _build_replay_buffer(self) -> ReplayBuffer:
        """Instantiate the replay buffer (or a user-provided custom sampler).

        Set ``trainer.v1.sampler.custom_sampler.{path,name}`` to plug in a custom
        ``ReplayBuffer`` subclass; otherwise the built-in implementation is used.
        """
        sampler_config = self.config.trainer.v1.sampler
        custom_sampler = sampler_config.get("custom_sampler", None)
        sampler_cls = ReplayBuffer
        if custom_sampler is not None and custom_sampler.get("path") and custom_sampler.get("name"):
            sampler_cls = load_extern_type(custom_sampler.path, custom_sampler.name)

        return sampler_cls(
            trainer_mode=self.trainer_mode,
            trainer_config=self.config.trainer.v1.get(self.trainer_mode, {}),
            max_off_policy_threshold=sampler_config.max_off_policy_threshold,
            max_off_policy_strategy=sampler_config.max_off_policy_strategy,
            sampler_kwargs=sampler_config.sampler_kwargs,
        )

    def init(self):
        """Initialize all components of the trainer.

        1. WorkerGroup: actor, critic, reference with model engine: FSDP/Megatron/VeOmni/...
        2. LLMServerManager: launch and manage LLM server replicas for generation.
        3. CheckpointEngineManager: sync weights between worker group and LLM server replicas.
        4. RewardLoopManager: reward workers for rule-based reward, optional LLM server for model-based reward.
        5. [Optional] MultiTeacherModelManager: LLM teacher servers for on-policy distillation.
        """
        self._setup()
        self.on_init_end()

    def _setup(self):
        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()
        self._init_resource_pool_mgr()
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 1. define actor and rollout class
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        # 2. define critic class
        if self.use_critic:
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_cfg.model_config,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # 3. create worker group for actor rollout and critic
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.config.trainer.device
        logger.info(f"worker group kwargs: {wg_kwargs}")

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            logger.info(f"create worker group {spawn_wg.keys()}")

        # 5. initialize critic model engine
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)
            logger.info("critic model engine initialized")

        # 6. initialize actor and ref model engine
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        logger.info("actor and ref model engine initialized")

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if self.use_reference_policy:
            self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # 7. initialize reward loop manager
        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        logger.info("reward loop manager initialized")

        # 8. initialize teacher loop manager
        if self.use_teacher_policy:
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # 9. initialize agent loop manager
        self.llm_server_manager: LLMServerManager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # 10. initialize checkpoint engine manager
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        checkpoint_engine_config.backend = "naive"
        self.checkpoint_manager: CheckpointEngineManager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )
        logger.info("checkpoint engine manager initialized")

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()
        self._load_checkpoint()

        logger.info("all initialize finished, ready to fit")

    def get_llm_client(self) -> LLMServerClient:
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client()

    def get_teacher_client(self) -> Optional[dict[str, LLMServerClient]]:
        """Get the On-Policy Distillation teacher server clients.

        Returns:
            dict[str, LLMServerClient]: The teacher server clients.
        """
        return self.teacher_model_manager.get_client() if self.use_teacher_policy else None

    def get_reward_handles(self) -> list[ray.actor.ActorHandle]:
        """Get the handles of reward loop workers."""
        return self.reward_loop_manager.reward_loop_worker_handles

    def fit(self, agent_loop_manager: AgentLoopManager):
        """Fit the trainer with the agent loop manager.

        Args:
            agent_loop_manager: The agent loop manager to generate sequences.
        """
        self.agent_loop_manager = agent_loop_manager

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        exit_code = 0
        try:
            self.validation_generations_logger = ValidationGenerationsLogger(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
            )
            result = self._fit()
            self.on_train_end()
            return result
        except BaseException:
            exit_code = 1
            raise
        finally:
            self.logger.finish(exit_code=exit_code)

    def _fit(self):
        """Run the training loop inside the tracking lifecycle."""

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            self.on_validate_begin()
            val_metrics = self._validate()
            self.on_validate_end()
            self._drain_dump_futures()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        current_epoch = self.global_steps // len(self.train_dataloader)
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        self.on_train_begin()
        last_val_metrics = None
        while current_epoch < self.config.trainer.total_epochs and self.global_steps <= self.total_training_steps:
            is_last_step = self.global_steps >= self.total_training_steps
            metrics = {}
            self.timing_raw = {}
            checkpoint_saved = False

            # 1. perform rollout and actor/critic training
            with marked_timer("step", self.timing_raw):
                self.on_step_begin()

                self._start_profiling()
                batch = self.step(metrics, self.timing_raw)
                self._stop_profiling()

                # 2. save checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", self.timing_raw, color="green"):
                        self._save_checkpoint()
                    checkpoint_saved = True

                self.on_step_end()

            # 4. validate
            if self.config.trainer.test_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.test_freq == 0
            ):
                with marked_timer("testing", self.timing_raw, color="green"):
                    self.on_validate_begin()
                    val_metrics: dict = self._validate()
                    self.on_validate_end()
                    # Validation predictions are part of the checkpoint-selection
                    # record. Surface write/split failures before upload or pruning.
                    self._drain_dump_futures()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

                upload = self.config.trainer.get("modelscope_upload", {})
                if bool(upload.get("enabled", False)):
                    if bool(upload.get("async_enabled", False)):
                        queue = self._get_modelscope_upload_queue()
                        queue.record_validation(
                            step=self.global_steps,
                            validation_metrics=val_metrics,
                            checkpoint_saved=checkpoint_saved,
                        )
                        metrics["modelscope_upload/enqueued"] = 0.0
                        if checkpoint_saved:
                            queue.enqueue(
                                step=self.global_steps,
                                validation_metrics=val_metrics,
                                terminal=is_last_step,
                            )
                            metrics["modelscope_upload/enqueued"] = 1.0
                        metrics["modelscope_upload/pending"] = float(
                            queue.pending_count()
                        )
                    else:
                        if not checkpoint_saved:
                            raise RuntimeError(
                                "Synchronous ModelScope upload is enabled, but validation ran "
                                "without a saved checkpoint"
                            )
                        token_env = str(upload.get("token_env", "MODELSCOPE_TOKEN"))
                        upload_result = upload_checkpoint_to_modelscope(
                            self.config.trainer.default_local_dir,
                            step=self.global_steps,
                            repo_id=str(upload["repo_id"]),
                            experiment_id=str(upload["experiment_id"]),
                            token=os.environ[token_env],
                            path_prefix=str(
                                upload.get("path_prefix", "qwen3-1.7b-verpo-zpd")
                            ),
                            revision=str(upload.get("revision", "master")),
                            max_workers=int(upload.get("max_workers", 8)),
                            audit_dir=upload.get(
                                "audit_dir",
                                str(
                                    Path(self.config.trainer.default_local_dir).resolve().parent
                                    / "modelscope_uploads"
                                ),
                            ),
                            validation_metrics=val_metrics,
                        )
                        metrics["modelscope_upload/completed"] = 1.0
                        metrics["modelscope_upload/file_count"] = upload_result["file_count"]

                retention = self.config.trainer.get("best_checkpoint_retention", {})
                keep_best = int(retention.get("keep", 0))
                async_upload = bool(upload.get("enabled", False)) and bool(
                    upload.get("async_enabled", False)
                )
                if checkpoint_saved and keep_best > 0 and not async_upload:
                    retention_result = retain_best_verpo_contrastive_checkpoints(
                        self.config.trainer.default_local_dir,
                        step=self.global_steps,
                        metrics=val_metrics,
                        keep_best=keep_best,
                        keep_current=bool(retention.get("keep_current", True)),
                        terminal=is_last_step,
                    )
                    metrics["checkpoint_retention/selection_score"] = retention_result["history"][-1]["score"]
                    metrics["checkpoint_retention/retained_count"] = len(retention_result["retained_steps"])

            # 5. record metrics
            self._compute_metrics(batch, metrics, self.timing_raw, global_steps=self.global_steps, epoch=current_epoch)

            # 6. dump rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                self._log_rollout_data(batch, self.timing_raw, rollout_data_dir)

            # 7. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

            logging_freq = int(self.config.trainer.get("logging_freq", 1))
            if self.global_steps % logging_freq == 0 or is_last_step:
                self.logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)
            self.global_steps += 1
            current_epoch = (self.global_steps - 1) // len(self.train_dataloader)
            if is_last_step:
                try:
                    self._shutdown_modelscope_upload_queue()
                finally:
                    self._shutdown_dump_executor()
                    progress_bar.close()
                pprint(f"Final validation metrics: {last_val_metrics}")
                return

        try:
            self._shutdown_modelscope_upload_queue()
        finally:
            # Ensure dump executor is shut down when training loop ends without
            # reaching is_last_step, even if terminal upload drain fails.
            self._shutdown_dump_executor()

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        filter_config = self.config.algorithm.get("filter_groups", None)
        filter_enabled = bool(filter_config and filter_config.get("enable", False))
        if filter_enabled:
            with marked_timer("gen", timing_raw, color="red"):
                batch = self._sample_reward_variable_batch(metrics)
        else:
            # 1. add batch to generate
            self._add_batch_to_generate()

            # 2. sample batch from replay buffer
            with marked_timer("gen", timing_raw, color="red"):
                self.on_sample_begin()
                batch, off_policy_metrics = self.replay_buffer.sample(
                    global_steps=self.global_steps,
                    partition_id="train",
                    batch_size=self.config.data.train_batch_size,
                )
                metrics.update(off_policy_metrics)
                batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                self.on_sample_end()

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                with marked_timer("reward", timing_raw, color="yellow"):
                    batch = self._compute_reward_colocate(batch, metrics=metrics)

        # 4. balance batch across data parallel groups
        batch = self._balance_batch(batch, metrics=metrics)

        # 5. compute old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] compute ref_log_prob
        if self.use_reference_policy and not self._is_verpo_enabled():
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        # 7. [OPTIONAL] compute critic values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        # 8. compute advantage and return
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        # 9. [OPTIONAL] update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        # 10. update actor
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)

        return batch

    def _sample_reward_variable_batch(self, metrics: dict) -> KVBatchMeta:
        """Regenerate constant-reward groups until one optimizer batch is full."""
        config = self.config.algorithm.filter_groups
        metric = str(config.get("metric", "seq_final_reward"))
        if metric != "seq_final_reward":
            raise ValueError(
                "The beta-OPSD dynamic group filter must use metric=seq_final_reward; "
                f"got {metric!r}"
            )
        min_range = float(config.get("min_reward_range", 0.0))
        max_batches = int(config.get("max_num_gen_batches", 0))
        target_groups = int(self.config.data.train_batch_size)
        selected_keys: list[str] = []
        selected_tags: list[dict[str, Any]] = []
        accepted_groups = 0
        generated_groups = 0
        constant_groups = 0
        excess_groups = 0
        num_gen_batches = 0

        while accepted_groups < target_groups:
            num_gen_batches += 1
            if max_batches > 0 and num_gen_batches > max_batches:
                raise RuntimeError(
                    "Dynamic final-reward group filtering exhausted "
                    f"max_num_gen_batches={max_batches}: accepted "
                    f"{accepted_groups}/{target_groups} groups"
                )
            self._add_batch_to_generate()
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=target_groups,
            )
            metrics.update(off_policy_metrics)
            self.on_sample_end()
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                batch = self._compute_reward_colocate(batch, metrics=metrics)

            reward_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["rm_scores"],
            )
            sequence_rewards = (
                reward_data["rm_scores"].to_padded_tensor(padding=0).sum(dim=-1).detach().cpu().tolist()
            )
            remaining = target_groups - accepted_groups
            kept, dropped, summary = select_reward_variable_groups(
                list(batch.keys),
                sequence_rewards,
                min_reward_range=min_range,
                max_groups=remaining,
                expected_group_size=int(self.config.actor_rollout_ref.rollout.n),
            )
            tag_by_key = dict(zip(batch.keys, batch.tags, strict=True))
            selected_keys.extend(kept)
            selected_tags.extend(tag_by_key[key] for key in kept)
            if dropped:
                tq.kv_clear(partition_id=batch.partition_id, keys=dropped)
            accepted_groups += summary["accepted_groups"]
            generated_groups += summary["generated_groups"]
            constant_groups += summary["constant_groups"]
            excess_groups += summary["excess_variable_groups"]

        metrics.update(
            {
                "dynamic_group_filter/gen_batches": float(num_gen_batches),
                "dynamic_group_filter/generated_groups": float(generated_groups),
                "dynamic_group_filter/accepted_groups": float(accepted_groups),
                "dynamic_group_filter/constant_groups": float(constant_groups),
                "dynamic_group_filter/excess_variable_groups": float(excess_groups),
                "dynamic_group_filter/acceptance_rate": float(accepted_groups / max(generated_groups, 1)),
                "dynamic_group_filter/min_reward_range": min_range,
            }
        )
        result = KVBatchMeta(partition_id="train", keys=selected_keys, tags=selected_tags)
        result.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return result

    # ------------------------------ abstract methods ------------------------------

    def on_init_end(self):
        """Called after the initialization ends."""
        return

    def on_train_begin(self):
        """Called before the training loop starts."""
        return

    def on_train_end(self):
        """Called after the training loop ends."""
        return

    def on_validate_begin(self):
        """Called before the validation loop starts."""
        return

    def on_validate_end(self):
        """Called after the validation loop ends."""
        return

    def on_step_begin(self):
        """Called at the beginning of each training step."""
        return

    @abstractmethod
    def on_step_end(self):
        """Called at the end of each training step."""
        return

    def on_sample_begin(self):
        """Called at the beginning of sampling batch from replay buffer."""
        return

    @abstractmethod
    def on_sample_end(self):
        """Called after sampling a batch from replay buffer."""
        return

    # ------------------------------ common methods ------------------------------

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            self.config.actor_rollout_ref.model.path, use_shm=self.config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    def _init_dataloader(self):
        """Initialize train and validate dataloader."""
        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.train_dataloader_it = None
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size or len(self.val_dataset),
            num_workers=self.config.data["dataloader_num_workers"],
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        logger.info(
            f"train and validate dataloader initialized, train dataset size: "
            f"{len(self.train_dataset)}, val dataset size: {len(self.val_dataset)}"
        )

        # adjust total_training_steps
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            logger.warning(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _init_resource_pool_mgr(self):
        config = self.config
        # role => worker class
        self.role_worker_mapping = {}
        # role => resource pool
        self.mapping = {}

        # Add actor rollout worker to mapping
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        verpo_enabled = bool(config.actor_rollout_ref.actor.get("verpo", {}).get("enabled", False))
        role = (
            Role.ActorRolloutRef
            if need_reference_policy(config) and (verpo_enabled or not ref_in_actor)
            else Role.ActorRollout
        )
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"

        # Add critic worker to mapping.
        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

        # Global resource pool is used for actor, rollout, critic, ref
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # Add separate resource pool for reward model if enabled
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool
            self.mapping[Role.TeacherModel] = "teacher_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def _load_checkpoint(self):
        self.global_steps = 0

        # 1. find latest checkpoint folder
        if self.config.trainer.resume_mode == "disable":
            return
        elif self.config.trainer.resume_mode == "auto":
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            if global_step_folder is None:
                logger.info("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            logger.exception(f"Unknown resume mode {self.config.trainer.resume_mode}")

        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        logger.info(f"Resuming from {global_step_folder}, setting global step to {self.global_steps}")

        # 2. load actor checkpoint
        self.actor_rollout_wg.load_checkpoint(
            local_path=os.path.join(global_step_folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # 3. load critic checkpoint
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, str(Role.Critic)),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # 4. load dataloader checkpoint
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logger.warning(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _save_checkpoint(self):
        """Save actor, critic, and dataloader checkpoints to local (and optionally remote) storage."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        logger.info(f"Saving checkpoint to {local_global_step_folder}")

        # resolve max checkpoints to keep
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            logger.warning(
                "remove_previous_ckpt_in_save is deprecated, "
                "set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # save actor
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # save critic
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader state
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        # write latest checkpointed iteration tracker for atomic resume
        actor_ckpt_cfg = self.config.actor_rollout_ref.actor.get("checkpoint", {})
        if actor_ckpt_cfg.get("async_save", False):
            logger.info("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _validate(self) -> dict[str, float]:
        # Lists to collect samples for the table
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        session_to_sample_idx: dict[str, int] = {}

        for batch_dict in self.val_dataloader:
            # 1. put batch to agent loop manager
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            # Register each prompt (GRPO group) in TransferQueue as a tag-only status marker.
            # global_steps is required by ReplayBuffer's metadata sync / staleness ordering.
            tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps}] * len(batch)
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=tags)
            self.agent_loop_manager.generate_sequences(batch)

            # 2. sample batch from replay buffer: one prompt (GRPO group) per submitted row.
            batch, _ = self.replay_buffer.sample(
                global_steps=self.global_steps, partition_id="val", batch_size=len(batch)
            )

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()

            # 4. collect necessary data for logging
            # For multi-output agent loops, only use the final output per session for metrics.
            # Keys have format {uid}_{session_id}_{index}; keep only the highest index per session.
            session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
            for pos, key in enumerate(batch.keys):
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    session_key = f"{parts[0]}_{parts[1]}"
                    index = int(parts[2])
                    if session_key not in session_max or index > session_max[session_key][0]:
                        session_max[session_key] = (index, pos)
                else:
                    session_max[key] = (0, pos)
            sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
            final_indices = [pos for _, (_, pos) in sorted_sessions]
            final_keys = [batch.keys[i] for i in final_indices]
            base_offset = len(sample_scores)
            session_to_sample_idx.update(
                {session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)}
            )

            text_data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
            )
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            all_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
            all_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

            fields = ["uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]
            data = tq.kv_batch_get(keys=final_keys, partition_id=batch.partition_id, select_fields=fields)

            sample_uids.extend(data.pop("uid").tolist())
            sample_outputs.extend(all_outputs[i] for i in final_indices)
            sample_inputs.extend(all_inputs[i] for i in final_indices)
            scores = data["rm_scores"].sum(dim=1).tolist()
            sample_scores.extend(scores)
            sample_turns.extend(data.pop("num_turns").tolist())
            reward_extra_infos_dict["reward"].extend(scores)

            extra_fields_list = data.pop("extra_fields", None)
            if extra_fields_list is not None:
                n_prior = len(reward_extra_infos_dict["reward"]) - len(extra_fields_list.tolist())
                for extra_field in extra_fields_list.tolist():
                    reward_extra_info = (
                        extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                    )
                    for key in reward_extra_infos_dict:
                        if key != "reward" and key not in reward_extra_info:
                            reward_extra_infos_dict[key].append(None)
                    for key, value in reward_extra_info.items():
                        if key not in reward_extra_infos_dict:
                            reward_extra_infos_dict[key] = [None] * n_prior
                        reward_extra_infos_dict[key].append(value)
                    n_prior += 1

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                sample_gts.extend([item.get("ground_truth", None) for item in reward_model.tolist()])
            else:
                sample_gts.extend([None] * len(final_indices))

            data_source = data.pop("data_source", None)
            if data_source is not None:
                data_sources.extend(data_source.tolist())
            else:
                data_sources.extend(["unknown"] * len(final_indices))

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(batch.keys)

            # 5. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

        # logger to wandb
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump to local dir
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                parts = key.rsplit("_", 2)
                sort_keys.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 else (key, 0, 0))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values in the
            # lists for the final samples of each session
            dump_all_sessions = [
                f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key
                for key in dump_all_keys
                for parts in [key.rsplit("_", 2)]
            ]
            session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
            self._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in session_final_indices],
                scores=[sample_scores[i] for i in session_final_indices],
                reward_extra_infos_dict={
                    k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()
                }
                | {
                    "uid": dump_all_keys,
                    "data_source": [data_sources[i] for i in session_final_indices],
                },
                dump_path=val_data_dir,
                split_by_data_source=True,
                expected_data_sources=self.config.trainer.get(
                    "expected_validation_data_sources",
                    ["amc23", "aime24", "aime25"],
                ),
            )

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    @staticmethod
    def _write_generations(
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        global_steps,
        split_by_data_source=False,
        expected_data_sources=None,
    ):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        data_sources = base_data.get("data_source", ["unknown"] * n)
        prompt_record_ids = [
            hashlib.sha256(
                (
                    f"{data_sources[index]}\0{inputs[index]}\0"
                    + json.dumps(
                        gts[index],
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    )
                ).encode("utf-8")
            ).hexdigest()[:24]
            for index in range(n)
        ]
        occurrence: dict[str, int] = defaultdict(int)
        sample_record_ids = []
        for prompt_record_id in prompt_record_ids:
            rollout_index = occurrence[prompt_record_id]
            occurrence[prompt_record_id] += 1
            sample_record_ids.append(f"{prompt_record_id}:{rollout_index:02d}")
        base_data["prompt_record_id"] = prompt_record_ids
        base_data["record_id"] = sample_record_ids

        def json_encode_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=json_encode_default) + "\n")

        print(f"Dumped generations to {filename}")

        if split_by_data_source:
            data_sources = base_data.get("data_source")
            if data_sources is None:
                raise RuntimeError(
                    "Per-dataset validation output requires data_source for every row"
                )
            by_source: dict[str, list[int]] = defaultdict(list)
            for index, source in enumerate(data_sources):
                by_source[str(source).lower()].append(index)
            expected_sources = {
                str(source).lower()
                for source in (
                    expected_data_sources
                    if expected_data_sources is not None
                    else ("amc23", "aime24", "aime25")
                )
            }
            if set(by_source) != expected_sources:
                raise RuntimeError(
                    f"Validation datasets must be exactly {sorted(expected_sources)}; "
                    f"found {sorted(by_source)}"
                )
            metrics = {"step": global_steps, "datasets": {}}
            for source in sorted(by_source):
                indices = by_source[source]
                source_dir = os.path.join(dump_path, "by_dataset", source)
                os.makedirs(source_dir, exist_ok=True)
                source_filename = os.path.join(source_dir, f"{global_steps}.jsonl")
                with open(source_filename, "w") as f:
                    for index in indices:
                        entry = {k: v[index] for k, v in base_data.items()}
                        f.write(
                            json.dumps(
                                entry,
                                ensure_ascii=False,
                                default=json_encode_default,
                            )
                            + "\n"
                        )
                accuracy_values = base_data.get("acc", scores)
                format_values = base_data.get("formatted")
                metrics["datasets"][source] = {
                    "count": len(indices),
                    "accuracy_mean": float(
                        np.mean([float(accuracy_values[index]) for index in indices])
                    ),
                    "reward_mean": float(
                        np.mean([float(scores[index]) for index in indices])
                    ),
                    "format_validity_mean": (
                        float(
                            np.mean(
                                [float(format_values[index]) for index in indices]
                            )
                        )
                        if format_values is not None
                        else None
                    ),
                    "prediction_file": source_filename,
                }
            metrics_dir = os.path.join(dump_path, "by_dataset")
            metrics_filename = os.path.join(
                metrics_dir,
                f"{global_steps}.metrics.json",
            )
            with open(metrics_filename, "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"Dumped per-dataset validation metrics to {metrics_filename}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        split_by_data_source=False,
        expected_data_sources=None,
    ):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
            split_by_data_source,
            expected_data_sources,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _get_modelscope_upload_queue(self) -> AsyncModelScopeCheckpointUploader:
        if self._modelscope_upload_queue is not None:
            return self._modelscope_upload_queue
        upload = self.config.trainer.get("modelscope_upload", {})
        token_env = str(upload.get("token_env", "MODELSCOPE_TOKEN"))
        if token_env not in os.environ:
            raise RuntimeError(
                f"ModelScope token environment variable is missing: {token_env}"
            )
        retention = self.config.trainer.get("best_checkpoint_retention", {})
        self._modelscope_upload_queue = AsyncModelScopeCheckpointUploader(
            self.config.trainer.default_local_dir,
            repo_id=str(upload["repo_id"]),
            experiment_id=str(upload["experiment_id"]),
            token=os.environ[token_env],
            path_prefix=str(upload.get("path_prefix", "qwen3-1.7b-verpo-zpd")),
            revision=str(upload.get("revision", "master")),
            max_workers=int(upload.get("max_workers", 8)),
            audit_dir=upload.get(
                "audit_dir",
                str(
                    Path(self.config.trainer.default_local_dir).resolve().parent
                    / "modelscope_uploads"
                ),
            ),
            retention_keep_best=int(retention.get("keep", 0)),
            retention_keep_current=bool(retention.get("keep_current", True)),
        )
        return self._modelscope_upload_queue

    def _shutdown_modelscope_upload_queue(self) -> None:
        """Terminally drain uploads; this is the only intentional wait point."""

        if self._modelscope_upload_queue is None:
            return
        try:
            self._modelscope_upload_queue.close()
        finally:
            self._modelscope_upload_queue = None

    def _drain_dump_futures(self):
        """Wait for pending generation artifacts without closing the executor."""
        for future in self._dump_futures:
            future.result()
        self._dump_futures.clear()

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        self._drain_dump_futures()
        self._dump_executor.shutdown(wait=True)

    def _log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        """Fetch rollout data from TransferQueue and dump sorted by uid."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            fields = [
                "uid",
                "prompts",
                "responses",
                "rm_scores",
                "reward_model",
                "data_source",
                "extra_fields",
            ]
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
            data["prompts"] = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            data["responses"] = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

            uids = data.pop("uid").tolist()
            inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["prompts"]]
            outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["responses"]]
            scores = data["rm_scores"].sum(dim=1).tolist()

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                gts = [item.get("ground_truth", None) for item in reward_model.tolist()]
            else:
                gts = [None] * len(uids)
            data_source = data.pop("data_source", None)
            data_sources = data_source.tolist() if data_source is not None else ["unknown"] * len(uids)
            extra_fields = data.pop("extra_fields", None)
            reward_infos = []
            for item in extra_fields.tolist() if extra_fields is not None else [None] * len(uids):
                reward_infos.append(
                    item.get("reward_extra_info", {}) if isinstance(item, dict) else {}
                )
            if any("acc" not in info for info in reward_infos):
                raise RuntimeError(
                    "Reward-ranked rollout persistence requires reward_extra_info.acc"
                )

            # Sort by uid key ({sample}_{rollout}_{output})
            sort_keys = []
            for key in batch.keys:
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    sort_keys.append((parts[0], int(parts[1]), int(parts[2])))
                else:
                    sort_keys.append((key, 0, 0))
            sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

            inputs = [inputs[i] for i in sorted_indices]
            outputs = [outputs[i] for i in sorted_indices]
            gts = [gts[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            data_sources = [data_sources[i] for i in sorted_indices]
            reward_infos = [reward_infos[i] for i in sorted_indices]

            reward_extra_infos_dict = {
                "uid": [batch.keys[i] for i in sorted_indices],
                "data_source": data_sources,
                "acc": [info["acc"] for info in reward_infos],
                "formatted": [info.get("formatted") for info in reward_infos],
                "response_length": [info.get("response_length") for info in reward_infos],
            }

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
            )

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns) -> dict[str, float]:
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.array(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _start_profiling(self) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        do_profile = (
            not self.prev_step_profile and self.curr_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )

        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        self.next_step_profile = (
            self.global_steps + 1 in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        do_profile = (
            self.curr_step_profile and not self.next_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )
        self.prev_step_profile = self.curr_step_profile
        self.curr_step_profile = self.next_step_profile

        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _add_batch_to_generate(self):
        """Sample a batch from dataloader and add to AgentLoopManager."""
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)

        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)

        # Register each prompt (GRPO group) in TransferQueue as a tag-only status marker
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps}] * len(batch)
        tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)

        # add batch to agent loop manager
        self.agent_loop_manager.generate_sequences(batch)

    def _compute_reward_colocate(self, batch: KVBatchMeta, metrics: dict | None = None) -> KVBatchMeta:
        """Compute the reward score with a colocated reward model."""
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"

        # 1. read the fields required by the reward model from TransferQueue.
        fields = ["prompts", "responses", "raw_prompt"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        prompt_lengths = data["prompts"].offsets().diff()
        response_lengths = data["responses"].offsets().diff()
        prompts = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
        responses = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

        # 2. rebuild the attention mask aligned with the [prompts | responses] layout.
        prompt_mask = self._lengths_to_mask(prompt_lengths, prompts.size(1))
        response_mask = self._lengths_to_mask(response_lengths, responses.size(1))
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

        # `raw_prompt` is a non-tensor field; depending on the TransferQueue backend it
        # comes back as a tensordict LinkedList (a `list` subclass), a NonTensorStack or a
        # numpy array. `list(...)` normalizes all of them to a plain list where each element
        # is one sample's chat-message list (whereas `.tolist()` only exists on numpy/tensors).
        raw_prompts = list(data["raw_prompt"])
        raw_prompt_arr = np.empty(len(raw_prompts), dtype=object)
        raw_prompt_arr[:] = raw_prompts

        rm_input = DataProto(
            batch=TensorDict(
                {"prompts": prompts, "responses": responses, "attention_mask": attention_mask},
                batch_size=len(batch),
            ),
            non_tensor_batch={"raw_prompt": raw_prompt_arr},
        )

        # 3. run the reward model (wakes/sleeps the reward model internally).
        rm_output = self.reward_loop_manager.compute_rm_score(rm_input)

        # 4. write rm_scores (and reward extra info) back to TransferQueue.
        padded_rm_scores = rm_output.batch["rm_scores"]
        rm_scores = torch.nested.as_nested_tensor(
            [padded_rm_scores[i, : response_lengths[i]] for i in range(len(batch))],
            layout=torch.jagged,
        )
        write_back = {"rm_scores": rm_scores}
        for key in rm_output.meta_info.get("reward_extra_keys", []):
            write_back[key] = rm_output.non_tensor_batch[key]
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=tu.get_tensordict(write_back),
        )

        return batch

    @staticmethod
    def _lengths_to_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
        """Build a right-padded mask of shape (len(lengths), width) from per-row valid lengths."""
        positions = torch.arange(width, device=lengths.device).unsqueeze(0)
        return (positions < lengths.unsqueeze(1)).to(torch.int64)

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        """Return the global batch multiple required by downstream train steps(e.g. critics, actors)."""
        required_multiple = dp_size

        # If enabled with critic training, the batch should align with critic PPO mini-batches.
        if self.use_critic:
            critic_global_mini_batch_size = self.config.critic.ppo_mini_batch_size
            critic_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, critic_global_mini_batch_size)

        # If there is an actor update, the batch should align with actor PPO mini-batches too.
        if self.config.trainer.critic_warmup <= self.global_steps:
            actor_global_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            actor_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, actor_global_mini_batch_size)

        # Notice lcm(a, b, c) == lcm(lcm(a, b), c), so it is optimal.
        return required_multiple

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens."""
        # get actor dp size
        role, worker_group = "actor", self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        dp_size = max(dp_rank_mapping) + 1

        # Upsampling the batch with padding sequences
        batch_multiple = self._get_required_batch_multiple(dp_size)
        batch = upsample_batch_to_divisible_size(batch, batch_multiple, self.tokenizer.eos_token_id)
        global_seqlen_lst = torch.tensor([tag["seq_len"] for tag in batch.tags], dtype=torch.int64)
        workload_lst = calculate_workload(global_seqlen_lst)

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        batch.reorder([j for partition in global_partition_lst for j in partition])
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return batch

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the old log prob of the batch."""
        # Operating Mode Selection:
        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
            data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"]
            )
            data["old_log_probs"] = data.pop("rollout_log_probs")
            tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
            return batch

        # 1. compute log probs
        batch.extra_info.update(
            {
                "calculate_entropy": True,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )

        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
        assert len(output) == len(batch)

        fields = ["entropy", "log_probs", "response_mask"]
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.extend(["responses", "rollout_log_probs"])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        # 2. write old_log_probs and entropy back to TransferQueue
        data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
        batch = tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")
        )

        data = DataProto(batch=data.to_padded_tensor())

        # 3. calculate actor entroy metrics
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=data.batch["entropy"],
            loss_mask=data.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
            # "perf/mfu/actor_infer": old_log_prob_mfu,
        }
        metrics.update(old_log_prob_metrics)

        # 4. calculate rollout vs actor logprobs diff
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(calculate_debug_metrics(data))

        return batch

    def _compute_ref_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reference log prob of the batch."""
        # 1. compute log probs
        metadata = {
            "calculate_entropy": False,
            "compute_loss": False,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        batch.extra_info.update(metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch)
        assert len(output) == len(batch)

        # 2. write ref_log_prob and entropy back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["log_probs", "response_mask"]
        )
        data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("ref_log_prob"))

        return batch

    def _compute_values(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the values of the batch."""
        # 1. compute value
        output = self.critic_wg.infer_batch(batch)
        # TODO: DataProtoFuture support KVBatchMeta
        ray.get(output.futures)

        # 2. write value back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["values", "response_mask"]
        )
        data["values"] = response_from_nested(data.pop("values"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("values"))

        return batch

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the advantage of the batch."""
        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        response_mask = data["response_mask"]
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)

        # 1. apply kl penalty to rewards
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # 2. Compute rollout correction: IS weights, rejection sampling, and metrics
        # Only runs in decoupled mode (computes once per batch using stable π_old)
        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        # 3. compute advantages
        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        # 4. write nested advantages and returns back to TransferQueue
        fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            fields.append("token_level_rewards")
        if rollout_correction:
            fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                fields.append("rollout_is_weights")

        output = {}
        for field in fields:
            output[field] = response_to_nested(data.batch[field], response_mask)
        output = TensorDict(output, batch_size=len(batch))

        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

        return batch

    def _update_critic(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the critic network."""
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        extra_info = {
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.critic.ppo_epochs,
            "seed": self.config.critic.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.critic.shuffle},
        }
        batch.extra_info.update(extra_info)

        output: DataProtoFuture = self.critic_wg.train_mini_batch(batch)
        output: TensorDict = output.get()
        output = rename_dict(output["metrics"], "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_metrics = reduce_metrics(output)
        metrics.update(critic_metrics)

        return batch

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the actor network."""
        if self._is_verpo_enabled():
            batch = self._prepare_verpo_teacher_inputs(batch, metrics)
        elif self._is_paper_baseline_enabled():
            batch = self._prepare_paper_baseline_teacher_inputs(batch, metrics)
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        extra_info = {
            "calculate_entropy": calculate_entropy,
            "distillation_use_topk": distillation_use_topk,
            "verpo_use_full_logits": self._is_verpo_enabled(),
            "actor_loss_use_full_logits": self._is_paper_baseline_enabled(),
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
            "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        batch.extra_info.update(extra_info)

        output: TensorDict = self.actor_rollout_wg.update_actor(batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        actor_metrics = reduce_metrics(output)
        metrics.update(actor_metrics)
        for key, value in actor_metrics.items():
            if key.startswith("actor/verpo/") or key.startswith("actor/verpo_zpd/"):
                metrics.setdefault(key.removeprefix("actor/"), value)
            if key.startswith("actor/paper_baseline/"):
                metrics.setdefault(key.removeprefix("actor/"), value)

        return batch

    def _is_verpo_enabled(self) -> bool:
        verpo = self.config.actor_rollout_ref.actor.get("verpo", {})
        return bool(verpo.get("enabled", False))

    def _is_paper_baseline_enabled(self) -> bool:
        baseline = self.config.actor_rollout_ref.actor.get("paper_baseline", {})
        return bool(baseline.get("enabled", False))

    def _prepare_paper_baseline_teacher_inputs(
        self, batch: KVBatchMeta, metrics: dict
    ) -> KVBatchMeta:
        """Materialize the independent SDPO/SRPO sibling replay batch."""

        baseline = self.config.actor_rollout_ref.actor.paper_baseline
        custom = self.config.actor_rollout_ref.rollout.custom or {}
        rollout_temperature = float(self.config.actor_rollout_ref.rollout.temperature)
        if abs(rollout_temperature - float(baseline.temperature)) > 1e-8:
            raise ValueError(
                "paper SDPO/SRPO requires rollout.temperature == "
                "actor.paper_baseline.temperature"
            )
        teacher_chat_template_kwargs = dict(
            custom.get("teacher_chat_template_kwargs", {}) or {}
        )
        fields = ["responses", "response_mask", "rm_scores", "raw_prompt"]
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        rewards = torch.tensor(
            [float(row.sum().item()) for row in data["rm_scores"].unbind()],
            dtype=torch.float32,
        )
        group_ids = rollout_group_ids(batch.keys)
        routing = build_paper_routing(
            rewards,
            group_ids,
            objective=str(baseline.objective),
            success_reward_threshold=float(baseline.success_reward_threshold),
        )
        teacher_fields = build_paper_teacher_fields(
            tokenizer=self.tokenizer,
            responses=data["responses"],
            raw_prompts=data["raw_prompt"],
            routing=routing,
            total_token_budget=int(baseline.teacher_max_token_len_per_gpu),
            max_reprompt_tokens=int(baseline.max_reprompt_len),
            chat_template_kwargs=teacher_chat_template_kwargs,
            remove_thinking_from_demonstration=bool(
                baseline.remove_thinking_from_demonstration
            ),
        )
        teacher_data = TensorDict(teacher_fields, batch_size=len(batch))
        batch = tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=teacher_data,
        )
        metrics["paper_baseline/reward_mean"] = float(rewards.mean().item())
        metrics["paper_baseline/correct_fraction"] = float(
            routing.correct.float().mean().item()
        )
        metrics["paper_baseline/teacher_available_fraction"] = float(
            routing.teacher_available.float().mean().item()
        )
        metrics["paper_baseline/sdpo_route_fraction"] = float(
            routing.sdpo_route.float().mean().item()
        )
        metrics["paper_baseline/grpo_route_fraction"] = float(
            routing.grpo_route.float().mean().item()
        )
        metrics["paper_baseline/teacher_prompt_truncated_fraction"] = float(
            teacher_fields["paper_teacher_prompt_truncated"].float().mean().item()
        )
        metrics["paper_baseline/teacher_prompt_truncated_tokens_mean"] = float(
            teacher_fields["paper_teacher_prompt_truncated_tokens"].mean().item()
        )
        return batch

    def _prepare_verpo_teacher_inputs(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Materialize fixed or sibling-contrast Teacher prompts with the sampled suffix."""
        verpo = self.config.actor_rollout_ref.actor.verpo
        custom = self.config.actor_rollout_ref.rollout.custom or {}
        displacement_mode = str(verpo.displacement_mode)
        if str(verpo.evidence_source) != "rollout_group":
            raise ValueError("public VERPO requires evidence_source=rollout_group")
        if bool(custom.get("thinking_system_prompt", False)):
            raise ValueError("The VERPO_CONTRASTIVE protocol requires thinking_system_prompt=false")
        teacher_chat_template_kwargs = dict(custom.get("teacher_chat_template_kwargs", {}) or {})
        legacy_teacher_thinking = custom.get("teacher_enable_thinking")
        if legacy_teacher_thinking is not None:
            teacher_chat_template_kwargs.setdefault("enable_thinking", bool(legacy_teacher_thinking))
        rollout_temperature = float(self.config.actor_rollout_ref.rollout.temperature)
        if abs(rollout_temperature - float(verpo.temperature)) > 1e-8:
            raise ValueError(
                "VERPO requires rollout.temperature == actor.verpo.temperature; "
                f"got {rollout_temperature} and {verpo.temperature}"
            )
        fields = [
            "responses",
            "response_mask",
            "rm_scores",
            "uid",
            "raw_prompt",
            "reward_model",
            "extra_info",
            "extra_fields",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        rewards = torch.tensor(
            [float(row.sum().item()) for row in data["rm_scores"].unbind()],
            dtype=torch.float32,
        )
        # TransferQueue row keys are {prompt_uid}_{session_id}_{output_index}.
        # VERPO must group sibling sessions by the original prompt UID, not by
        # the unique per-rollout key stored in the uid field.
        uids = []
        for key in batch.keys:
            parts = str(key).rsplit("_", 2)
            if len(parts) != 3:
                raise ValueError(f"Unexpected rollout key format for VERPO grouping: {key}")
            uids.append(parts[0])
        reward_group_state = classify_reward_ranked_groups(
            rewards,
            group_ids=uids,
        )
        group_zpd_mode = str(verpo.group_zpd_mode)
        sibling_selection_mode = str(verpo.sibling_selection_mode)
        evidence_rollout_scope = str(verpo.evidence_rollout_scope)
        require_independent_acc = (
            (bool(verpo.group_zpd_enabled) and group_zpd_mode == "binary_mixed")
            or (sibling_selection_mode == "correctness")
            or evidence_rollout_scope == "wrong_only"
        )
        outcome_positive = _resolve_verpo_outcome_positive(
            rewards,
            data.get("extra_fields"),
            require_independent_acc=require_independent_acc,
        )
        if bool(verpo.group_zpd_enabled):
            group_gate_values = outcome_positive.float() if group_zpd_mode == "binary_mixed" else rewards
            raw_group_gate = compute_group_zpd_gate_by_uid(
                group_gate_values,
                uids,
                mode=group_zpd_mode,
                epsilon=float(verpo.group_zpd_epsilon),
            )
        else:
            raw_group_gate = torch.ones_like(rewards, dtype=torch.bool)
        teacher_fields = build_contrastive_evidence_teacher_fields(
            tokenizer=self.tokenizer,
            responses=data["responses"],
            raw_prompts=data["raw_prompt"],
            rewards=rewards,
            correctness=outcome_positive,
            uids=uids,
            rollout_ids=[str(key) for key in batch.keys],
            require_negative=displacement_mode != "evidence_vs_none",
            total_token_budget=int(verpo.teacher_max_token_len_per_gpu),
            max_reprompt_tokens=int(verpo.teacher_max_reprompt_len),
            reward_models=data["reward_model"],
            num_negative_hints=int(verpo.contrastive_num_negative_hints),
            selection_mode=sibling_selection_mode,
            chat_template_kwargs=teacher_chat_template_kwargs,
            allow_unboxed_candidates=bool(verpo.smoke_allow_unboxed_contrastive),
        )
        contrastive_available = teacher_fields["verpo_contrastive_available"]
        audit_root = self.config.trainer.get("rollout_data_dir")
        if audit_root:
            from risk_aware_opsd.rollout_evidence import REASONS

            audit_path = Path(audit_root) / "evidence_selection"
            audit_path.mkdir(parents=True, exist_ok=True)
            with (audit_path / f"step_{self.global_steps}.jsonl").open("w", encoding="utf-8") as handle:
                for row, target_id in enumerate(batch.keys):
                    pos = int(teacher_fields["verpo_positive_sibling_index"][row])
                    negs = teacher_fields["verpo_negative_sibling_indices"][row].tolist()
                    selected = [pos] + negs
                    record = {
                        "evidence_source": "rollout_group",
                        "trainer_step": self.global_steps,
                        "target_id": str(target_id),
                        "group_id": uids[row],
                        "target_correct": bool(outcome_positive[row]),
                        "target_format_valid": bool(teacher_fields["verpo_candidate_format_valid"][row]),
                        "positive_id": str(batch.keys[pos]) if pos >= 0 else None,
                        "negative_ids": [str(batch.keys[i]) if i >= 0 else None for i in negs],
                        "selected_correct": [bool(outcome_positive[i]) if i >= 0 else None for i in selected],
                        "selected_format_valid": [
                            bool(teacher_fields["verpo_candidate_format_valid"][i]) if i >= 0 else None
                            for i in selected
                        ],
                        "available": bool(contrastive_available[row]),
                        "reason": REASONS[int(teacher_fields["verpo_evidence_reason"][row])],
                        "prompt_truncated_tokens": int(teacher_fields["verpo_teacher_prompt_truncated_tokens"][row]),
                    }
                    handle.write(json.dumps(record) + "\n")
        available_group_gate = raw_group_gate & contrastive_available
        evidence_rollout_gate = apply_evidence_rollout_scope(
            available_group_gate,
            outcome_positive,
            scope=evidence_rollout_scope,
        )
        response_token_counts = torch.tensor(
            [int(row.bool().sum().item()) for row in data["response_mask"].unbind()],
            dtype=torch.long,
        )
        routed_token_count = int((response_token_counts * evidence_rollout_gate.long()).sum().item())
        routed_row_count = int((evidence_rollout_gate & response_token_counts.gt(0)).sum().item())
        fec_correct_token_count = int(
            (response_token_counts * (evidence_rollout_gate & outcome_positive).long()).sum().item()
        )
        fec_wrong_token_count = int(
            (response_token_counts * (evidence_rollout_gate & ~outcome_positive).long()).sum().item()
        )

        def repeated_count(value: int) -> torch.Tensor:
            return torch.full_like(rewards, int(value), dtype=torch.long)

        teacher_fields["verpo_group_gate"] = raw_group_gate
        teacher_fields["verpo_available_group_gate"] = available_group_gate
        teacher_fields["verpo_contrastive_available"] = contrastive_available
        teacher_fields["verpo_evidence_rollout_gate"] = evidence_rollout_gate
        teacher_fields["verpo_outcome_positive"] = outcome_positive
        teacher_fields["verpo_evidence_batch_num_tokens"] = repeated_count(routed_token_count)
        teacher_fields["verpo_evidence_global_batch_size"] = repeated_count(routed_row_count)
        teacher_fields["verpo_fec_correct_token_count"] = repeated_count(fec_correct_token_count)
        teacher_fields["verpo_fec_wrong_token_count"] = repeated_count(fec_wrong_token_count)
        teacher_data = TensorDict(teacher_fields, batch_size=len(batch))
        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=teacher_data)

        truncated = teacher_fields["verpo_teacher_prompt_truncated"].float()
        group_count = len(set(uids))

        def group_fraction(mask: torch.Tensor) -> float:
            if not group_count:
                return 0.0
            tagged = {uid for uid, active in zip(uids, mask.tolist(), strict=True) if bool(active)}
            return float(len(tagged) / group_count)

        metrics["verpo/group_gate_rate"] = float(raw_group_gate.float().mean().item())
        metrics["verpo/available_group_gate_rate"] = float(available_group_gate.float().mean().item())
        metrics["verpo/evidence_rollout_gate_rate"] = float(evidence_rollout_gate.float().mean().item())
        metrics["verpo/contrastive_available_rate"] = float(contrastive_available.float().mean().item())
        metrics["verpo/policy_reward_mean"] = float(rewards.mean().item())
        metrics["verpo/policy_reward_min"] = float(rewards.min().item())
        metrics["verpo/policy_reward_max"] = float(rewards.max().item())
        metrics["verpo/accuracy_mean"] = float(outcome_positive.float().mean().item())
        if "verpo_positive_sibling_index" in teacher_fields:
            positive_indices = teacher_fields["verpo_positive_sibling_index"]
            negative_indices = teacher_fields["verpo_negative_sibling_indices"][:, 0]
            valid_indices = contrastive_available & positive_indices.ge(0) & negative_indices.ge(0)
            if bool(valid_indices.any().item()):
                preferred = rewards[positive_indices[valid_indices]]
                dispreferred = rewards[negative_indices[valid_indices]]
                metrics["verpo/preferred_reward_mean"] = float(preferred.mean().item())
                metrics["verpo/dispreferred_reward_mean"] = float(dispreferred.mean().item())
                metrics["verpo/selected_reward_gap_mean"] = float((preferred - dispreferred).mean().item())
        metrics["verpo/group_all_one_fraction"] = group_fraction(reward_group_state.all_one)
        metrics["verpo/group_all_zero_fraction"] = group_fraction(reward_group_state.all_zero)
        metrics["verpo/group_all_negative_one_fraction"] = group_fraction(reward_group_state.all_negative_one)
        metrics["verpo/group_zero_variance_fraction"] = group_fraction(reward_group_state.zero_variance)
        metrics["verpo/group_reward_ranked_zpd_fraction"] = group_fraction(reward_group_state.zpd_gate)
        grouped_outcomes: dict[str, list[bool]] = defaultdict(list)
        for uid, positive in zip(uids, outcome_positive.tolist(), strict=True):
            grouped_outcomes[uid].append(bool(positive))
        all_positive_groups = sum(all(values) for values in grouped_outcomes.values())
        all_negative_groups = sum(not any(values) for values in grouped_outcomes.values())
        mixed_groups = group_count - all_positive_groups - all_negative_groups
        metrics["verpo/group_all_positive_fraction"] = float(all_positive_groups / group_count) if group_count else 0.0
        metrics["verpo/group_all_negative_fraction"] = float(all_negative_groups / group_count) if group_count else 0.0
        metrics["verpo/group_mixed_fraction"] = float(mixed_groups / group_count) if group_count else 0.0
        metrics["verpo_zpd/group_gate_fraction"] = metrics["verpo/group_gate_rate"]
        metrics["verpo_zpd/evidence_rollout_gate_fraction"] = metrics["verpo/evidence_rollout_gate_rate"]
        metrics["verpo_zpd/contrastive_available_fraction"] = metrics["verpo/contrastive_available_rate"]
        metrics["verpo/teacher_prompt_truncated_rate"] = float(truncated.mean().item())
        metrics["verpo/teacher_prompt_truncated_tokens_mean"] = float(
            teacher_fields["verpo_teacher_prompt_truncated_tokens"].mean().item()
        )
        reference_identity = str(self.config.actor_rollout_ref.model.path)
        reference_fingerprint = hashlib.sha256(reference_identity.encode("utf-8")).hexdigest()
        teacher_mode = str(getattr(verpo, "teacher_mode", "fixed_initial"))
        if teacher_mode in {"snapshot", "ema"}:
            reference_fingerprint = "actor_moving_teacher_checkpoint_sidecar"
        metrics["verpo/group_zpd_enabled"] = float(bool(verpo.group_zpd_enabled))
        run_root = os.path.dirname(self.config.trainer.default_local_dir.rstrip(os.sep))
        os.makedirs(run_root, exist_ok=True)
        truncation_tokens = teacher_fields["verpo_teacher_prompt_truncated_tokens"].tolist()
        stats = {
            "step": int(self.global_steps),
            "qref_parameter_fingerprint": reference_fingerprint,
            "q0_parameter_fingerprint": reference_fingerprint,
            "qe_parameter_fingerprint": reference_fingerprint,
            "parameter_fingerprint_source": (
                "per_rank_actor_teacher_state" if teacher_mode in {"snapshot", "ema"} else "reference_model_path_sha256"
            ),
            "teacher_mode": teacher_mode,
            "measurement_provenance": "training_rollout",
            "weight_quantile_implementation_fingerprint": "exact_global_token_quantile_v1",
            "displacement_mode": displacement_mode,
            "group_zpd_enabled": bool(verpo.group_zpd_enabled),
            "group_zpd_mode": str(verpo.group_zpd_mode),
            "sibling_selection_mode": str(verpo.sibling_selection_mode),
            "rows": len(truncation_tokens),
            "policy_reward_mean": float(rewards.mean().item()),
            "policy_reward_min": float(rewards.min().item()),
            "policy_reward_max": float(rewards.max().item()),
            "accuracy_mean": float(outcome_positive.float().mean().item()),
            "group_gate_rows": int(raw_group_gate.sum().item()),
            "available_group_gate_rows": int(available_group_gate.sum().item()),
            "evidence_rollout_gate_rows": int(evidence_rollout_gate.sum().item()),
            "evidence_routed_tokens": routed_token_count,
            "contrastive_available_rows": int(contrastive_available.sum().item()),
            "all_one_rows": int(reward_group_state.all_one.sum().item()),
            "all_zero_rows": int(reward_group_state.all_zero.sum().item()),
            "all_negative_one_rows": int(reward_group_state.all_negative_one.sum().item()),
            "zero_variance_rows": int(reward_group_state.zero_variance.sum().item()),
            "truncated_rows": int(truncated.sum().item()),
            "truncated_tokens": int(sum(truncation_tokens)),
            "max_truncated_tokens": int(max(truncation_tokens, default=0)),
        }
        stats_path = os.path.join(run_root, "verpo_teacher_stats.jsonl")
        with open(stats_path, "a", encoding="utf-8") as handle:
            json.dump(stats, handle, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
        return batch

    def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
        # 1. collect necessary fields from TransferQueue for computing metrics
        non_padding_mask = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "values",
            "advantages",
            "returns",
            "rm_scores",
            "token_level_rewards",
            "num_turns",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        num_turns = np.array(data.pop("num_turns").tolist())
        prompt_length = data["prompts"].offsets().diff()
        response_length = data["responses"].offsets().diff()
        global_token_num = (prompt_length + response_length).tolist()
        min_global_steps = np.array([tag["min_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]
        max_global_steps = np.array([tag["max_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]

        # Only fetch speculative decoding stats when rollout writes them.
        spec_drafts = spec_accepts = spec_verifies = None
        mtp_config = getattr(self.config.actor_rollout_ref.model, "mtp", None)
        if mtp_config is not None and mtp_config.enable and mtp_config.enable_rollout:
            spec_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields = spec_data["extra_fields"].tolist()
            spec_drafts = [extra_field["spec_num_draft_tokens"] for extra_field in extra_fields]
            spec_accepts = [extra_field["spec_num_accepted_tokens"] for extra_field in extra_fields]
            spec_verifies = [extra_field["spec_num_verify_steps"] for extra_field in extra_fields]

        data = data.to_padded_tensor()
        data["token_level_scores"] = data["rm_scores"]
        if "token_level_rewards" not in data:
            data["token_level_rewards"] = data["rm_scores"]
        data["prompt_length"] = prompt_length.float()
        data["response_length"] = response_length.float()
        batch = DataProto(batch=data, meta_info={"global_token_num": global_token_num})
        metrics_batch = batch.select_idxs(non_padding_mask) if non_padding_mask.any() else batch

        # 2. compute metrics
        metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
        metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))

        # 3. other auxiliary metrics
        if non_padding_mask.any():
            num_turns = num_turns[non_padding_mask]
        metrics.update(
            {
                "training/num_turns/mean": num_turns.mean(),
                "training/num_turns/max": num_turns.max(),
                "training/num_turns/min": num_turns.min(),
            }
        )

        # 4. per-request speculative-decoding aggregation (same metrics async PPO logs;
        # see compute_spec_decode_metrics in verl/trainer/ppo/ray_trainer.py).
        metrics.update(compute_spec_decode_metrics(spec_drafts, spec_accepts, spec_verifies, non_padding_mask))

        # 5. off-policy staleness metrics
        #   - trajectory_spans: how many distinct model versions a single trajectory was
        #     generated across (1 == fully generated on a single version). This captures the
        #     within-trajectory policy inconsistency caused by partial rollout / continuation.
        #   - trajectory_staleness: how many training steps the trajectory lags behind the
        #     *current* policy. A trajectory spans versions [min_global_steps, max_global_steps],
        #     so the lag is a range: the freshest weights used give the lower bound
        #     (global_steps - max_global_steps) and the oldest weights the worst case
        #     (global_steps - min_global_steps). We log the lower bound as the primary metric.
        trajectory_spans = (max_global_steps - min_global_steps + 1) / self.parameter_sync_step
        trajectory_staleness = ((global_steps - 1) - max_global_steps) / self.parameter_sync_step
        trajectory_staleness_worst = ((global_steps - 1) - min_global_steps) / self.parameter_sync_step
        metrics.update(
            {
                "training/off_policy/trajectory_spans/mean": trajectory_spans.mean(),
                "training/off_policy/trajectory_spans/max": trajectory_spans.max(),
                "training/off_policy/trajectory_spans/min": trajectory_spans.min(),
                "training/off_policy/trajectory_staleness/mean": trajectory_staleness.mean(),
                "training/off_policy/trajectory_staleness/max": trajectory_staleness.max(),
                "training/off_policy/trajectory_staleness/min": trajectory_staleness.min(),
                "training/off_policy/trajectory_staleness_worst/mean": trajectory_staleness_worst.mean(),
                "training/off_policy/trajectory_staleness_worst/max": trajectory_staleness_worst.max(),
                "training/off_policy/trajectory_staleness_worst/min": trajectory_staleness_worst.min(),
            }
        )


TRAINER_REGISTRY: dict[str, type[PPOTrainer]] = {}


def register_trainer(name: str):
    """Class decorator that registers a :class:`PPOTrainer` subclass under ``name``."""

    def decorator(cls: type[PPOTrainer]) -> type[PPOTrainer]:
        if not (isinstance(cls, type) and issubclass(cls, PPOTrainer)):
            raise TypeError(f"register_trainer expected a PPOTrainer subclass, got {cls!r}")
        existing = TRAINER_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Trainer name '{name}' is already registered to {existing.__name__}; "
                f"cannot re-register it to {cls.__name__}."
            )
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def get_trainer_cls(name: str) -> type[PPOTrainer]:
    """Return the :class:`PPOTrainer` subclass registered under ``name``."""
    try:
        return TRAINER_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(TRAINER_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown trainer '{name}'. Available trainers: {available}.") from None
