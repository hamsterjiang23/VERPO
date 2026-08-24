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
import functools
import logging
import os
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from functools import partial
from itertools import chain
from typing import Optional

import psutil
import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh

from risk_aware_opsd.verpo_zpd import VERPO_EFFECTIVE_WEIGHT_THRESHOLD

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.distillation import distillation_ppo_loss, is_distillation_enabled
from verl.trainer.distillation.snapshot_teacher import ActorSideTeacher
from verl.trainer.distillation.verpo_gradient_audit import VerpoGradientAuditor
from verl.trainer.distillation.verpo_zpd import verpo_zpd_ppo_loss
from verl.trainer.baselines.ema_teacher import PaperBaselineEMATeacher
from verl.trainer.baselines.sdpo_srpo_loss import paper_sdpo_srpo_loss
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_torch_device, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import (
    ActorConfig,
    DistillationConfig,
    HFModelConfig,
    MtpConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.utils.losses import ppo_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _with_routing_replay_flag(enabled: bool):
    """Decorator to set 'enable_routing_replay' flag on the data TensorDict."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, data: TensorDict, *args, **kwargs):
            if self.enable_routing_replay:
                tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
            return func(self, data, *args, **kwargs)

        return wrapper

    return decorator


def _refresh_verpo_global_routing_counts(data: TensorDict, dp_group) -> None:
    """Attach per-mini-batch routed denominators before engine micro-batching."""
    if "verpo_evidence_rollout_gate" not in data:
        return
    response_mask = data["response_mask"]
    response_token_counts = torch.tensor(
        [int(row.bool().sum().item()) for row in response_mask.unbind()],
        dtype=torch.long,
    )
    evidence_gate = data["verpo_evidence_rollout_gate"].detach().cpu().bool()
    outcome_positive = data["verpo_outcome_positive"].detach().cpu().bool()
    if response_token_counts.shape != evidence_gate.shape:
        raise ValueError("VERPO response masks and rollout gates must have the same batch size")
    local_counts = torch.tensor(
        [
            int((response_token_counts * evidence_gate.long()).sum().item()),
            int((evidence_gate & response_token_counts.gt(0)).sum().item()),
            int(
                (
                    response_token_counts
                    * (evidence_gate & outcome_positive).long()
                ).sum().item()
            ),
            int(
                (
                    response_token_counts
                    * (evidence_gate & ~outcome_positive).long()
                ).sum().item()
            ),
        ],
        dtype=torch.long,
        device=get_device_id(),
    )
    torch.distributed.all_reduce(
        local_counts,
        op=torch.distributed.ReduceOp.SUM,
        group=dp_group,
    )
    global_counts = local_counts.cpu().tolist()
    for key, value in zip(
        (
            "verpo_evidence_batch_num_tokens",
            "verpo_evidence_global_batch_size",
            "verpo_fec_correct_token_count",
            "verpo_fec_wrong_token_count",
        ),
        global_counts,
        strict=True,
    ):
        data[key] = torch.full_like(
            data["verpo_evidence_rollout_gate"], int(value), dtype=torch.long
        )


def _refresh_paper_baseline_global_routing_counts(data: TensorDict, dp_group) -> None:
    """Attach the routed-token denominator for pure SDPO token-mean loss."""

    if "paper_sdpo_route" not in data:
        return
    response_mask = data["response_mask"]
    response_token_counts = torch.tensor(
        [int(row.bool().sum().item()) for row in response_mask.unbind()],
        dtype=torch.long,
        device=get_device_id(),
    )
    sdpo_route = data["paper_sdpo_route"].detach().to(
        device=get_device_id(), dtype=torch.bool
    )
    if response_token_counts.shape != sdpo_route.shape:
        raise ValueError("paper baseline response masks and routes must align")
    routed_tokens = (response_token_counts * sdpo_route.long()).sum()
    torch.distributed.all_reduce(
        routed_tokens,
        op=torch.distributed.ReduceOp.SUM,
        group=dp_group,
    )
    data["paper_sdpo_batch_num_tokens"] = torch.full_like(
        data["paper_sdpo_route"], int(routed_tokens.item()), dtype=torch.long
    )


def _flatten_metric_numbers(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_metric_numbers(item))
        return flattened
    if isinstance(value, torch.Tensor):
        return value.detach().float().reshape(-1).cpu().tolist()
    return [float(value)]


def _finalize_verpo_monitoring(metrics: dict) -> None:
    """Convert internal VERPO raw totals into TRL-compatible monitoring keys."""
    prefix = "_verpo_internal/"
    if not any(key.startswith(prefix) for key in metrics):
        return

    def total(name: str) -> float:
        values = metrics.pop(f"{prefix}{name}", [])
        return float(sum(_flatten_metric_numbers(values)))

    reference_count = total("count/reference")
    evidence_count = total("count/evidence")
    available_count = total("count/available")
    weight_count = total("count/weight")
    fec_count = total("count/fec")
    fec_correct_count = total("count/fec_correct")
    fec_wrong_count = total("count/fec_wrong")
    mixed_success_count = total("count/mixed_success")
    mixed_failure_count = total("count/mixed_failure")
    benefit_negative_count = total("count/benefit_negative")
    benefit_positive_count = total("count/benefit_positive")

    def ratio_with_count(numerator: str, count: float) -> float:
        value = total(numerator)
        return value / count if count > 0 else 0.0

    reference_loss = ratio_with_count("sum/reference_loss", reference_count)
    evidence_loss = ratio_with_count("sum/evidence_loss", evidence_count)
    selected_benefit = ratio_with_count("sum/selected_benefit", evidence_count)
    selected_cost = ratio_with_count("sum/selected_cost", evidence_count)
    weight_values = _flatten_metric_numbers(
        metrics.pop(f"{prefix}weight_values", [])
    )
    if len(weight_values) != int(round(weight_count)):
        raise ValueError(
            "VERPO weight count does not match the gathered token weights"
        )
    weight_tensor = torch.tensor(weight_values, dtype=torch.float32)
    if weight_tensor.numel() and not bool(torch.isfinite(weight_tensor).all().item()):
        raise ValueError("VERPO weight values must be finite")

    def weight_quantile(q: float) -> float:
        return (
            float(torch.quantile(weight_tensor, q).item())
            if weight_tensor.numel()
            else 0.0
        )

    finalized = {
        "verpo/reference_loss": reference_loss,
        "verpo/evidence_loss_signed": evidence_loss,
        "verpo/evidence_loss": evidence_loss,
        "verpo/lambda_ref_scaled_loss": ratio_with_count(
            "sum/scaled_reference_loss", reference_count
        ),
        "verpo/lambda_evi_scaled_loss": ratio_with_count(
            "sum/scaled_evidence_loss", evidence_count
        ),
        "verpo/reference_fkl": reference_loss,
        "verpo/reference_l1": ratio_with_count("sum/reference_l1", reference_count),
        "verpo/reference_l2": ratio_with_count("sum/reference_l2", reference_count),
        "verpo/evidence_displacement_l2": ratio_with_count(
            "sum/evidence_displacement", evidence_count
        ),
        "verpo/accepted_evidence_movement": ratio_with_count(
            "sum/accepted_movement", evidence_count
        ),
        "verpo/weight_mean": ratio_with_count("sum/weight", weight_count),
        "verpo/weight_effective_threshold": VERPO_EFFECTIVE_WEIGHT_THRESHOLD,
        "verpo/weight_median": weight_quantile(0.5),
        "verpo/weight_p10": weight_quantile(0.1),
        "verpo/weight_p90": weight_quantile(0.9),
        "verpo/weight_zero_fraction": ratio_with_count("count/weight_zero", weight_count),
        "verpo/weight_nonzero_fraction": ratio_with_count(
            "count/weight_nonzero", weight_count
        ),
        "verpo/weight_effective_coverage": ratio_with_count(
            "count/weight_effective", weight_count
        ),
        "verpo/weight_gt_05_fraction": ratio_with_count("count/weight_gt_05", weight_count),
        "verpo/weight_negative_fraction": ratio_with_count(
            "count/weight_negative", weight_count
        ),
        "verpo/weight_ge_one_fraction": ratio_with_count(
            "count/weight_ge_one", weight_count
        ),
        "verpo/benefit_mean": ratio_with_count("sum/benefit", evidence_count),
        "verpo/benefit_negative_fraction": (
            benefit_negative_count / weight_count if weight_count > 0 else 0.0
        ),
        "verpo/benefit_positive_fraction": (
            benefit_positive_count / weight_count if weight_count > 0 else 0.0
        ),
        "verpo/negative_benefit_mean": ratio_with_count(
            "sum/negative_benefit", benefit_negative_count
        ),
        "verpo/negative_benefit_weight_mean": ratio_with_count(
            "sum/negative_benefit_weight", benefit_negative_count
        ),
        "verpo/negative_benefit_effective_coverage": ratio_with_count(
            "count/negative_benefit_effective", benefit_negative_count
        ),
        "verpo/fisher_cost_mean": ratio_with_count("sum/fisher_cost", evidence_count),
        "verpo/fec_task_displacement_l2": ratio_with_count("sum/fec_task_norm", fec_count),
        "verpo/fec_nuisance_displacement_l2": ratio_with_count(
            "sum/fec_nuisance_norm", fec_count
        ),
        "verpo/fec_task_nuisance_fisher_cosine": ratio_with_count(
            "sum/fec_cosine", fec_count
        ),
        "verpo/fec_nuisance_projection_coefficient": ratio_with_count(
            "sum/fec_projection", fec_count
        ),
        "verpo/fec_residual_nuisance_fisher_covariance": ratio_with_count(
            "sum/fec_residual_covariance", fec_count
        ),
        "verpo/fec_correct_alignment_mean": ratio_with_count(
            "sum/fec_correct_alignment", fec_correct_count
        ),
        "verpo/fec_wrong_alignment_mean": ratio_with_count(
            "sum/fec_wrong_alignment", fec_wrong_count
        ),
        "verpo/fec_joint_sign_pass_rate": ratio_with_count(
            "count/fec_sign_pass", fec_correct_count + fec_wrong_count
        ),
        "verpo/benefit_cost_ratio_mean": ratio_with_count(
            "sum/benefit_cost_ratio", evidence_count
        ),
        "verpo/mixed_success_weight_mean": ratio_with_count(
            "sum/mixed_success_weight", mixed_success_count
        ),
        "verpo/mixed_failure_weight_mean": ratio_with_count(
            "sum/mixed_failure_weight", mixed_failure_count
        ),
        "verpo/student_forward_time_seconds": total("time/student"),
        "verpo/qref_forward_time_seconds": total("time/qref"),
        "verpo/q0_forward_time_seconds": total("time/q0"),
        "verpo/qe_forward_time_seconds": total("time/qe"),
        "verpo/peak_memory_allocated_gb": max(
            _flatten_metric_numbers(metrics.pop(f"{prefix}max/memory_allocated", [0.0]))
        ),
        "verpo/peak_memory_reserved_gb": max(
            _flatten_metric_numbers(metrics.pop(f"{prefix}max/memory_reserved", [0.0]))
        ),
        "verpo/reference_token_count": int(round(reference_count)),
        "verpo/evidence_token_count": int(round(evidence_count)),
        "verpo/weight_token_count": int(round(available_count)),
        "verpo_zpd/selected_benefit": selected_benefit,
        "verpo_zpd/selected_cost": selected_cost,
        "verpo_zpd/selected_displacement_norm": ratio_with_count(
            "sum/selected_displacement", evidence_count
        ),
        "verpo_zpd/selected_weight": ratio_with_count("sum/selected_weight", evidence_count),
        "verpo_zpd/selected_token_count": int(round(evidence_count)),
        "verpo_zpd/benefit_cost_ratio": (
            selected_benefit / selected_cost if selected_cost > 0 else 0.0
        ),
    }
    correct_alignment = finalized["verpo/fec_correct_alignment_mean"]
    wrong_alignment = finalized["verpo/fec_wrong_alignment_mean"]
    finalized["verpo/fec_alignment_margin"] = correct_alignment - wrong_alignment
    for name in (
        "topk_support_size",
        "topk_reference_support_mass",
        "topk_evidence_support_mass",
        "topk_negative_support_mass",
        "topk_student_support_mass",
    ):
        numerator_key = f"sum/{name}"
        if f"{prefix}{numerator_key}" in metrics:
            finalized[f"verpo_zpd/{name}"] = ratio_with_count(
                numerator_key, evidence_count
            )
    for key, value in finalized.items():
        metrics[key] = [value]


class TrainingWorker(Worker, DistProfilerExtension):
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.
    """

    def __init__(self, config: TrainingWorkerConfig):
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        initialize_global_process_group_ray(timeout_second=None)

        set_numa_affinity()

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        if self.engine_config is None:
            assert self.optimizer_config is None
            if self.config.auto_select_engine_optim_fn is None:
                raise ValueError(
                    "engine_config is not provided and auto_select_engine_optim_fn is not set. "
                    "Cannot determine engine backend."
                )
            # Support automatically select engine backend given model config
            self.engine_config, self.optimizer_config = self.config.auto_select_engine_optim_fn(
                self.model_config, self.device_name
            )

        # we use the one defined in model
        # TODO: this is not elegant and should refactor later
        self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding", False)
        self.engine_config.use_fused_kernels = self.model_config.get("use_fused_kernels", False)

        self.profiler_config = self.config.profiler_config
        if self.profiler_config is not None:
            self.profiler_tool_config = self.profiler_config.tool_config.get(self.profiler_config.tool, {})
        else:
            self.profiler_tool_config = None

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=self.profiler_tool_config)
        )

        self.model_config.model_type = self.config.model_type
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        if hasattr(self.model_config, "hf_config"):
            self.flops_counter = FlopsCounter(self.model_config.hf_config)
        else:
            self.flops_counter = None

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states
        """
        self.engine.initialize()

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        """

        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        """

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        dp_group = self.engine.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach().item()
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group (only if DP > 1)
        if dp_group is not None:
            final_metrics = allgather_dict_into_dict(data=metrics, group=dp_group)
        else:
            final_metrics = metrics
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # log memory
        final_metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        final_metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        final_metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        if global_token_num is not None and self.flops_counter is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        """Split a batch into N mini-batches run for multiple epochs

        Args:
            data:

        Returns:

        """
        maybe_fix_3d_position_ids(data)
        batch_size_per_dp = data.shape[0]
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

        assert mini_batch_size is not None or num_mini_batch is not None

        if mini_batch_size is None:
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
        else:
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
            )
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

        # make iterator
        dataloader = tu.make_iterator(
            data,
            mini_batch_size=mini_batch_size_per_gpu,
            epochs=epochs,
            seed=seed + self.engine.get_data_parallel_rank(),
            dataloader_kwargs=dataloader_kwargs,
        )

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            # update
            output_lst = []
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

            for batch_idx, mini_batch_td in enumerate(dataloader):
                # add global token num
                if "input_ids" in mini_batch_td:
                    global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)
                    # allgather from dp rank
                    global_token_num_output = [None] * torch.distributed.get_world_size(
                        self.engine.get_data_parallel_group()
                    )
                    torch.distributed.all_gather_object(
                        global_token_num_output, global_token_num, self.engine.get_data_parallel_group()
                    )
                    global_token_num = [x for xs in global_token_num_output for x in xs]
                else:
                    global_token_num = None

                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                actor_output = self.train_batch(mini_batch_td)
                output_lst.append(actor_output)

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        # flattn dp and micro batch
                        if isinstance(val, list):
                            output[key] = (
                                Metric.aggregate_dp(val)
                                if isinstance(val[0], Metric)
                                else list(chain.from_iterable(val))
                            )
                    append_to_dict(metrics, output)

                _finalize_verpo_monitoring(metrics)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="train_batch")
    def train_batch(self, data: TensorDict) -> TensorDict:
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        _refresh_verpo_global_routing_counts(
            data,
            self.engine.get_data_parallel_group(),
        )
        _refresh_paper_baseline_global_routing_counts(
            data,
            self.engine.get_data_parallel_group(),
        )

        gradient_auditor = getattr(self, "_verpo_gradient_auditor", None)
        if gradient_auditor is not None:
            gradient_auditor.begin_train_batch()

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
            actor_teacher = getattr(self, "_verpo_actor_teacher", None)
            if actor_teacher is not None:
                teacher_synced = actor_teacher.after_optimizer_step(
                    update_applied=bool(
                        getattr(self.engine, "_last_optimizer_step_applied", False)
                    )
                )
                if self.engine.is_mp_src_rank_with_outputs():
                    output["metrics"]["verpo/teacher_update_count"] = [
                        float(actor_teacher.optimizer_update_count)
                    ]
                    output["metrics"]["verpo/teacher_last_sync_update"] = [
                        float(actor_teacher.last_sync_update)
                    ]
                    output["metrics"]["verpo/teacher_synced"] = [
                        float(teacher_synced)
                    ]
                    output["metrics"][
                        "verpo/teacher_fingerprint_prefix"
                    ] = [float(int(actor_teacher.fingerprint()[:12], 16))]
                    output["metrics"]["verpo/teacher_is_ema"] = [
                        float(actor_teacher.mode == "ema")
                    ]
            paper_teacher = getattr(self, "_paper_baseline_teacher", None)
            if paper_teacher is not None:
                teacher_synced = paper_teacher.after_optimizer_step(
                    update_applied=bool(
                        getattr(self.engine, "_last_optimizer_step_applied", False)
                    )
                )
                if self.engine.is_mp_src_rank_with_outputs():
                    output["metrics"]["paper_baseline/teacher_update_count"] = [
                        float(paper_teacher.optimizer_update_count)
                    ]
                    output["metrics"]["paper_baseline/teacher_last_sync_update"] = [
                        float(paper_teacher.last_sync_update)
                    ]
                    output["metrics"]["paper_baseline/teacher_synced"] = [
                        float(teacher_synced)
                    ]
                    output["metrics"][
                        "paper_baseline/teacher_fingerprint_prefix"
                    ] = [float(int(paper_teacher.fingerprint()[:12], 16))]
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
            with adapter_ctx:
                output = self.engine.infer_batch(data, loss_function=loss_function)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=True,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    actor_worker_cls = TrainingWorker
    ref_worker_cls = TrainingWorker

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.distillation_enabled = is_distillation_enabled(distillation_config)
        self.role = role
        self.actor: TrainingWorker | None = None
        self.ref: TrainingWorker | None = None
        self.rollout: BaseRollout = None
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        # Router replay is supported on the megatron engine and on the veomni
        # engine. Both expose `router_replay` on their per-strategy engine
        # config (the field lives on the shared `EngineConfig` base).
        actor_strategy = self.config.actor.strategy
        if actor_strategy == "megatron":
            rr_mode = self.config.actor.megatron.router_replay.mode
        elif actor_strategy == "veomni":
            rr_mode = self.config.actor.veomni.router_replay.mode
        else:
            rr_mode = "disabled"
        self.enable_routing_replay = rr_mode != "disabled"

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.actor.set_loss_fn(loss_fn=loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # 1. build reference model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)

            # The ref model does not need to enable MTP; force it to false.
            ref_config.model_config = deepcopy(model_config)
            ref_config.model_config.mtp = MtpConfig(enable=False)

            # construct TrainingWorkerConfig
            ref_training_config = TrainingWorkerConfig(
                model_type=ref_config.model_config.get("model_type", "language_model"),
                model_config=ref_config.model_config,
                engine_config=ref_config.engine,
                optimizer_config=ref_config.optim,
                checkpoint_config=ref_config.checkpoint,
            )

            # assign engine configs
            ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz
            ref_training_config.engine_config.infer_max_token_len_per_gpu = self.config.ref.ppo_max_token_len_per_gpu
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.ref.ppo_micro_batch_size_per_gpu
            )
            ref_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            self.ref = self.ref_worker_cls(config=ref_training_config)
            self.ref.reset()
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # 2. build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = deepcopy(model_config)
            actor_config.model_config.freeze_vision_tower = actor_config.freeze_vision_tower
            distillation_config: Optional[DistillationConfig] = (
                omega_conf_to_dataclass(self.distillation_config) if self.distillation_enabled else None
            )

            actor_training_config = TrainingWorkerConfig(
                model_type=actor_config.model_config.get("model_type", "language_model"),
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )

            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz

            # assign engine configs
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
            actor_training_config.engine_config.micro_batch_size_per_gpu = (
                self.config.actor.ppo_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)
            if actor_config.freeze_vision_tower and actor_config.strategy == "fsdp":
                # FSDP1 needs original parameters when one wrapped module mixes
                # frozen vision parameters with trainable language parameters.
                actor_training_config.engine_config = replace(
                    actor_training_config.engine_config, use_orig_params=True
                )

            if self.config.actor.use_dynamic_bsz:
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None
                assert self.config.actor.ppo_max_token_len_per_gpu is not None
            else:
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None
                assert self.config.actor.ppo_micro_batch_size_per_gpu is not None
            if actor_config.verpo.enabled:
                if self.distillation_enabled:
                    raise ValueError(
                        "VERPO-ZPD uses the colocated fixed Reference and cannot enable teacher-manager distillation"
                    )
                if self.ref is None:
                    raise ValueError("VERPO-ZPD requires role=actor_rollout_ref with a fixed Reference model")
                if actor_config.strategy not in {"fsdp", "fsdp2"}:
                    raise NotImplementedError("VERPO-ZPD currently supports native FSDP/FSDP2 actors only")
                if actor_config.ulysses_sequence_parallel_size != 1:
                    raise NotImplementedError(
                        "VERPO-ZPD exact response-token alignment currently requires "
                        "actor.ulysses_sequence_parallel_size=1"
                    )
                if self.ref.engine.optimizer is not None:
                    raise ValueError("VERPO-ZPD infrastructure Reference must be forward-only")
                gradient_auditor = None
                if actor_config.verpo.gradient_audit_enabled:
                    gradient_auditor = VerpoGradientAuditor(
                        max_train_batches=actor_config.verpo.gradient_audit_max_steps,
                        max_parameter_elements=(
                            actor_config.verpo.gradient_audit_max_parameter_elements
                        ),
                        max_parameter_tensors=(
                            actor_config.verpo.gradient_audit_max_parameter_tensors
                        ),
                    )
                self._verpo_gradient_auditor = gradient_auditor
                self.loss_fn = partial(
                    verpo_zpd_ppo_loss,
                    config=actor_config,
                    reference_engine=self.ref.engine,
                    actor_teacher=None,
                    gradient_auditor=gradient_auditor,
                )
            elif actor_config.paper_baseline.enabled:
                if self.distillation_enabled:
                    raise ValueError(
                        "paper SDPO/SRPO owns its colocated EMA Teacher and cannot "
                        "enable the generic distillation manager"
                    )
                if actor_config.strategy not in {"fsdp", "fsdp2"}:
                    raise NotImplementedError(
                        "paper SDPO/SRPO currently supports native FSDP/FSDP2 actors"
                    )
                if actor_config.ulysses_sequence_parallel_size != 1:
                    raise NotImplementedError(
                        "paper SDPO/SRPO exact response alignment requires "
                        "actor.ulysses_sequence_parallel_size=1"
                    )
                # Replaced with the paper loss after the actor engine exists.
                self.loss_fn = partial(ppo_loss, config=actor_config)
            elif self.distillation_enabled:
                self.loss_fn = partial(
                    distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
                )
            else:
                self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor = self.actor_worker_cls(config=actor_training_config)
            self.actor.reset()
            if actor_config.paper_baseline.enabled:
                paper_teacher = PaperBaselineEMATeacher(
                    self.actor.engine,
                    objective=actor_config.paper_baseline.objective,
                    ema_decay=actor_config.paper_baseline.teacher_ema_decay,
                )
                self.actor._paper_baseline_teacher = paper_teacher
                self.loss_fn = partial(
                    paper_sdpo_srpo_loss,
                    config=actor_config,
                    baseline_teacher=paper_teacher,
                )
            if (
                actor_config.verpo.enabled
                and actor_config.verpo.teacher_mode in {"snapshot", "ema"}
            ):
                actor_teacher = ActorSideTeacher(
                    self.actor.engine,
                    mode=actor_config.verpo.teacher_mode,
                    sync_interval=actor_config.verpo.teacher_sync_interval,
                    ema_decay=actor_config.verpo.teacher_ema_decay,
                )
                self.actor._verpo_actor_teacher = actor_teacher
                self.loss_fn = partial(
                    verpo_zpd_ppo_loss,
                    config=actor_config,
                    reference_engine=self.ref.engine,
                    actor_teacher=actor_teacher,
                    gradient_auditor=gradient_auditor,
                )
            gradient_auditor = getattr(self, "_verpo_gradient_auditor", None)
            if gradient_auditor is not None:
                gradient_auditor.bind(self.actor.engine.module, self.actor.engine.optimizer)
                self.actor._verpo_gradient_auditor = gradient_auditor
            self.actor.set_loss_fn(self.loss_fn)
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )

            # used for LoRA (base_sync_done is unused in merge-only mode but kept for Phase 2 adapter path)
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # 4. build checkpoint engine
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            # If custom_backend_module is set, import it so plugins can register
            # in CheckpointEngineRegistry before the backend is instantiated.
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.actor.infer_batch(data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        verpo_enabled = bool(self.config.actor.get("verpo", {}).get("enabled", False))
        reference_context = self.ref.engine.eval_mode() if verpo_enabled else nullcontext()
        with reference_context:
            output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)
        actor_teacher = getattr(self.actor, "_verpo_actor_teacher", None)
        if actor_teacher is not None:
            actor_teacher.load(local_path)
        paper_teacher = getattr(self.actor, "_paper_baseline_teacher", None)
        if paper_teacher is not None:
            paper_teacher.load(local_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        actor_teacher = getattr(self.actor, "_verpo_actor_teacher", None)
        if actor_teacher is not None:
            actor_teacher.save(local_path)
        paper_teacher = getattr(self.actor, "_paper_baseline_teacher", None)
        if paper_teacher is not None:
            paper_teacher.save(local_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        """Update weights from trainer to rollout.

        1. For sync training with colocated trainer and rollout, update rollout directly from model engine.
           - before update_weights: rollout should be in sleep mode.
           - after update_weights: rollout should be in wake_up mode.
        2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.

        LoRA handling: when model.lora.merge=True (peft_merge), LoRA is merged into
        base weights before sync. The engine returns full HF-keyed params with
        peft_config=None, so the rollout receives a standard weight update.

        Args:
            global_steps: Current global training step count, passed to rollout for logging/tracking.
            mode: Weight update strategy. Supported values:
                - ``"auto"``: Automatically resolve to the backend configured in
                  ``config.rollout.checkpoint_engine.backend`` (default).
                - ``"naive"``: Direct in-process weight sync between colocated trainer
                  and rollout. Used for synchronous training where both share the same
                  process. Rollout must be in sleep mode before this call.
                - Any other value: Delegates to
                  :meth:`checkpoint_engine.send_weights` for asynchronous weight
                  transfer via checkpoint engine, suitable for disaggregated
                  trainer/rollout deployments.
        """

        # Resolve mode: "auto" falls back to config, explicit values take precedence
        effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

        # 0. send_weights only for async training with disaggregated trainer and rollout
        if effective_mode != "naive":
            per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
            return

        set_expandable_segments(False)
        log_gpu_memory_usage("Before resume weights", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
        if self.config.rollout.free_cache_engine and self.rollout.sleep_level != 1:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        # 2. determine if we need a base weight sync (adapter path only)
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = not self.base_sync_done

        # 3. sync weights: For SGLang, we need base first (when needed), then adapter/merged
        if do_lora_base_sync:
            per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=False
            )
            await self.rollout.update_weights(
                per_tensor_param_base, peft_config=peft_config, base_sync_done=False, global_steps=global_steps
            )

        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
        )

        log_gpu_memory_usage("After update_weights", logger=logger)

        # 3. offload model to cpu
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        # 4. resume kv_cache
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Execute checkpoint engine method.

        Args:
            method (str): Checkpoint engine method name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
