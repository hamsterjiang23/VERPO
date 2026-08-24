from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import verl.trainer.distillation.verpo_zpd as verpo_zpd_module
from verl.trainer.distillation.snapshot_teacher import ActorSideTeacher
from verl.trainer.distillation.verpo_zpd import _teacher_forward, _verpo_logits_processor
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.config.actor import ActorConfig, VerpoZPDConfig


def _manager(
    sync_interval: int = 10,
    *,
    mode: str = "snapshot",
    ema_decay: float = 0.95,
):
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    engine = SimpleNamespace(module=module)
    return module, ActorSideTeacher(
        engine,
        mode=mode,
        sync_interval=sync_interval,
        ema_decay=ema_decay,
    )


def test_verpo_teacher_mode_config_supports_fixed_snapshot_and_ema():
    assert VerpoZPDConfig(enabled=True).teacher_mode == "fixed_initial"
    assert VerpoZPDConfig(enabled=True, teacher_mode="snapshot").teacher_mode == "snapshot"
    assert VerpoZPDConfig(enabled=True, teacher_mode="snapshot_anchor").teacher_mode == "snapshot"
    assert VerpoZPDConfig(enabled=True, teacher_mode="ema").teacher_mode == "ema"
    assert VerpoZPDConfig(enabled=True).teacher_sync_interval == 10
    assert VerpoZPDConfig(enabled=True).teacher_ema_decay == 0.95
    with pytest.raises(ValueError, match="teacher_mode"):
        VerpoZPDConfig(enabled=True, teacher_mode="unknown")
    with pytest.raises(ValueError, match="teacher_ema_decay"):
        VerpoZPDConfig(enabled=True, teacher_mode="ema", teacher_ema_decay=1.0)


def test_snapshot_teacher_refreshes_after_tenth_successful_update():
    module, teacher = _manager()

    with torch.no_grad():
        module.weight.fill_(2.0)
    for _ in range(9):
        assert not teacher.after_optimizer_step(update_applied=True)
    assert teacher.optimizer_update_count == 9

    with teacher.forward_context():
        torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
    torch.testing.assert_close(module.weight, torch.full_like(module.weight, 2.0))

    assert teacher.after_optimizer_step(update_applied=True)
    assert teacher.last_sync_update == 10
    with teacher.forward_context():
        torch.testing.assert_close(module.weight, torch.full_like(module.weight, 2.0))


def test_snapshot_teacher_does_not_count_skipped_updates():
    _, teacher = _manager(sync_interval=2)
    assert not teacher.after_optimizer_step(update_applied=False)
    assert teacher.optimizer_update_count == 0


def _cpu_fsdp_engine_for_optimizer_step():
    engine = object.__new__(FSDPEngine)
    engine.module = torch.nn.Linear(2, 1, bias=False)
    engine.optimizer = torch.optim.SGD(engine.module.parameters(), lr=0.1)
    engine.optimizer_config = SimpleNamespace(clip_grad=1.0)
    engine.scaler = None
    engine._qat_enabled = False
    return engine


def test_fsdp_optimizer_step_exposes_success_for_teacher_counter():
    engine = _cpu_fsdp_engine_for_optimizer_step()
    engine.module(torch.ones(1, 2)).sum().backward()
    engine.optimizer_step()
    assert engine._last_optimizer_step_applied is True

    teacher = ActorSideTeacher(engine, mode="snapshot", sync_interval=2)
    assert not teacher.after_optimizer_step(
        update_applied=engine._last_optimizer_step_applied
    )
    assert teacher.optimizer_update_count == 1


def test_fsdp_nonfinite_step_does_not_advance_teacher_counter():
    engine = _cpu_fsdp_engine_for_optimizer_step()
    for parameter in engine.module.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    engine.optimizer_step()
    assert engine._last_optimizer_step_applied is False

    teacher = ActorSideTeacher(engine, mode="snapshot", sync_interval=2)
    assert not teacher.after_optimizer_step(
        update_applied=engine._last_optimizer_step_applied
    )
    assert teacher.optimizer_update_count == 0


def test_ema_teacher_updates_after_each_successful_optimizer_update():
    module, teacher = _manager(mode="ema", ema_decay=0.5)
    with torch.no_grad():
        module.weight.fill_(3.0)
    assert teacher.after_optimizer_step(update_applied=True)
    assert teacher.optimizer_update_count == 1
    assert teacher.last_sync_update == 1
    with teacher.forward_context():
        torch.testing.assert_close(module.weight, torch.full_like(module.weight, 2.0))
    torch.testing.assert_close(module.weight, torch.full_like(module.weight, 3.0))


def test_snapshot_teacher_checkpoint_round_trip_preserves_next_sync(tmp_path):
    module, teacher = _manager(sync_interval=10)
    with torch.no_grad():
        module.weight.fill_(3.0)
    for _ in range(17):
        teacher.after_optimizer_step(update_applied=True)
    teacher.save(tmp_path)

    restored_module, restored = _manager(sync_interval=10)
    restored.load(tmp_path)
    assert restored.optimizer_update_count == 17
    assert restored.last_sync_update == 10
    for _ in range(2):
        assert not restored.after_optimizer_step(update_applied=True)
    with torch.no_grad():
        restored_module.weight.fill_(4.0)
    assert restored.after_optimizer_step(update_applied=True)
    assert restored.last_sync_update == 20


def test_snapshot_teacher_resume_fails_closed_on_interval_mismatch(tmp_path):
    _, teacher = _manager(sync_interval=10)
    teacher.save(tmp_path)
    _, restored = _manager(sync_interval=5)
    with pytest.raises(ValueError, match="sync interval mismatch"):
        restored.load(tmp_path)


def test_actor_teacher_resume_fails_closed_on_mode_mismatch(tmp_path):
    _, teacher = _manager(mode="snapshot")
    teacher.save(tmp_path)
    _, restored = _manager(mode="ema")
    with pytest.raises(ValueError, match="mode mismatch"):
        restored.load(tmp_path)


class _LogitModule(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))

    def forward(self, input_ids, use_cache=False):
        del use_cache
        logits = input_ids.float().unsqueeze(-1) * self.weight
        return SimpleNamespace(logits=logits)


class _TeacherEngine:
    def __init__(self, value: float):
        self.module = _LogitModule(value)

    @staticmethod
    def prepare_model_inputs(micro_batch):
        return {"input_ids": micro_batch["input_ids"]}, {}


def test_actor_teacher_replaces_fixed_reference_for_qref_forward():
    actor_engine = _TeacherEngine(1.0)
    fixed_reference = _TeacherEngine(9.0)
    snapshot = ActorSideTeacher(actor_engine, sync_interval=10)
    with torch.no_grad():
        actor_engine.module.weight.fill_(3.0)
    data = TensorDict({"input_ids": torch.tensor([[2]])}, batch_size=[1])

    fixed_logits = _teacher_forward(fixed_reference, data)
    snapshot_logits = _teacher_forward(fixed_reference, data, snapshot)
    torch.testing.assert_close(fixed_logits, torch.tensor([[18.0]]))
    torch.testing.assert_close(snapshot_logits, torch.tensor([[2.0]]))
    torch.testing.assert_close(actor_engine.module.weight, torch.tensor(3.0))


def _jagged(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def test_qref_q0_positive_and_negative_share_actor_teacher(monkeypatch):
    actor_teacher = object()
    teacher_calls = []

    def fake_teacher_forward(reference_engine, data, selected_teacher=None):
        del reference_engine
        teacher_calls.append(selected_teacher)
        token_count = data["input_ids"].values().numel()
        return torch.zeros(token_count, 3)

    monkeypatch.setattr(verpo_zpd_module, "_teacher_forward", fake_teacher_forward)
    config = ActorConfig(
        strategy="fsdp2",
        loss_mode="verpo_zpd",
        rollout_n=1,
        use_dynamic_bsz=True,
        use_kl_loss=True,
        kl_loss_coef=0.0,
        verpo=VerpoZPDConfig(
            enabled=True,
            teacher_mode="snapshot",
            displacement_mode="correct_vs_incorrect",
            contrastive_num_negative_hints=1,
        ),
    )
    prompts = _jagged([torch.tensor([0, 1])])
    responses = _jagged([torch.tensor([1, 2])])
    input_ids = _jagged([torch.tensor([0, 1, 1, 2])])
    position_ids = _jagged([torch.arange(4)])
    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "response_mask": _jagged([torch.tensor([True, True])]),
            "advantages": _jagged([torch.tensor([0.5, 0.5])]),
            "verpo_evidence_rollout_gate": torch.tensor([True]),
            "verpo_positive_input_ids": input_ids.clone(),
            "verpo_positive_position_ids": position_ids.clone(),
            "verpo_positive_prompt_lengths": torch.tensor([2]),
            "verpo_negative_0_input_ids": input_ids.clone(),
            "verpo_negative_0_position_ids": position_ids.clone(),
            "verpo_negative_0_prompt_lengths": torch.tensor([2]),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor_data(data, "temperature", 1.0)

    outputs = _verpo_logits_processor(
        config=config,
        reference_engine=object(),
        actor_teacher=actor_teacher,
        student_logits=torch.zeros(1, 4, 3),
        data=data,
    )

    assert teacher_calls == [actor_teacher, actor_teacher, actor_teacher]
    assert "verpo_q0_sample_log_prob" in outputs


def _assert_dtensor_value(parameter, expected: float) -> None:
    local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
    torch.testing.assert_close(local, torch.full_like(local, expected))


def _fsdp2_snapshot_teacher_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        module = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            module.weight.fill_(1.0)
        fully_shard(module, mesh=mesh)
        engine = SimpleNamespace(module=module)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        teacher = ActorSideTeacher(engine, mode="snapshot", sync_interval=1)

        module(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        assert teacher.after_optimizer_step(update_applied=True)
        teacher_local = module.weight.to_local().detach().clone()
        with torch.no_grad():
            module.weight.fill_(5.0)

        with teacher.forward_context():
            torch.testing.assert_close(module.weight.to_local(), teacher_local)
        _assert_dtensor_value(module.weight, 5.0)

        state_path = teacher.save(checkpoint_dir)
        assert state_path.name == (
            f"verpo_actor_teacher_world_size_{world_size}_rank_{rank}.pt"
        )
        dist.barrier()

        restored = ActorSideTeacher(engine, mode="snapshot", sync_interval=1)
        restored.load(checkpoint_dir)
        assert restored.optimizer_update_count == 1
        assert restored.last_sync_update == 1
        with restored.forward_context():
            torch.testing.assert_close(module.weight.to_local(), teacher_local)
        _assert_dtensor_value(module.weight, 5.0)
    finally:
        dist.destroy_process_group()


def test_snapshot_teacher_round_trips_two_rank_fsdp2_dtensors(tmp_path):
    world_size = 2
    init_path = Path(tmp_path) / "fsdp2_teacher_init"
    init_method = f"file:///{init_path.as_posix()}"
    checkpoint_dir = str(Path(tmp_path) / "teacher_checkpoint")

    mp.spawn(
        _fsdp2_snapshot_teacher_worker,
        args=(world_size, init_method, checkpoint_dir),
        nprocs=world_size,
        join=True,
    )

    checkpoint_files = sorted(Path(checkpoint_dir).glob("verpo_actor_teacher_*.pt"))
    assert [path.name for path in checkpoint_files] == [
        "verpo_actor_teacher_world_size_2_rank_0.pt",
        "verpo_actor_teacher_world_size_2_rank_1.pt",
    ]
