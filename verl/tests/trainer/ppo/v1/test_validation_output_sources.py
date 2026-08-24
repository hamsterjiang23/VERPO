from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _LifecycleTrainer(PPOTrainer):
    def on_step_end(self):
        return

    def on_sample_end(self):
        return


def _make_lifecycle_trainer(loop: MagicMock) -> _LifecycleTrainer:
    trainer = object.__new__(_LifecycleTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            project_name="project",
            experiment_name="experiment",
            logger=["swanlab"],
        )
    )
    trainer._fit = loop
    trainer.on_train_end = MagicMock()
    return trainer


def test_fit_finishes_tracking_after_success() -> None:
    trainer = _make_lifecycle_trainer(MagicMock(return_value="complete"))
    tracking = MagicMock()

    with (
        patch("verl.trainer.ppo.v1.trainer_base.Tracking", return_value=tracking),
        patch("verl.trainer.ppo.v1.trainer_base.ValidationGenerationsLogger"),
        patch("verl.trainer.ppo.v1.trainer_base.OmegaConf.to_container", return_value={}),
    ):
        result = trainer.fit(MagicMock())

    assert result == "complete"
    trainer.on_train_end.assert_called_once_with()
    tracking.finish.assert_called_once_with(exit_code=0)


def test_fit_finishes_tracking_after_failure() -> None:
    trainer = _make_lifecycle_trainer(MagicMock(side_effect=RuntimeError("boom")))
    tracking = MagicMock()

    with (
        patch("verl.trainer.ppo.v1.trainer_base.Tracking", return_value=tracking),
        patch("verl.trainer.ppo.v1.trainer_base.ValidationGenerationsLogger"),
        patch("verl.trainer.ppo.v1.trainer_base.OmegaConf.to_container", return_value={}),
        pytest.raises(RuntimeError, match="boom"),
    ):
        trainer.fit(MagicMock())

    trainer.on_train_end.assert_not_called()
    tracking.finish.assert_called_once_with(exit_code=1)


def _write_sciknoweval_validation(tmp_path, *, expected_data_sources):
    PPOTrainer._write_generations(
        inputs=["question"],
        outputs=["<answer>A</answer>"],
        gts=["A"],
        scores=[1.0],
        reward_extra_infos_dict={
            "data_source": ["sciknoweval"],
            "acc": [1.0],
            "formatted": [1.0],
        },
        dump_path=tmp_path,
        global_steps=1,
        split_by_data_source=True,
        expected_data_sources=expected_data_sources,
    )


def test_validation_output_accepts_registered_sdpo_source(tmp_path) -> None:
    _write_sciknoweval_validation(
        tmp_path,
        expected_data_sources=["sciknoweval"],
    )

    metrics = json.loads((tmp_path / "by_dataset" / "1.metrics.json").read_text(encoding="utf-8"))
    assert metrics["datasets"]["sciknoweval"]["count"] == 1
    assert (tmp_path / "by_dataset" / "sciknoweval" / "1.jsonl").is_file()


def test_validation_output_keeps_math_default_fail_closed(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="aime24.*aime25.*amc23"):
        _write_sciknoweval_validation(tmp_path, expected_data_sources=None)
