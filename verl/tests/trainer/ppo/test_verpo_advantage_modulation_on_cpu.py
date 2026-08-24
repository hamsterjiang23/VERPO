import pytest
import torch

from verl.workers.config import VerpoZPDConfig
from verl.workers.utils.losses import modulate_advantages


def test_modulate_advantages_preserves_baseline_and_sign() -> None:
    advantages = torch.tensor([[2.0, -3.0, 0.0]])
    weights = torch.tensor([[0.0, 0.5, 0.25]], requires_grad=True)

    result = modulate_advantages(advantages, weights, coefficient=1.0)

    assert torch.equal(result, torch.tensor([[2.0, -4.5, 0.0]]))
    assert torch.equal(torch.sign(result), torch.sign(advantages))
    assert not result.requires_grad


def test_modulate_advantages_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        modulate_advantages(torch.ones(2), torch.ones(2, 1), coefficient=1.0)


def test_advantage_modulation_config_is_fec_only() -> None:
    config = VerpoZPDConfig(
        enabled=True,
        displacement_mode="fec",
        advantage_modulation="multiplicative_w",
        advantage_modulation_lambda=1.0,
    )
    assert config.advantage_modulation == "multiplicative_w"

    with pytest.raises(ValueError, match="requires displacement_mode=fec"):
        VerpoZPDConfig(
            enabled=True,
            displacement_mode="correct_vs_incorrect",
            advantage_modulation="multiplicative_w",
            advantage_modulation_lambda=1.0,
        )
