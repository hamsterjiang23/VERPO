"""Compatibility exports for the standalone pure VERPO trainer.

The former shared risk-routing Trainer is intentionally absent.  This module
keeps the stable import path for callers that only need the backend-neutral
VERPO loss façade.
"""

from .verpo_launch_config import VERPOConfig
from .verpo_trainer import VERPOTrainer, grpo_policy_loss, masked_mean

__all__ = ["VERPOConfig", "VERPOTrainer", "grpo_policy_loss", "masked_mean"]
