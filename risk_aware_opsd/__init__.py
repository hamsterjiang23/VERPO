"""Shared, backend-neutral VERPO-ZPD building blocks.

The package deliberately contains no model-specific probing or span routing.
Both the JSONL TRL path and the native veRL path import the same probability
space loss helpers from :mod:`verpo_zpd`.
"""

__all__ = [
    "VERPOConfig",
    "classify_reward_ranked_groups",
    "length_aware_outcome_rewards",
    "soft_overlong_penalty",
]


def __getattr__(name: str):
    if name == "VERPOConfig":
        from .verpo_launch_config import VERPOConfig

        return VERPOConfig
    if name in {"classify_reward_ranked_groups", "length_aware_outcome_rewards", "soft_overlong_penalty"}:
        from . import length_aware_reward

        return getattr(length_aware_reward, name)
    raise AttributeError(name)
