"""Repository-native paper baselines kept separate from VERPO-ZPD."""

from .paper_prompts import (
    SDPO_OFFICIAL_PROMPT_PROFILE,
    SRPO_PAPER_PROMPT_PROFILE,
    apply_paper_prompt_profile,
)

__all__ = [
    "SDPO_OFFICIAL_PROMPT_PROFILE",
    "SRPO_PAPER_PROMPT_PROFILE",
    "apply_paper_prompt_profile",
]
