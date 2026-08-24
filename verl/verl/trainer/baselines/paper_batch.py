"""Paper-faithful routing and Teacher replay fields for SDPO and SRPO."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .paper_prompts import build_self_teacher_messages


_THINKING_TRACE_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class PaperRouting:
    group_ids: tuple[str, ...]
    correct: torch.Tensor
    teacher_available: torch.Tensor
    selected_sibling_index: torch.Tensor
    sdpo_route: torch.Tensor
    grpo_route: torch.Tensor


def rollout_group_ids(keys: Sequence[Any]) -> tuple[str, ...]:
    """Recover the original prompt UID from TransferQueue rollout keys."""

    group_ids: list[str] = []
    for key in keys:
        parts = str(key).rsplit("_", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Unexpected rollout key format for SDPO/SRPO grouping: {key}"
            )
        group_ids.append(parts[0])
    return tuple(group_ids)


def build_paper_routing(
    rewards: torch.Tensor,
    group_ids: Sequence[str],
    *,
    objective: str,
    success_reward_threshold: float = 0.5,
) -> PaperRouting:
    """Select the first target-excluded correct sibling and route samples."""

    objective = str(objective).strip().lower()
    if objective not in {"sdpo", "srpo"}:
        raise ValueError("paper baseline objective must be 'sdpo' or 'srpo'")
    rewards = rewards.detach().float().cpu().reshape(-1)
    if rewards.numel() != len(group_ids):
        raise ValueError("rewards and group_ids must have the same number of rows")
    correct = rewards.ge(float(success_reward_threshold))
    selected = torch.full((rewards.numel(),), -1, dtype=torch.long)
    grouped_correct: dict[str, list[int]] = {}
    for index, (group_id, is_correct) in enumerate(
        zip(group_ids, correct.tolist(), strict=True)
    ):
        if is_correct:
            grouped_correct.setdefault(str(group_id), []).append(index)
    for index, group_id in enumerate(group_ids):
        candidates = [
            candidate
            for candidate in grouped_correct.get(str(group_id), [])
            if candidate != index
        ]
        if candidates:
            # This matches the official SDPO implementation. Rollout ordering is
            # sampled, while the replay decision remains deterministic/auditable.
            selected[index] = candidates[0]
    available = selected.ge(0)
    sdpo_route = available if objective == "sdpo" else (~correct & available)
    grpo_route = torch.zeros_like(sdpo_route) if objective == "sdpo" else ~sdpo_route
    return PaperRouting(
        group_ids=tuple(str(value) for value in group_ids),
        correct=correct,
        teacher_available=available,
        selected_sibling_index=selected,
        sdpo_route=sdpo_route,
        grpo_route=grpo_route,
    )


def _plain_rows(values: Any, *, expected: int, name: str) -> list[Any]:
    rows = list(values)
    if len(rows) != expected:
        raise ValueError(f"{name} has {len(rows)} rows, expected {expected}")
    return rows


def _render_prompt_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    chat_template_kwargs: dict[str, Any],
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def _remove_thinking_trace(text: str) -> str:
    return _THINKING_TRACE_RE.sub("", str(text or ""))


def build_paper_teacher_fields(
    *,
    tokenizer,
    responses: torch.Tensor,
    raw_prompts: Any,
    routing: PaperRouting,
    total_token_budget: int,
    max_reprompt_tokens: int,
    chat_template_kwargs: dict[str, Any] | None = None,
    remove_thinking_from_demonstration: bool = True,
) -> dict[str, torch.Tensor]:
    """Render exact Teacher prompts and append the unchanged Student suffix.

    Official SDPO uses right truncation for reprompts.  Accordingly, only the
    right side of an overlong rendered Teacher prompt is removed; the sampled
    Student response is never truncated or regenerated here.
    """

    if not responses.is_nested:
        raise ValueError("SDPO/SRPO Teacher construction requires nested responses")
    if int(total_token_budget) <= 0 or int(max_reprompt_tokens) <= 0:
        raise ValueError("SDPO/SRPO Teacher token budgets must be positive")
    response_rows = list(responses.unbind())
    batch_size = len(response_rows)
    prompt_rows = _plain_rows(raw_prompts, expected=batch_size, name="raw_prompt")
    if routing.selected_sibling_index.numel() != batch_size:
        raise ValueError("routing and responses must have the same batch size")
    response_texts = [
        tokenizer.decode(response.detach().cpu().long().tolist(), skip_special_tokens=True)
        for response in response_rows
    ]
    chat_kwargs = dict(chat_template_kwargs or {})

    input_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    prompt_lengths_before: list[int] = []
    truncated_tokens: list[int] = []
    for index, response in enumerate(response_rows):
        response_ids = response.detach().cpu().long().tolist()
        if not response_ids:
            raise ValueError(f"Sample {index} has an empty response")
        if len(response_ids) >= int(total_token_budget):
            raise ValueError(
                f"Sample {index} response has {len(response_ids)} tokens, exceeding "
                f"Teacher budget {total_token_budget}"
            )
        sibling_index = int(routing.selected_sibling_index[index].item())
        if sibling_index >= 0:
            sibling_response = response_texts[sibling_index]
            if remove_thinking_from_demonstration:
                sibling_response = _remove_thinking_trace(sibling_response)
            messages = build_self_teacher_messages(
                prompt_rows[index], sibling_response
            )
        else:
            # Unrouted rows are never forwarded through the Teacher. Keeping the
            # original prompt here preserves a rectangular TensorDict contract.
            messages = [dict(message) for message in prompt_rows[index]]
        prompt_ids = _render_prompt_ids(
            tokenizer,
            messages,
            chat_template_kwargs=chat_kwargs,
        )
        prompt_budget = min(
            int(max_reprompt_tokens),
            int(total_token_budget) - len(response_ids),
        )
        if prompt_budget <= 0:
            raise ValueError(f"Sample {index} has no remaining Teacher prompt budget")
        # The official code sets tokenizer.truncation_side='right'.
        kept_prompt_ids = prompt_ids[:prompt_budget]
        combined = torch.tensor(kept_prompt_ids + response_ids, dtype=torch.long)
        if combined[-len(response_ids) :].tolist() != response_ids:
            raise AssertionError("SDPO/SRPO Teacher response suffix changed")
        input_rows.append(combined)
        position_rows.append(torch.arange(combined.numel(), dtype=torch.long))
        prompt_lengths.append(len(kept_prompt_ids))
        prompt_lengths_before.append(len(prompt_ids))
        truncated_tokens.append(len(prompt_ids) - len(kept_prompt_ids))

    return {
        "paper_teacher_input_ids": torch.nested.as_nested_tensor(
            input_rows, layout=torch.jagged
        ),
        "paper_teacher_position_ids": torch.nested.as_nested_tensor(
            position_rows, layout=torch.jagged
        ),
        "paper_teacher_prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        "paper_teacher_prompt_tokens_before_trunc": torch.tensor(
            prompt_lengths_before, dtype=torch.float32
        ),
        "paper_teacher_prompt_truncated_tokens": torch.tensor(
            truncated_tokens, dtype=torch.float32
        ),
        "paper_teacher_prompt_truncated": torch.tensor(
            [value > 0 for value in truncated_tokens], dtype=torch.bool
        ),
        "paper_correct": routing.correct.clone(),
        "paper_teacher_available": routing.teacher_available.clone(),
        "paper_selected_sibling_index": routing.selected_sibling_index.clone(),
        "paper_sdpo_route": routing.sdpo_route.clone(),
        "paper_grpo_route": routing.grpo_route.clone(),
    }


__all__ = [
    "PaperRouting",
    "build_paper_routing",
    "build_paper_teacher_fields",
    "rollout_group_ids",
]
