# Copyright 2026 BPM Team of Tsinghua University
# Copyright 2026 VERPO-ZPD contributors
#
# Prompt and grading semantics are adapted from THU-BPM/VERPO_CONTRASTIVE commit
# 3da75d2209317d7f8fc7a89f3f65b8ad5cc4e4c0 under the MIT License.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to inclusion of this copyright and permission
# notice. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

"""Pinned VERPO_CONTRASTIVE math prompt, evidence-prompt, and boxed-answer protocol."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

import torch

BOXED_ANSWER_INSTRUCTION = "Present your final answer inside \\boxed{}, for example \\boxed{42}."
LEGACY_BOXED_ANSWER_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
TEACHER_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you understand the reasoning behind each step.\n"
)
INCORRECT_CANDIDATE_LABEL = "Incorrect candidate solution:"
UNAVAILABLE_CONTRASTIVE_HINT = "The output below is an incorrect answer."
TRUNCATED_CONTRASTIVE_HINT_MARKER = (
    "\n...[middle of candidate hint truncated to fit Teacher context]...\n"
)
EVAL_DATA_SOURCES = {"amc23", "aime24", "aime25"}
VERPO_CONTRASTIVE_COMMIT = "3da75d2209317d7f8fc7a89f3f65b8ad5cc4e4c0"
SDPO_TEACHER_TEMPLATE_ID = "sdpo_official_correct_solution_v1"
VERPO_CONTRASTIVE_CANDIDATE_HINT_TEMPLATE_ID = "verpo_contrastive_candidate_hint"
SDPO_REPROMPT_TEMPLATE = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
SDPO_SOLUTION_TEMPLATE = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"

_SIMPLE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_DEGREE_SUFFIX_RE = re.compile(r"(?:\^\{?\\circ\}?|\\degree|\\deg|°)$")
_SDPO_MCQ_FORMAT_RE = re.compile(r"<answer>\s*(A|B|C|D)\s*</answer>$")
_SDPO_TOOLUSE_FORMAT_RE = re.compile(r"Action:.*?\nAction Input:.*?", re.DOTALL)
_SDPO_MCQ_STYLES = {
    "mcq",
    "science",
    "sciknoweval",
    "biology",
    "chemistry",
    "material",
    "physics",
}
_SDPO_TOOLUSE_STYLES = {"tooluse", "tool_use", "toolalpaca"}


def strip_official_math_prompt(text: str) -> str:
    text = str(text or "").strip()
    for instruction in (BOXED_ANSWER_INSTRUCTION, LEGACY_BOXED_ANSWER_INSTRUCTION):
        suffix = f"\n\n{instruction}"
        while text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    if text.startswith("Problem: "):
        return text[len("Problem: ") :].strip()
    return text


def build_rollout_messages(problem: str, data_source: str | None = None) -> list[dict[str, str]]:
    problem = strip_official_math_prompt(problem)
    prefix = "" if str(data_source or "").lower() in EVAL_DATA_SOURCES else "Problem: "
    content = f"{prefix}{problem}\n\n{BOXED_ANSWER_INSTRUCTION}"
    return [{"role": "user", "content": content}]


def build_teacher_messages(problem: str, answer: str, solution: str) -> list[dict[str, str]]:
    problem = strip_official_math_prompt(problem)
    solution = str(solution or "").strip()
    answer = str(answer or "").strip()
    if not solution:
        raise ValueError("VERPO_CONTRASTIVE solution_answer Teacher requires a non-empty reference solution")
    if not answer:
        raise ValueError("VERPO_CONTRASTIVE Teacher requires a non-empty ground-truth answer")
    content = (
        f"Problem: {problem}\n\n"
        "Here is a reference solution to this problem:\n"
        "=== Reference Solution Begin ===\n"
        f"{solution}\n\nCorrect final answer: {answer}\n"
        "=== Reference Solution End ==="
        f"{TEACHER_TRANSITION_PROMPT}\n"
        f"{BOXED_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def build_sdpo_teacher_messages(raw_prompt: Any, solution: str) -> list[dict[str, str]]:
    """Apply the official SDPO reprompt/solution templates to raw messages."""
    solution = str(solution or "").strip()
    if not solution:
        raise ValueError("SDPO Correct solution Teacher requires a non-empty solution")
    if isinstance(raw_prompt, (list, tuple)) and raw_prompt:
        if any(not isinstance(message, dict) for message in raw_prompt):
            raise ValueError("SDPO Teacher raw prompt messages must all be dictionaries")
        raw_messages = [dict(message) for message in raw_prompt]
    elif isinstance(raw_prompt, dict):
        raw_messages = [dict(raw_prompt)]
    else:
        raw_messages = [{"role": "user", "content": _content_to_text(raw_prompt)}]
    if not raw_messages:
        raise ValueError("SDPO Teacher requires at least one raw prompt message")
    if raw_messages[-1].get("role") != "user":
        raise ValueError("SDPO Teacher requires the final raw prompt message to have role=user")
    prompt_text = _content_to_text(raw_messages[-1].get("content", ""))
    if not prompt_text.strip():
        raise ValueError("SDPO Teacher requires a non-empty final prompt message")
    solution_section = SDPO_SOLUTION_TEMPLATE.format(successful_previous_attempt=solution)
    reprompt_text = SDPO_REPROMPT_TEMPLATE.format(
        prompt=prompt_text,
        solution=solution_section,
        feedback="",
    )
    return raw_messages[:-1] + [{"role": "user", "content": reprompt_text}]


def is_sdpo_candidate_format_valid(text: str, reward_model: Any) -> bool:
    """Apply the upstream SDPO format gate for the registered task family."""
    reward_model = reward_model if isinstance(reward_model, dict) else {}
    style = str(reward_model.get("style", "")).strip().lower()
    if style in _SDPO_MCQ_STYLES:
        return _SDPO_MCQ_FORMAT_RE.search(str(text or "")) is not None
    if style in _SDPO_TOOLUSE_STYLES:
        return _SDPO_TOOLUSE_FORMAT_RE.search(str(text or "")) is not None
    raise ValueError(f"Unsupported SDPO reward_model.style for format validation: {style!r}")


def build_contrastive_teacher_messages(
    problem: str,
    candidate_response: str,
    *,
    raw_prompt: Any | None = None,
    template_variant: str = VERPO_CONTRASTIVE_CANDIDATE_HINT_TEMPLATE_ID,
) -> list[dict[str, str]]:
    """Build either the official SDPO q+ prompt or explicit q- prompt."""
    problem = strip_official_math_prompt(problem)
    candidate_response = str(candidate_response or "").strip()
    if not candidate_response:
        raise ValueError("Contrastive Teacher requires a non-empty candidate response")
    if template_variant == SDPO_TEACHER_TEMPLATE_ID:
        source_prompt = raw_prompt if raw_prompt is not None else [{"role": "user", "content": problem}]
        return build_sdpo_teacher_messages(source_prompt, candidate_response)
    if template_variant != VERPO_CONTRASTIVE_CANDIDATE_HINT_TEMPLATE_ID:
        raise ValueError(f"Unsupported Teacher prompt template: {template_variant}")
    if raw_prompt is not None:
        # Keep the dataset's original system message and output-format contract
        # (for example, SDPO's <reasoning>/<answer> format).  The sibling is
        # explicitly marked as an incorrect candidate, never as a reference.
        if isinstance(raw_prompt, (list, tuple)) and raw_prompt:
            if any(not isinstance(message, dict) for message in raw_prompt):
                raise ValueError("Contrastive Teacher raw prompt messages must all be dictionaries")
            raw_messages = [dict(message) for message in raw_prompt]
        elif isinstance(raw_prompt, dict):
            raw_messages = [dict(raw_prompt)]
        else:
            raw_messages = [{"role": "user", "content": _content_to_text(raw_prompt)}]
        if not raw_messages or raw_messages[-1].get("role") != "user":
            raise ValueError("Contrastive Teacher requires the final raw prompt message to have role=user")
        prompt_text = _content_to_text(raw_messages[-1].get("content", ""))
        if not prompt_text.strip():
            raise ValueError("Contrastive Teacher requires a non-empty final raw prompt message")
        hint = f"\n\n{INCORRECT_CANDIDATE_LABEL}\n\n{candidate_response}"
        return raw_messages[:-1] + [{"role": "user", "content": prompt_text + hint}]
    content = (
        f"Problem: {problem}\n\n"
        f"{INCORRECT_CANDIDATE_LABEL}\n\n"
        f"{candidate_response}\n\n"
        f"{BOXED_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def _budget_contrastive_teacher_prompt(
    *,
    tokenizer,
    problem: str,
    candidate_response: str,
    max_prompt_tokens: int,
    chat_template_kwargs: dict[str, Any] | None,
    raw_prompt: Any | None = None,
    template_variant: str = VERPO_CONTRASTIVE_CANDIDATE_HINT_TEMPLATE_ID,
) -> tuple[list[int], int, int]:
    """Keep the problem/scaffold intact and truncate only the sibling hint.

    The sampled completion suffix is budgeted by the caller and is never
    changed. If the fixed problem plus label-blind scaffold cannot fit, fail
    closed instead of silently deleting the problem from the Teacher input.
    """

    if int(max_prompt_tokens) <= 0:
        raise ValueError("Contrastive Teacher prompt has no token budget")

    def render(candidate: str) -> list[int]:
        prompt_text = tokenizer.apply_chat_template(
            build_contrastive_teacher_messages(
                problem,
                candidate,
                raw_prompt=raw_prompt,
                template_variant=template_variant,
            ),
            tokenize=False,
            add_generation_prompt=True,
            **dict(chat_template_kwargs or {}),
        )
        return list(tokenizer.encode(prompt_text, add_special_tokens=False))

    full_prompt_ids = render(candidate_response)
    if len(full_prompt_ids) <= int(max_prompt_tokens):
        return full_prompt_ids, len(full_prompt_ids), len(full_prompt_ids)

    candidate_ids = list(tokenizer.encode(candidate_response, add_special_tokens=False))

    def truncated_candidate(keep_tokens: int) -> str:
        if keep_tokens >= len(candidate_ids):
            return candidate_response
        head_tokens = (int(keep_tokens) + 1) // 2
        tail_tokens = int(keep_tokens) // 2
        head = tokenizer.decode(candidate_ids[:head_tokens], skip_special_tokens=True)
        tail = (
            tokenizer.decode(candidate_ids[-tail_tokens:], skip_special_tokens=True)
            if tail_tokens
            else ""
        )
        return f"{head}{TRUNCATED_CONTRASTIVE_HINT_MARKER}{tail}".strip()

    minimal_prompt_ids = render(truncated_candidate(0))
    if len(minimal_prompt_ids) > int(max_prompt_tokens):
        raise ValueError(
            "Fixed contrastive Teacher problem/template plus the exact sampled "
            "suffix exceeds the Teacher token budget; refusing to truncate the problem"
        )

    low = 0
    high = len(candidate_ids)
    best_prompt_ids = minimal_prompt_ids
    while low <= high:
        keep_tokens = (low + high) // 2
        prompt_ids = render(truncated_candidate(keep_tokens))
        if len(prompt_ids) <= int(max_prompt_tokens):
            best_prompt_ids = prompt_ids
            low = keep_tokens + 1
        else:
            high = keep_tokens - 1

    return best_prompt_ids, len(full_prompt_ids), len(best_prompt_ids)


def _content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in value)
    return str(value or "")


def raw_prompt_to_problem(raw_prompt: Any) -> str:
    content: str
    if isinstance(raw_prompt, (list, tuple)) and raw_prompt:
        message = raw_prompt[-1]
        if isinstance(message, dict):
            content = _content_to_text(message.get("content", ""))
        else:
            content = _content_to_text(raw_prompt)
    elif isinstance(raw_prompt, dict):
        content = _content_to_text(raw_prompt.get("content", ""))
    else:
        content = _content_to_text(raw_prompt)
    if "[TEACHER_CONTEXT_TOKEN]" in content:
        from sdpg_reproduction.src.sdpg_prompt import split_sdpg_prompt

        content, _ = split_sdpg_prompt(content)
    return strip_official_math_prompt(content)


def _as_plain_list(values: Any, expected: int, name: str) -> list[Any]:
    items = list(values)
    if len(items) != expected:
        raise ValueError(f"{name} has {len(items)} rows, expected {expected}")
    return items


def _resolve_chat_template_kwargs(
    chat_template_kwargs: dict[str, Any] | None,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    """Merge the legacy Qwen3 flag into the model-neutral kwargs interface."""
    resolved = dict(chat_template_kwargs or {})
    if enable_thinking is None:
        return resolved
    if "enable_thinking" in resolved and bool(resolved["enable_thinking"]) != bool(enable_thinking):
        raise ValueError("Conflicting enable_thinking chat-template settings")
    resolved.setdefault("enable_thinking", bool(enable_thinking))
    return resolved


def build_evidence_teacher_fields(
    *,
    tokenizer,
    responses: torch.Tensor,
    raw_prompts: Any,
    reward_models: Any,
    extra_infos: Any,
    total_token_budget: int,
    max_reprompt_tokens: int = 0,
    chat_template_kwargs: dict[str, Any] | None = None,
    enable_thinking: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Build left-truncated evidence prompts with an unchanged response suffix."""
    chat_template_kwargs = _resolve_chat_template_kwargs(chat_template_kwargs, enable_thinking)
    if not responses.is_nested:
        raise ValueError("VERPO evidence construction requires nested response tensors")
    response_rows = list(responses.unbind())
    batch_size = len(response_rows)
    raw_prompt_rows = _as_plain_list(raw_prompts, batch_size, "raw_prompt")
    reward_rows = _as_plain_list(reward_models, batch_size, "reward_model")
    extra_rows = _as_plain_list(extra_infos, batch_size, "extra_info")

    input_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    kept_prompt_lens: list[int] = []
    raw_prompt_lens: list[int] = []
    truncated_tokens: list[int] = []
    for index, response in enumerate(response_rows):
        reward_model = reward_rows[index] if isinstance(reward_rows[index], dict) else {}
        extra_info = extra_rows[index] if isinstance(extra_rows[index], dict) else {}
        answer = reward_model.get("ground_truth", "")
        problem = _content_to_text(extra_info.get("problem", "")).strip()
        if not problem:
            problem = raw_prompt_to_problem(raw_prompt_rows[index])
        solution = _content_to_text(extra_info.get("solution", "")).strip()
        template_variant = str(extra_info.get("teacher_prompt_template", "verpo_contrastive_reference_solution"))
        if template_variant == SDPO_TEACHER_TEMPLATE_ID:
            messages = build_sdpo_teacher_messages(raw_prompt_rows[index], solution)
        elif template_variant == "verpo_contrastive_reference_solution":
            messages = build_teacher_messages(problem=problem, answer=str(answer), solution=solution)
        else:
            raise ValueError(f"Unsupported Teacher prompt template: {template_variant}")
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **dict(chat_template_kwargs or {}),
        )
        prompt_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
        response_ids = response.detach().cpu().long().tolist()
        if not response_ids:
            raise ValueError(f"Sample {index} has an empty response; VERPO requires a sampled completion")
        if len(response_ids) >= total_token_budget:
            raise ValueError(
                f"Sample {index} response has {len(response_ids)} tokens, exceeding Teacher budget {total_token_budget}"
            )
        max_prompt_tokens = total_token_budget - len(response_ids)
        if int(max_reprompt_tokens) > 0:
            max_prompt_tokens = min(max_prompt_tokens, int(max_reprompt_tokens))
        kept_prompt_ids = prompt_ids[-max_prompt_tokens:] if len(prompt_ids) > max_prompt_tokens else prompt_ids
        combined = torch.tensor(kept_prompt_ids + response_ids, dtype=torch.long)
        if combined[-len(response_ids) :].tolist() != response_ids:
            raise AssertionError("Evidence Teacher completion suffix changed during construction")
        input_rows.append(combined)
        position_rows.append(torch.arange(combined.numel(), dtype=torch.long))
        kept_prompt_lens.append(len(kept_prompt_ids))
        raw_prompt_lens.append(len(prompt_ids))
        truncated_tokens.append(len(prompt_ids) - len(kept_prompt_ids))

    return {
        "verpo_evidence_input_ids": torch.nested.as_nested_tensor(input_rows, layout=torch.jagged),
        "verpo_evidence_position_ids": torch.nested.as_nested_tensor(position_rows, layout=torch.jagged),
        "verpo_evidence_prompt_lengths": torch.tensor(kept_prompt_lens, dtype=torch.long),
        "verpo_teacher_prompt_truncated": torch.tensor([value > 0 for value in truncated_tokens], dtype=torch.bool),
        "verpo_teacher_prompt_truncated_tokens": torch.tensor(truncated_tokens, dtype=torch.float32),
        "verpo_teacher_prompt_tokens_before_trunc": torch.tensor(raw_prompt_lens, dtype=torch.float32),
        "verpo_teacher_prompt_tokens_after_trunc": torch.tensor(kept_prompt_lens, dtype=torch.float32),
    }


def build_contrastive_evidence_teacher_fields(
    *,
    tokenizer,
    responses: torch.Tensor,
    raw_prompts: Any,
    rewards: torch.Tensor,
    correctness: torch.Tensor | None = None,
    uids: list[str],
    total_token_budget: int,
    max_reprompt_tokens: int = 0,
    reward_models: Any | None = None,
    extra_infos: Any | None = None,
    num_negative_hints: int = 4,
    selection_mode: str = "correctness",
    chat_template_kwargs: dict[str, Any] | None = None,
    enable_thinking: bool | None = None,
    allow_unboxed_candidates: bool = False,
) -> dict[str, torch.Tensor]:
    """Build contrastive Teacher inputs under an explicit selection mode.

    ``correctness`` uses the dataset reference solution plus ground-truth answer
    for q+ and target-excluded format-valid incorrect siblings for q-. It never
    infers correctness from a length-shaped reward. ``reward_ranked`` retains
    the explicit highest/lowest final-reward sibling contrast. The q+ scaffold
    follows the configured positive-template metadata for correctness mode;
    every q- branch uses the explicit incorrect-candidate scaffold. All
    branches keep the identical sampled suffix.
    """
    chat_template_kwargs = _resolve_chat_template_kwargs(chat_template_kwargs, enable_thinking)
    if not responses.is_nested:
        raise ValueError("VERPO contrastive construction requires nested response tensors")
    if int(num_negative_hints) <= 0:
        raise ValueError("num_negative_hints must be positive")
    response_rows = list(responses.unbind())
    batch_size = len(response_rows)
    raw_prompt_rows = _as_plain_list(raw_prompts, batch_size, "raw_prompt")
    uid_rows = _as_plain_list(uids, batch_size, "uid")
    reward_rows = rewards.detach().float().flatten().cpu()
    if reward_rows.numel() != batch_size:
        raise ValueError("rewards must have one value per response")

    if selection_mode not in {"correctness", "reward_ranked"}:
        raise ValueError("selection_mode must be correctness or reward_ranked")
    response_texts = [
        tokenizer.decode(response.detach().cpu().long().tolist(), skip_special_tokens=True).strip()
        for response in response_rows
    ]
    if selection_mode == "correctness" and correctness is None:
        raise ValueError(
            "selection_mode=correctness requires independent per-rollout correctness; "
            "refusing to infer correctness from length-shaped rewards"
        )
    external_positive_candidates: list[str] | None = None
    teacher_template_variants = [VERPO_CONTRASTIVE_CANDIDATE_HINT_TEMPLATE_ID] * batch_size
    reward_model_rows: list[Any] | None = None
    if selection_mode == "correctness":
        if reward_models is None or extra_infos is None:
            raise ValueError(
                "selection_mode=correctness requires reward_models and extra_infos "
                "to construct the external positive evidence"
            )
        reward_model_rows = _as_plain_list(reward_models, batch_size, "reward_model")
        extra_info_rows = _as_plain_list(extra_infos, batch_size, "extra_info")
        external_positive_candidates = []
        for row, (reward_model_value, extra_info_value) in enumerate(
            zip(reward_model_rows, extra_info_rows, strict=True)
        ):
            reward_model = reward_model_value if isinstance(reward_model_value, dict) else {}
            extra_info = extra_info_value if isinstance(extra_info_value, dict) else {}
            answer = _content_to_text(reward_model.get("ground_truth", "")).strip()
            solution = _content_to_text(extra_info.get("solution", "")).strip()
            teacher_template_variants[row] = str(
                extra_info.get("teacher_prompt_template", VERPO_CONTRASTIVE_CANDIDATE_HINT_TEMPLATE_ID)
            )
            if not solution:
                raise ValueError(f"Sample {row} requires a non-empty external reference solution for q+")
            if not answer:
                raise ValueError(f"Sample {row} requires a non-empty ground-truth answer for q+")
            if teacher_template_variants[row] == SDPO_TEACHER_TEMPLATE_ID:
                # Official SDPO inserts the successful solution verbatim into
                # ``solution_template``. The generated SDPO solution already
                # contains its validated final answer, so adding VERPO_CONTRASTIVE's
                # wrapper here would silently change the training reprompt.
                external_positive_candidates.append(solution)
            else:
                external_positive_candidates.append(f"{solution}\n\nFinal answer: {answer}")
    if bool(allow_unboxed_candidates):
        format_valid = [True] * batch_size
    elif selection_mode == "correctness":
        assert reward_model_rows is not None
        format_valid = [
            is_sdpo_candidate_format_valid(text, reward_model)
            if teacher_template_variants[row] == SDPO_TEACHER_TEMPLATE_ID
            else extract_boxed_answer(text) is not None
            for row, (text, reward_model) in enumerate(
                zip(response_texts, reward_model_rows, strict=True)
            )
        ]
    else:
        format_valid = [extract_boxed_answer(text) is not None for text in response_texts]
    if correctness is not None:
        correctness_rows = correctness.detach().bool().flatten().cpu()
        if correctness_rows.numel() != batch_size:
            raise ValueError("correctness must have one value per response")
        correct = correctness_rows.tolist()
    else:
        # Reward-ranked selection deliberately ignores correctness and orders
        # siblings by the final scalar reward used by GRPO and the group gate.
        correct = [False] * batch_size
    groups: dict[str, list[int]] = {}
    for index, uid in enumerate(uid_rows):
        groups.setdefault(str(uid), []).append(index)

    positive_indices: list[int] = []
    negative_indices: list[list[int]] = []
    available: list[bool] = []
    for row, uid in enumerate(uid_rows):
        group = groups[str(uid)]
        if selection_mode == "reward_ranked":
            siblings = [index for index in group if index != row]
            sibling_rewards = [float(reward_rows[index].item()) for index in siblings]
            row_available = bool(
                siblings and max(sibling_rewards) > min(sibling_rewards) + 1e-8
            )
        else:
            negative_pool = [
                index
                for index in group
                if index != row and format_valid[index] and not correct[index]
            ]
            row_available = bool(negative_pool)
        available.append(row_available)
        if row_available and selection_mode == "reward_ranked":
            highest = max(sibling_rewards)
            lowest = min(sibling_rewards)
            positive_pool = [
                index
                for index in siblings
                if abs(float(reward_rows[index].item()) - highest) <= 1e-8
            ]
            negative_pool = [
                index
                for index in siblings
                if abs(float(reward_rows[index].item()) - lowest) <= 1e-8
            ]
        if row_available:
            positive_index = (
                positive_pool[row % len(positive_pool)]
                if selection_mode == "reward_ranked"
                else -1
            )
            offset = row % len(negative_pool)
            ordered_negative = negative_pool[offset:] + negative_pool[:offset]
            row_negative_indices = [
                ordered_negative[index % len(ordered_negative)] for index in range(int(num_negative_hints))
            ]
        else:
            positive_index = -1
            row_negative_indices = [-1] * int(num_negative_hints)
        positive_indices.append(positive_index)
        negative_indices.append(row_negative_indices)

    fields: dict[str, torch.Tensor] = {
        "verpo_contrastive_available": torch.tensor(available, dtype=torch.bool),
        "verpo_positive_sibling_index": torch.tensor(positive_indices, dtype=torch.long),
        "verpo_negative_sibling_indices": torch.tensor(negative_indices, dtype=torch.long),
    }
    truncation_by_teacher: list[list[int]] = [[] for _ in range(batch_size)]
    prompt_before_by_teacher: list[list[int]] = [[] for _ in range(batch_size)]
    prompt_after_by_teacher: list[list[int]] = [[] for _ in range(batch_size)]

    def build_branch(prefix: str, candidates: list[str], template_variants: list[str]) -> None:
        if len(template_variants) != batch_size:
            raise ValueError(f"{prefix} template variants must have one entry per response")
        input_rows: list[torch.Tensor] = []
        position_rows: list[torch.Tensor] = []
        kept_prompt_lens: list[int] = []
        for row, response in enumerate(response_rows):
            problem = raw_prompt_to_problem(raw_prompt_rows[row])
            candidate = candidates[row]
            response_ids = response.detach().cpu().long().tolist()
            if not response_ids:
                raise ValueError(f"Sample {row} has an empty response; VERPO requires a sampled completion")
            if len(response_ids) >= total_token_budget:
                raise ValueError(
                    f"Sample {row} response has {len(response_ids)} tokens, exceeding Teacher budget "
                    f"{total_token_budget}"
                )
            max_prompt_tokens = total_token_budget - len(response_ids)
            if int(max_reprompt_tokens) > 0:
                max_prompt_tokens = min(max_prompt_tokens, int(max_reprompt_tokens))
            prompt_ids, prompt_tokens_before, prompt_tokens_after = _budget_contrastive_teacher_prompt(
                tokenizer=tokenizer,
                problem=problem,
                candidate_response=candidate,
                max_prompt_tokens=max_prompt_tokens,
                chat_template_kwargs=chat_template_kwargs,
                raw_prompt=raw_prompt_rows[row],
                template_variant=template_variants[row],
            )
            combined = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
            if combined[-len(response_ids) :].tolist() != response_ids:
                raise AssertionError("Contrastive Teacher completion suffix changed during construction")
            input_rows.append(combined)
            position_rows.append(torch.arange(combined.numel(), dtype=torch.long))
            kept_prompt_lens.append(len(prompt_ids))
            truncation_by_teacher[row].append(prompt_tokens_before - prompt_tokens_after)
            prompt_before_by_teacher[row].append(prompt_tokens_before)
            prompt_after_by_teacher[row].append(prompt_tokens_after)
        fields[f"{prefix}_input_ids"] = torch.nested.as_nested_tensor(input_rows, layout=torch.jagged)
        fields[f"{prefix}_position_ids"] = torch.nested.as_nested_tensor(position_rows, layout=torch.jagged)
        fields[f"{prefix}_prompt_lengths"] = torch.tensor(kept_prompt_lens, dtype=torch.long)

    if selection_mode == "correctness":
        assert external_positive_candidates is not None
        positive_candidates = external_positive_candidates
        positive_template_variants = teacher_template_variants
    else:
        positive_candidates = [
            response_texts[index] if index >= 0 else UNAVAILABLE_CONTRASTIVE_HINT
            for index in positive_indices
        ]
        # Reward-ranked siblings are not guaranteed to be correct, so they
        # use the explicit candidate scaffold rather than SDPO's
        # ``Correct solution`` scaffold.
        positive_template_variants = [VERPO_CONTRASTIVE_CANDIDATE_HINT_TEMPLATE_ID] * batch_size
    build_branch("verpo_positive", positive_candidates, positive_template_variants)
    # A target-excluded incorrect sibling is never a ``Correct solution``.
    # Use the explicit incorrect-candidate scaffold for every q- branch,
    # including datasets whose q+ metadata selects the official SDPO template.
    negative_template_variants = [VERPO_CONTRASTIVE_CANDIDATE_HINT_TEMPLATE_ID] * batch_size
    for negative_index in range(int(num_negative_hints)):
        build_branch(
            f"verpo_negative_{negative_index}",
            [
                response_texts[indices[negative_index]]
                if indices[negative_index] >= 0
                else UNAVAILABLE_CONTRASTIVE_HINT
                for indices in negative_indices
            ],
            negative_template_variants,
        )

    max_truncation = [max(values, default=0) for values in truncation_by_teacher]
    max_before = [max(values, default=0) for values in prompt_before_by_teacher]
    min_after = [min(values, default=0) for values in prompt_after_by_teacher]
    fields.update(
        {
            "verpo_teacher_prompt_truncated": torch.tensor([value > 0 for value in max_truncation], dtype=torch.bool),
            "verpo_teacher_prompt_truncated_tokens": torch.tensor(max_truncation, dtype=torch.float32),
            "verpo_teacher_prompt_tokens_before_trunc": torch.tensor(max_before, dtype=torch.float32),
            "verpo_teacher_prompt_tokens_after_trunc": torch.tensor(min_after, dtype=torch.float32),
        }
    )
    return fields


def extract_boxed_answer(text: str | None) -> str | None:
    if text is None:
        return None
    index = text.rfind("\\boxed")
    if index < 0:
        return None
    depth = 0
    right = None
    for cursor in range(index, len(text)):
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
            if depth == 0:
                right = cursor
                break
    if right is None:
        return None
    boxed = text[index : right + 1]
    return boxed[7:-1].strip() if boxed.startswith("\\boxed{") and boxed.endswith("}") else None


def _normalize_fallback_string(text: str) -> str:
    return str(text).replace("$", "").replace(" ", "").lower().strip()


def _strip_answer_wrappers(text: str) -> str:
    value = str(text).strip()
    boxed = extract_boxed_answer(value)
    if boxed is not None:
        value = boxed
    value = value.strip().strip("$").strip()
    if value.startswith(r"\(") and value.endswith(r"\)"):
        value = value[2:-2].strip()
    if value.startswith(r"\[") and value.endswith(r"\]"):
        value = value[2:-2].strip()
    return value.replace(r"\left", "").replace(r"\right", "").strip()


def _simple_number(text: str) -> Decimal | None:
    value = _strip_answer_wrappers(text)
    value = _DEGREE_SUFFIX_RE.sub("", value.replace(" ", "").replace(",", "").replace("−", "-"))
    if not _SIMPLE_NUMBER_RE.fullmatch(value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def grade_boxed_answer(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    pred_number, gt_number = _simple_number(predicted), _simple_number(str(ground_truth))
    if pred_number is not None and gt_number is not None:
        return pred_number == gt_number
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ModuleNotFoundError("math_verify is required for VERPO_CONTRASTIVE-aligned grading; install math-verify") from exc
    try:
        pred_text = predicted if "$" in predicted else f"${predicted}$"
        gt_text = str(ground_truth) if "$" in str(ground_truth) else f"${ground_truth}$"
        return bool(
            verify(
                parse(gt_text, fallback_mode="no_fallback"),
                parse(pred_text, fallback_mode="no_fallback"),
                timeout_seconds=5,
            )
        )
    except Exception:
        return _normalize_fallback_string(predicted) == _normalize_fallback_string(ground_truth)
