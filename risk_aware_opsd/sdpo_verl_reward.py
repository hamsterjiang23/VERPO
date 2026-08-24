"""SDPO Section 3 format-gated rewards for SciKnowEval and ToolAlpaca.

Semantic accuracy stays independent for CTR/FEC sibling labels, while the
optimization score also requires the registered output format. The module is
dependency-free so native veRL can load it as a custom reward.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

SCIENCE_SOURCES = {
    "sciknoweval",
    "biology",
    "chemistry",
    "material",
    "physics",
}


def _science_score(solution: str, ground_truth: str) -> dict[str, Any]:
    answer = solution.split("<answer>")[-1].split("</answer>")[0].strip()
    correct_format = (
        re.search(r"<answer>\s*(A|B|C|D)\s*</answer>$", solution) is not None
    )
    correct = float(answer == str(ground_truth).strip())
    reward = correct * float(correct_format)
    return {
        "score": reward,
        # Keep semantic correctness independent for CTR/FEC sibling selection.
        "acc": correct,
        "pred": answer,
        "incorrect_format": 0 if correct_format else 1,
        "feedback": "",
    }


def _extract_actions(text: str) -> list[str]:
    return re.findall(r"Action:\s*(\w+)", text)


def _extract_action_inputs(text: str) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for block in re.findall(r"Action Input:\s*({.*?})", text, re.DOTALL):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            combined.update(parsed)
    return combined


def _tooluse_score(solution: str, ground_truth: Any) -> dict[str, Any]:
    try:
        gold = (
            json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        )
    except json.JSONDecodeError:
        gold = None
    if not isinstance(gold, list):
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": "",
            "incorrect_format": 1,
            "feedback": "Failed to parse ground truth JSON",
        }

    gold_actions: list[str] = []
    gold_inputs: dict[str, Any] = {}
    for item in gold:
        if not isinstance(item, dict):
            continue
        gold_actions.append(str(item.get("Action", "")))
        raw_input = item.get("Action_Input")
        try:
            parsed_input = (
                json.loads(raw_input) if isinstance(raw_input, str) else raw_input
            )
        except json.JSONDecodeError:
            parsed_input = None
        if isinstance(parsed_input, dict):
            gold_inputs.update(parsed_input)

    pred_actions = _extract_actions(solution)
    pred_inputs = _extract_action_inputs(solution)
    actions_correct = Counter(pred_actions) == Counter(gold_actions)
    inputs_correct = pred_inputs == gold_inputs
    correct_format = (
        re.search(r"Action:.*?\nAction Input:.*?", solution, re.DOTALL) is not None
    )
    correct = float(actions_correct and inputs_correct)
    reward = correct * float(correct_format)
    feedback_parts: list[str] = []
    if not actions_correct:
        feedback_parts.append(
            f"Actions mismatch: predicted {pred_actions}, expected {gold_actions}"
        )
    if not inputs_correct:
        feedback_parts.append(
            f"Action inputs mismatch: predicted {pred_inputs}, expected {gold_inputs}"
        )
    return {
        "score": reward,
        # Keep semantic correctness independent for CTR/FEC sibling selection.
        "acc": correct,
        "pred": f"Actions: {pred_actions}, Inputs: {pred_inputs}",
        "incorrect_format": 0 if correct_format else 1,
        "feedback": "; ".join(feedback_parts),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch the exact Section 3 task-family reward."""

    del extra_info
    source = str(data_source).strip().lower()
    if source in SCIENCE_SOURCES:
        return _science_score(solution_str, str(ground_truth))
    if source == "tooluse":
        return _tooluse_score(solution_str, ground_truth)
    raise ValueError(f"SDPO Section 3 reward style {data_source!r} not found")


__all__ = ["compute_score"]
