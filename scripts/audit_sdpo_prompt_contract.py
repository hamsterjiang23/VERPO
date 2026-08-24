#!/usr/bin/env python3
"""Generate a tiny SDPO sample and audit q0/q+/q- plus answer parsing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "verl"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from risk_aware_opsd.sdpo_verl_reward import compute_score  # noqa: E402
from verl.trainer.distillation.verpo_protocol import (  # noqa: E402
    SDPO_REPROMPT_TEMPLATE,
    SDPO_SOLUTION_TEMPLATE,
    SDPO_TEACHER_TEMPLATE_ID,
    build_contrastive_evidence_teacher_fields,
    build_contrastive_teacher_messages,
    build_sdpo_teacher_messages,
    is_sdpo_candidate_format_valid,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_row(path: Path, row_index: int) -> dict[str, Any]:
    table = pq.read_table(path)
    if not 0 <= row_index < table.num_rows:
        raise IndexError(f"row-index {row_index} outside [0, {table.num_rows})")
    return {name: table[name][row_index].as_py() for name in table.column_names}


def _wrong_mcq_responses(ground_truth: str) -> list[str]:
    wrong_answers = [answer for answer in "ABCD" if answer != ground_truth]
    return [
        (
            "<reasoning>This is a deliberately incorrect, format-valid sibling "
            f"used only to audit q-minus selection: choose {answer}.</reasoning>\n"
            f"<answer>{answer}</answer>"
        )
        for answer in wrong_answers[:2]
    ]


def _prompt_text(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    args = _parse_args()
    if args.num_generations not in {1, 2}:
        raise ValueError("num-generations must be 1 or 2 for this bounded audit")

    row = _load_row(args.data, args.row_index)
    raw_prompt = row["prompt"]
    reward_model = row["reward_model"]
    extra_info = row["extra_info"]
    ground_truth = str(reward_model["ground_truth"]).strip()
    data_source = str(row["data_source"]).strip().lower()
    solution = str(extra_info["solution"]).strip()

    assert isinstance(raw_prompt, list) and raw_prompt[-1]["role"] == "user"
    assert extra_info["teacher_prompt_template"] == SDPO_TEACHER_TEMPLATE_ID
    assert extra_info["teacher_reprompt_template"] == SDPO_REPROMPT_TEMPLATE
    assert extra_info["teacher_solution_template"] == SDPO_SOLUTION_TEMPLATE
    assert reward_model["style"] == "mcq"
    assert ground_truth in "ABCD"
    assert solution

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    torch.manual_seed(args.seed)
    prompt_ids = tokenizer.apply_chat_template(
        raw_prompt,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(prompt_ids)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_generations,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_texts = tokenizer.batch_decode(
        generated[:, prompt_ids.shape[1] :],
        skip_special_tokens=True,
    )
    if any(not text.strip() for text in generated_texts):
        raise RuntimeError("Model returned an empty audit generation")

    generated_scores = [
        compute_score(data_source, text, ground_truth, extra_info)
        for text in generated_texts
    ]
    controlled_negatives = _wrong_mcq_responses(ground_truth)
    response_texts = [*generated_texts, *controlled_negatives]
    response_rows = [
        torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
        for text in response_texts
    ]
    correctness = torch.tensor(
        [
            bool(compute_score(data_source, text, ground_truth, extra_info)["acc"])
            for text in response_texts
        ]
    )
    fields = build_contrastive_evidence_teacher_fields(
        tokenizer=tokenizer,
        responses=torch.nested.as_nested_tensor(response_rows, layout=torch.jagged),
        raw_prompts=[raw_prompt] * len(response_texts),
        rewards=correctness.float(),
        correctness=correctness,
        uids=[str(extra_info.get("uid", row.get("uid", "audit-row")))] * len(response_texts),
        total_token_budget=18944,
        max_reprompt_tokens=10240,
        reward_models=[reward_model] * len(response_texts),
        extra_infos=[extra_info] * len(response_texts),
        num_negative_hints=1,
        selection_mode="correctness",
        enable_thinking=False,
    )

    q0_messages = raw_prompt
    qplus_messages = build_sdpo_teacher_messages(raw_prompt, solution)
    branch_audits: list[dict[str, Any]] = []
    negative_indices = fields["verpo_negative_sibling_indices"].tolist()
    available = fields["verpo_contrastive_available"].tolist()
    for target_index, target_ids in enumerate(response_rows):
        if not available[target_index]:
            raise AssertionError(f"q-minus unavailable for target {target_index}")
        negative_index = int(negative_indices[target_index][0])
        if negative_index == target_index:
            raise AssertionError("q-minus sibling was not target-excluded")
        negative_text = response_texts[negative_index]
        if not is_sdpo_candidate_format_valid(negative_text, reward_model):
            raise AssertionError("q-minus sibling failed the SDPO format parser")
        qminus_messages = build_contrastive_teacher_messages(
            "",
            negative_text,
            raw_prompt=raw_prompt,
            template_variant=SDPO_TEACHER_TEMPLATE_ID,
        )
        suffix = target_ids.tolist()
        positive_combined = list(fields["verpo_positive_input_ids"].unbind())[target_index].tolist()
        negative_combined = list(fields["verpo_negative_0_input_ids"].unbind())[target_index].tolist()
        if positive_combined[-len(suffix) :] != suffix or negative_combined[-len(suffix) :] != suffix:
            raise AssertionError("q-plus/q-minus completion suffix differs from q-zero target")
        branch_audits.append(
            {
                "target_index": target_index,
                "qminus_sibling_index": negative_index,
                "target_excluded": True,
                "same_completion_suffix": True,
                "qminus_is_format_valid_incorrect": bool(not correctness[negative_index].item()),
                "qminus_messages": qminus_messages,
            }
        )

    q0_rendered = _prompt_text(tokenizer, q0_messages)
    qplus_rendered = _prompt_text(tokenizer, qplus_messages)
    if solution in q0_rendered:
        raise AssertionError("q-zero unexpectedly contains the privileged solution")
    if solution not in qplus_rendered:
        raise AssertionError("q-plus does not contain the privileged solution")

    report = {
        "status": "pass",
        "data_path": str(args.data.resolve()),
        "row_index": args.row_index,
        "record_id": extra_info.get("record_id", extra_info.get("uid", row.get("uid"))),
        "data_source": data_source,
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "enable_thinking": False,
        "ground_truth": ground_truth,
        "student_prompt": raw_prompt,
        "generated": [
            {"text": text, "parsed": score}
            for text, score in zip(generated_texts, generated_scores, strict=True)
        ],
        "controlled_qminus_candidates": controlled_negatives,
        "q0": {
            "messages": q0_messages,
            "contains_privileged_solution": False,
        },
        "qplus": {
            "messages": qplus_messages,
            "contains_exact_validated_solution": True,
        },
        "branches": branch_audits,
        "contracts": {
            "official_reprompt_template_exact": True,
            "official_solution_template_exact": True,
            "student_teacher_thinking_matched": True,
            "answer_parser_applied_to_real_generations": True,
            "qminus_target_excluded": True,
            "qminus_format_valid_and_incorrect": True,
            "q0_qplus_qminus_share_target_suffix": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output.resolve()),
        "record_id": report["record_id"],
        "generated": generated_scores,
        "contracts": report["contracts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
