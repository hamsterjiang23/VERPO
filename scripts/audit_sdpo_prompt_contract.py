#!/usr/bin/env python3
"""Generate a tiny SDPO sample and audit q0/q+/q- plus answer parsing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "verl"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from risk_aware_opsd.sdpo_verl_reward import compute_score


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_row(path: Path, row_index: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if not 0 <= row_index < table.num_rows:
        raise IndexError(f"row-index {row_index} outside [0, {table.num_rows})")
    return {name: table[name][row_index].as_py() for name in table.column_names}


def _prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    args = _parse_args()
    if not 1 <= args.num_generations <= 8:
        raise ValueError("num-generations must be in [1, 8] for this bounded audit")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.trainer.distillation.verpo_protocol import (
        VERPO_CONTRASTIVE_CANDIDATE_HINT_TEMPLATE_ID,
        build_contrastive_evidence_teacher_fields,
        build_contrastive_teacher_messages,
        build_sdpo_teacher_messages,
    )

    from risk_aware_opsd.rollout_evidence import REASONS

    row = _load_row(args.data, args.row_index)
    raw_prompt = row["prompt"]
    reward_model = row["reward_model"]
    extra_info = row["extra_info"]
    ground_truth = str(reward_model["ground_truth"]).strip()
    data_source = str(row["data_source"]).strip().lower()
    assert isinstance(raw_prompt, list) and raw_prompt[-1]["role"] == "user"

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
    response_texts = generated_texts
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
        uids=[str(extra_info.get("uid", row.get("uid", "audit-row")))]
        * len(response_texts),
        rollout_ids=[f"audit_{i}_0" for i in range(len(response_texts))],
        total_token_budget=18944,
        max_reprompt_tokens=10240,
        reward_models=[reward_model] * len(response_texts),
        extra_infos=[extra_info] * len(response_texts),
        num_negative_hints=1,
        selection_mode="correctness",
        enable_thinking=False,
    )

    branches: list[dict[str, Any]] = []
    positives = fields["verpo_positive_sibling_index"].tolist()
    negatives = fields["verpo_negative_sibling_indices"].tolist()
    available = fields["verpo_contrastive_available"].tolist()
    for target, suffix_tensor in enumerate(response_rows):
        pos, neg = positives[target], negatives[target][0]
        if target in (pos, neg):
            raise AssertionError("Teacher evidence is not target-excluded")
        suffix = suffix_tensor.tolist()
        for name in ("verpo_positive_input_ids", "verpo_negative_0_input_ids"):
            combined = list(fields[name].unbind())[target].tolist()
            if combined[-len(suffix) :] != suffix:
                raise AssertionError("Teacher replay changed the target suffix")
        branches.append(
            {
                "target_index": target,
                "positive_index": pos,
                "negative_index": neg,
                "available": available[target],
                "reason": REASONS[int(fields["verpo_evidence_reason"][target])],
                "qplus_messages": build_sdpo_teacher_messages(
                    raw_prompt, response_texts[pos]
                )
                if pos >= 0
                else None,
                "qminus_messages": build_contrastive_teacher_messages(
                    "",
                    response_texts[neg],
                    raw_prompt=raw_prompt,
                    template_variant=VERPO_CONTRASTIVE_CANDIDATE_HINT_TEMPLATE_ID,
                )
                if neg >= 0
                else None,
                "same_completion_suffix": True,
            }
        )
    report = {
        "status": "pass" if any(available) else "no_eligible_targets",
        "evidence_source": "rollout_group",
        "data_path": str(args.data.resolve()),
        "record_id": row.get("record_id"),
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "student_prompt": raw_prompt,
        "generated": [
            {"text": text, "parsed": score}
            for text, score in zip(generated_texts, generated_scores, strict=True)
        ],
        "branches": branches,
        "eligible_targets": sum(available),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.resolve()),
                "eligible_targets": sum(available),
            }
        )
    )


if __name__ == "__main__":
    main()
