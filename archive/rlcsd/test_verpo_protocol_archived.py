import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.distillation.rlcsd_protocol import (
    BOXED_ANSWER_INSTRUCTION,
    build_contrastive_evidence_teacher_fields,
    build_contrastive_teacher_messages,
    build_evidence_teacher_fields,
    build_sdpo_teacher_messages,
    build_rollout_messages,
    build_teacher_messages,
    extract_boxed_answer,
    grade_boxed_answer,
    is_sdpo_candidate_format_valid,
    raw_prompt_to_problem,
)
from verl.trainer.distillation.verpo_zpd import compute_group_zpd_gate_by_uid
from verl.trainer.ppo.utils import need_reference_policy
from verl.trainer.ppo.v1.trainer_base import (
    _resolve_verpo_outcome_positive,
    select_reward_variable_groups,
)
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.config.actor import VerpoZPDConfig


class _CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **chat_template_kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        assert set(chat_template_kwargs).issubset({"enable_thinking"})
        if "enable_thinking" in chat_template_kwargs:
            assert chat_template_kwargs["enable_thinking"] is False
        return messages[0]["content"] + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(chr(int(token_id)) for token_id in token_ids)


class _PlainCharacterTokenizer(_CharacterTokenizer):
    """Qwen2.5-style tokenizer stub that rejects model-specific template kwargs."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"] + "<assistant>"


class _MessageCharacterTokenizer(_CharacterTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **chat_template_kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        assert set(chat_template_kwargs).issubset({"enable_thinking"})
        return "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        ) + "<assistant>"


def _external_evidence_rows(batch_size: int, *, solution: str = "Add the two ones."):
    return (
        [{"ground_truth": "2"} for _ in range(batch_size)],
        [{"problem": "1+1?", "solution": solution} for _ in range(batch_size)],
    )


def test_official_prompt_bytes():
    train_content = build_rollout_messages("1+1?", "deepmath")[0]["content"]
    eval_content = build_rollout_messages("1+1?", "aime24")[0]["content"]
    assert train_content == (f"Problem: 1+1?\n\n{BOXED_ANSWER_INSTRUCTION}")
    assert eval_content == (f"1+1?\n\n{BOXED_ANSWER_INSTRUCTION}")
    assert "Reference Solution" not in train_content
    assert "Reference Solution" not in eval_content
    tokenizer = _CharacterTokenizer()
    assert tokenizer.encode(train_content) == [ord(char) for char in train_content]
    assert tokenizer.encode(eval_content) == [ord(char) for char in eval_content]
    teacher = build_teacher_messages("1+1?", "2", "Add the two ones.")[0]["content"]
    assert teacher == (
        "Problem: 1+1?\n\n"
        "Here is a reference solution to this problem:\n"
        "=== Reference Solution Begin ===\n"
        "Add the two ones.\n\nCorrect final answer: 2\n"
        "=== Reference Solution End ===\n\n"
        "After reading the reference solution above, make sure you understand the reasoning behind each step.\n\n"
        f"{BOXED_ANSWER_INSTRUCTION}"
    )


def test_official_sdpo_teacher_prompt_template_preserves_system_message():
    raw_prompt = [
        {"role": "system", "content": "Return <reasoning> and <answer>."},
        {"role": "user", "content": "Question with four options."},
    ]
    messages = build_sdpo_teacher_messages(raw_prompt, "Reference reasoning with answer B.")
    assert messages[0] == raw_prompt[0]
    assert messages[1] == {
        "role": "user",
        "content": (
            "Question with four options.\n"
            "Correct solution:\n\n"
            "Reference reasoning with answer B.\n\n\n\n"
            "Correctly solve the original question.\n"
        ),
    }


@pytest.mark.parametrize(
    ("text", "style", "expected"),
    [
        ("<reasoning>work</reasoning>\n<answer> A </answer>", "mcq", True),
        ("<reasoning>work</reasoning>\n<answer>E</answer>", "mcq", False),
        ("<answer>A</answer> trailing", "mcq", False),
        ('Action: search\nAction Input: {"query": "x"}', "tooluse", True),
        ('Action Input: {"query": "x"}', "tooluse", False),
    ],
)
def test_sdpo_candidate_format_matches_upstream_rules(text, style, expected):
    assert is_sdpo_candidate_format_valid(text, {"style": style}) is expected


def test_sdpo_candidate_format_fails_closed_for_unknown_style():
    with pytest.raises(ValueError, match="Unsupported SDPO reward_model.style"):
        is_sdpo_candidate_format_valid("<answer>A</answer>", {"style": "unknown"})


def test_registered_boxed_suffix_is_stripped_and_rebuilt_exactly_once():
    raw_prompt = [
        {
            "role": "user",
            "content": f"Problem: 1+1?\n\n{BOXED_ANSWER_INSTRUCTION}\n\n{BOXED_ANSWER_INSTRUCTION}",
        }
    ]
    assert raw_prompt_to_problem(raw_prompt) == "1+1?"
    rollout = build_rollout_messages(raw_prompt[0]["content"], "deepmath")[0]["content"]
    teacher = build_teacher_messages(raw_prompt[0]["content"], "2", "Add.")[0]["content"]
    assert rollout.count(BOXED_ANSWER_INSTRUCTION) == 1
    assert teacher.count(BOXED_ANSWER_INSTRUCTION) == 1
    assert "Problem: 1+1?" in rollout
    assert "Problem: 1+1?" in teacher


def test_raw_prompt_to_problem_strips_sdpg_teacher_context():
    raw_prompt = [
        {
            "role": "user",
            "content": (
                f"Problem: 1+1?\n\n{BOXED_ANSWER_INSTRUCTION}"
                "[TEACHER_CONTEXT_TOKEN]privileged reference solution and answer"
            ),
        }
    ]

    assert raw_prompt_to_problem(raw_prompt) == "1+1?"


def test_rlhf_dataset_appends_registered_suffix_exactly_once_without_mutation():
    dataset = object.__new__(RLHFDataset)
    dataset.actor_prompt_suffix = BOXED_ANSWER_INSTRUCTION
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.processor = None
    source = {
        "prompt": [
            {
                "role": "user",
                "content": f"1+1?\n\n{BOXED_ANSWER_INSTRUCTION}\n\n{BOXED_ANSWER_INSTRUCTION}",
            }
        ]
    }

    messages = dataset._build_messages(source, key="prompt")

    assert messages[0]["content"] == f"1+1?\n\n{BOXED_ANSWER_INSTRUCTION}"
    assert messages[0]["content"].count(BOXED_ANSWER_INSTRUCTION) == 1
    assert source["prompt"][0]["content"].count(BOXED_ANSWER_INSTRUCTION) == 2


def test_dynamic_filter_uses_final_reward_range_not_correctness_labels():
    keys = [
        f"{group}_{rollout}_0"
        for group in ("g0", "g1", "gn", "gw", "gc", "gx", "gbad")
        for rollout in range(2)
    ]
    rewards = [
        0.0,
        0.0,  # all zero: closed
        1.0,
        1.0,  # all one: closed
        -1.0,
        -1.0,  # all negative one: closed
        0.0,
        -0.25,  # all wrong with length-shaped variation: admitted
        1.0,
        0.8,  # all correct with length-shaped variation: admitted
        0.3,
        0.1,  # variable but beyond max_groups: drop as a complete group
        float("nan"),
        0.0,  # non-finite: closed
    ]

    kept, dropped, summary = select_reward_variable_groups(
        keys,
        rewards,
        min_reward_range=1e-6,
        max_groups=2,
        expected_group_size=2,
    )

    assert kept == ["gw_0_0", "gw_1_0", "gc_0_0", "gc_1_0"]
    assert {key.split("_", 1)[0] for key in dropped} == {"g0", "g1", "gn", "gx", "gbad"}
    assert all(key.startswith(("gw_", "gc_")) for key in kept)
    assert summary == {
        "generated_groups": 7,
        "accepted_groups": 2,
        "constant_groups": 4,
        "excess_variable_groups": 1,
    }


def test_teacher_left_truncation_preserves_exact_sampled_suffix():
    responses = torch.nested.as_nested_tensor([torch.tensor([11, 12, 13]), torch.tensor([21, 22])], layout=torch.jagged)
    fields = build_evidence_teacher_fields(
        tokenizer=_CharacterTokenizer(),
        responses=responses,
        raw_prompts=[build_rollout_messages("1+1?"), build_rollout_messages("2+2?")],
        reward_models=[{"ground_truth": "2"}, {"ground_truth": "4"}],
        extra_infos=[
            {"problem": "1+1?", "solution": "Add."},
            {"problem": "2+2?", "solution": "Add again."},
        ],
        total_token_budget=64,
        chat_template_kwargs={"enable_thinking": False},
    )
    rows = list(fields["verpo_evidence_input_ids"].unbind())
    assert rows[0][-3:].tolist() == [11, 12, 13]
    assert rows[1][-2:].tolist() == [21, 22]
    assert fields["verpo_teacher_prompt_truncated"].all()

    teacher_text = _CharacterTokenizer().apply_chat_template(
        build_teacher_messages("1+1?", "2", "Add."),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    teacher_token_ids = _CharacterTokenizer().encode(teacher_text, add_special_tokens=False)
    expected_prefix = teacher_token_ids[-(64 - 3) :]
    assert rows[0].tolist() == expected_prefix + [11, 12, 13]


def test_boxed_grader_and_config_validation():
    assert extract_boxed_answer(r"work \\boxed{2}") == "2"
    assert grade_boxed_answer("2.0", "2")
    assert grade_boxed_answer("2", r"\boxed{2}")
    assert VerpoZPDConfig(enabled=True).vocab_chunk_size == 4096
    assert VerpoZPDConfig(enabled=True).teacher_mode == "fixed_initial"
    assert VerpoZPDConfig(enabled=True, teacher_mode="snapshot").teacher_mode == "snapshot"
    assert VerpoZPDConfig(enabled=True, teacher_mode="snapshot_anchor").teacher_mode == "snapshot"
    assert VerpoZPDConfig(enabled=True, teacher_mode="ema").teacher_mode == "ema"
    assert VerpoZPDConfig(enabled=True).teacher_ema_decay == 0.95
    assert VerpoZPDConfig(enabled=True).divergence == "forward_kl"
    assert VerpoZPDConfig(enabled=True).displacement_mode == "evidence_vs_none"
    assert VerpoZPDConfig(enabled=True).group_zpd_enabled is False
    assert VerpoZPDConfig(enabled=True).group_zpd_epsilon == 0.0
    assert VerpoZPDConfig(enabled=True).evidence_rollout_scope == "all"
    assert VerpoZPDConfig(enabled=True).vocab_mode == "topk_truncated"
    assert VerpoZPDConfig(enabled=True).top_k == 128
    assert VerpoZPDConfig(enabled=True, vocab_mode="topk_truncated", top_k=64).top_k == 64
    with pytest.raises(ValueError, match="vocab_mode"):
        VerpoZPDConfig(enabled=True, vocab_mode="bad")
    with pytest.raises(ValueError, match="top_k"):
        VerpoZPDConfig(enabled=True, top_k=0)
    assert (
        VerpoZPDConfig(
            enabled=True,
            evidence_rollout_scope="wrong_only",
        ).evidence_rollout_scope
        == "wrong_only"
    )
    assert VerpoZPDConfig(enabled=True, displacement_mode="correct_vs_incorrect").displacement_mode == (
        "correct_vs_incorrect"
    )
    assert VerpoZPDConfig(enabled=True, displacement_mode="fec").displacement_mode == "fec"
    assert VerpoZPDConfig(enabled=True, divergence="reverse_kl").divergence == "reverse_kl"
    assert (
        VerpoZPDConfig(enabled=True, divergence="reverse_kl", displacement_mode="correct_vs_incorrect").divergence
        == "reverse_kl"
    )
    with pytest.raises(ValueError, match="divergence"):
        VerpoZPDConfig(enabled=True, divergence="unknown")
    with pytest.raises(ValueError, match="displacement_mode"):
        VerpoZPDConfig(enabled=True, displacement_mode="unknown")
    with pytest.raises(ValueError, match="evidence_rollout_scope"):
        VerpoZPDConfig(enabled=True, evidence_rollout_scope="unknown")
    assert (
        VerpoZPDConfig(
            enabled=True,
            divergence="reverse_kl",
            displacement_mode="fec",
        ).displacement_mode
        == "fec"
    )
    with pytest.raises(ValueError, match="negative_hints"):
        VerpoZPDConfig(enabled=True, contrastive_num_negative_hints=0)
    with pytest.raises(ValueError, match="finite and positive"):
        VerpoZPDConfig(enabled=True, projection_epsilon=float("nan"))
    with pytest.raises(ValueError, match="group_zpd_epsilon"):
        VerpoZPDConfig(enabled=True, group_zpd_epsilon=-0.1)


def test_reward_ranked_config_is_fail_closed_and_explicit():
    with pytest.raises(ValueError, match="reward-ranked protocol mismatch"):
        VerpoZPDConfig(enabled=True, displacement_mode="reward_ranked")

    config = VerpoZPDConfig(
        enabled=True,
        displacement_mode="reward_ranked",
        group_zpd_mode="reward_ranked",
        sibling_selection_mode="reward_ranked",
        evidence_rollout_scope="all",
        contrastive_num_negative_hints=1,
    )
    assert config.group_zpd_mode == "reward_ranked"


def test_reward_ranked_group_gate_allows_correctness_siblings_for_ctr_and_fec():
    for displacement_mode in ("correct_vs_incorrect", "fec"):
        config = VerpoZPDConfig(
            enabled=True,
            displacement_mode=displacement_mode,
            group_zpd_enabled=True,
            group_zpd_mode="reward_ranked",
            sibling_selection_mode="correctness",
            evidence_rollout_scope="all",
            contrastive_num_negative_hints=1,
        )
        assert config.group_zpd_mode == "reward_ranked"
        assert config.sibling_selection_mode == "correctness"

    for group_zpd_enabled in (False, True):
        scoped = VerpoZPDConfig(
            enabled=True,
            displacement_mode="fec",
            group_zpd_enabled=group_zpd_enabled,
            group_zpd_mode="reward_ranked",
            sibling_selection_mode="correctness",
            evidence_rollout_scope="wrong_only",
        )
        assert scoped.group_zpd_enabled is group_zpd_enabled
        assert scoped.evidence_rollout_scope == "wrong_only"


def test_reward_ranked_group_gate_uses_final_reward_not_correctness_labels():
    group_ids = ["varying-final-reward"] * 2 + ["constant-final-reward"] * 2
    final_rewards = torch.tensor([0.0, -0.5, 0.0, 0.0])
    independent_acc = torch.tensor([True, False, True, False])

    gate = compute_group_zpd_gate_by_uid(
        final_rewards,
        group_ids,
        mode="reward_ranked",
    )

    assert independent_acc.tolist() == [True, False, True, False]
    assert gate.tolist() == [True, True, False, False]


def test_correctness_dependent_routing_requires_independent_acc():
    shaped_rewards = torch.tensor([0.0, 0.0])

    with pytest.raises(ValueError, match="reward_extra_info.acc"):
        _resolve_verpo_outcome_positive(
            shaped_rewards,
            None,
            require_independent_acc=True,
        )

    outcomes = _resolve_verpo_outcome_positive(
        shaped_rewards,
        [
            {"reward_extra_info": {"acc": 1.0}},
            {"reward_extra_info": {"acc": 0.0}},
        ],
        require_independent_acc=True,
    )

    assert outcomes.tolist() == [True, False]
    gate = compute_group_zpd_gate_by_uid(
        outcomes.float(),
        ["same-prompt", "same-prompt"],
        mode="binary_mixed",
    )
    assert gate.tolist() == [True, True]


def test_correctness_dependent_routing_rejects_partial_acc_metadata():
    with pytest.raises(ValueError, match="rollout 1"):
        _resolve_verpo_outcome_positive(
            torch.tensor([1.0, 0.0]),
            [
                {"reward_extra_info": {"acc": 1.0}},
                {"reward_extra_info": {}},
            ],
            require_independent_acc=True,
        )


def test_verpo_keeps_reference_teacher_without_legacy_kl_beta():
    config = OmegaConf.create(
        {
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": False,
                    "kl_loss_coef": 0.0,
                    "verpo": {"enabled": True},
                }
            },
        }
    )

    assert need_reference_policy(config)


def test_contrastive_teacher_fields_are_target_excluded_and_suffix_exact():
    tokenizer = _PlainCharacterTokenizer()
    response_texts = [
        r"first path \boxed{2}",
        r"second path \boxed{2}",
        r"third path \boxed{3}",
        r"fourth path \boxed{4}",
    ]
    response_rows = [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in response_texts]
    responses = torch.nested.as_nested_tensor(response_rows, layout=torch.jagged)
    reward_models, extra_infos = _external_evidence_rows(4)
    fields = build_contrastive_evidence_teacher_fields(
        tokenizer=tokenizer,
        responses=responses,
        raw_prompts=[build_rollout_messages("1+1?")] * 4,
        rewards=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        correctness=torch.tensor([True, True, False, False]),
        uids=["same-problem"] * 4,
        total_token_budget=1024,
        reward_models=reward_models,
        extra_infos=extra_infos,
        num_negative_hints=4,
    )

    assert fields["verpo_contrastive_available"].tolist() == [True] * 4
    positive_indices = fields["verpo_positive_sibling_index"].tolist()
    negative_indices = fields["verpo_negative_sibling_indices"].tolist()
    for row in range(4):
        assert positive_indices[row] == -1
        assert row not in negative_indices[row]
        assert set(negative_indices[row]).issubset({2, 3})
        target_suffix = response_rows[row].tolist()
        positive_row = list(fields["verpo_positive_input_ids"].unbind())[row].tolist()
        assert positive_row[-len(target_suffix) :] == target_suffix
        for negative_index in range(4):
            negative_row = list(fields[f"verpo_negative_{negative_index}_input_ids"].unbind())[row].tolist()
            assert negative_row[-len(target_suffix) :] == target_suffix

    external_positive = "Add the two ones.\n\nFinal answer: 2"
    positive_prompt = build_contrastive_teacher_messages("1+1?", external_positive)[0]["content"]
    negative_prompt = build_contrastive_teacher_messages("1+1?", response_texts[2])[0]["content"]
    assert positive_prompt.replace(external_positive, "<candidate>") == negative_prompt.replace(
        response_texts[2], "<candidate>"
    )
    assert "Incorrect candidate solution:" in positive_prompt
    assert "Incorrect candidate solution:" in negative_prompt


def test_sdpo_mcq_qplus_uses_official_prompt_qminus_uses_neutral_hint_and_valid_incorrect_siblings():
    tokenizer = _MessageCharacterTokenizer()
    system_prompt = (
        "Solve the question and return <reasoning>...</reasoning> followed by "
        "<answer>A-D</answer>."
    )
    question = "Which option is chemically correct?\nA. alpha\nB. beta\nC. gamma\nD. delta"
    raw_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    response_texts = [
        "<reasoning>correct route one</reasoning>\n<answer>A</answer>",
        "<reasoning>correct route two</reasoning>\n<answer> A </answer>",
        "<reasoning>wrong route B</reasoning>\n<answer>B</answer>",
        "<reasoning>wrong route C</reasoning>\n<answer>C</answer>",
        "<reasoning>malformed wrong route</reasoning>\nanswer D",
    ]
    response_rows = [
        torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in response_texts
    ]
    solution = "Chemical reference reasoning. Correct final answer: A"
    reward_models = [{"ground_truth": "A", "style": "mcq"} for _ in response_texts]
    extra_infos = [
        {
            "solution": solution,
            "teacher_prompt_template": "sdpo_official_correct_solution_v1",
        }
        for _ in response_texts
    ]

    fields = build_contrastive_evidence_teacher_fields(
        tokenizer=tokenizer,
        responses=torch.nested.as_nested_tensor(response_rows, layout=torch.jagged),
        raw_prompts=[raw_prompt] * len(response_texts),
        rewards=torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
        correctness=torch.tensor([True, True, False, False, False]),
        uids=["same-sdpo-question"] * len(response_texts),
        total_token_budget=4096,
        reward_models=reward_models,
        extra_infos=extra_infos,
        num_negative_hints=1,
        selection_mode="correctness",
        enable_thinking=False,
    )

    assert fields["verpo_contrastive_available"].tolist() == [True] * len(response_texts)
    assert fields["verpo_positive_sibling_index"].tolist() == [-1] * len(response_texts)
    for row, (negative_index,) in enumerate(
        fields["verpo_negative_sibling_indices"].tolist()
    ):
        assert negative_index in {2, 3}
        assert negative_index != row

    for row, target_response in enumerate(response_rows):
        target_suffix = target_response.tolist()
        positive_combined = list(fields["verpo_positive_input_ids"].unbind())[row].tolist()
        negative_combined = list(fields["verpo_negative_0_input_ids"].unbind())[row].tolist()
        assert positive_combined[-len(target_suffix) :] == target_suffix
        assert negative_combined[-len(target_suffix) :] == target_suffix

        positive_prompt_length = int(fields["verpo_positive_prompt_lengths"][row].item())
        negative_prompt_length = int(fields["verpo_negative_0_prompt_lengths"][row].item())
        positive_prompt = tokenizer.decode(positive_combined[:positive_prompt_length])
        negative_prompt = tokenizer.decode(negative_combined[:negative_prompt_length])
        negative_index = fields["verpo_negative_sibling_indices"][row, 0].item()

        assert f"<system>{system_prompt}</system>" in positive_prompt
        assert f"<system>{system_prompt}</system>" in negative_prompt
        assert question in positive_prompt
        assert question in negative_prompt
        assert f"Correct solution:\n\n{solution}" in positive_prompt
        assert f"Incorrect candidate solution:\n\n{response_texts[negative_index]}" in negative_prompt
        assert "Correctly solve the original question." in positive_prompt
        assert "Correctly solve the original question." not in negative_prompt
        assert "Correct solution:" not in negative_prompt
        assert response_texts[4] not in negative_prompt


@pytest.mark.parametrize(
    ("correctness", "expected_available"),
    [
        ([False, False, False, False], [True, True, True, True]),
        ([True, True, True, True], [False, False, False, False]),
    ],
)
def test_correctness_external_positive_only_requires_an_incorrect_sibling(
    correctness, expected_available
):
    tokenizer = _CharacterTokenizer()
    response_texts = [
        rf"path {index} \\boxed{{{3 + index}}}" for index in range(4)
    ]
    responses = torch.nested.as_nested_tensor(
        [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in response_texts],
        layout=torch.jagged,
    )
    reward_models, extra_infos = _external_evidence_rows(4)

    fields = build_contrastive_evidence_teacher_fields(
        tokenizer=tokenizer,
        responses=responses,
        raw_prompts=[build_rollout_messages("1+1?")] * 4,
        rewards=torch.tensor([0.0] * 4),
        correctness=torch.tensor(correctness),
        uids=["same-problem"] * 4,
        total_token_budget=1024,
        reward_models=reward_models,
        extra_infos=extra_infos,
        num_negative_hints=1,
        selection_mode="correctness",
    )

    assert fields["verpo_contrastive_available"].tolist() == expected_available
    assert fields["verpo_positive_sibling_index"].tolist() == [-1] * 4
    if not any(correctness):
        for row, negative_indices in enumerate(
            fields["verpo_negative_sibling_indices"].tolist()
        ):
            assert negative_indices[0] != row


def test_reward_ranked_teacher_fields_use_shaped_reward_extrema():
    tokenizer = _CharacterTokenizer()
    response_texts = [
        r"short correct \boxed{2}",
        r"longer correct \boxed{2}",
        r"short wrong \boxed{3}",
        r"long wrong \boxed{4}",
    ]
    response_rows = [
        torch.tensor(tokenizer.encode(text), dtype=torch.long)
        for text in response_texts
    ]
    fields = build_contrastive_evidence_teacher_fields(
        tokenizer=tokenizer,
        responses=torch.nested.as_nested_tensor(response_rows, layout=torch.jagged),
        raw_prompts=[build_rollout_messages("1+1?")] * 4,
        rewards=torch.tensor([1.0, 0.5, 0.0, -0.5]),
        uids=["same-problem"] * 4,
        total_token_budget=1024,
        num_negative_hints=1,
        selection_mode="reward_ranked",
        enable_thinking=False,
    )

    assert fields["verpo_contrastive_available"].tolist() == [True] * 4
    assert fields["verpo_positive_sibling_index"].tolist()[1] == 0
    assert fields["verpo_negative_sibling_indices"].tolist()[1] == [3]


def test_correctness_teacher_fields_use_acc_not_length_shaped_reward():
    tokenizer = _CharacterTokenizer()
    response_texts = [
        r"short correct \boxed{2}",
        r"overlong correct \boxed{2}",
        r"short wrong \boxed{3}",
        r"overlong wrong \boxed{4}",
    ]
    response_rows = [
        torch.tensor(tokenizer.encode(text), dtype=torch.long)
        for text in response_texts
    ]
    reward_models, extra_infos = _external_evidence_rows(4)
    fields = build_contrastive_evidence_teacher_fields(
        tokenizer=tokenizer,
        responses=torch.nested.as_nested_tensor(response_rows, layout=torch.jagged),
        raw_prompts=[build_rollout_messages("1+1?")] * 4,
        # The second correct response has the same shaped reward as the short
        # wrong response; only the independent acc tensor distinguishes them.
        rewards=torch.tensor([1.0, 0.0, 0.0, -0.5]),
        correctness=torch.tensor([True, True, False, False]),
        uids=["same-problem"] * 4,
        total_token_budget=1024,
        reward_models=reward_models,
        extra_infos=extra_infos,
        num_negative_hints=1,
        selection_mode="correctness",
        enable_thinking=False,
    )

    assert fields["verpo_contrastive_available"].tolist() == [True] * 4
    assert fields["verpo_positive_sibling_index"].tolist() == [-1] * 4
    assert fields["verpo_negative_sibling_indices"].tolist()[0][0] in {2, 3}


def test_correctness_teacher_fields_fail_closed_without_independent_acc():
    tokenizer = _CharacterTokenizer()
    response_texts = [r"correct \boxed{2}", r"wrong \boxed{3}"]
    responses = torch.nested.as_nested_tensor(
        [
            torch.tensor(tokenizer.encode(text), dtype=torch.long)
            for text in response_texts
        ],
        layout=torch.jagged,
    )

    with pytest.raises(ValueError, match="requires independent per-rollout correctness"):
        build_contrastive_evidence_teacher_fields(
            tokenizer=tokenizer,
            responses=responses,
            raw_prompts=[build_rollout_messages("1+1?")] * 2,
            # A correct overlong response can have the same shaped reward as
            # an incorrect short response, so this tensor is not a label.
            rewards=torch.tensor([0.0, 0.0]),
            uids=["same-problem"] * 2,
            total_token_budget=1024,
            num_negative_hints=1,
            selection_mode="correctness",
        )


def test_contrastive_teacher_budget_preserves_problem_and_suffix_by_truncating_only_hint():
    tokenizer = _CharacterTokenizer()
    problem = "ORIGINAL-PROBLEM-MUST-STAY: compute 1+1."
    response_texts = [
        "correct-start " + "A" * 500 + r" correct-tail \boxed{2}",
        "correct-start " + "B" * 500 + r" correct-tail \boxed{2}",
        "wrong-start " + "C" * 500 + r" wrong-tail \boxed{3}",
        "wrong-start " + "D" * 500 + r" wrong-tail \boxed{4}",
    ]
    response_rows = [
        torch.tensor(tokenizer.encode(text), dtype=torch.long)
        for text in response_texts
    ]
    responses = torch.nested.as_nested_tensor(response_rows, layout=torch.jagged)
    reward_models, extra_infos = _external_evidence_rows(4, solution="E" * 500)
    fields = build_contrastive_evidence_teacher_fields(
        tokenizer=tokenizer,
        responses=responses,
        raw_prompts=[build_rollout_messages(problem)] * 4,
        rewards=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        correctness=torch.tensor([True, True, False, False]),
        uids=["same-problem"] * 4,
        total_token_budget=1200,
        reward_models=reward_models,
        extra_infos=extra_infos,
        num_negative_hints=4,
    )

    assert fields["verpo_teacher_prompt_truncated"].all()
    prefixes = ["verpo_positive", *(f"verpo_negative_{index}" for index in range(4))]
    for prefix in prefixes:
        for row, combined in enumerate(fields[f"{prefix}_input_ids"].unbind()):
            target_suffix = response_rows[row].tolist()
            assert combined[-len(target_suffix) :].tolist() == target_suffix
            prompt = tokenizer.decode(combined[: -len(target_suffix)].tolist())
            assert f"Problem: {problem}" in prompt
            assert "Incorrect candidate solution:" in prompt
            assert "middle of candidate hint truncated" in prompt


def test_contrastive_teacher_budget_fails_closed_instead_of_truncating_problem():
    tokenizer = _CharacterTokenizer()
    response_texts = [r"a \boxed{2}", r"b \boxed{2}", r"c \boxed{3}", r"d \boxed{4}"]
    responses = torch.nested.as_nested_tensor(
        [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in response_texts],
        layout=torch.jagged,
    )

    reward_models, extra_infos = _external_evidence_rows(4)
    with pytest.raises(ValueError, match="refusing to truncate the problem"):
        build_contrastive_evidence_teacher_fields(
            tokenizer=tokenizer,
            responses=responses,
            raw_prompts=[build_rollout_messages("P" * 1000)] * 4,
            rewards=torch.tensor([1.0, 1.0, 0.0, 0.0]),
            correctness=torch.tensor([True, True, False, False]),
            uids=["same-problem"] * 4,
            total_token_budget=200,
            reward_models=reward_models,
            extra_infos=extra_infos,
            num_negative_hints=4,
        )
