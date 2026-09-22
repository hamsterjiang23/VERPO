"""Optional tensor checks: collected as skipped when torch is not installed."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip(
    "torch", reason="torch is intentionally not installed for this CPU-only delivery"
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verpo_protocol_under_test",
    ROOT / "verl/verl/trainer/distillation/verpo_protocol.py",
)
assert spec is not None and spec.loader is not None
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


class CharacterTokenizer:
    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, **kwargs: Any
    ) -> str | list[int]:
        text = "\n".join(message["content"] for message in messages)
        return self.encode(text) if tokenize else text


@pytest.mark.parametrize("require_negative", [False, True])
def test_replay_preserves_suffix_and_ignores_ground_truth(
    require_negative: bool,
) -> None:
    tokenizer = CharacterTokenizer()
    texts = [
        "good <answer>A</answer>",
        "also good <answer>A</answer>",
        "wrong <answer>B</answer>",
    ]
    responses = torch.nested.as_nested_tensor(
        [torch.tensor(tokenizer.encode(text)) for text in texts], layout=torch.jagged
    )
    fields = protocol.build_contrastive_evidence_teacher_fields(
        tokenizer=tokenizer,
        responses=responses,
        raw_prompts=[[{"role": "user", "content": "Original question"}]] * 3,
        rewards=torch.tensor([1.0, 1.0, 0.0]),
        correctness=torch.tensor([True, True, False]),
        uids=["g"] * 3,
        rollout_ids=["g_0_0", "g_1_0", "g_2_0"],
        reward_models=[{"style": "mcq", "ground_truth": "SECRET_GT"}] * 3,
        extra_infos=[{"solution": "PRIVATE_SOLUTION"}] * 3,
        total_token_budget=2000,
        require_negative=require_negative,
    )
    for i, tokens in enumerate(fields["verpo_positive_input_ids"].unbind()):
        decoded = tokenizer.decode(tokens.tolist())
        assert decoded.endswith(texts[i])
        assert "SECRET_GT" not in decoded and "PRIVATE_SOLUTION" not in decoded
        assert fields["verpo_positive_sibling_index"][i] != i
    assert fields["verpo_contrastive_available"].tolist() == (
        [True, True, False] if require_negative else [True, True, True]
    )


def test_zero_evidence_keeps_reference_gradient() -> None:
    from risk_aware_opsd.verpo_launch_config import VERPOConfig
    from risk_aware_opsd.verpo_trainer import VERPOTrainer

    student = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
    reference = torch.tensor([[[1.0, 0.0, -1.0]]], requires_grad=True)
    positive = torch.tensor([[[-1.0, 1.0, 0.0]]], requires_grad=True)
    trainer = VERPOTrainer(config=VERPOConfig(lambda_ref=0.1, lambda_evi=1.0))
    loss, _ = trainer.compute_verpo_loss(
        student,
        reference_logits=reference,
        base_teacher_logits=reference,
        evidence_teacher_logits=positive,
        evidence_weights=torch.zeros(1, 1),
    )
    loss.backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0
    assert reference.grad is None and positive.grad is None
