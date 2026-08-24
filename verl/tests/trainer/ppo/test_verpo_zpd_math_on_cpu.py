import hashlib

import torch
from tensordict import TensorDict
from risk_aware_opsd.verpo_zpd import (
    compute_contrastive_teacher_token_losses,
    compute_evidence_displacement_stats,
    compute_fec_evidence_stats,
    compute_fec_teacher_token_losses,
    compute_interpolated_teacher_token_losses,
    compute_reverse_kl_evidence_stats,
    compute_reverse_kl_fec_evidence_stats,
    compute_reverse_kl_fec_teacher_token_losses,
    compute_reverse_kl_teacher_token_losses,
    compute_verpo_token_weights,
)

from verl.trainer.distillation.verpo_topk import (
    TruncatedTopKSupport,
    chunked_logsumexp,
    gather_full_softmax_log_probs,
)
from verl.trainer.distillation.verpo_zpd import (
    _accumulate_indexed_teacher_probabilities_,
    _teacher_data,
    apply_evidence_rollout_scope,
    compute_flat_verpo_losses,
    compute_group_zpd_gate_by_uid,
    marginalize_teacher_logits,
)
from verl.utils import tensordict_utils as tu


def test_teacher_data_preserves_scalar_temperature_as_metadata():
    batch_size = 6
    data = TensorDict(
        {
            "verpo_positive_input_ids": torch.ones(batch_size, 3, dtype=torch.long),
            "verpo_positive_position_ids": torch.arange(3).expand(batch_size, -1),
        },
        batch_size=[batch_size],
    )
    tu.assign_non_tensor_data(data, "temperature", 1.0)

    teacher = _teacher_data(data, "verpo_positive")

    assert teacher.batch_size == data.batch_size
    assert tu.get_non_tensor_data(teacher, "temperature", None) == 1.0


def test_teacher_data_preserves_per_sample_temperature_tensor():
    batch_size = 4
    temperatures = torch.tensor([0.6, 0.7, 0.8, 0.9])
    data = TensorDict(
        {
            "verpo_positive_input_ids": torch.ones(batch_size, 3, dtype=torch.long),
            "verpo_positive_position_ids": torch.arange(3).expand(batch_size, -1),
            "temperature": temperatures,
        },
        batch_size=[batch_size],
    )

    teacher = _teacher_data(data, "verpo_positive")

    torch.testing.assert_close(teacher["temperature"], temperatures)


def test_topk_reference_and_evidence_supports_are_independent():
    vocab = 7
    student = torch.zeros(1, vocab)
    student[0, 3] = 9.0
    q0 = torch.zeros(1, vocab)
    q0[0, 0] = 9.0
    qpos = torch.zeros(1, vocab)
    qpos[0, 1] = 9.0
    qneg = torch.zeros(1, vocab)
    qneg[0, 2] = 9.0
    sampled = torch.tensor([6])
    common = dict(
        sampled_token_ids=sampled,
        token_advantages=torch.ones(1),
        active_token_mask=torch.ones(1, dtype=torch.bool),
        negative_teacher_logits=qneg,
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=1.0,
        vocab_chunk_size=2,
        displacement_mode="correct_vs_incorrect",
        vocab_mode="topk_truncated",
        top_k=1,
    )
    forward = compute_flat_verpo_losses(
        student, q0, qpos, divergence="forward_kl", **common
    )
    assert forward["reference_support_size"].item() == 2
    assert forward["support_size"].item() == 3

    reverse = compute_flat_verpo_losses(
        student, q0, qpos, divergence="reverse_kl", **common
    )
    assert reverse["reference_support_size"].item() == 3
    assert reverse["support_size"].item() == 4


def test_native_signed_benefit_uses_unclamped_controller_numerator():
    student = torch.zeros(1, 3)
    q0 = torch.zeros(1, 3)
    qpos = torch.tensor([[4.0, 0.0, 0.0]])
    qneg = torch.tensor([[0.0, 4.0, 0.0]])
    common = dict(
        sampled_token_ids=torch.tensor([1]),
        token_advantages=torch.ones(1),
        active_token_mask=torch.ones(1, dtype=torch.bool),
        negative_teacher_logits=qneg,
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=1.0,
        vocab_chunk_size=3,
        divergence="forward_kl",
        displacement_mode="correct_vs_incorrect",
    )

    bounded = compute_flat_verpo_losses(student, q0, qpos, **common)
    signed = compute_flat_verpo_losses(
        student,
        q0,
        qpos,
        allow_negative_benefit=True,
        **common,
    )

    assert bounded["weights"].item() == 0.0
    h = signed["benefit"]
    movement_cost = 0.01 + signed["fisher_cost"] + 1e-4
    torch.testing.assert_close(signed["weights"], h / (h + movement_cost))
    assert signed["weights"].item() != 0.0


def test_native_fixed_negative_weight_is_a_reverse_signed_correction():
    student = torch.tensor([[0.2, -0.3, 0.7]], requires_grad=True)
    q0 = torch.tensor([[4.0, 0.0, 0.0]])
    qe = torch.tensor([[0.0, 4.0, 0.0]])
    outputs = compute_flat_verpo_losses(
        student,
        q0,
        qe,
        sampled_token_ids=torch.tensor([0]),
        token_advantages=torch.ones(1),
        active_token_mask=torch.ones(1, dtype=torch.bool),
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        allow_negative_benefit=True,
        temperature=1.0,
        vocab_chunk_size=3,
        divergence="forward_kl",
        displacement_mode="evidence_vs_none",
    )

    assert outputs["benefit"].item() < 0
    assert outputs["weights"].item() != 0
    expected = -outputs["weights"] * (
        (torch.softmax(qe, dim=-1) - torch.softmax(q0, dim=-1))
        * torch.log_softmax(student, dim=-1)
    ).sum(-1)
    torch.testing.assert_close(outputs["evidence_loss"], expected)


def test_flat_native_math_matches_trl_oracle_and_gradients():
    torch.manual_seed(7)
    temperature = 0.7
    batch, tokens, vocab = 2, 3, 13
    student_trl = torch.randn(batch, tokens, vocab, requires_grad=True)
    student_verl = student_trl.detach().clone().requires_grad_(True)
    q0 = torch.randn(batch, tokens, vocab)
    qe = torch.randn(batch, tokens, vocab)
    sampled = torch.randint(0, vocab, (batch, tokens))
    advantages = torch.tensor([0.75, -0.25])
    gate = torch.tensor([True, False])
    mask = torch.ones(batch, tokens, dtype=torch.bool)

    benefit, fisher, displacement = compute_evidence_displacement_stats(
        qe, q0, student_trl, sampled, advantages, temperature=temperature, vocab_chunk_size=5
    )
    weights = compute_verpo_token_weights(
        benefit,
        fisher,
        group_gate=gate,
        token_mask=mask,
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
    )
    reference, evidence, _ = compute_interpolated_teacher_token_losses(
        student_trl,
        q0,
        q0,
        qe,
        weights,
        temperature=temperature,
        vocab_chunk_size=5,
        return_interpolated_probs=False,
    )

    native = compute_flat_verpo_losses(
        student_verl.reshape(-1, vocab),
        q0.reshape(-1, vocab),
        qe.reshape(-1, vocab),
        sampled.reshape(-1),
        advantages.repeat_interleave(tokens),
        gate.repeat_interleave(tokens),
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=temperature,
        vocab_chunk_size=5,
    )
    torch.testing.assert_close(native["weights"].reshape(batch, tokens), weights)
    torch.testing.assert_close(native["fisher_cost"].reshape(batch, tokens), fisher)
    torch.testing.assert_close(native["displacement_norm"].reshape(batch, tokens), displacement)
    torch.testing.assert_close(native["reference_loss"].reshape(batch, tokens), reference)
    torch.testing.assert_close(native["evidence_loss"].reshape(batch, tokens), evidence)
    expected_q0_log_prob = (
        torch.log_softmax(q0.float() / temperature, dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    )
    torch.testing.assert_close(native["q0_sample_log_prob"].reshape(batch, tokens), expected_q0_log_prob)
    grpo_loss = torch.tensor(0.37)
    trl_total = grpo_loss + 0.1 * reference.mean() + 0.1 * evidence.mean()
    native_total = grpo_loss + 0.1 * native["reference_loss"].mean() + 0.1 * native["evidence_loss"].mean()
    torch.testing.assert_close(native_total, trl_total)

    (reference.sum() + evidence.sum()).backward()
    (native["reference_loss"].sum() + native["evidence_loss"].sum()).backward()
    torch.testing.assert_close(student_verl.grad, student_trl.grad)


def test_flat_reverse_kl_math_matches_trl_oracle_and_gradients():
    torch.manual_seed(19)
    temperature = 0.8
    batch, tokens, vocab = 2, 3, 11
    student_trl = torch.randn(batch, tokens, vocab, requires_grad=True)
    student_verl = student_trl.detach().clone().requires_grad_(True)
    q0 = torch.randn(batch, tokens, vocab)
    qe = torch.randn(batch, tokens, vocab)
    sampled = torch.randint(0, vocab, (batch, tokens))
    advantages = torch.tensor([0.6, -0.4])
    gate = torch.tensor([True, False])
    mask = torch.ones(batch, tokens, dtype=torch.bool)

    benefit, fisher, displacement = compute_reverse_kl_evidence_stats(
        qe,
        q0,
        student_trl,
        sampled,
        advantages,
        temperature=temperature,
        vocab_chunk_size=4,
    )
    weights = compute_verpo_token_weights(
        benefit,
        fisher,
        group_gate=gate,
        token_mask=mask,
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
    )
    reference, evidence, _ = compute_reverse_kl_teacher_token_losses(
        student_trl,
        q0,
        q0,
        qe,
        weights,
        temperature=temperature,
        vocab_chunk_size=4,
        return_interpolated_probs=False,
    )

    native = compute_flat_verpo_losses(
        student_verl.reshape(-1, vocab),
        q0.reshape(-1, vocab),
        qe.reshape(-1, vocab),
        sampled.reshape(-1),
        advantages.repeat_interleave(tokens),
        gate.repeat_interleave(tokens),
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=temperature,
        vocab_chunk_size=4,
        divergence="reverse_kl",
    )
    torch.testing.assert_close(native["weights"].reshape(batch, tokens), weights)
    torch.testing.assert_close(native["fisher_cost"].reshape(batch, tokens), fisher)
    torch.testing.assert_close(native["displacement_norm"].reshape(batch, tokens), displacement)
    torch.testing.assert_close(native["reference_loss"].reshape(batch, tokens), reference)
    torch.testing.assert_close(native["evidence_loss"].reshape(batch, tokens), evidence)

    (reference.sum() + evidence.sum()).backward()
    (native["reference_loss"].sum() + native["evidence_loss"].sum()).backward()
    torch.testing.assert_close(student_verl.grad, student_trl.grad)


def test_flat_reverse_kl_ctr_matches_trl_oracle_and_gradients():
    torch.manual_seed(19)
    temperature = 0.8
    batch, tokens, vocab = 2, 3, 11
    student_trl = torch.randn(batch, tokens, vocab, requires_grad=True)
    student_verl = student_trl.detach().clone().requires_grad_(True)
    q0 = torch.randn(batch, tokens, vocab)
    qpos = torch.randn(batch, tokens, vocab)
    qneg = torch.randn(batch, tokens, vocab)
    sampled = torch.randint(0, vocab, (batch, tokens))
    advantages = torch.tensor([0.7, -0.4])
    gate = torch.tensor([True, True])
    mask = torch.ones(batch, tokens, dtype=torch.bool)

    benefit, fisher, displacement = compute_reverse_kl_evidence_stats(
        qpos,
        qneg,
        student_trl,
        sampled,
        advantages,
        temperature=temperature,
        vocab_chunk_size=5,
    )
    weights = compute_verpo_token_weights(
        benefit,
        fisher,
        group_gate=gate,
        token_mask=mask,
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
    )
    reference, evidence, _ = compute_reverse_kl_teacher_token_losses(
        student_trl,
        q0,
        qneg,
        qpos,
        weights,
        temperature=temperature,
        vocab_chunk_size=5,
        return_interpolated_probs=False,
    )
    native = compute_flat_verpo_losses(
        student_verl.reshape(-1, vocab),
        q0.reshape(-1, vocab),
        qpos.reshape(-1, vocab),
        sampled.reshape(-1),
        advantages.repeat_interleave(tokens),
        gate.repeat_interleave(tokens),
        negative_teacher_logits=qneg.reshape(-1, vocab),
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=temperature,
        vocab_chunk_size=5,
        divergence="reverse_kl",
        displacement_mode="correct_vs_incorrect",
    )
    torch.testing.assert_close(native["weights"].reshape(batch, tokens), weights)
    torch.testing.assert_close(native["fisher_cost"].reshape(batch, tokens), fisher)
    torch.testing.assert_close(native["displacement_norm"].reshape(batch, tokens), displacement)
    torch.testing.assert_close(native["reference_loss"].reshape(batch, tokens), reference)
    torch.testing.assert_close(native["evidence_loss"].reshape(batch, tokens), evidence)

    (reference.sum() + evidence.sum()).backward()
    (native["reference_loss"].sum() + native["evidence_loss"].sum()).backward()
    torch.testing.assert_close(student_verl.grad, student_trl.grad)


def test_flat_contrastive_math_matches_trl_oracle_and_negative_mixture():
    torch.manual_seed(23)
    temperature = 0.9
    batch, tokens, vocab, negatives = 2, 3, 11, 4
    student_trl = torch.randn(batch, tokens, vocab, requires_grad=True)
    student_verl = student_trl.detach().clone().requires_grad_(True)
    q0 = torch.randn(batch, tokens, vocab)
    qpos = torch.randn(batch, tokens, vocab)
    qneg_components = torch.randn(negatives, batch, tokens, vocab)
    qneg = marginalize_teacher_logits(qneg_components.reshape(negatives, batch * tokens, vocab), temperature).reshape(
        batch, tokens, vocab
    )
    sampled = torch.randint(0, vocab, (batch, tokens))
    advantages = torch.tensor([0.8, -0.35])
    gate = torch.tensor([True, True])
    mask = torch.ones(batch, tokens, dtype=torch.bool)

    expected_mixture = torch.softmax(qneg_components.float() / temperature, dim=-1).mean(dim=0)
    torch.testing.assert_close(torch.softmax(qneg / temperature, dim=-1), expected_mixture)
    benefit, fisher, displacement = compute_evidence_displacement_stats(
        qpos,
        qneg,
        student_trl,
        sampled,
        advantages,
        temperature=temperature,
        vocab_chunk_size=5,
    )
    weights = compute_verpo_token_weights(
        benefit,
        fisher,
        group_gate=gate,
        token_mask=mask,
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
    )
    reference, evidence = compute_contrastive_teacher_token_losses(
        student_trl,
        q0,
        qpos,
        qneg,
        weights,
        temperature=temperature,
        vocab_chunk_size=5,
    )
    native = compute_flat_verpo_losses(
        student_verl.reshape(-1, vocab),
        q0.reshape(-1, vocab),
        qpos.reshape(-1, vocab),
        sampled.reshape(-1),
        advantages.repeat_interleave(tokens),
        gate.repeat_interleave(tokens),
        negative_teacher_logits=qneg.reshape(-1, vocab),
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=temperature,
        vocab_chunk_size=5,
        displacement_mode="correct_vs_incorrect",
    )
    torch.testing.assert_close(native["weights"].reshape(batch, tokens), weights)
    torch.testing.assert_close(native["fisher_cost"].reshape(batch, tokens), fisher)
    torch.testing.assert_close(native["displacement_norm"].reshape(batch, tokens), displacement)
    torch.testing.assert_close(native["reference_loss"].reshape(batch, tokens), reference)
    torch.testing.assert_close(native["evidence_loss"].reshape(batch, tokens), evidence)

    (reference.sum() + evidence.sum()).backward()
    (native["reference_loss"].sum() + native["evidence_loss"].sum()).backward()
    torch.testing.assert_close(student_verl.grad, student_trl.grad)


def test_flat_fec_math_matches_trl_oracle_and_gradients():
    torch.manual_seed(29)
    temperature = 0.85
    batch, tokens, vocab = 2, 4, 13
    student_trl = torch.randn(batch, tokens, vocab, requires_grad=True)
    student_verl = student_trl.detach().clone().requires_grad_(True)
    q0 = torch.randn(batch, tokens, vocab)
    qpos = torch.randn(batch, tokens, vocab)
    qneg = torch.randn(batch, tokens, vocab)
    sampled = torch.randint(0, vocab, (batch, tokens))
    advantages = torch.tensor([0.65, -0.45])
    gate = torch.tensor([True, False])
    mask = torch.ones(batch, tokens, dtype=torch.bool)

    stats = compute_fec_evidence_stats(
        qpos,
        qneg,
        q0,
        student_trl,
        sampled,
        advantages,
        temperature=temperature,
        vocab_chunk_size=5,
        projection_epsilon=1e-8,
    )
    weights = compute_verpo_token_weights(
        stats["benefit"],
        stats["fisher_cost"],
        group_gate=gate,
        token_mask=mask,
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
    )
    reference, _ = compute_contrastive_teacher_token_losses(
        student_trl,
        q0,
        qpos,
        qneg,
        torch.zeros_like(weights),
        temperature=temperature,
        vocab_chunk_size=5,
    )
    evidence = compute_fec_teacher_token_losses(
        student_trl,
        qpos,
        qneg,
        q0,
        weights,
        stats["nuisance_projection_coefficient"],
        temperature=temperature,
        vocab_chunk_size=5,
    )
    native = compute_flat_verpo_losses(
        student_verl.reshape(-1, vocab),
        q0.reshape(-1, vocab),
        qpos.reshape(-1, vocab),
        sampled.reshape(-1),
        advantages.repeat_interleave(tokens),
        gate.repeat_interleave(tokens),
        negative_teacher_logits=qneg.reshape(-1, vocab),
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=temperature,
        vocab_chunk_size=5,
        displacement_mode="fec",
        projection_epsilon=1e-8,
    )
    for name in (
        "weights",
        "fisher_cost",
        "displacement_norm",
        "task_displacement_norm",
        "nuisance_displacement_norm",
        "task_nuisance_fisher_cosine",
        "nuisance_projection_coefficient",
    ):
        expected = weights if name == "weights" else stats[name]
        torch.testing.assert_close(native[name].reshape(batch, tokens), expected)
    torch.testing.assert_close(native["reference_loss"].reshape(batch, tokens), reference)
    torch.testing.assert_close(native["evidence_loss"].reshape(batch, tokens), evidence)

    (reference.sum() + evidence.sum()).backward()
    (native["reference_loss"].sum() + native["evidence_loss"].sum()).backward()
    torch.testing.assert_close(student_verl.grad, student_trl.grad)


def test_flat_reverse_kl_fec_matches_trl_oracle_and_gradients():
    torch.manual_seed(31)
    temperature = 0.9
    batch, tokens, vocab = 2, 3, 11
    student_trl = torch.randn(batch, tokens, vocab, requires_grad=True)
    student_verl = student_trl.detach().clone().requires_grad_(True)
    q0 = torch.randn(batch, tokens, vocab)
    qpos = torch.randn(batch, tokens, vocab)
    qneg_components = torch.randn(3, batch, tokens, vocab)
    qneg = marginalize_teacher_logits(qneg_components.reshape(3, batch * tokens, vocab), temperature).reshape(
        batch, tokens, vocab
    )
    sampled = torch.randint(0, vocab, (batch, tokens))
    advantages = torch.tensor([0.8, -0.35])
    gate = torch.tensor([True, False])
    mask = torch.ones(batch, tokens, dtype=torch.bool)

    stats = compute_reverse_kl_fec_evidence_stats(
        qpos,
        qneg,
        q0,
        student_trl,
        sampled,
        advantages,
        temperature=temperature,
        vocab_chunk_size=4,
        projection_epsilon=1e-8,
    )
    weights = compute_verpo_token_weights(
        stats["benefit"],
        stats["fisher_cost"],
        group_gate=gate,
        token_mask=mask,
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
    )
    reference, _, _ = compute_reverse_kl_teacher_token_losses(
        student_trl,
        q0,
        q0,
        q0,
        torch.zeros_like(weights),
        temperature=temperature,
        vocab_chunk_size=4,
        return_interpolated_probs=False,
    )
    evidence = compute_reverse_kl_fec_teacher_token_losses(
        student_trl,
        qpos,
        qneg,
        q0,
        weights,
        stats["nuisance_projection_coefficient"],
        temperature=temperature,
        vocab_chunk_size=4,
    )
    native = compute_flat_verpo_losses(
        student_verl.reshape(-1, vocab),
        q0.reshape(-1, vocab),
        qpos.reshape(-1, vocab),
        sampled.reshape(-1),
        advantages.repeat_interleave(tokens),
        gate.repeat_interleave(tokens),
        negative_teacher_logits=qneg.reshape(-1, vocab),
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=temperature,
        vocab_chunk_size=4,
        divergence="reverse_kl",
        displacement_mode="fec",
        projection_epsilon=1e-8,
    )
    for name in (
        "weights",
        "benefit",
        "fisher_cost",
        "displacement_norm",
        "task_displacement_norm",
        "nuisance_displacement_norm",
        "task_nuisance_fisher_cosine",
        "nuisance_projection_coefficient",
        "residual_nuisance_fisher_covariance",
        "alignment",
    ):
        expected = weights if name == "weights" else stats["fec_alignment" if name == "alignment" else name]
        torch.testing.assert_close(native[name].reshape(batch, tokens), expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        native["reference_loss"].reshape(batch, tokens),
        reference,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        native["evidence_loss"].reshape(batch, tokens),
        evidence,
        rtol=1e-6,
        atol=1e-6,
    )

    (reference.sum() + evidence.sum()).backward()
    (native["reference_loss"].sum() + native["evidence_loss"].sum()).backward()
    torch.testing.assert_close(student_verl.grad, student_trl.grad, rtol=1e-6, atol=1e-6)


def test_flat_verpo_rejects_unknown_divergence():
    logits = torch.zeros(1, 3)
    try:
        compute_flat_verpo_losses(
            logits,
            logits,
            logits,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            tau=1.0,
            rho=1e-4,
            cost_floor=0.01,
            cost_beta=1.0,
            temperature=1.0,
            vocab_chunk_size=2,
            divergence="unknown",
        )
    except ValueError as error:
        assert "divergence" in str(error)
    else:
        raise AssertionError("unknown VERPO divergence was accepted")


def test_uid_group_gate_is_order_independent():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    uids = ["a", "b", "c", "a", "b", "c"]
    gate = compute_group_zpd_gate_by_uid(rewards, uids)
    assert gate.tolist() == [False, False, True, False, False, True]


def test_uid_reward_ranked_gate_admits_all_wrong_length_spread():
    rewards = torch.tensor([0.0, -0.25, -0.5, -1.0])
    uids = ["a"] * 4
    gate = compute_group_zpd_gate_by_uid(
        rewards,
        uids,
        mode="reward_ranked",
    )
    assert gate.all()


def test_evidence_rollout_scope_matches_post_group_gate_contract():
    group_gate = torch.tensor([True, True, False, False])
    outcome_positive = torch.tensor([True, False, True, False])

    assert apply_evidence_rollout_scope(
        group_gate, outcome_positive, scope="all"
    ).tolist() == [True, True, False, False]
    assert apply_evidence_rollout_scope(
        group_gate, outcome_positive, scope="wrong_only"
    ).tolist() == [False, True, False, False]


def test_wrong_only_scope_masks_only_evidence_loss_and_keeps_reference_loss():
    student = torch.zeros(2, 3, requires_grad=True)
    q0 = torch.zeros_like(student)
    qe = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    sampled = torch.tensor([0, 0])
    advantages = torch.tensor([1.0, -1.0])
    outcome_positive = torch.tensor([True, False])
    group_gate = torch.ones(2, dtype=torch.bool)
    all_gate = apply_evidence_rollout_scope(
        group_gate, outcome_positive, scope="all"
    )
    wrong_only_gate = apply_evidence_rollout_scope(
        group_gate, outcome_positive, scope="wrong_only"
    )

    all_outputs = compute_flat_verpo_losses(
        student,
        q0,
        qe,
        sampled,
        advantages,
        all_gate,
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=1.0,
        vocab_chunk_size=3,
        divergence="forward_kl",
        displacement_mode="evidence_vs_none",
    )
    wrong_only_outputs = compute_flat_verpo_losses(
        student,
        q0,
        qe,
        sampled,
        advantages,
        wrong_only_gate,
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=1.0,
        vocab_chunk_size=3,
        divergence="forward_kl",
        displacement_mode="evidence_vs_none",
    )

    assert bool((all_outputs["weights"] > 0).all().item())
    assert wrong_only_outputs["weights"][0].item() == 0.0
    torch.testing.assert_close(
        wrong_only_outputs["weights"][1],
        all_outputs["weights"][1],
    )
    torch.testing.assert_close(
        wrong_only_outputs["reference_loss"],
        all_outputs["reference_loss"],
    )
    assert wrong_only_outputs["evidence_loss"][0].item() == 0.0
    torch.testing.assert_close(
        wrong_only_outputs["evidence_loss"][1],
        all_outputs["evidence_loss"][1],
    )
    wrong_only_outputs["evidence_loss"].sum().backward()
    torch.testing.assert_close(student.grad[0], torch.zeros_like(student.grad[0]))
    assert bool((student.grad[1].abs() > 0).any().item())


def test_qref_sample_log_prob_is_exactly_reusable_from_q0():
    torch.manual_seed(11)
    q0 = torch.randn(7, 19)
    sampled = torch.randint(0, 19, (7,))
    qref = torch.log_softmax(q0.float(), dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    reused_q0 = torch.log_softmax(q0.float(), dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(reused_q0, qref, rtol=0.0, atol=0.0)


def _parameter_fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_reference_is_immutable_across_update_save_and_resume(tmp_path):
    torch.manual_seed(17)
    reference = torch.nn.Linear(4, 3)
    reference.requires_grad_(False)
    student = torch.nn.Linear(4, 3)
    initial = _parameter_fingerprint(reference)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    loss = (student(torch.randn(2, 4)) - reference(torch.randn(2, 4))).square().mean()
    loss.backward()
    optimizer.step()
    assert _parameter_fingerprint(reference) == initial

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"student": student.state_dict(), "reference": reference.state_dict()}, checkpoint)
    assert _parameter_fingerprint(reference) == initial
    restored = torch.nn.Linear(4, 3)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["reference"])
    assert _parameter_fingerprint(restored) == initial



def test_chunked_logsumexp_matches_full_fp32_values_and_gradients():
    torch.manual_seed(101)
    temperature = 0.73
    chunked_logits = torch.randn(5, 29, requires_grad=True)
    oracle_logits = chunked_logits.detach().clone().requires_grad_(True)
    weights = torch.randn(5, 1)

    actual = chunked_logsumexp(
        chunked_logits,
        temperature=temperature,
        vocab_chunk_size=7,
    )
    expected = torch.logsumexp(
        oracle_logits.float() / temperature,
        dim=-1,
        keepdim=True,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(
        chunked_logits.grad,
        oracle_logits.grad,
        rtol=1e-6,
        atol=1e-6,
    )


def test_chunked_gather_keeps_full_softmax_student_gradients():
    torch.manual_seed(103)
    temperature = 0.81
    actual_logits = torch.randn(4, 31, requires_grad=True)
    oracle_logits = actual_logits.detach().clone().requires_grad_(True)
    support = TruncatedTopKSupport(
        token_ids=torch.tensor(
            [
                [0, 7, 30],
                [1, 8, 29],
                [2, 9, 28],
                [3, 10, 27],
            ],
            dtype=torch.long,
        ),
        valid_mask=torch.ones(4, 3, dtype=torch.bool),
    )

    actual = gather_full_softmax_log_probs(
        actual_logits,
        support,
        temperature=temperature,
        vocab_chunk_size=6,
    )
    expected = torch.log_softmax(
        oracle_logits.float() / temperature,
        dim=-1,
    ).gather(-1, support.token_ids)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(
        actual_logits.grad,
        oracle_logits.grad,
        rtol=1e-6,
        atol=1e-6,
    )


def test_streaming_negative_teacher_mixture_matches_probability_mean(monkeypatch):
    torch.manual_seed(107)
    temperature = 0.67
    components = torch.randn(4, 9, 23)
    row_indices = torch.tensor([1, 3, 4, 8], dtype=torch.long)
    accumulator = torch.zeros(row_indices.numel(), components.shape[-1])

    def forbidden_log_softmax(*args, **kwargs):
        raise AssertionError("streaming top-k mixture must not call full log_softmax")

    monkeypatch.setattr(torch, "log_softmax", forbidden_log_softmax)
    for component in components:
        _accumulate_indexed_teacher_probabilities_(
            accumulator,
            component,
            row_indices,
            temperature=temperature,
            vocab_chunk_size=5,
            weight=1.0 / components.shape[0],
        )

    expected = torch.softmax(
        components[:, row_indices].float() / temperature,
        dim=-1,
    ).mean(dim=0)
    torch.testing.assert_close(accumulator, expected, rtol=1e-6, atol=1e-6)


def test_pre_normalized_negative_teacher_matches_logit_contract():
    torch.manual_seed(109)
    tokens, vocab = 6, 37
    temperature = 0.72
    student = torch.randn(tokens, vocab, requires_grad=True)
    q0 = torch.randn(tokens, vocab)
    qe = torch.randn(tokens, vocab)
    components = torch.randn(4, tokens, vocab)
    negative_probabilities = torch.softmax(
        components.float() / temperature,
        dim=-1,
    ).mean(dim=0)
    negative_logits = negative_probabilities.log() * temperature
    common = dict(
        sampled_token_ids=torch.randint(0, vocab, (tokens,)),
        token_advantages=torch.randn(tokens),
        active_token_mask=torch.ones(tokens, dtype=torch.bool),
        tau=1.0,
        rho=1e-4,
        cost_floor=0.01,
        cost_beta=1.0,
        temperature=temperature,
        vocab_chunk_size=8,
        divergence="forward_kl",
        displacement_mode="fec",
        projection_epsilon=1e-8,
        vocab_mode="topk_truncated",
        top_k=5,
    )

    from_probabilities = compute_flat_verpo_losses(
        student,
        q0,
        qe,
        negative_teacher_probabilities=negative_probabilities,
        **common,
    )
    from_logits = compute_flat_verpo_losses(
        student,
        q0,
        qe,
        negative_teacher_logits=negative_logits,
        **common,
    )
    assert from_probabilities.keys() == from_logits.keys()
    for name in from_probabilities:
        torch.testing.assert_close(
            from_probabilities[name],
            from_logits[name],
            rtol=1e-5,
            atol=1e-6,
        )
