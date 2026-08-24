import pytest
import torch

from verl.trainer.distillation.verpo_gradient_audit import VerpoGradientAuditor


def test_verpo_gradient_audit_reports_scaled_norms_and_pairwise_cosines():
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    auditor = VerpoGradientAuditor(
        max_train_batches=1,
        max_parameter_elements=16,
        max_parameter_tensors=2,
    )
    auditor.bind(model, optimizer)
    auditor.begin_train_batch()

    output = model(torch.tensor([[1.0, -2.0]])).sum()
    metrics = auditor.audit(
        grpo_loss=output,
        reference_scaled_loss=2.0 * output,
        evidence_scaled_loss=-output,
    )

    assert metrics["verpo_grad/reference_scaled_to_grpo_ratio"] == pytest.approx(2.0)
    assert metrics["verpo_grad/evidence_scaled_to_grpo_ratio"] == pytest.approx(1.0)
    assert metrics["verpo_grad/grpo_reference_cosine"] == pytest.approx(1.0)
    assert metrics["verpo_grad/grpo_evidence_cosine"] == pytest.approx(-1.0)
    assert metrics["verpo_grad/reference_evidence_cosine"] == pytest.approx(-1.0)
    assert all(parameter.grad is None for parameter in model.parameters())

    # The retained graph remains available for the ordinary combined backward.
    (output + 2.0 * output - output).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_verpo_gradient_audit_runs_only_for_configured_train_batches():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    auditor = VerpoGradientAuditor(
        max_train_batches=1,
        max_parameter_elements=4,
        max_parameter_tensors=1,
    )
    auditor.bind(model, optimizer)

    auditor.begin_train_batch()
    first = model(torch.ones(1, 1)).sum()
    assert auditor.audit(
        grpo_loss=first,
        reference_scaled_loss=first,
        evidence_scaled_loss=first,
    )

    auditor.begin_train_batch()
    second = model(torch.ones(1, 1)).sum()
    assert auditor.audit(
        grpo_loss=second,
        reference_scaled_loss=second,
        evidence_scaled_loss=second,
    ) == {}
