import pytest

from risk_aware_opsd.rlcsd_verl_reward import compute_score as compute_real_score
from risk_aware_opsd.rlcsd_verl_smoke_reward import compute_score as compute_smoke_score


def test_real_reward_requires_a_correct_boxed_answer():
    assert compute_real_score("deepmath_filtered", r"The answer is \boxed{2}.", "2")["score"] == 1.0
    assert compute_real_score("deepmath_filtered", "The answer is 2.", "2")["score"] == 0.0


def test_smoke_reward_forces_mixed_session_parity_and_reports_real_grade():
    positive = compute_smoke_score(
        "deepmath_filtered",
        r"The answer is \boxed{2}.",
        "2",
        extra_info={"session_id": 0},
    )
    negative = compute_smoke_score(
        "deepmath_filtered",
        r"The answer is \boxed{2}.",
        "2",
        extra_info={"session_id": 1},
    )
    assert positive["score"] == 1.0
    assert negative["score"] == 0.0
    assert positive["real_acc"] == negative["real_acc"] == 1.0
    assert positive["smoke_forced_mixed"] == negative["smoke_forced_mixed"] == 1.0


def test_smoke_reward_requires_session_id():
    with pytest.raises(ValueError, match="session_id"):
        compute_smoke_score("deepmath_filtered", "answer", "2", extra_info={})
