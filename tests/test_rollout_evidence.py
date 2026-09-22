from itertools import permutations

import pytest

from risk_aware_opsd.rollout_evidence import select_rollout_evidence


@pytest.mark.parametrize(
    "correct,negative,reasons",
    [
        ([False, False], True, (1, 1)),
        ([True, True], True, (2, 2)),
        ([True, True], False, (0, 0)),
        ([True, False], True, (1, 2)),
        ([True, False], False, (1, 0)),
        ([True], False, (1,)),
        ([False], True, (1,)),
        ([True, True, False, False], True, (0, 0, 0, 0)),
    ],
)
def test_availability(
    correct: list[bool], negative: bool, reasons: tuple[int, ...]
) -> None:
    n = len(correct)
    result = select_rollout_evidence(
        ["g"] * n,
        [f"g_{i}" for i in range(n)],
        correct,
        [True] * n,
        ["answer"] * n,
        require_negative=negative,
    )
    assert result.reason == reasons
    for i, selected in enumerate(result.positive):
        assert selected != i
        if selected >= 0:
            assert correct[selected]
    for i, selected in enumerate(result.negative):
        assert i not in selected
        assert all(index < 0 or not correct[index] for index in selected)


def test_selection_is_independent_of_batch_order() -> None:
    ids = ["g_10", "g_2", "g_3", "g_4"]
    correct = [True, True, False, False]
    expected: dict[str, tuple[str | None, tuple[str | None, ...]]] | None = None
    for order in permutations(range(4)):
        current = [ids[i] for i in order]
        result = select_rollout_evidence(
            ["g"] * 4, current, [correct[i] for i in order], [True] * 4, ["x"] * 4
        )
        selected = {
            current[i]: (
                current[p] if p >= 0 else None,
                tuple(current[n] if n >= 0 else None for n in result.negative[i]),
            )
            for i, p in enumerate(result.positive)
        }
        if expected is None:
            expected = selected
        assert selected == expected
    assert expected is not None and expected["g_3"][0] == "g_2"


def test_no_cross_group_empty_or_invalid_candidates() -> None:
    result = select_rollout_evidence(
        ["a", "b", "a", "a"],
        ["a_0", "b_0", "a_1", "a_2"],
        [False, True, True, True],
        [True, True, False, True],
        ["x", "x", "x", " "],
        require_negative=False,
    )
    assert result.positive[0] == -1


def test_duplicate_rollout_ids_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_rollout_evidence(
            ["a", "a"], ["same", "same"], [True, False], [True, True], ["x", "y"]
        )


def test_transferqueue_session_ordinal_is_numeric() -> None:
    result = select_rollout_evidence(
        ["g"] * 3,
        ["g_0_0", "g_10_0", "g_2_0"],
        [False, True, True],
        [True] * 3,
        ["x"] * 3,
        require_negative=False,
    )
    assert result.positive[0] == 2
