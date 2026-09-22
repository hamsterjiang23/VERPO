from pathlib import Path

import pytest

from risk_aware_opsd.sdpo_data import convert_rows, parse_rows
from scripts.summarize_results import TASKS, summarize, summarize_run


def test_data_has_verifier_answer_but_no_solution() -> None:
    row = {
        "idx": 7,
        "prompt": "question",
        "system": "system",
        "answer": "SECRET_GT",
        "extra_info": {"solution": "PRIVATE_SOLUTION"},
    }
    result = convert_rows([row], "biology", "train")[0]
    assert result["record_id"] == "sdpo_biology_train_7"
    assert result["reward_model"]["ground_truth"] == "SECRET_GT"
    assert "SECRET_GT" not in str(result["prompt"])
    assert "PRIVATE_SOLUTION" not in str(result)
    assert "solution" not in result["extra_info"]
    with pytest.raises(ValueError, match="duplicate"):
        convert_rows([row, row], "biology", "train")


def test_parquet_preserves_messages_and_ids(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = convert_rows(
        [{"idx": 0, "prompt": "Question: use tool", "answer": "[]"}], "tooluse", "test"
    )
    path = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert pq.read_table(path).to_pylist() == rows


def test_jsonl_and_array_inputs() -> None:
    assert parse_rows(b'[{"x":1}]') == parse_rows(b'{"x":1}\n')


def test_selection_ties_fixed_late_and_missing() -> None:
    run = {
        "points": [
            {"step": 0, "score": 1},
            {"step": 150, "score": 0.8},
            {"step": 155, "score": 0.8},
            {"step": 200, "score": 0.6},
        ],
        "status": "collapsed",
    }
    result = summarize_run(run)
    assert result["best"]["step"] == 150
    assert result["fixed"]["score"] == 0.6
    assert result["late_count"] == 3
    assert result["late_sample_std"] == pytest.approx(0.1154700538)
    assert not result["rankable"]
    assert summarize_run({"points": []})["best"] is None


def test_average_requires_all_tasks_and_keeps_seed_provenance() -> None:
    runs = [
        {
            "model": "m",
            "method": "lw",
            "task": task,
            "seed": 42,
            "evidence_source": "rollout_group",
            "points": [{"step": 200, "score": (i + 1) / 10}],
        }
        for i, task in enumerate(TASKS)
    ]
    result = summarize(runs)
    assert result["averages"][0]["best_average"] == 0.3
    assert summarize(runs[:-1])["averages"][0]["best_average"] is None
    with pytest.raises(ValueError, match="multiple runs"):
        summarize(runs + runs[:1])


def test_nonfinite_metric_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        summarize_run({"points": [{"step": 1, "score": float("nan")}]})
