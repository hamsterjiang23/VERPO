"""Summarize explicit per-run evaluation points without inventing missing evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path
from typing import Any

TASKS = ("biology", "chemistry", "material", "physics", "tooluse")


def summarize_run(
    run: dict[str, Any], *, fixed_step: int = 200, late_start: int = 150
) -> dict[str, Any]:
    points = run.get("points", [])
    if len({point["step"] for point in points}) != len(points):
        raise ValueError("duplicate evaluation step within one run")
    for point in points:
        if not isinstance(point["step"], int) or point["step"] < 0:
            raise ValueError("evaluation steps must be nonnegative integers")
        if (
            not math.isfinite(float(point["score"]))
            or not 0 <= float(point["score"]) <= 1
        ):
            raise ValueError("scores must be finite accuracy fractions in [0, 1]")
    trained = [point for point in points if point["step"] > 0]
    best = (
        min(trained, key=lambda point: (-Decimal(str(point["score"])), point["step"]))
        if trained
        else None
    )
    fixed = next((point for point in trained if point["step"] == fixed_step), None)
    late = [
        float(point["score"])
        for point in trained
        if late_start <= point["step"] <= fixed_step
    ]
    result = {key: value for key, value in run.items() if key != "points"}
    result.update(
        {
            "best": best,
            "fixed": fixed,
            "late_count": len(late),
            "late_mean": statistics.mean(late) if late else None,
            "late_sample_std": statistics.stdev(late) if len(late) > 1 else None,
            "rankable": bool(trained) and run.get("status", "ok") == "ok",
            "score_source": "provided_evaluation_points" if points else "unavailable",
        }
    )
    return result


def summarize(
    runs: list[dict[str, Any]], *, fixed_step: int = 200, late_start: int = 150
) -> dict[str, Any]:
    rows = [
        summarize_run(run, fixed_step=fixed_step, late_start=late_start) for run in runs
    ]
    groups: dict[tuple[str, str, int | None, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["model"],
            row["method"],
            row.get("seed"),
            row.get("evidence_source", "unknown"),
        )
        groups.setdefault(key, []).append(row)
    averages = []
    for (model, method, seed, evidence_source), group in groups.items():
        if len({row["task"] for row in group}) != len(group):
            raise ValueError(
                "multiple runs for one task/seed/protocol: select a run explicitly"
            )
        complete = {row["task"] for row in group} == set(TASKS)
        averages.append(
            {
                "model": model,
                "method": method,
                "seed": seed,
                "evidence_source": evidence_source,
                "rankable": complete and all(row["rankable"] for row in group),
                **{
                    f"{kind}_average": float(
                        sum(Decimal(str(row[kind]["score"])) for row in group)
                        / Decimal(5)
                    )
                    if complete and all(row[kind] is not None for row in group)
                    else None
                    for kind in ("best", "fixed")
                },
            }
        )
    return {
        "schema_version": 1,
        "fixed_step": fixed_step,
        "late_start": late_start,
        "runs": rows,
        "averages": averages,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Evaluation summary",
        "",
        "Scores are accuracy fractions from supplied evaluation points. Best excludes step 0; ties select the earlier step.",
        "",
        "| Model | Method | Task | Status | Best step | Best | Fixed | Source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["runs"]:
        best = row["best"] or {}
        fixed = row["fixed"] or {}
        values = (
            row["model"],
            row["method"],
            row["task"],
            row.get("status", "ok"),
            best.get("step", "—"),
            best.get("score", "—"),
            fixed.get("score", "—"),
            row["score_source"],
        )
        lines.append(
            "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "## Five-task averages",
            "",
            "Incomplete task sets have no average. Collapsed/RH runs remain visible and are not rankable.",
            "",
        ]
    )
    for row in result["averages"]:
        lines.append(
            f"- {row['model']} / {row['method']} / seed {row['seed']}: best={row['best_average']}, fixed={row['fixed_average']}, rankable={row['rankable']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="JSON object with a runs list"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-step", type=int, default=200)
    parser.add_argument("--late-start", type=int, default=150)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.input.read_text(encoding="utf-8"))["runs"],
        fixed_step=args.fixed_step,
        late_start=args.late_start,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(result), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
