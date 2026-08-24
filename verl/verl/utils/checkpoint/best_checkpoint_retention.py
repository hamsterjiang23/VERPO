"""Validation-ranked checkpoint retention for storage-constrained formal runs."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

VERPO_CONTRASTIVE_MEAN_AT_12_KEYS = (
    "val-core/amc23/acc/mean@12",
    "val-core/aime24/acc/mean@12",
    "val-core/aime25/acc/mean@12",
)


def verpo_contrastive_macro_mean_at_12(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    missing = [key for key in VERPO_CONTRASTIVE_MEAN_AT_12_KEYS if key not in metrics]
    if missing:
        raise RuntimeError(f"Top-checkpoint selection is missing fixed-validation metrics: {missing}")
    selected = {key: float(metrics[key]) for key in VERPO_CONTRASTIVE_MEAN_AT_12_KEYS}
    nonfinite = [key for key, value in selected.items() if not math.isfinite(value)]
    if nonfinite:
        raise RuntimeError(f"Top-checkpoint selection has non-finite metrics: {nonfinite}")
    return sum(selected.values()) / len(selected), selected


def retain_best_verpo_contrastive_checkpoints(
    checkpoint_root: str | Path,
    *,
    step: int,
    metrics: dict[str, float],
    keep_best: int,
    keep_current: bool = True,
    terminal: bool,
) -> dict:
    if keep_best < 1:
        raise ValueError("keep_best must be positive")
    root = Path(checkpoint_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    current = root / f"global_step_{step}"
    if not current.is_dir():
        raise RuntimeError(f"Current checkpoint is missing before Top-{keep_best} retention: {current}")

    manifest_path = root / "best_checkpoint_retention.json"
    history: dict[int, dict] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(previous.get("keep_best", -1)) != keep_best:
            raise RuntimeError("Checkpoint-retention keep_best changed inside one run")
        if bool(previous.get("keep_current", True)) != keep_current:
            raise RuntimeError("Checkpoint-retention keep_current changed inside one run")
        history = {int(item["step"]): item for item in previous.get("history", [])}

    score, selected_metrics = verpo_contrastive_macro_mean_at_12(metrics)
    history[step] = {"step": step, "score": score, "metrics": selected_metrics}
    ranked = sorted(history.values(), key=lambda item: (-float(item["score"]), int(item["step"])))
    best_steps = [int(item["step"]) for item in ranked[:keep_best]]
    retained_steps = set(best_steps)
    if not terminal and keep_current:
        retained_steps.add(step)

    removed_steps = []
    for path in sorted(root.glob("global_step_*")):
        suffix = path.name.removeprefix("global_step_")
        if not path.is_dir() or not suffix.isdigit():
            continue
        candidate_step = int(suffix)
        if candidate_step > step or candidate_step in retained_steps:
            continue
        resolved = path.resolve()
        if resolved.parent != root or resolved.name != f"global_step_{candidate_step}":
            raise RuntimeError(f"Refusing to prune checkpoint outside the registered root: {resolved}")
        shutil.rmtree(resolved)
        removed_steps.append(candidate_step)

    existing_steps = sorted(
        int(path.name.removeprefix("global_step_"))
        for path in root.glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
    )
    if set(existing_steps) != retained_steps:
        raise RuntimeError(
            f"Checkpoint retention mismatch: expected={sorted(retained_steps)}, actual={existing_steps}"
        )
    tracker = root / "latest_checkpointed_iteration.txt"
    tracker.write_text(str(max(existing_steps)), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "selection_metric": "verpo_contrastive_macro_mean_at_12_accuracy",
        "keep_best": keep_best,
        "keep_current": keep_current,
        "peak_full_checkpoints": keep_best + 1,
        "history": sorted(history.values(), key=lambda item: int(item["step"])),
        "best_steps": best_steps,
        "retained_steps": existing_steps,
        "latest_evaluated_step": step,
        "terminal": terminal,
        "removed_at_latest_update": removed_steps,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return payload
