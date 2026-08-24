"""Fail-closed ModelScope uploads for completed training checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

MAX_FILE_BYTES = 50 * 1024**3
MAX_TOTAL_FILES = 100_000
MAX_FILES_PER_DIRECTORY = 10_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


def _clean_remote_component(value: str, *, field: str, allow_slashes: bool) -> str:
    candidate = value.strip().strip("/")
    if not candidate:
        raise ValueError(f"{field} must not be empty")
    path = PurePosixPath(candidate)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} contains an unsafe remote path: {value!r}")
    if not allow_slashes and len(path.parts) != 1:
        raise ValueError(f"{field} must be one path component: {value!r}")
    return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_files(checkpoint: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    counts_by_directory: dict[str, int] = {}
    for path in sorted(checkpoint.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"Refusing to upload a checkpoint containing a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(checkpoint).as_posix()
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise RuntimeError(
                f"ModelScope single-file limit exceeded ({size} > {MAX_FILE_BYTES}): {relative}"
            )
        parent = PurePosixPath(relative).parent.as_posix()
        counts_by_directory[parent] = counts_by_directory.get(parent, 0) + 1
        if counts_by_directory[parent] > MAX_FILES_PER_DIRECTORY:
            raise RuntimeError(
                "ModelScope per-directory file limit exceeded "
                f"({MAX_FILES_PER_DIRECTORY}): {parent}"
            )
        files.append({"path": relative, "size": size, "sha256": _sha256(path)})
        if len(files) > MAX_TOTAL_FILES:
            raise RuntimeError(
                f"ModelScope total file limit exceeded ({MAX_TOTAL_FILES})"
            )
    return files


def _validate_checkpoint(checkpoint_root: Path, step: int) -> Path:
    if step <= 0:
        raise ValueError("step must be positive")
    root = checkpoint_root.resolve()
    checkpoint = root / f"global_step_{step}"
    actor = checkpoint / "actor"
    tracker = root / "latest_checkpointed_iteration.txt"
    complete = (
        checkpoint.is_dir()
        and actor.is_dir()
        and any(path.is_file() for path in actor.rglob("*"))
        and (checkpoint / "data.pt").is_file()
        and tracker.is_file()
    )
    if complete:
        try:
            # An asynchronous uploader may inspect step N after training has
            # already saved N+1.  The global tracker therefore proves that N
            # was committed when it is at least N; equality is only valid for
            # a synchronous upload performed immediately after saving.
            complete = int(tracker.read_text(encoding="utf-8").strip()) >= step
        except ValueError:
            complete = False
    if not complete:
        raise RuntimeError(f"checkpoint is incomplete for step {step}: {checkpoint}")
    if checkpoint.is_symlink() or checkpoint.resolve().parent != root:
        raise RuntimeError(f"Refusing to upload a checkpoint outside its root: {checkpoint}")
    return checkpoint.resolve()


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(
        manifest["files"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _redact(value: Any, token: str) -> str:
    rendered = str(value)
    return rendered.replace(token, "<redacted>") if token else rendered


def _normalize_repo_id(repo_id: str) -> str:
    repo_parts = repo_id.strip().strip("/").split("/")
    if len(repo_parts) != 2 or any(part in {"", ".", ".."} for part in repo_parts):
        raise ValueError("repo_id must use the owner/repository form")
    return "/".join(repo_parts)


def _visibility_value(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    for key in ("visibility", "Visibility"):
        if key in value:
            try:
                return int(value[key])
            except (TypeError, ValueError):
                return None
    for key in ("data", "Data", "model", "Model"):
        nested = _visibility_value(value.get(key))
        if nested is not None:
            return nested
    return None


def _is_not_found(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message for marker in ("404", "not found", "not exist", "不存在")
    )


def _ensure_private_repository_with_api(
    api: Any,
    *,
    repo_id: str,
    audit_path: Path,
    token: str,
) -> dict[str, Any]:
    if audit_path.is_file():
        previous = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "complete"
            and previous.get("repo_id") == repo_id
            and previous.get("visibility") == "private"
        ):
            return {**previous, "reused": True}

    started_at = _utc_now()
    try:
        try:
            model_info = api.get_model(repo_id)
            visibility = _visibility_value(model_info)
            if visibility != 1:
                raise RuntimeError(
                    f"Existing ModelScope repository is not verified private: {repo_id}"
                )
            created = False
        except Exception as error:
            if not _is_not_found(error):
                raise
            api.create_model(repo_id, visibility=1)
            created = True
        result = {
            "schema_version": 1,
            "status": "complete",
            "repo_id": repo_id,
            "visibility": "private",
            "visibility_code": 1,
            "created": created,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "reused": False,
        }
        _write_json(audit_path, result)
        return result
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "repo_id": repo_id,
            "visibility": "private",
            "visibility_code": 1,
            "started_at": started_at,
            "failed_at": _utc_now(),
            "error_type": type(error).__name__,
            "error": _redact(error, token),
        }
        _write_json(audit_path, failure)
        raise RuntimeError(_redact(error, token)) from None


def ensure_private_modelscope_repository(
    *,
    repo_id: str,
    token: str,
    audit_path: str | Path,
    api_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Create a missing ModelScope model repository with private visibility."""

    if not token:
        raise RuntimeError("ModelScope upload token is empty")
    normalized_repo_id = _normalize_repo_id(repo_id)
    if api_factory is None:
        from modelscope.hub.api import HubApi

        api_factory = HubApi
    api = api_factory()
    api.login(token)
    return _ensure_private_repository_with_api(
        api,
        repo_id=normalized_repo_id,
        audit_path=Path(audit_path).resolve(),
        token=token,
    )


def upload_checkpoint_to_modelscope(
    checkpoint_root: str | Path,
    *,
    step: int,
    repo_id: str,
    experiment_id: str,
    token: str,
    path_prefix: str = "qwen3-1.7b-verpo-zpd",
    revision: str = "master",
    max_workers: int = 8,
    audit_dir: str | Path | None = None,
    validation_metrics: dict[str, Any] | None = None,
    api_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Upload one complete checkpoint and its SHA256 manifest.

    A completed matching audit is idempotently reused. Any SDK failure is
    recorded locally and re-raised so checkpoint retention cannot run.
    """

    if not token:
        raise RuntimeError("ModelScope upload token is empty")
    repo_id = _normalize_repo_id(repo_id)
    if not revision.strip():
        raise ValueError("revision must not be empty")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    checkpoint_root_path = Path(checkpoint_root).resolve()
    checkpoint = _validate_checkpoint(checkpoint_root_path, step)
    clean_prefix = _clean_remote_component(
        path_prefix, field="path_prefix", allow_slashes=True
    )
    clean_experiment = _clean_remote_component(
        experiment_id, field="experiment_id", allow_slashes=False
    )
    remote_path = f"{clean_prefix}/{clean_experiment}/global_step_{step}"
    audit_root = (
        Path(audit_dir).resolve()
        if audit_dir is not None
        else checkpoint_root_path.parent / "modelscope_uploads"
    )
    step_audit = audit_root / f"global_step_{step}"
    if audit_root == checkpoint or checkpoint in audit_root.parents:
        raise ValueError("audit_dir must be outside the uploaded checkpoint directory")
    manifest_path = step_audit / "upload_manifest.json"
    result_path = step_audit / "upload_result.json"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "step": step,
        "repo_id": repo_id,
        "revision": revision,
        "remote_path": remote_path,
        "checkpoint_directory": checkpoint.name,
        "validation_metrics": _json_safe(validation_metrics or {}),
        "files": _checkpoint_files(checkpoint),
    }
    manifest["checkpoint_fingerprint"] = _manifest_fingerprint(manifest)

    if manifest_path.is_file() and result_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_result = json.loads(result_path.read_text(encoding="utf-8"))
        reusable = (
            previous_result.get("status") == "complete"
            and previous_manifest.get("checkpoint_fingerprint")
            == manifest["checkpoint_fingerprint"]
            and previous_result.get("repo_id") == repo_id
            and previous_result.get("revision") == revision
            and previous_result.get("remote_path") == remote_path
        )
        if reusable:
            return {**previous_result, "reused": True}

    _write_json(manifest_path, manifest)
    started_at = _utc_now()
    try:
        if api_factory is None:
            from modelscope.hub.api import HubApi

            api_factory = HubApi
        api = api_factory()
        api.login(token)
        _ensure_private_repository_with_api(
            api,
            repo_id=repo_id,
            audit_path=audit_root / "repository_result.json",
            token=token,
        )
        folder_response = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(checkpoint),
            path_in_repo=remote_path,
            commit_message=f"Upload {clean_experiment} checkpoint step {step}",
            max_workers=max_workers,
            revision=revision,
        )
        manifest_response = api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=f"{remote_path}/upload_manifest.json",
            repo_id=repo_id,
            commit_message=f"Upload checkpoint manifest for step {step}",
            revision=revision,
        )
        result = {
            "schema_version": 1,
            "status": "complete",
            "step": step,
            "repo_id": repo_id,
            "revision": revision,
            "remote_path": remote_path,
            "checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
            "file_count": len(manifest["files"]),
            "total_bytes": sum(item["size"] for item in manifest["files"]),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "folder_response": _redact(folder_response, token),
            "manifest_response": _redact(manifest_response, token),
            "reused": False,
        }
        _write_json(result_path, result)
        return result
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "step": step,
            "repo_id": repo_id,
            "revision": revision,
            "remote_path": remote_path,
            "checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
            "started_at": started_at,
            "failed_at": _utc_now(),
            "error_type": type(error).__name__,
            "error": _redact(error, token),
        }
        _write_json(result_path, failure)
        raise RuntimeError(_redact(error, token)) from None


def _retention_score(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    keys = (
        "val-core/amc23/acc/mean@12",
        "val-core/aime24/acc/mean@12",
        "val-core/aime25/acc/mean@12",
    )
    missing = [key for key in keys if key not in metrics]
    if missing:
        raise RuntimeError(
            f"Top-checkpoint selection is missing fixed-validation metrics: {missing}"
        )
    selected = {key: float(metrics[key]) for key in keys}
    nonfinite = [key for key, value in selected.items() if not math.isfinite(value)]
    if nonfinite:
        raise RuntimeError(
            f"Top-checkpoint selection has non-finite metrics: {nonfinite}"
        )
    return sum(selected.values()) / len(selected), selected


def _successful_upload_steps(audit_root: Path, repo_id: str) -> set[int]:
    successful: set[int] = set()
    for result_path in audit_root.glob("global_step_*/upload_result.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            step = int(payload["step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("status") == "complete" and payload.get("repo_id") == repo_id:
            successful.add(step)
    return successful


def _retain_after_verified_upload(
    checkpoint_root: Path,
    *,
    audit_root: Path,
    repo_id: str,
    step: int,
    metrics: dict[str, Any],
    keep_best: int,
    keep_current: bool,
    terminal: bool,
    observed_checkpoint_count: int,
) -> dict[str, Any]:
    """Prune only checkpoints with a completed matching upload audit.

    Unlike the synchronous retention helper, this function deliberately
    tolerates newer local checkpoints.  Such checkpoints are an asynchronous
    upload backlog and are protected until their own upload succeeds.
    """

    if keep_best < 1:
        raise ValueError("keep_best must be positive")
    root = checkpoint_root.resolve()
    current = root / f"global_step_{step}"
    if not current.is_dir():
        raise RuntimeError(
            f"Uploaded checkpoint disappeared before retention: {current}"
        )
    result_path = audit_root / f"global_step_{step}" / "upload_result.json"
    manifest_path = audit_root / f"global_step_{step}" / "upload_manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"Missing completed upload audit before retention for step {step}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "complete"
        or int(result.get("step", -1)) != step
        or result.get("repo_id") != repo_id
    ):
        raise RuntimeError(
            f"Upload audit is not complete and matching before retention for step {step}"
        )

    retention_path = root / "best_checkpoint_retention.json"
    previous_observed_peak = 0
    if retention_path.is_file():
        previous = json.loads(retention_path.read_text(encoding="utf-8"))
        if int(previous.get("keep_best", -1)) != keep_best:
            raise RuntimeError("Checkpoint-retention keep_best changed inside one run")
        if bool(previous.get("keep_current", True)) != keep_current:
            raise RuntimeError("Checkpoint-retention keep_current changed inside one run")
        previous_observed_peak = int(
            previous.get("observed_peak_full_checkpoints", 0)
        )

    validation_root = audit_root / "validation_selection"
    history: dict[int, dict[str, Any]] = {}
    for validation_path in sorted(validation_root.glob("global_step_*.json")):
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation_step = int(validation["step"])
        score, selected_metrics = _retention_score(validation["validation_metrics"])
        history[validation_step] = {
            "step": validation_step,
            "score": score,
            "metrics": selected_metrics,
            "checkpoint_available": bool(validation.get("checkpoint_saved", False)),
            "upload_complete": False,
        }
    if step not in history or not history[step]["checkpoint_available"]:
        raise RuntimeError(
            f"Missing checkpoint-bearing validation-selection record for step {step}"
        )

    successful_steps = _successful_upload_steps(audit_root, repo_id)
    for successful_step in successful_steps:
        if successful_step in history and history[successful_step]["checkpoint_available"]:
            history[successful_step]["upload_complete"] = True
    ranked = sorted(
        (
            item
            for item in history.values()
            if item["checkpoint_available"] and item["upload_complete"]
        ),
        key=lambda item: (-float(item["score"]), int(item["step"])),
    )
    best_steps = [int(item["step"]) for item in ranked[:keep_best]]

    existing_before = {
        int(path.name.removeprefix("global_step_"))
        for path in root.glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
    }
    pending_protected = existing_before - successful_steps
    retained_uploaded = set(best_steps)
    if not terminal and keep_current:
        retained_uploaded.add(step)

    removed_steps: list[int] = []
    for candidate_step in sorted(successful_steps - retained_uploaded):
        path = root / f"global_step_{candidate_step}"
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved.parent != root or resolved.name != f"global_step_{candidate_step}":
            raise RuntimeError(
                f"Refusing to prune checkpoint outside the registered root: {resolved}"
            )
        import shutil

        shutil.rmtree(resolved)
        removed_steps.append(candidate_step)

    existing_after = sorted(
        int(path.name.removeprefix("global_step_"))
        for path in root.glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
    )
    expected_after = sorted((existing_before - set(removed_steps)))
    if existing_after != expected_after:
        raise RuntimeError(
            "Asynchronous checkpoint retention mismatch: "
            f"expected={expected_after}, actual={existing_after}"
        )
    if terminal and pending_protected:
        raise RuntimeError(
            "Terminal asynchronous retention still has unuploaded checkpoints: "
            f"{sorted(pending_protected)}"
        )

    tracker = root / "latest_checkpointed_iteration.txt"
    if existing_after:
        tracker.write_text(str(max(existing_after)), encoding="utf-8")
    payload = {
        "schema_version": 2,
        "selection_metric": "verpo_contrastive_macro_mean_at_12_accuracy",
        "keep_best": keep_best,
        "keep_current": keep_current,
        # This remains the registered steady-state target used by the formal
        # profile.  The observed field below records temporary async backlog.
        "peak_full_checkpoints": keep_best + 1,
        "observed_peak_full_checkpoints": max(
            previous_observed_peak, observed_checkpoint_count, len(existing_before)
        ),
        "async_upload": True,
        "history": sorted(history.values(), key=lambda item: int(item["step"])),
        "best_steps": best_steps,
        "retained_steps": existing_after,
        "pending_protected_steps": sorted(pending_protected),
        "latest_evaluated_step": max(history, default=step),
        "latest_uploaded_step": step,
        "terminal": terminal,
        "removed_at_latest_update": removed_steps,
    }
    _write_json(retention_path, payload)
    return payload


class AsyncModelScopeCheckpointUploader:
    """Durable single-worker upload queue for checkpoint/manifest uploads.

    Enqueueing never waits for hashing or network I/O.  Queue records are
    persisted without the token before work is submitted.  Upload failures are
    retained in the audit and surfaced by :meth:`drain`; failed or pending
    checkpoints are never eligible for pruning.
    """

    def __init__(
        self,
        checkpoint_root: str | Path,
        *,
        repo_id: str,
        experiment_id: str,
        token: str,
        path_prefix: str = "qwen3-1.7b-verpo-zpd",
        revision: str = "master",
        max_workers: int = 8,
        audit_dir: str | Path | None = None,
        retention_keep_best: int = 0,
        retention_keep_current: bool = True,
        api_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not token:
            raise RuntimeError("ModelScope upload token is empty")
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.repo_id = _normalize_repo_id(repo_id)
        self.experiment_id = _clean_remote_component(
            experiment_id, field="experiment_id", allow_slashes=False
        )
        self.token = token
        self.path_prefix = _clean_remote_component(
            path_prefix, field="path_prefix", allow_slashes=True
        )
        self.revision = revision
        self.max_workers = max_workers
        self.audit_root = (
            Path(audit_dir).resolve()
            if audit_dir is not None
            else self.checkpoint_root.parent / "modelscope_uploads"
        )
        self.queue_root = self.audit_root / "queue"
        self.retention_keep_best = retention_keep_best
        self.retention_keep_current = retention_keep_current
        self.api_factory = api_factory
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="modelscope-checkpoint-upload"
        )
        self._futures: dict[int, Future] = {}
        self._durable_failures: dict[int, str] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._recover_interrupted_jobs()

    def _job_path(self, step: int) -> Path:
        return self.queue_root / f"global_step_{step}.json"

    def _validation_path(self, step: int) -> Path:
        return self.audit_root / "validation_selection" / f"global_step_{step}.json"

    def _write_job(self, job: dict[str, Any]) -> None:
        _write_json(self._job_path(int(job["step"])), job)

    def _load_job(self, path: Path) -> dict[str, Any]:
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("repo_id") != self.repo_id:
            raise RuntimeError(
                f"Durable upload job repository mismatch at {path}: "
                f"{job.get('repo_id')} != {self.repo_id}"
            )
        if job.get("experiment_id") != self.experiment_id:
            raise RuntimeError(
                f"Durable upload job experiment mismatch at {path}"
            )
        return job

    def _recover_interrupted_jobs(self) -> None:
        if not self.queue_root.is_dir():
            return
        for path in sorted(self.queue_root.glob("global_step_*.json")):
            job = self._load_job(path)
            if job.get("status") == "failed":
                self._durable_failures[int(job["step"])] = str(
                    job.get("error", "previous asynchronous upload failure")
                )
                continue
            if job.get("status") not in {"queued", "uploading"}:
                continue
            step = int(job["step"])
            checkpoint = self.checkpoint_root / f"global_step_{step}"
            result_path = self.audit_root / f"global_step_{step}" / "upload_result.json"
            if not checkpoint.is_dir() and result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("status") == "complete" and result.get("repo_id") == self.repo_id:
                    retention_ok = self.retention_keep_best == 0
                    retention_path = (
                        self.checkpoint_root / "best_checkpoint_retention.json"
                    )
                    if retention_path.is_file():
                        retention = json.loads(
                            retention_path.read_text(encoding="utf-8")
                        )
                        retained_history = {
                            int(item["step"])
                            for item in retention.get("history", [])
                        }
                        retention_ok = step in retained_history and (
                            not bool(job.get("terminal", False))
                            or bool(retention.get("terminal", False))
                        )
                    if retention_ok:
                        job.update(
                            status="complete", recovered=True, completed_at=_utc_now()
                        )
                        self._write_job(job)
                        continue
                    message = (
                        "uploaded checkpoint is missing before its retention audit "
                        f"completed for step {step}"
                    )
                    job.update(
                        status="failed",
                        recovered=True,
                        failed_at=_utc_now(),
                        error_type="RuntimeError",
                        error=message,
                    )
                    self._durable_failures[step] = message
                    self._write_job(job)
                    continue
            job.update(status="queued", recovered=True, queued_at=_utc_now())
            self._write_job(job)
            self._submit(job)

    def enqueue(
        self,
        *,
        step: int,
        validation_metrics: dict[str, Any],
        terminal: bool,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("ModelScope upload queue is closed")
        _validate_checkpoint(self.checkpoint_root, step)
        path = self._job_path(step)
        if path.is_file():
            previous = self._load_job(path)
            if previous.get("status") == "complete":
                return previous
            if step in self._futures:
                return previous
        observed_count = sum(
            1
            for item in self.checkpoint_root.glob("global_step_*")
            if item.is_dir() and item.name.removeprefix("global_step_").isdigit()
        )
        job = {
            "schema_version": 1,
            "status": "queued",
            "step": step,
            "repo_id": self.repo_id,
            "experiment_id": self.experiment_id,
            "path_prefix": self.path_prefix,
            "revision": self.revision,
            "max_workers": self.max_workers,
            "validation_metrics": _json_safe(validation_metrics),
            "terminal": bool(terminal),
            "retention_keep_best": self.retention_keep_best,
            "retention_keep_current": self.retention_keep_current,
            "observed_checkpoint_count": observed_count,
            "queued_at": _utc_now(),
        }
        self._write_job(job)
        self._submit(job)
        return job

    def record_validation(
        self,
        *,
        step: int,
        validation_metrics: dict[str, Any],
        checkpoint_saved: bool,
    ) -> dict[str, Any]:
        """Durably record every fixed validation, including non-save steps."""

        payload = {
            "schema_version": 1,
            "step": step,
            "repo_id": self.repo_id,
            "experiment_id": self.experiment_id,
            "checkpoint_saved": bool(checkpoint_saved),
            "validation_metrics": _json_safe(validation_metrics),
            "recorded_at": _utc_now(),
        }
        path = self._validation_path(step)
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            comparable_keys = (
                "step",
                "repo_id",
                "experiment_id",
                "checkpoint_saved",
                "validation_metrics",
            )
            if all(previous.get(key) == payload.get(key) for key in comparable_keys):
                return previous
            raise RuntimeError(f"Validation-selection record changed for step {step}")
        _write_json(path, payload)
        return payload

    def _submit(self, job: dict[str, Any]) -> None:
        step = int(job["step"])
        with self._lock:
            if step in self._futures:
                return
            self._futures[step] = self._executor.submit(self._run_job, dict(job))

    def _run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        step = int(job["step"])
        job.update(status="uploading", started_at=_utc_now())
        self._write_job(job)
        try:
            upload_result = upload_checkpoint_to_modelscope(
                self.checkpoint_root,
                step=step,
                repo_id=self.repo_id,
                experiment_id=self.experiment_id,
                token=self.token,
                path_prefix=self.path_prefix,
                revision=self.revision,
                max_workers=self.max_workers,
                audit_dir=self.audit_root,
                validation_metrics=job["validation_metrics"],
                api_factory=self.api_factory,
            )
            retention_result = None
            if self.retention_keep_best > 0:
                retention_result = _retain_after_verified_upload(
                    self.checkpoint_root,
                    audit_root=self.audit_root,
                    repo_id=self.repo_id,
                    step=step,
                    metrics=job["validation_metrics"],
                    keep_best=self.retention_keep_best,
                    keep_current=self.retention_keep_current,
                    terminal=bool(job["terminal"]),
                    observed_checkpoint_count=int(job["observed_checkpoint_count"]),
                )
            job.update(
                status="complete",
                completed_at=_utc_now(),
                upload_result={
                    key: upload_result[key]
                    for key in (
                        "status",
                        "step",
                        "repo_id",
                        "revision",
                        "remote_path",
                        "checkpoint_fingerprint",
                        "file_count",
                        "total_bytes",
                    )
                    if key in upload_result
                },
                retention_result=retention_result,
            )
            self._write_job(job)
            return job
        except Exception as error:
            job.update(
                status="failed",
                failed_at=_utc_now(),
                error_type=type(error).__name__,
                error=_redact(error, self.token),
            )
            self._write_job(job)
            raise RuntimeError(_redact(error, self.token)) from None

    def pending_count(self) -> int:
        return sum(1 for future in self._futures.values() if not future.done())

    def drain(self) -> list[dict[str, Any]]:
        """Wait for all queued uploads and surface every failure at terminal."""

        failures: list[str] = []
        results: list[dict[str, Any]] = []
        failures.extend(
            f"step {step}: {message}"
            for step, message in sorted(self._durable_failures.items())
        )
        for step, future in sorted(self._futures.items()):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(f"step {step}: {_redact(error, self.token)}")
        if failures:
            raise RuntimeError(
                "ModelScope asynchronous upload queue failed; local checkpoints were preserved: "
                + "; ".join(failures)
            )
        return results

    def close(self) -> list[dict[str, Any]]:
        if self._closed:
            return []
        try:
            return self.drain()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True)
