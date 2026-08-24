#!/usr/bin/env python3
"""Preserve every completed Trainer checkpoint outside rotation scope."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")
TERMINAL_STATUSES = {
    "early_stopped",
    "max_optimizer_steps_reached",
    "max_rounds_reached",
}


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def checkpoint_is_complete(path: Path) -> bool:
    step = checkpoint_step(path)
    if step is None or not path.is_dir():
        return False
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if int(state.get("global_step", -1)) != step:
        return False
    model_files = list(path.glob("model*.safetensors")) + list(
        path.glob("pytorch_model*.bin")
    )
    if not model_files:
        return False
    latest = path / "latest"
    global_step_dir = path / f"global_step{step}"
    if latest.exists() or global_step_dir.exists():
        if not latest.is_file() or not global_step_dir.is_dir():
            return False
        try:
            if latest.read_text(encoding="utf-8").strip() != global_step_dir.name:
                return False
        except OSError:
            return False
        if not any(global_step_dir.iterdir()):
            return False
    return True


def _link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_dev == destination_stat.st_dev
            and source_stat.st_ino == destination_stat.st_ino
        ):
            return
        if (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _sync_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for root, directories, files in os.walk(source):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        target_root = destination / relative
        target_root.mkdir(parents=True, exist_ok=True)
        directories.sort()
        files.sort()
        for filename in files:
            source_file = root_path / filename
            target_file = target_root / filename
            if source_file.is_symlink():
                link_target = os.readlink(source_file)
                if target_file.is_symlink() and os.readlink(target_file) == link_target:
                    continue
                if target_file.exists() or target_file.is_symlink():
                    target_file.unlink()
                target_file.symlink_to(link_target)
            else:
                _link_or_copy_file(source_file, target_file)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_manifest(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()]
    trainer_state = path / "trainer_state.json"
    return {
        "step": checkpoint_step(path),
        "path": str(path),
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "trainer_state_sha256": _sha256(trainer_state),
        "model_files": [
            {"name": item.name, "bytes": item.stat().st_size}
            for item in sorted(path.glob("model*.safetensors"))
        ],
    }


def _write_manifest(run_dir: Path, archive_dir: Path) -> dict[str, Any]:
    checkpoints = sorted(
        (
            path
            for path in archive_dir.glob("checkpoint-*")
            if checkpoint_is_complete(path)
        ),
        key=lambda path: checkpoint_step(path) or -1,
    )
    manifest = {
        "schema_version": "all_training_checkpoints_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "archive_dir": str(archive_dir),
        "retention_policy": "keep_all",
        "storage_method": "hardlink_same_filesystem_copy_fallback",
        "archived_steps": [checkpoint_step(path) for path in checkpoints],
        "checkpoints": [_checkpoint_manifest(path) for path in checkpoints],
    }
    temporary = archive_dir / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(archive_dir / "manifest.json")
    return manifest


def archive_available_checkpoints(
    run_dir: Path, archive_dir: Path | None = None
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    archive_dir = (
        Path(archive_dir).resolve()
        if archive_dir is not None
        else run_dir / "all_checkpoints"
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    lock_path = archive_dir / ".archive.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        sources = sorted(
            (
                path
                for path in run_dir.glob("checkpoint-*")
                if checkpoint_is_complete(path)
            ),
            key=lambda path: checkpoint_step(path) or -1,
        )
        for source in sources:
            step = checkpoint_step(source)
            assert step is not None
            destination = archive_dir / source.name
            if destination.exists():
                _sync_tree(source, destination)
                continue
            partial = archive_dir / f".{source.name}.partial"
            _sync_tree(source, partial)
            if checkpoint_is_complete(source):
                partial.replace(destination)
        manifest = _write_manifest(run_dir, archive_dir)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return manifest


def _terminal_status(summary_path: Path) -> str | None:
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = payload.get("status")
    return str(status) if status in TERMINAL_STATUSES else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--archive-dir")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stop-when-terminal", action="store_true")
    parser.add_argument("--terminal-stable-polls", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    archive_dir = Path(args.archive_dir) if args.archive_dir else None
    summary_path = run_dir / "early_stopping_summary.json"
    previous_steps: list[int] | None = None
    stable_terminal_polls = 0
    while True:
        manifest = archive_available_checkpoints(run_dir, archive_dir)
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
        if args.once:
            return
        status = _terminal_status(summary_path) if args.stop_when_terminal else None
        steps = list(manifest["archived_steps"])
        if status and steps == previous_steps:
            stable_terminal_polls += 1
            if stable_terminal_polls >= args.terminal_stable_polls:
                return
        else:
            stable_terminal_polls = 0
        previous_steps = steps
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
