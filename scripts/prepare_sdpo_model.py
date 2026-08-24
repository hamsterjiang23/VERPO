#!/usr/bin/env python3
"""Resolve or download the pinned Hugging Face model used by SDPO training."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path


def validate_snapshot(path: Path, revision: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"model snapshot does not exist: {resolved}")
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"model snapshot is missing config.json: {resolved}")
    if revision != "main" and resolved.name != revision:
        raise ValueError(
            f"model snapshot revision mismatch: expected directory {revision!r}, got {resolved}"
        )
    return resolved


def resolve_model(
    *,
    repo_id: str,
    revision: str,
    cache_dir: Path,
    model_path: Path | None,
    downloader: Callable[..., str] | None = None,
) -> Path:
    if model_path is not None:
        return validate_snapshot(model_path, revision)
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    downloaded = Path(
        downloader(repo_id=repo_id, revision=revision, cache_dir=str(cache_dir.expanduser()))
    )
    return validate_snapshot(downloaded, revision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = resolve_model(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
    )
    print(path)


if __name__ == "__main__":
    main()
