#!/usr/bin/env python3
"""Create and verify the versioned SDPO training-data bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sdpo.privileged_context import DATASET_SPECS  # noqa: E402


BUNDLE_SCHEMA = "sdpo_training_bundle_v1"
DEFAULT_VERSION = "sdpo_privileged_context_v1"
ARCHIVE_NAME = "sdpo_verl.tar.gz"
MANIFEST_NAME = "bundle_manifest.json"
CHECKSUM_NAME = "SHA256SUMS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def required_files() -> dict[str, int | None]:
    files: dict[str, int | None] = {"manifest.json": None}
    for name, spec in DATASET_SPECS.items():
        files[f"{name}.train.annotated.jsonl"] = spec.expected_train_rows
        files[f"{name}.train.annotated.parquet"] = None
        files[f"{name}.test.jsonl"] = spec.expected_test_rows
        files[f"{name}.test.parquet"] = None
    return files


def inspect_data_dir(data_dir: Path, *, allow_partial: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, expected_rows in sorted(required_files().items()):
        path = data_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"required SDPO export is missing or empty: {path}")
        record: dict[str, Any] = {
            "name": name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if expected_rows is not None:
            actual_rows = count_jsonl(path)
            partial_train = name.endswith(".train.annotated.jsonl") and 0 < actual_rows <= expected_rows
            if actual_rows != expected_rows and not (allow_partial and partial_train):
                raise ValueError(
                    f"row-count mismatch for {path}: expected {expected_rows}, got {actual_rows}"
                )
            record["rows"] = actual_rows
        records.append(record)
    return records


def write_archive(data_dir: Path, output: Path, files: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
                with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                    for record in files:
                        source = data_dir / str(record["name"])
                        info = archive.gettarinfo(str(source), arcname=source.name)
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        with source.open("rb") as source_handle:
                            archive.addfile(info, source_handle)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def create_bundle(
    data_dir: Path, output_dir: Path, version: str, *, allow_partial: bool = False
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = inspect_data_dir(data_dir, allow_partial=allow_partial)
    archive_path = output_dir / ARCHIVE_NAME
    write_archive(data_dir, archive_path, files)
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "data_version": version,
        "allow_partial": allow_partial,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data_dir": str(data_dir),
        "archive": {
            "name": ARCHIVE_NAME,
            "size": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
        "files": files,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_path = output_dir / CHECKSUM_NAME
    checksum_path.write_text(
        f"{manifest['archive']['sha256']}  {ARCHIVE_NAME}\n"
        f"{sha256_file(manifest_path)}  {MANIFEST_NAME}\n",
        encoding="utf-8",
    )
    return manifest


def verify_bundle(data_dir: Path, manifest_path: Path, expected_version: str | None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError(f"unsupported SDPO bundle schema: {manifest.get('schema_version')!r}")
    if expected_version and manifest.get("data_version") != expected_version:
        raise ValueError(
            f"SDPO data version mismatch: expected {expected_version!r}, "
            f"got {manifest.get('data_version')!r}"
        )
    expected_records = {record["name"]: record for record in manifest.get("files", [])}
    required = required_files()
    if set(expected_records) != set(required):
        raise ValueError("bundle manifest file set does not match the registered SDPO export")
    for name, registered_rows in required.items():
        record = expected_records[name]
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"bundle file is missing: {path}")
        if path.stat().st_size != int(record["size"]):
            raise ValueError(f"size mismatch for {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"SHA256 mismatch for {path}")
        if registered_rows is not None:
            actual_rows = count_jsonl(path)
            partial_train = name.endswith(".train.annotated.jsonl") and 0 < actual_rows <= registered_rows
            allowed = actual_rows == registered_rows or (
                manifest.get("allow_partial") is True and partial_train
            )
            if not allowed or actual_rows != int(record.get("rows", -1)):
                raise ValueError(f"row-count mismatch for {path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/SDPO/verl")
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--version", default=DEFAULT_VERSION)
    create.add_argument("--allow-partial", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-version")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create":
        result = create_bundle(
            args.data_dir, args.output_dir, args.version, allow_partial=args.allow_partial
        )
    else:
        result = verify_bundle(args.data_dir, args.manifest, args.expected_version)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
