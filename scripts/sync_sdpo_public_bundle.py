#!/usr/bin/env python3
"""Install the pinned, public SDPO train/evaluation bundle atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.package_sdpo_training_data import (
    ARCHIVE_NAME,
    CHECKSUM_NAME,
    DEFAULT_VERSION,
    MANIFEST_NAME,
    sha256_file,
    verify_bundle,
)


@dataclass(frozen=True)
class PublicDriveArtifact:
    file_id: str
    sha256: str

    @property
    def url(self) -> str:
        return f"https://drive.google.com/uc?export=download&id={self.file_id}"


# These IDs and digests define the immutable sdpo_privileged_context_v1 release.
PUBLIC_ARTIFACTS: dict[str, PublicDriveArtifact] = {
    ARCHIVE_NAME: PublicDriveArtifact(
        file_id="1XIr92tESjEkBgbsBTa5WmpL0_DOhcyxG",
        sha256="212dcde5a933a7d15cebdf5708b473b727a721463e660735b6b239ca7727baf3",
    ),
    MANIFEST_NAME: PublicDriveArtifact(
        file_id="1VPTUJDUbbKocXoSOrVHnvAZPzH4jGP2r",
        sha256="06f51ef1c59dd65cfd03e49a546d7f25a5759885064fc37a17b4e94944d50004",
    ),
    CHECKSUM_NAME: PublicDriveArtifact(
        file_id="1ChLTIQeRrZqfUaLlBxFjcPYyLXNeBPHK",
        sha256="e5fb1a35391addbfa86c65edb3cd283df7817d2460369c092e3d22891e50f342",
    ),
}


def _download(artifact: PublicDriveArtifact, destination: Path, retries: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib_request.Request(
        artifact.url,
        headers={"User-Agent": "VERPO-ZPD-SDPO-Bootstrap/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary = destination.with_name(f".{destination.name}.download.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        try:
            digest = hashlib.sha256()
            with urllib_request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise ValueError(
                    f"registered SHA256 mismatch for {destination.name}: "
                    f"expected {artifact.sha256}, got {digest.hexdigest()}"
                )
            os.replace(temporary, destination)
            return
        except (OSError, ValueError, urllib_error.URLError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"failed to download public SDPO artifact: {destination.name}") from last_error


def _copy_source_bundle(source_dir: Path, destination: Path) -> None:
    for name in PUBLIC_ARTIFACTS:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"offline SDPO bundle artifact is missing: {source}")
        shutil.copy2(source, destination / name)


def _verify_registered_download(download_dir: Path) -> None:
    for name, artifact in PUBLIC_ARTIFACTS.items():
        actual = sha256_file(download_dir / name)
        if actual != artifact.sha256:
            raise ValueError(
                f"registered SHA256 mismatch for {name}: expected {artifact.sha256}, got {actual}"
            )

    expected_lines = {
        f"{PUBLIC_ARTIFACTS[ARCHIVE_NAME].sha256}  {ARCHIVE_NAME}",
        f"{PUBLIC_ARTIFACTS[MANIFEST_NAME].sha256}  {MANIFEST_NAME}",
    }
    actual_lines = {
        line.strip()
        for line in (download_dir / CHECKSUM_NAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if actual_lines != expected_lines:
        raise ValueError("downloaded SHA256SUMS does not match the registered SDPO release")


def _safe_extract(archive_path: Path, manifest: dict[str, Any], staged_dir: Path) -> None:
    expected = {str(record["name"]) for record in manifest.get("files", [])}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        if names != expected:
            raise ValueError("SDPO archive file set does not match bundle_manifest.json")
        for member in members:
            pure = PurePosixPath(member.name)
            if (
                not member.isfile()
                or pure.is_absolute()
                or len(pure.parts) != 1
                or pure.name in {"", ".", ".."}
            ):
                raise ValueError(f"unsafe SDPO archive member: {member.name!r}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read SDPO archive member: {member.name!r}")
            target = staged_dir / pure.name
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=1024 * 1024)


def sync_bundle(
    data_dir: Path,
    *,
    expected_version: str = DEFAULT_VERSION,
    force: bool = False,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    installed_manifest = data_dir / MANIFEST_NAME
    if not force and installed_manifest.is_file():
        try:
            manifest = verify_bundle(data_dir, installed_manifest, expected_version)
            return {"status": "already_verified", "data_dir": str(data_dir), "manifest": manifest}
        except (OSError, ValueError, KeyError, TypeError):
            pass

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sdpo-public-sync-", dir=data_dir.parent) as temporary:
        temporary_dir = Path(temporary)
        download_dir = temporary_dir / "download"
        staged_dir = temporary_dir / "staged"
        download_dir.mkdir()
        staged_dir.mkdir()

        if source_dir is None:
            for name, artifact in PUBLIC_ARTIFACTS.items():
                _download(artifact, download_dir / name)
            _verify_registered_download(download_dir)
        else:
            _copy_source_bundle(source_dir.expanduser().resolve(), download_dir)

        manifest = json.loads((download_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("data_version") != expected_version:
            raise ValueError(
                f"SDPO data version mismatch: expected {expected_version!r}, "
                f"got {manifest.get('data_version')!r}"
            )
        archive_record = manifest.get("archive", {})
        if archive_record.get("name") != ARCHIVE_NAME:
            raise ValueError("bundle manifest names an unexpected SDPO archive")
        if sha256_file(download_dir / ARCHIVE_NAME) != archive_record.get("sha256"):
            raise ValueError("SDPO archive digest does not match bundle_manifest.json")

        _safe_extract(download_dir / ARCHIVE_NAME, manifest, staged_dir)
        shutil.copy2(download_dir / MANIFEST_NAME, staged_dir / MANIFEST_NAME)
        verify_bundle(staged_dir, staged_dir / MANIFEST_NAME, expected_version)

        data_dir.mkdir(parents=True, exist_ok=True)
        install_order = [str(record["name"]) for record in manifest["files"]] + [MANIFEST_NAME]
        for name in install_order:
            temporary_target = data_dir / f".{name}.install.{os.getpid()}"
            shutil.copy2(staged_dir / name, temporary_target)
            os.replace(temporary_target, data_dir / name)

    verified = verify_bundle(data_dir, installed_manifest, expected_version)
    return {"status": "installed", "data_dir": str(data_dir), "manifest": verified}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "SDPO" / "verl")
    parser.add_argument("--expected-version", default=DEFAULT_VERSION)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Offline test/recovery directory containing the three release artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = sync_bundle(
        args.data_dir,
        expected_version=args.expected_version,
        force=args.force,
        source_dir=args.source_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
