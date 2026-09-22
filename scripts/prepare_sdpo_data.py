"""Prepare pinned public SDPO splits without generating privileged solutions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from risk_aware_opsd.sdpo_data import (
    DATASET_SPECS,
    SCHEMA_VERSION,
    SOURCE_REVISION,
    SOURCE_ROOT,
    convert_rows,
    parse_rows,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/sdpo_source_manifest.json"


def verify_installed(data_dir: Path, *, parquet: bool = True) -> dict:
    manifest = json.loads(
        (data_dir / "rollout_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("source_revision") != SOURCE_REVISION
    ):
        raise ValueError("installed data belongs to another source/protocol")
    expected = {
        f"{name}.{split}.{ext}"
        for name in DATASET_SPECS
        for split in ("train", "test")
        for ext in (("jsonl", "parquet") if parquet else ("jsonl",))
    }
    files = manifest["files"]
    if not expected.issubset(files):
        raise ValueError("installed data manifest is incomplete")
    for name in expected:
        record = files[name]
        path = data_dir / name
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"installed data checksum mismatch: {name}")
        dataset, split, ext = name.split(".")
        expected_count = getattr(DATASET_SPECS[dataset], f"expected_{split}_rows")
        if record["rows"] != expected_count:
            raise ValueError(f"installed row count mismatch: {name}")
        if ext == "jsonl" and len(parse_rows(path.read_bytes())) != expected_count:
            raise ValueError(f"actual JSONL row count mismatch: {name}")
    return manifest


def prepare(
    data_dir: Path, *, source_dir: Path | None = None, parquet: bool = True
) -> dict:
    if (data_dir / "rollout_manifest.json").is_file():
        try:
            return verify_installed(data_dir, parquet=parquet)
        except (OSError, ValueError, KeyError):
            # Keep the previous files until every replacement is staged and checked.
            pass
    if parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="sdpo-rollout-", dir=data_dir.parent
    ) as temp:
        staging = Path(temp)
        records: dict[str, dict] = {}
        for dataset, spec in DATASET_SPECS.items():
            for split in ("train", "test"):
                relative = f"{spec.path}/{split}.json"
                if source_dir is not None:
                    payload = (source_dir / relative).read_bytes()
                else:
                    with urllib.request.urlopen(
                        f"{SOURCE_ROOT}/{relative}", timeout=120
                    ) as response:
                        payload = response.read()
                raw_path = staging / "raw.json"
                raw_path.write_bytes(payload)
                if sha256_file(raw_path) != lock["files"][relative]["sha256"]:
                    raise ValueError(f"upstream checksum mismatch: {relative}")
                rows = convert_rows(parse_rows(payload), dataset, split)
                count = getattr(spec, f"expected_{split}_rows")
                if len(rows) != count:
                    raise ValueError(
                        f"expected {count} rows in {relative}, got {len(rows)}"
                    )
                name = f"{dataset}.{split}.jsonl"
                (staging / name).write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
                names = [name]
                if parquet:
                    name = f"{dataset}.{split}.parquet"
                    pq.write_table(pa.Table.from_pylist(rows), staging / name)
                    names.append(name)
                for name in names:
                    records[name] = {
                        "rows": count,
                        "sha256": sha256_file(staging / name),
                        "source": relative,
                    }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_revision": SOURCE_REVISION,
            "evidence_source": "rollout_group",
            "files": records,
        }
        (staging / "rollout_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        for name in [*records, "rollout_manifest.json"]:
            os.replace(staging / name, data_dir / name)
    return verify_installed(data_dir, parquet=parquet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/SDPO/verl")
    parser.add_argument(
        "--source-dir", type=Path, help="Offline copy of pinned upstream files"
    )
    parser.add_argument("--jsonl-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = (
        verify_installed(args.data_dir, parquet=not args.jsonl_only)
        if args.verify_only
        else prepare(
            args.data_dir, source_dir=args.source_dir, parquet=not args.jsonl_only
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
