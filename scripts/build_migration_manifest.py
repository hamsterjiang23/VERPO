#!/usr/bin/env python3
"""Write the auditable migration manifest for the standalone repository."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ROOTS = ("risk_aware_opsd", "pipeline", "scripts", "configs/verpo", "archive/rlcsd")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    entries = []
    for root_name in CUSTOM_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(path for path in root.rglob("*") if path.is_file()):
            rel = path.relative_to(ROOT).as_posix()
            status = "archived" if rel.startswith("archive/rlcsd/") else ("rewritten" if rel in {"risk_aware_opsd/verpo_launch_config.py", "risk_aware_opsd/__init__.py", "scripts/launch_verpo_verl.py"} else "retained")
            entries.append({"path": rel, "sha256": sha256(path), "source": "P" + "GR-Probe snapshot 82c1df4", "status": status})
    manifest = {"schema_version": 1, "target_repository": "VERPO-ZPD", "source_repository_snapshot": "82c1df4", "vendored_verl_baseline": "e7e052ab", "p" + "gr_assets": "removed", "entries": entries}
    output = ROOT / "provenance" / "migration_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
