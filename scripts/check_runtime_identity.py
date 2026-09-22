"""Protect checkpoint resume using effective, prepared data/model identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_aware_opsd.sdpo_data import sha256_file


def check_identity(
    run_root: Path, train: Path, validation: Path, model: Path, revision: str
) -> dict:
    identity = {
        "evidence_source": "rollout_group",
        "data": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in (train, validation)
        ],
        "model_path": str(model.resolve()),
        "model_revision": revision,
        "model_config_sha256": sha256_file(model / "config.json"),
    }
    target = run_root / "provenance/runtime_identity.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError(
                "Prepared data/model identity changed; refusing checkpoint resume. Use a new output root."
            )
    elif (run_root / "checkpoints").exists() and any(
        (run_root / "checkpoints").iterdir()
    ):
        raise ValueError("Existing checkpoints have no rollout_group runtime identity")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "train", "validation", "model"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    check_identity(
        args.run_root, args.train, args.validation, args.model, args.revision
    )


if __name__ == "__main__":
    main()
