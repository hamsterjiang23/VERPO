"""Run existing tensor tests only in an already-provisioned environment."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    missing = [
        name
        for name in (
            "torch",
            "tensordict",
            "transformers",
            "ray",
            "omegaconf",
            "codetiming",
        )
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        print(
            json.dumps(
                {
                    "status": "not_run",
                    "reason": "missing_existing_dependencies",
                    "missing": missing,
                }
            )
        )
        return
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "verl"), env.get("PYTHONPATH", "")]
    )
    files = [
        "test_verpo_zpd_math_on_cpu.py",
        "test_verpo_advantage_modulation_on_cpu.py",
        "test_verpo_topk_memory_on_cpu.py",
    ]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *[str(root / "verl/tests/trainer/ppo" / name) for name in files],
        ],
        env=env,
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
