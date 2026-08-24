#!/usr/bin/env python3
"""Fail-closed preflight for a private ModelScope experiment repository."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from verl.utils.checkpoint.modelscope_upload import ensure_private_modelscope_repository


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--audit-path", type=Path, required=True)
    parser.add_argument("--token-env", default="MODELSCOPE_TOKEN")
    args = parser.parse_args()
    result = ensure_private_modelscope_repository(
        repo_id=args.repo_id,
        token=os.environ[args.token_env],
        audit_path=args.audit_path,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "repo_id": result["repo_id"],
                "visibility": result["visibility"],
                "created": result["created"],
                "reused": result["reused"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
