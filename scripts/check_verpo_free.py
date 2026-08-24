#!/usr/bin/env python3
"""Fail closed when removed probing/runtime names re-enter the code tree."""

from __future__ import annotations

import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (ROOT / "risk_aware_opsd", ROOT / "pipeline", ROOT / "scripts", ROOT / "configs", ROOT / "verl" / "verl", ROOT / "verl" / "tests")
EXCLUDED = {"archive", "provenance"}
FORBIDDEN = (re.compile(r"(?<![A-Za-z])pgr(?:[-_]|\b)"), re.compile(r"hidden_replay"))


def main() -> int:
    findings = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.name == "check_verpo_free.py" or any(part in EXCLUDED for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lowered = text.lower()
            if any(term.search(lowered) for term in FORBIDDEN):
                findings.append(path.relative_to(ROOT).as_posix())
    if findings:
        print("forbidden runtime references:", *sorted(findings), sep="\n", file=sys.stderr)
        return 1
    print("VERPO runtime scan clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
