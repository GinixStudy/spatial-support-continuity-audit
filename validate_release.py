#!/usr/bin/env python3
"""Run lightweight validation that does not repeat the scientific experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(
    label: str, command: list[str], *, environment: dict[str, str] | None = None
) -> bool:
    print(f"\n[{label}]")
    completed = subprocess.run(command, cwd=ROOT, check=False, env=environment)
    return completed.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run lightweight release checks without repeating experiments."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional final TAR to verify against the 20 locked-result assertions.",
    )
    args = parser.parse_args()

    test_environment = os.environ.copy()
    source_path = str(ROOT / "src")
    existing_pythonpath = test_environment.get("PYTHONPATH")
    test_environment["PYTHONPATH"] = (
        source_path + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else source_path
    )

    checks = [
        run(
            "companion-tool unit tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            environment=test_environment,
        ),
        run("pipeline source check", [sys.executable, "run_pipeline.py", "check"]),
        run(
            "reference audit report check",
            [sys.executable, "validation/verify_reference_audit.py"],
        ),
    ]
    if args.archive is not None:
        checks.append(
            run(
                "locked-result archive check",
                [
                    sys.executable,
                    "verify_frozen_results.py",
                    "--archive",
                    str(args.archive.resolve()),
                ],
            )
        )
    print("\nPASS: release validation" if all(checks) else "\nFAIL: release validation")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
