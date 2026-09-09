#!/usr/bin/env python3
"""Validate the container without rerunning the scientific experiments."""

from __future__ import annotations

import csv
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> None:
    print(f"\n[{label}]", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify_environment() -> None:
    expected: dict[str, str] = {}
    with (ROOT / "software_versions_core.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            expected[row["package"]] = row["version"]

    actual_python = platform.python_version()
    if actual_python != expected.pop("python"):
        raise RuntimeError(
            f"Python version mismatch: expected 3.12.13, observed {actual_python}"
        )

    mismatches = []
    for distribution, expected_version in sorted(expected.items()):
        actual_version = metadata.version(distribution)
        if actual_version != expected_version:
            mismatches.append(
                f"{distribution}: expected {expected_version}, observed {actual_version}"
            )
    if mismatches:
        raise RuntimeError("Core environment mismatch:\n" + "\n".join(mismatches))

    toolkit_version = metadata.version("spatial-support-audit")
    if toolkit_version != "0.1.0":
        raise RuntimeError(
            f"Toolkit version mismatch: expected 0.1.0, observed {toolkit_version}"
        )
    print("PASS: Python, core dependencies, and toolkit versions match")


def verify_manifest() -> None:
    failures = []
    lines = (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected_hash, relative_path = line.split("  ", 1)
        path = ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing: {relative_path}")
            continue
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            failures.append(f"hash mismatch: {relative_path}")
    if failures:
        raise RuntimeError("Manifest verification failed:\n" + "\n".join(failures))
    print(f"PASS: {len(lines)} release files match SHA256SUMS.txt")


def verify_real_data_reference() -> None:
    with tempfile.TemporaryDirectory(prefix="spatial-support-audit-") as tmp:
        output = Path(tmp) / "diagnostics.csv"
        report = Path(tmp) / "validation_report.json"
        run(
            "43-group real-data diagnostic",
            [
                sys.executable,
                "-m",
                "spatial_support_audit.cli",
                "real_data/three_state_support_counts.csv",
                "--output",
                str(output),
                "--group-column",
                "analysis_group",
                "--minimum-effort",
                "10",
                "--minimum-stable-cells",
                "3",
            ],
        )
        run(
            "locked real-data Jaccard comparison",
            [
                sys.executable,
                "utilities/validate_real_three_state_output.py",
                "--reference",
                "real_data/locked_reference_cell_jaccard.csv",
                "--diagnostics",
                str(output),
                "--report",
                str(report),
            ],
        )


def main() -> int:
    verify_environment()
    verify_manifest()
    run("installed-package dependency check", [sys.executable, "-m", "pip", "check"])
    run("lightweight release validation", [sys.executable, "validate_release.py"])
    verify_real_data_reference()
    print("\nPASS: container validation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
