#!/usr/bin/env python3
"""Verify the machine-readable report from the successful frozen run."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reference_reports"
AUDIT_CSV = REPORTS / "locked_result_audit.csv"
MARKER_JSON = REPORTS / "FINAL_REPRODUCTION_AUDIT_COMPLETE.json"


def main() -> int:
    with AUDIT_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    failed = [row for row in rows if row.get("passed") != "True"]
    marker = json.loads(MARKER_JSON.read_text(encoding="utf-8"))

    assertions = {
        "CSV contains 138 checks": len(rows) == 138,
        "CSV contains no failed checks": not failed,
        "completion status is complete": marker.get("status") == "complete",
        "completion marker reports 138 checks": marker.get("n_checks") == 138,
        "completion marker reports 138 passes": marker.get("n_passed") == 138,
        "completion marker reports zero failures": marker.get("n_failed") == 0,
        "scientific checks passed": marker.get("all_scientific_checks_passed") is True,
    }

    for label, passed in assertions.items():
        print(f"{'PASS' if passed else 'FAIL'}: {label}")

    if failed:
        for row in failed:
            print(f"FAILED AUDIT ROW: {row.get('group')} / {row.get('check')}")

    return 0 if all(assertions.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
