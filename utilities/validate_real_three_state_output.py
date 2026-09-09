"""Validate real-data diagnostics against the locked manuscript Jaccard table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_by_group(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        key = "analysis_group" if "analysis_group" in fields else "group"
        if key not in fields:
            raise ValueError(f"{path} has neither analysis_group nor group")
        return {row[key]: row for row in reader}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    reference = read_by_group(args.reference)
    diagnostics = read_by_group(args.diagnostics)
    failures: list[str] = []
    if len(reference) != 43:
        failures.append(f"reference contains {len(reference)} groups instead of 43")
    if set(reference) != set(diagnostics):
        missing = sorted(set(reference) - set(diagnostics))
        extra = sorted(set(diagnostics) - set(reference))
        failures.append(f"group mismatch; missing={missing}, extra={extra}")

    maximum_absolute_difference = 0.0
    for group in sorted(set(reference) & set(diagnostics)):
        expected = float(reference[group]["expected_species_jaccard"])
        observed = float(diagnostics[group]["species_jaccard"])
        difference = abs(expected - observed)
        maximum_absolute_difference = max(maximum_absolute_difference, difference)
        if not math.isclose(expected, observed, rel_tol=0.0, abs_tol=1e-12):
            failures.append(
                f"{group}: expected species_jaccard={expected}, observed={observed}"
            )

    report = {
        "status": "PASS" if not failures else "FAIL",
        "groups_checked": len(set(reference) & set(diagnostics)),
        "expected_groups": 43,
        "jaccard_absolute_tolerance": 1e-12,
        "maximum_absolute_difference": maximum_absolute_difference,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{report['status']}: {report['groups_checked']} real state-species groups; "
        f"maximum Jaccard difference={maximum_absolute_difference:.3g}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
