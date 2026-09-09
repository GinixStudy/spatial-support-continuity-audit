"""Command-line interface for spatial-support diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

from .diagnostics import CellPeriodCount, diagnose_support


def _number(value: str, *, column: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"row {row_number}: {column} is not numeric") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"row {row_number}: {column} must be non-negative and finite")
    return parsed


def _load_groups(args: argparse.Namespace) -> dict[str, list[CellPeriodCount]]:
    required = {
        args.period_column,
        args.cell_column,
        args.effort_column,
        args.focal_column,
    }
    if args.group_column:
        required.add(args.group_column)

    groups: dict[str, list[CellPeriodCount]] = {}
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            group = row[args.group_column] if args.group_column else "all"
            groups.setdefault(group, []).append(
                CellPeriodCount(
                    period=row[args.period_column],
                    cell_id=row[args.cell_column],
                    effort_count=_number(
                        row[args.effort_column],
                        column=args.effort_column,
                        row_number=row_number,
                    ),
                    focal_count=_number(
                        row[args.focal_column],
                        column=args.focal_column,
                        row_number=row_number,
                    ),
                )
            )
    if not groups:
        raise ValueError("input contains no data rows")
    return groups


def _write_results(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial-support-audit",
        description=(
            "Calculate two-period support overlap and correction-availability "
            "diagnostics from an aggregated cell-period CSV."
        ),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-column")
    parser.add_argument("--period-column", default="period")
    parser.add_argument("--cell-column", default="cell_id")
    parser.add_argument("--effort-column", default="effort_count")
    parser.add_argument("--focal-column", default="focal_count")
    parser.add_argument("--early-label", default="early")
    parser.add_argument("--late-label", default="late")
    parser.add_argument("--minimum-effort", type=float, default=10.0)
    parser.add_argument("--minimum-stable-cells", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        groups = _load_groups(args)
        results = [
            diagnose_support(
                rows,
                group=group,
                early_label=args.early_label,
                late_label=args.late_label,
                minimum_effort_per_cell_period=args.minimum_effort,
                minimum_stable_cells=args.minimum_stable_cells,
            ).to_dict()
            for group, rows in sorted(groups.items())
        ]
        _write_results(args.output, results)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    unavailable = sum(not bool(result["corrected_estimable"]) for result in results)
    print(f"Wrote {len(results)} diagnostic row(s) to {args.output}")
    print(f"Corrected estimator unavailable for {unavailable} group(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
