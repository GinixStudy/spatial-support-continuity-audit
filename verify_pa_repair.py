#!/usr/bin/env python3
"""Validate the PA descriptive reconstruction without rerunning it."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED = {
    "candidate_pairs_before_tree_filter": 2226,
    "matched_live_tree_physical_plots": 2199,
    "tree_rows_scanned": 557397,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("matched_plot_parquet", type=Path)
    args = parser.parse_args()

    with args.summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one summary row, found {len(rows)}")

    failures = []
    for column, expected in EXPECTED.items():
        actual = int(rows[0][column])
        ok = actual == expected
        print(f"{'PASS' if ok else 'FAIL'}\t{column}\t{actual}")
        if not ok:
            failures.append((column, actual, expected))

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to validate the matched-plot Parquet; "
            "install requirements_core.txt"
        ) from error

    parquet_rows = pq.ParquetFile(args.matched_plot_parquet).metadata.num_rows
    ok = parquet_rows == EXPECTED["matched_live_tree_physical_plots"]
    print(f"{'PASS' if ok else 'FAIL'}\tparquet_rows\t{parquet_rows}")
    if not ok:
        failures.append(("parquet_rows", parquet_rows, 2199))

    if failures:
        return 1
    print("OK: PA descriptive reconstruction is internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
