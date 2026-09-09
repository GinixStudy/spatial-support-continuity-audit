#!/usr/bin/env python3
"""Verify locked CSV values in the legacy TAR or a reproduced output tree."""

from __future__ import annotations

import argparse
import csv
from io import TextIOWrapper
import json
import math
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parent


def normalize(value: str):
    text = value.strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        return float(text)
    except ValueError:
        return text


def select_row(rows: list[dict[str, str]], where: dict[str, str]) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in where.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {where}, found {len(matches)}")
    return matches[0]


def check_value(actual, expected, atol: float | None) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol or 0)
    return actual == expected


def archive_rows(tf: tarfile.TarFile, path_contains: str) -> list[dict[str, str]]:
    members = [m for m in tf.getmembers() if path_contains in m.name]
    if len(members) != 1:
        raise FileNotFoundError(f"archive match for {path_contains}: {len(members)}")
    handle = tf.extractfile(members[0])
    if handle is None:
        raise FileNotFoundError(members[0].name)
    return list(csv.DictReader(TextIOWrapper(handle, encoding="utf-8")))


def tree_rows(root: Path, path_contains: str) -> list[dict[str, str]]:
    normalized = path_contains.replace("/", str(Path("/")).replace("/", "\\"))
    matches = [p for p in root.rglob("*.csv") if path_contains in p.as_posix()]
    if not matches and normalized != path_contains:
        matches = [p for p in root.rglob("*.csv") if normalized in str(p)]
    if len(matches) != 1:
        raise FileNotFoundError(f"filesystem match for {path_contains}: {len(matches)}")
    with matches[0].open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--expected", type=Path, default=ROOT / "expected_results.json"
    )
    args = parser.parse_args()

    specification = json.loads(args.expected.read_text(encoding="utf-8"))
    cache: dict[str, list[dict[str, str]]] = {}
    tf = tarfile.open(args.archive, "r") if args.archive else None
    failures = []
    try:
        for check in specification["checks"]:
            key = check["path_contains"]
            if key not in cache:
                cache[key] = (
                    archive_rows(tf, key)
                    if tf is not None
                    else tree_rows(args.output_root, key)
                )
            row = select_row(cache[key], check.get("where", {}))
            actual = normalize(row[check["column"]])
            expected = check["expected"]
            ok = check_value(actual, expected, check.get("atol"))
            label = f"{key}:{check['column']}"
            print(f"{'PASS' if ok else 'FAIL'}\t{label}\t{actual}")
            if not ok:
                failures.append((label, actual, expected))
    finally:
        if tf is not None:
            tf.close()

    if failures:
        print(f"FAILED: {len(failures)} locked checks")
        return 1
    print(f"OK: {len(specification['checks'])} locked checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
