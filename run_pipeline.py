#!/usr/bin/env python3
"""Run the manuscript reproduction scripts in their locked order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"
ORDER_FILE = ROOT / "pipeline_order.tsv"


def load_stages() -> list[tuple[str, str]]:
    stages = []
    for raw in ORDER_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        policy, script = line.split("\t", 1)
        stages.append((policy, script))
    return stages


def select_stages(
    stages: list[tuple[str, str]], start: int | None, stop: int | None
) -> list[tuple[int, str, str]]:
    numbered = [(i, policy, script) for i, (policy, script) in enumerate(stages, 1)]
    if start is not None:
        numbered = [item for item in numbered if item[0] >= start]
    if stop is not None:
        numbered = [item for item in numbered if item[0] <= stop]
    return numbered


def check_scripts(stages: list[tuple[str, str]]) -> int:
    missing = [script for _, script in stages if not (PIPELINE / script).is_file()]
    if missing:
        for script in missing:
            print(f"MISSING: {script}")
        return 1

    for _, script in stages:
        source = (PIPELINE / script).read_text(encoding="utf-8")
        compile(source, str(PIPELINE / script), "exec")
    print(f"OK: {len(stages)} scripts are present and syntactically valid.")
    return 0


def run_stage(
    number: int,
    policy: str,
    script: str,
    log_dir: Path,
    env: dict[str, str],
) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{number:02d}_{Path(script).stem}.log"
    print(f"\n[{number:02d}] {script} ({policy})")
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            [sys.executable, str(PIPELINE / script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check, list, or run the locked Kaggle-native manuscript pipeline. "
            "A full run includes the 600,000-replicate confirmation."
        )
    )
    parser.add_argument("command", choices=("check", "list", "run"))
    parser.add_argument("--start", type=int, help="First 1-based stage to run")
    parser.add_argument("--stop", type=int, help="Last 1-based stage to run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-heavy", action="store_true")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/kaggle/working/reproduction_logs"),
    )
    args = parser.parse_args()

    stages = load_stages()
    if args.command == "check":
        return check_scripts(stages)

    selected = select_stages(stages, args.start, args.stop)
    if args.command == "list" or args.dry_run:
        for number, policy, script in selected:
            print(f"{number:02d}\t{policy}\t{script}")
        return 0

    if not args.confirm_heavy:
        parser.error("run requires --confirm-heavy; use --dry-run to inspect the order")
    if os.name == "nt" or not Path("/kaggle/working").exists():
        parser.error(
            "the extracted notebook code is Kaggle-native and requires /kaggle/working"
        )

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("MPLBACKEND", "Agg")

    for number, policy, script in selected:
        code = run_stage(number, policy, script, args.log_dir, env)
        if code == 0:
            continue
        if policy == "repairable":
            print(
                f"Stage {number:02d} returned {code}; continuing to its locked repair."
            )
            continue
        print(f"STOP: stage {number:02d} returned {code}.", file=sys.stderr)
        return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
