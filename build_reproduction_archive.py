#!/usr/bin/env python3
"""Build a new reproduction archive while preserving the legacy TAR."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent

REQUIRED_PROJECT_FILES = [
    "derived/final_three_region_confirmation/state_correction_context.csv",
    "derived/final_three_region_confirmation/state_feature_confirmation.csv",
    "derived/final_three_region_confirmation/cross_region_meta_analysis.csv",
    "derived/final_three_region_confirmation/final_feature_decision.csv",
    "derived/cell_jaccard_threshold_free_validation/stratified_threshold_free_inference.csv",
    "derived/cell_jaccard_species_cluster_robustness/species_cluster_bootstrap_summary.csv",
    "derived/cell_jaccard_species_cluster_robustness/cluster_robust_models.csv",
    "derived/full_orthogonal_phase_diagram_plan/full_orthogonal_phase_confirmation_plan.json",
    "derived/full_orthogonal_support_phase_confirmation/full_phase_confirmation_summary.csv",
    "derived/full_orthogonal_support_phase_confirmation/scenario_label_permutation_summary.csv",
    "derived/full_orthogonal_support_phase_confirmation/species_region_cluster_bootstrap_summary.csv",
    "derived/full_orthogonal_support_phase_confirmation/negative_control_summary.csv",
    "derived/full_orthogonal_support_phase_confirmation/marginal_effort_overlap_curve.csv",
    "derived/full_orthogonal_support_phase_confirmation/marginal_species_overlap_curve.csv",
]

LARGE_PROJECT_FILES = [
    "derived/full_orthogonal_support_phase_confirmation/full_core_phase_replicates.parquet",
    "derived/full_orthogonal_support_phase_confirmation/full_marginal_phase_replicates.parquet",
    "derived/full_orthogonal_support_phase_confirmation/scenario_label_permutation_nulls.parquet",
    "derived/full_orthogonal_support_phase_confirmation/species_region_cluster_bootstrap_values.parquet",
]

PA_FILES = [
    "PA_exact_matched_plot_count_summary.csv",
    "PA_same_window_matched_live_tree_physical_plots.parquet",
]


def unique_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_required(source: Path, relative: str, destination: Path) -> None:
    src = source / relative
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = destination / "analysis_outputs" / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-large-outputs", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    legacy = (ROOT.parent / "20260711q1test.tar").resolve()
    if output == legacy:
        parser.error("refusing to overwrite the legacy 20260711q1test.tar")
    if output.exists():
        parser.error(f"output already exists: {output}")

    with tempfile.TemporaryDirectory(prefix="reproduction_archive_") as tmp:
        stage = Path(tmp) / "manuscript_reproduction"
        stage.mkdir(parents=True)

        code_dest = stage / "code"
        shutil.copytree(
            ROOT,
            code_dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "SHA256SUMS.txt"),
        )

        for relative in REQUIRED_PROJECT_FILES:
            copy_required(args.project_root, relative, stage)
        if args.include_large_outputs:
            for relative in LARGE_PROJECT_FILES:
                copy_required(args.project_root, relative, stage)

        repair_dest = stage / "submission_repair"
        repair_dest.mkdir(parents=True)
        for name in PA_FILES:
            shutil.copy2(unique_file(args.repair_root, name), repair_dest / name)

        manifest_path = stage / "MANIFEST_SHA256.csv"
        rows = []
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            if path == manifest_path:
                continue
            rows.append(
                {
                    "relative_path": path.relative_to(stage).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("relative_path", "size_bytes", "sha256")
            )
            writer.writeheader()
            writer.writerows(rows)

        output.parent.mkdir(parents=True, exist_ok=True)
        mode = "w:gz" if output.name.endswith((".tar.gz", ".tgz")) else "w"
        with tarfile.open(output, mode) as archive:
            archive.add(stage, arcname=stage.name)

    print(f"ARCHIVE: {output}")
    print(f"SHA256: {sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
