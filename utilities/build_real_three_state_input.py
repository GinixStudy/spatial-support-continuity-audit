"""Build a real three-state input table for spatial-support-audit.

This utility performs deterministic filtering, thinning, gridding, and counting
only. It does not rerun the paper's estimators, simulations, or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd


STATES = ("PA", "VA", "NC")
EARLY_YEARS = tuple(range(2013, 2019))
LATE_YEARS = tuple(range(2020, 2026))
THINNING_RESOLUTION = 0.05
PRIMARY_GRID_RESOLUTION = 0.50
MAX_COORDINATE_UNCERTAINTY_M = 10_000
MIN_TARGET_EFFORT_PER_CELL_PERIOD = 10
EXPECTED_ARCHIVE_SHA256 = (
    "e23945880a15679b057aa51172bf8828332da27fb87433775eaa6c444402716a"
)

STATE_BOUNDS = {
    "PA": {"lat_min": 39.5, "lat_max": 42.6, "lon_min": -81.0, "lon_max": -74.4},
    "VA": {"lat_min": 36.4, "lat_max": 39.7, "lon_min": -83.8, "lon_max": -75.0},
    "NC": {"lat_min": 33.7, "lat_max": 36.7, "lon_min": -84.5, "lon_max": -75.2},
}

PARQUET_MEMBER_TEMPLATE = (
    "./fia_temporal_observation_drift/derived/"
    "registered_gbif_three_region_download/normalized/"
    "{state}_registered_gbif.parquet"
)
REFERENCE_MEMBER = (
    "./fia_temporal_observation_drift/derived/"
    "three_region_cell_jaccard_synthesis/"
    "three_region_harmonized_species_table.csv"
)
STATE_PROCESSING_MEMBER = (
    "./fia_temporal_observation_drift/derived/"
    "registered_gbif_formal_parity/formal_state_processing_summary.csv"
)
GRID_SUMMARY_MEMBER = (
    "./fia_temporal_observation_drift/derived/"
    "registered_gbif_formal_parity/formal_grid_summary.csv"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_member_bytes(archive: tarfile.TarFile, member_name: str) -> bytes:
    try:
        member = archive.getmember(member_name)
    except KeyError as error:
        raise FileNotFoundError(f"archive member not found: {member_name}") from error
    extracted = archive.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"archive member is not a regular file: {member_name}")
    return extracted.read()


def build_event_day(frame: pd.DataFrame) -> pd.Series:
    event_time = pd.to_datetime(frame["eventDate"], errors="coerce", utc=True)
    fallback = pd.to_datetime(
        pd.DataFrame(
            {
                "year": pd.to_numeric(frame["year"], errors="coerce"),
                "month": pd.to_numeric(frame["month"], errors="coerce"),
                "day": pd.to_numeric(frame["day"], errors="coerce"),
            }
        ),
        errors="coerce",
        utc=True,
    )
    result = event_time.fillna(fallback).dt.strftime("%Y-%m-%d").astype("string")
    missing = result.isna()
    result.loc[missing] = "unknown_" + frame.loc[missing, "gbifID"].astype(str)
    return result


def prepare_formal_state(frame: pd.DataFrame, state: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    numeric_columns = (
        "fia_species_code",
        "year",
        "month",
        "day",
        "decimalLatitude",
        "decimalLongitude",
        "coordinateUncertaintyInMeters",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["gbifID"] = frame["gbifID"].astype("string")

    bounds = STATE_BOUNDS[state]
    valid = (
        frame["year"].isin(EARLY_YEARS + LATE_YEARS)
        & frame["fia_species_code"].notna()
        & frame["decimalLatitude"].between(bounds["lat_min"], bounds["lat_max"])
        & frame["decimalLongitude"].between(bounds["lon_min"], bounds["lon_max"])
        & (
            frame["coordinateUncertaintyInMeters"].isna()
            | frame["coordinateUncertaintyInMeters"].le(
                MAX_COORDINATE_UNCERTAINTY_M
            )
        )
    )
    clean = frame.loc[valid].copy()
    clean["fia_species_code"] = clean["fia_species_code"].astype(int)
    clean["year"] = clean["year"].astype(int)
    clean["event_day"] = build_event_day(clean)
    clean["thin_lat_index"] = np.floor(
        clean["decimalLatitude"] / THINNING_RESOLUTION
    ).astype("int32")
    clean["thin_lon_index"] = np.floor(
        clean["decimalLongitude"] / THINNING_RESOLUTION
    ).astype("int32")
    clean["uncertainty_sort"] = clean["coordinateUncertaintyInMeters"].fillna(
        np.inf
    )

    thinned = (
        clean.sort_values(
            [
                "fia_species_code",
                "event_day",
                "thin_lat_index",
                "thin_lon_index",
                "uncertainty_sort",
                "gbifID",
            ]
        )
        .drop_duplicates(
            [
                "fia_species_code",
                "event_day",
                "thin_lat_index",
                "thin_lon_index",
            ],
            keep="first",
        )
        .drop(columns=["uncertainty_sort"])
        .reset_index(drop=True)
    )
    thinned["period"] = np.where(
        thinned["year"].isin(EARLY_YEARS), "early", "late"
    )
    thinned["grid_lat_index"] = np.floor(
        thinned["decimalLatitude"] / PRIMARY_GRID_RESOLUTION
    ).astype("int32")
    thinned["grid_lon_index"] = np.floor(
        thinned["decimalLongitude"] / PRIMARY_GRID_RESOLUTION
    ).astype("int32")
    thinned["grid_id"] = (
        thinned["grid_lat_index"].astype(str)
        + "_"
        + thinned["grid_lon_index"].astype(str)
    )
    return clean, thinned


def set_jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return float(len(first & second) / len(union)) if union else float("nan")


def build_state_rows(
    state: str,
    frame: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    clean, thinned = prepare_formal_state(frame, state)
    effort = (
        thinned.groupby(["period", "grid_id"], as_index=False)
        .agg(effort_count=("gbifID", "size"))
        .sort_values(["period", "grid_id"])
        .reset_index(drop=True)
    )

    state_reference = (
        reference.loc[reference["region"] == state, ["fia_species_code", "scientific_name", "cell_jaccard"]]
        .copy()
        .sort_values("fia_species_code")
        .reset_index(drop=True)
    )
    state_reference["fia_species_code"] = state_reference["fia_species_code"].astype(int)

    rows: list[pd.DataFrame] = []
    for item in state_reference.itertuples(index=False):
        species_counts = (
            thinned.loc[thinned["fia_species_code"] == item.fia_species_code]
            .groupby(["period", "grid_id"])
            .size()
            .rename("focal_count")
            .reset_index()
        )
        species_rows = effort.merge(
            species_counts, on=["period", "grid_id"], how="left"
        )
        species_rows["focal_count"] = species_rows["focal_count"].fillna(0).astype(int)
        species_rows.insert(0, "scientific_name", item.scientific_name)
        species_rows.insert(0, "fia_species_code", int(item.fia_species_code))
        species_rows.insert(0, "state", state)
        species_rows.insert(
            0,
            "analysis_group",
            f"{state}|{int(item.fia_species_code)}|{item.scientific_name}",
        )
        rows.append(species_rows)

    output = pd.concat(rows, ignore_index=True)
    output["effort_count"] = output["effort_count"].astype(int)
    output = output.rename(columns={"grid_id": "cell_id"})

    early_effort = set(effort.loc[effort["period"] == "early", "grid_id"])
    late_effort = set(effort.loc[effort["period"] == "late", "grid_id"])
    effort_wide = effort.pivot(
        index="grid_id", columns="period", values="effort_count"
    ).fillna(0)
    for period in ("early", "late"):
        if period not in effort_wide.columns:
            effort_wide[period] = 0
    stable_count = int(
        (
            (effort_wide["early"] >= MIN_TARGET_EFFORT_PER_CELL_PERIOD)
            & (effort_wide["late"] >= MIN_TARGET_EFFORT_PER_CELL_PERIOD)
        ).sum()
    )
    summary = {
        "state": state,
        "registered_rows": int(len(frame)),
        "quality_filtered_rows": int(len(clean)),
        "thinned_rows": int(len(thinned)),
        "effort_taxa": int(thinned["fia_species_code"].nunique()),
        "evaluation_species": int(len(state_reference)),
        "effort_cell_period_rows": int(len(effort)),
        "input_rows": int(len(output)),
        "effort_early_cells": int(len(early_effort)),
        "effort_late_cells": int(len(late_effort)),
        "effort_union_cells": int(len(early_effort | late_effort)),
        "effort_jaccard": set_jaccard(early_effort, late_effort),
        "stable_cells_at_effort_10": stable_count,
    }
    return output, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-archive-hash", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive_path = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_hash = None
    if args.verify_archive_hash:
        archive_hash = file_sha256(archive_path)
        if archive_hash.lower() != EXPECTED_ARCHIVE_SHA256:
            raise ValueError(
                "archive SHA-256 does not match the locked final archive: "
                f"{archive_hash}"
            )

    parquet_members = {
        state: PARQUET_MEMBER_TEMPLATE.format(state=state) for state in STATES
    }
    with tarfile.open(archive_path, mode="r") as archive:
        reference = pd.read_csv(io.BytesIO(read_member_bytes(archive, REFERENCE_MEMBER)))
        locked_state_processing = pd.read_csv(
            io.BytesIO(read_member_bytes(archive, STATE_PROCESSING_MEMBER))
        )
        locked_grid_summary = pd.read_csv(
            io.BytesIO(read_member_bytes(archive, GRID_SUMMARY_MEMBER))
        )
        state_frames = {
            state: pd.read_parquet(
                io.BytesIO(read_member_bytes(archive, member)), engine="pyarrow"
            )
            for state, member in parquet_members.items()
        }

    all_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for state in STATES:
        state_rows, summary = build_state_rows(
            state, state_frames[state], reference
        )
        all_rows.append(state_rows)
        summaries.append(summary)

    combined = pd.concat(all_rows, ignore_index=True)
    expected_groups = int(reference[["region", "fia_species_code"]].drop_duplicates().shape[0])
    actual_groups = int(combined["analysis_group"].nunique())
    if expected_groups != 43 or actual_groups != expected_groups:
        raise ValueError(
            f"unexpected analysis-group count: expected {expected_groups}, got {actual_groups}"
        )

    summary_frame = pd.DataFrame(summaries)
    parity_rows: list[dict[str, object]] = []
    for state in STATES:
        actual = summary_frame.loc[summary_frame["state"] == state].iloc[0]
        locked_state = locked_state_processing.loc[
            locked_state_processing["state_code"] == state
        ].iloc[0]
        locked_grid = locked_grid_summary.loc[
            (locked_grid_summary["state_code"] == state)
            & np.isclose(locked_grid_summary["grid_resolution"], PRIMARY_GRID_RESOLUTION)
        ].iloc[0]
        checks = {
            "registered_rows": (actual["registered_rows"], locked_state["raw_records"]),
            "quality_filtered_rows": (
                actual["quality_filtered_rows"],
                locked_state["clean_records"],
            ),
            "thinned_rows": (actual["thinned_rows"], locked_state["thinned_records"]),
            "effort_taxa": (actual["effort_taxa"], locked_state["effort_taxa"]),
            "evaluation_species": (
                actual["evaluation_species"],
                locked_state["evaluation_species"],
            ),
            "effort_union_cells": (
                actual["effort_union_cells"],
                locked_grid["n_effort_cells"],
            ),
            "stable_cells_at_effort_10": (
                actual["stable_cells_at_effort_10"],
                locked_grid["n_stable_cells"],
            ),
        }
        for check, (observed, expected) in checks.items():
            parity_rows.append(
                {
                    "state": state,
                    "check": check,
                    "observed": int(observed),
                    "locked_expected": int(expected),
                    "passed": int(observed) == int(expected),
                }
            )
    parity_frame = pd.DataFrame(parity_rows)
    if not bool(parity_frame["passed"].all()):
        failed = parity_frame.loc[~parity_frame["passed"]]
        raise ValueError(
            "real-data conversion does not match locked processing summaries:\n"
            + failed.to_string(index=False)
        )

    input_path = output_dir / "three_state_support_counts.csv"
    reference_path = output_dir / "locked_reference_cell_jaccard.csv"
    summary_path = output_dir / "conversion_summary.csv"
    parity_path = output_dir / "locked_conversion_parity.csv"
    provenance_path = output_dir / "provenance.json"

    combined.to_csv(input_path, index=False)
    reference_export = reference.loc[
        :, ["region", "fia_species_code", "scientific_name", "cell_jaccard"]
    ].copy()
    reference_export.insert(
        0,
        "analysis_group",
        reference_export.apply(
            lambda row: (
                f"{row['region']}|{int(row['fia_species_code'])}|{row['scientific_name']}"
            ),
            axis=1,
        ),
    )
    reference_export = reference_export.rename(
        columns={"cell_jaccard": "expected_species_jaccard"}
    )
    reference_export.to_csv(reference_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    parity_frame.to_csv(parity_path, index=False)

    provenance = {
        "purpose": "deterministic real-data input for spatial-support-audit",
        "scientific_experiments_rerun": False,
        "archive": archive_path.name,
        "archive_expected_sha256": EXPECTED_ARCHIVE_SHA256,
        "archive_verified_sha256": archive_hash,
        "source_members": [
            REFERENCE_MEMBER,
            STATE_PROCESSING_MEMBER,
            GRID_SUMMARY_MEMBER,
            *parquet_members.values(),
        ],
        "locked_conversion_rules": {
            "early_years": list(EARLY_YEARS),
            "late_years": list(LATE_YEARS),
            "thinning_resolution_degrees": THINNING_RESOLUTION,
            "grid_resolution_degrees": PRIMARY_GRID_RESOLUTION,
            "maximum_coordinate_uncertainty_m": MAX_COORDINATE_UNCERTAINTY_M,
            "minimum_target_effort_per_cell_period": MIN_TARGET_EFFORT_PER_CELL_PERIOD,
        },
        "software": {
            "python": __import__("sys").version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
        "outputs": {
            input_path.name: {
                "rows": int(len(combined)),
                "groups": actual_groups,
                "sha256": file_sha256(input_path),
            },
            reference_path.name: {"rows": int(len(reference_export)), "sha256": file_sha256(reference_path)},
            summary_path.name: {"rows": int(len(summaries)), "sha256": file_sha256(summary_path)},
            parity_path.name: {
                "rows": int(len(parity_frame)),
                "passed": int(parity_frame["passed"].sum()),
                "sha256": file_sha256(parity_path),
            },
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(combined):,} rows for {actual_groups} state-species groups")
    print(f"Input: {input_path}")
    print(f"Reference: {reference_path}")
    print(f"Summary: {summary_path}")
    print(f"Locked conversion parity: {int(parity_frame['passed'].sum())}/{len(parity_frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
