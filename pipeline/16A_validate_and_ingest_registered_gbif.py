
# Cell 16A — Validate and ingest formal registered GBIF download
# Input: /kaggle/input/datasets/nanjide/20260711gbif
# DOI: 10.15468/dl.sm6ygu
# Expected records: 143,355
# No analysis is rerun in this cell.

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import zipfile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display

INPUT_DIR = Path("/kaggle/input/datasets/nanjide/20260711gbif")
BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
OUT_DIR = BASE_DIR / "derived" / "registered_gbif_three_region_download"
EXTRACT_DIR = OUT_DIR / "extracted"
NORMALIZED_DIR = OUT_DIR / "normalized"

DOWNLOAD_KEY = "0032735-260623161305970"
DOWNLOAD_DOI = "10.15468/dl.sm6ygu"
EXPECTED_RECORDS = 143_355
EXPECTED_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
VALID_YEARS = set(range(2013, 2019)) | set(range(2020, 2026))
STATE_MAP = {
    "Pennsylvania": "PA",
    "Virginia": "VA",
    "North Carolina": "NC",
}
CHUNK_SIZE = 40_000

PLAN_PATHS = {
    "PA": BASE_DIR / "derived" / "pa_expanded_tree_taxon_audit" / "selected_expanded_tree_taxa.csv",
    "VA": BASE_DIR / "derived" / "va_confirmation_dataset" / "va_locked_selected_taxa.csv",
    "NC": BASE_DIR / "derived" / "nc_final_confirmation_dataset" / "nc_locked_selected_taxa.csv",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
RUN_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

print("RUN_UTC:", RUN_UTC)
print("INPUT_DIR:", INPUT_DIR)
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")


def append_readme(text):
    path = BASE_DIR / "README.md"
    old = path.read_text(encoding="utf-8") if path.exists() else "# Temporal Observation Drift\n"
    path.write_text(old + text, encoding="utf-8")


def resolve_column(columns, candidates):
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def safe_extract(zip_path, output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.infolist()

        for member in members:
            target = (output_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP path: {member.filename}")

        for member in tqdm(
            members,
            desc="Extracting GBIF ZIP",
            unit="file",
        ):
            archive.extract(member, path=output_dir)


def find_occurrence_table():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory missing: {INPUT_DIR}")

    files = [path for path in INPUT_DIR.rglob("*") if path.is_file()]

    if not files:
        raise FileNotFoundError(f"No files found in {INPUT_DIR}")

    display(
        pd.DataFrame(
            {
                "path": [str(path) for path in files],
                "size_mb": [
                    round(path.stat().st_size / (1024 ** 2), 3)
                    for path in files
                ],
            }
        )
    )

    zip_files = [path for path in files if path.suffix.lower() == ".zip"]

    if zip_files:
        zip_files.sort(
            key=lambda path: (
                DOWNLOAD_KEY not in path.name,
                -path.stat().st_size,
            )
        )
        archive_path = zip_files[0]
        safe_extract(archive_path, EXTRACT_DIR)
        root = EXTRACT_DIR
    else:
        archive_path = None
        root = INPUT_DIR

    exact_names = ["occurrence.txt", "occurrence.tsv", "occurrence.csv"]

    for name in exact_names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0], archive_path

    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".txt", ".tsv", ".csv"}
        and path.name.lower() not in {"citations.txt", "rights.txt"}
    ]

    if not candidates:
        raise FileNotFoundError("No occurrence TSV/CSV file found.")

    return max(candidates, key=lambda path: path.stat().st_size), archive_path


def detect_separator(path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    return "\t" if header.count("\t") >= header.count(",") else ","


def load_plan(state_code, path):
    if not path.exists():
        raise FileNotFoundError(f"Missing locked plan: {path}")

    frame = pd.read_csv(path)

    key_col = resolve_column(
        frame.columns,
        ["gbif_taxon_key", "taxonKey", "taxon_key"],
    )
    code_col = resolve_column(
        frame.columns,
        ["fia_species_code", "species_code"],
    )
    name_col = resolve_column(
        frame.columns,
        ["gbif_accepted_name", "scientific_name", "fia_canonical_name"],
    )

    if key_col is None:
        raise KeyError(f"{state_code}: no taxon-key column in {path}")

    keys = pd.to_numeric(frame[key_col], errors="coerce")

    result = pd.DataFrame(
        {
            "state_code": state_code,
            "locked_taxon_key": keys,
            "fia_species_code": (
                pd.to_numeric(frame[code_col], errors="coerce")
                if code_col
                else np.nan
            ),
            "locked_name": (
                frame[name_col].astype(str)
                if name_col
                else pd.NA
            ),
        }
    ).dropna(subset=["locked_taxon_key"])

    result["locked_taxon_key"] = result["locked_taxon_key"].astype(int)

    return result.drop_duplicates(
        ["state_code", "locked_taxon_key"]
    )


def failure(status, error, next_step):
    append_readme(
        f"""

## Registered GBIF ingest failure — {RUN_UTC}
- Status: **{status}**
- Error: `{error}`
- Next step: {next_step}
"""
    )
    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"ERROR: {error}")
    print(f"OUT_DIR: {OUT_DIR}")
    print("PASSED: False")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")


try:
    if not BASE_DIR.exists():
        raise FileNotFoundError(
            f"Restore the project first: {BASE_DIR}"
        )

    occurrence_path, archive_path = find_occurrence_table()
    separator = detect_separator(occurrence_path)

    print("SOURCE_ARCHIVE:", archive_path)
    print("OCCURRENCE_TABLE:", occurrence_path)
    print("SEPARATOR:", repr(separator))

    header = pd.read_csv(
        occurrence_path,
        sep=separator,
        nrows=0,
        low_memory=False,
    )

    columns = header.columns.tolist()

    column_map = {
        "gbifID": resolve_column(columns, ["gbifID", "key"]),
        "datasetKey": resolve_column(columns, ["datasetKey"]),
        "occurrenceID": resolve_column(columns, ["occurrenceID"]),
        "stateProvince": resolve_column(columns, ["stateProvince"]),
        "year": resolve_column(columns, ["year"]),
        "month": resolve_column(columns, ["month"]),
        "day": resolve_column(columns, ["day"]),
        "eventDate": resolve_column(columns, ["eventDate"]),
        "decimalLatitude": resolve_column(columns, ["decimalLatitude"]),
        "decimalLongitude": resolve_column(columns, ["decimalLongitude"]),
        "coordinateUncertaintyInMeters": resolve_column(
            columns,
            ["coordinateUncertaintyInMeters"],
        ),
        "recordedBy": resolve_column(columns, ["recordedBy"]),
        "basisOfRecord": resolve_column(columns, ["basisOfRecord"]),
        "occurrenceStatus": resolve_column(columns, ["occurrenceStatus"]),
        "taxonKey": resolve_column(columns, ["taxonKey"]),
        "speciesKey": resolve_column(columns, ["speciesKey"]),
        "acceptedTaxonKey": resolve_column(columns, ["acceptedTaxonKey"]),
        "scientificName": resolve_column(columns, ["scientificName"]),
        "acceptedScientificName": resolve_column(
            columns,
            ["acceptedScientificName"],
        ),
        "license": resolve_column(columns, ["license"]),
        "issues": resolve_column(columns, ["issues", "issue"]),
    }

    required = [
        "gbifID",
        "datasetKey",
        "stateProvince",
        "year",
        "decimalLatitude",
        "decimalLongitude",
        "basisOfRecord",
        "occurrenceStatus",
        "taxonKey",
    ]

    missing_required = [
        field for field in required if column_map[field] is None
    ]

    if missing_required:
        raise KeyError(
            "Missing required columns: "
            + " | ".join(missing_required)
        )

    usecols = list(
        dict.fromkeys(
            column for column in column_map.values() if column is not None
        )
    )

    parts = []

    reader = pd.read_csv(
        occurrence_path,
        sep=separator,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding_errors="replace",
        dtype={
            column_map["gbifID"]: "string",
            column_map["datasetKey"]: "string",
            column_map["stateProvince"]: "string",
        },
    )

    progress = tqdm(
        total=EXPECTED_RECORDS,
        desc="Reading registered GBIF records",
        unit="record",
    )

    for chunk in reader:
        chunk = chunk.rename(
            columns={
                source: target
                for target, source in column_map.items()
                if source is not None
            }
        )

        for column in [
            "year",
            "month",
            "day",
            "decimalLatitude",
            "decimalLongitude",
            "coordinateUncertaintyInMeters",
            "taxonKey",
            "speciesKey",
            "acceptedTaxonKey",
        ]:
            if column in chunk.columns:
                chunk[column] = pd.to_numeric(
                    chunk[column],
                    errors="coerce",
                )

        chunk["state_code"] = chunk["stateProvince"].map(STATE_MAP)
        parts.append(chunk)
        progress.update(len(chunk))

    progress.close()

    registered_df = pd.concat(parts, ignore_index=True)
    del parts

    actual_records = len(registered_df)
    duplicate_ids = int(registered_df["gbifID"].duplicated().sum())
    missing_coordinates = int(
        registered_df[
            ["decimalLatitude", "decimalLongitude"]
        ].isna().any(axis=1).sum()
    )
    invalid_year_rows = int(
        (~registered_df["year"].isin(VALID_YEARS)).sum()
    )
    unexpected_state_rows = int(
        registered_df["state_code"].isna().sum()
    )
    unexpected_dataset_rows = int(
        (
            registered_df["datasetKey"].astype(str)
            != EXPECTED_DATASET_KEY
        ).sum()
    )

    locked_plan_df = pd.concat(
        [
            load_plan(state_code, path)
            for state_code, path in PLAN_PATHS.items()
        ],
        ignore_index=True,
    )

    locked_plan_df.to_csv(
        OUT_DIR / "locked_three_region_taxon_plan.csv",
        index=False,
    )

    normalized_parts = []

    for state_code in tqdm(
        ["PA", "VA", "NC"],
        desc="Mapping formal records to locked taxa",
        unit="state",
    ):
        state_df = registered_df[
            registered_df["state_code"] == state_code
        ].copy()

        state_plan = locked_plan_df[
            locked_plan_df["state_code"] == state_code
        ].copy()

        code_map = state_plan.set_index(
            "locked_taxon_key"
        )["fia_species_code"].to_dict()

        name_map = state_plan.set_index(
            "locked_taxon_key"
        )["locked_name"].to_dict()

        matched_key = pd.Series(
            pd.NA,
            index=state_df.index,
            dtype="Int64",
        )
        mapped_code = pd.Series(
            np.nan,
            index=state_df.index,
            dtype=float,
        )

        for candidate in [
            "acceptedTaxonKey",
            "speciesKey",
            "taxonKey",
        ]:
            if candidate not in state_df.columns:
                continue

            keys = pd.to_numeric(
                state_df[candidate],
                errors="coerce",
            ).astype("Int64")

            candidate_codes = keys.map(code_map)

            fill = mapped_code.isna() & candidate_codes.notna()

            mapped_code.loc[fill] = candidate_codes.loc[fill]
            matched_key.loc[fill] = keys.loc[fill]

        state_df["matched_locked_taxon_key"] = matched_key
        state_df["fia_species_code"] = mapped_code
        state_df["locked_scientific_name"] = matched_key.map(name_map)

        normalized_parts.append(state_df)

    normalized_df = pd.concat(
        normalized_parts,
        ignore_index=True,
    )
    del normalized_parts

    unmatched_rows = int(
        normalized_df["fia_species_code"].isna().sum()
    )

    audit_rows = []

    for state_code in ["PA", "VA", "NC"]:
        state_df = normalized_df[
            normalized_df["state_code"] == state_code
        ]
        plan_df = locked_plan_df[
            locked_plan_df["state_code"] == state_code
        ]

        expected_keys = set(
            plan_df["locked_taxon_key"].astype(int)
        )
        observed_keys = set(
            pd.to_numeric(
                state_df["matched_locked_taxon_key"],
                errors="coerce",
            ).dropna().astype(int)
        )

        missing_keys = sorted(expected_keys - observed_keys)

        audit_rows.append(
            {
                "state_code": state_code,
                "n_records": len(state_df),
                "expected_locked_taxa": len(expected_keys),
                "observed_locked_taxa": len(observed_keys),
                "missing_locked_taxa": len(missing_keys),
                "missing_locked_taxon_keys": json.dumps(missing_keys),
                "unmatched_rows": int(
                    state_df["fia_species_code"].isna().sum()
                ),
                "n_years": int(state_df["year"].nunique()),
            }
        )

    audit_df = pd.DataFrame(audit_rows)

    print("\nREGISTERED DOWNLOAD STATE AUDIT")
    display(audit_df)

    all_path = (
        NORMALIZED_DIR
        / "registered_gbif_all_states.parquet"
    )
    normalized_df.to_parquet(
        all_path,
        index=False,
    )

    state_paths = {}

    for state_code in tqdm(
        ["PA", "VA", "NC"],
        desc="Saving formal per-state Parquet",
        unit="state",
    ):
        path = (
            NORMALIZED_DIR
            / f"{state_code}_registered_gbif.parquet"
        )
        normalized_df[
            normalized_df["state_code"] == state_code
        ].to_parquet(
            path,
            index=False,
        )
        state_paths[state_code] = str(path)

    audit_df.to_csv(
        OUT_DIR / "registered_download_state_audit.csv",
        index=False,
    )

    record_count_pass = actual_records == EXPECTED_RECORDS
    duplicate_pass = duplicate_ids == 0
    coordinate_pass = missing_coordinates == 0
    year_pass = invalid_year_rows == 0
    state_pass = (
        unexpected_state_rows == 0
        and set(normalized_df["state_code"].dropna().unique())
        == {"PA", "VA", "NC"}
    )
    dataset_pass = unexpected_dataset_rows == 0
    taxon_pass = (
        unmatched_rows == 0
        and audit_df["missing_locked_taxa"].sum() == 0
    )

    passed = all(
        [
            record_count_pass,
            duplicate_pass,
            coordinate_pass,
            year_pass,
            state_pass,
            dataset_pass,
            taxon_pass,
        ]
    )

    if passed:
        status = "REGISTERED_GBIF_DOWNLOAD_VALIDATED"
        next_step = (
            "Rerun the locked PA, VA and NC analyses using the formal "
            "per-state Parquet files and compare them with temporary API results."
        )
    elif all(
        [
            record_count_pass,
            coordinate_pass,
            year_pass,
            state_pass,
            dataset_pass,
        ]
    ):
        status = "REGISTERED_GBIF_VALIDATED_WITH_TAXON_WARNINGS"
        next_step = (
            "Inspect unmatched or missing taxon keys before replacing "
            "temporary API data."
        )
    else:
        status = "REGISTERED_GBIF_DOWNLOAD_INTEGRITY_FAILED"
        next_step = (
            "Do not replace temporary API data until failed checks are resolved."
        )

    manifest = {
        "run_utc": RUN_UTC,
        "download_key": DOWNLOAD_KEY,
        "download_doi": DOWNLOAD_DOI,
        "source_archive": str(archive_path) if archive_path else None,
        "occurrence_table": str(occurrence_path),
        "expected_records": EXPECTED_RECORDS,
        "actual_records": actual_records,
        "duplicate_ids": duplicate_ids,
        "missing_coordinates": missing_coordinates,
        "invalid_year_rows": invalid_year_rows,
        "unexpected_state_rows": unexpected_state_rows,
        "unexpected_dataset_rows": unexpected_dataset_rows,
        "unmatched_rows": unmatched_rows,
        "all_state_parquet": str(all_path),
        "per_state_parquets": state_paths,
        "status": status,
        "passed": passed,
    }

    (OUT_DIR / "registered_download_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    append_readme(
        f"""

## Formal registered GBIF download validation — {RUN_UTC}
- DOI: {DOWNLOAD_DOI}
- Download key: {DOWNLOAD_KEY}
- Expected records: {EXPECTED_RECORDS:,}
- Actual records: {actual_records:,}
- Duplicate GBIF IDs: {duplicate_ids:,}
- Missing coordinates: {missing_coordinates:,}
- Invalid-year rows: {invalid_year_rows:,}
- Unexpected-state rows: {unexpected_state_rows:,}
- Unexpected-dataset rows: {unexpected_dataset_rows:,}
- Unmatched locked-taxon rows: {unmatched_rows:,}
- Record count passed: {record_count_pass}
- State passed: {state_pass}
- Year passed: {year_pass}
- Dataset passed: {dataset_pass}
- Coordinates passed: {coordinate_pass}
- Duplicate check passed: {duplicate_pass}
- Taxon mapping passed: {taxon_pass}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}
"""
    )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"DOWNLOAD_KEY: {DOWNLOAD_KEY}")
    print(f"DOWNLOAD_DOI: {DOWNLOAD_DOI}")
    print(f"SOURCE_ARCHIVE: {archive_path}")
    print(f"OCCURRENCE_TABLE: {occurrence_path}")
    print(f"EXPECTED_RECORDS: {EXPECTED_RECORDS:,}")
    print(f"ACTUAL_RECORDS: {actual_records:,}")
    print(f"DUPLICATE_GBIF_IDS: {duplicate_ids:,}")
    print(f"MISSING_COORDINATES: {missing_coordinates:,}")
    print(f"INVALID_YEAR_ROWS: {invalid_year_rows:,}")
    print(f"UNEXPECTED_STATE_ROWS: {unexpected_state_rows:,}")
    print(f"UNEXPECTED_DATASET_ROWS: {unexpected_dataset_rows:,}")
    print(f"UNMATCHED_LOCKED_TAXON_ROWS: {unmatched_rows:,}")
    print(
        "STATE_AUDIT: "
        + json.dumps(
            audit_df.to_dict("records"),
            ensure_ascii=False,
        )
    )
    print(f"ALL_STATE_PARQUET: {all_path}")
    print(
        "PER_STATE_PARQUETS: "
        + json.dumps(
            state_paths,
            ensure_ascii=False,
        )
    )
    print(f"RECORD_COUNT_PASSED: {record_count_pass}")
    print(f"STATE_PASSED: {state_pass}")
    print(f"YEAR_PASSED: {year_pass}")
    print(f"DATASET_PASSED: {dataset_pass}")
    print(f"COORDINATE_PASSED: {coordinate_pass}")
    print(f"DUPLICATE_PASSED: {duplicate_pass}")
    print(f"TAXON_MAPPING_PASSED: {taxon_pass}")
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except Exception as exc:
    failure(
        "CELL16A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the displayed input-file table. "
            "Do not rerun PA/VA/NC analyses."
        ),
    )
