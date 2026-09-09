
# Cell 16B — Build the final North Carolina confirmation dataset
#
# Purpose
# -------
# Prepare the final independent dataset for confirming the two locked
# Pennsylvania transportability candidates:
#
#   1. cell_jaccard
#   2. abs_log_observer_growth
#
# This cell DOES NOT test either candidate.
#
# Locked windows
# --------------
# Early: 2013–2018
# Late : 2020–2025
#
# Locked processing
# -----------------
# - iNaturalist Research-grade observations through GBIF
# - coordinate uncertainty <=10 km or missing
# - species × event-day × 0.05-degree thinning
# - 0.5-degree effort grid
# - stable effort cell: >=10 target-frame records in each period
# - exact repeated FIA physical plots
# - species presence per physical plot and period
#
# Success
# -------
# - >=12 final benchmark species
# - >=500 repeated FIA physical plots
# - >=10 stable effort cells
# - >=20 effort-frame taxa retained
#
# Stop rule
# ---------
# The next cell is the final confirmation. If neither locked candidate confirms
# in North Carolina, the sensitivity-index route is permanently closed.

from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 220)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 340)

# ============================================================
# Locked configuration
# ============================================================
STATE = "NC"
STATE_NAME = "North Carolina"

EARLY_YEARS = list(range(2013, 2019))
LATE_YEARS = list(range(2020, 2026))
EARLY_MID_YEAR = float(np.mean(EARLY_YEARS))
LATE_MID_YEAR = float(np.mean(LATE_YEARS))

NC_LAT_MIN = 33.7
NC_LAT_MAX = 36.7
NC_LON_MIN = -84.4
NC_LON_MAX = -75.2

INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
INAT_DATASET_DOI = "10.15468/ab3s5x"
GBIF_API = "https://api.gbif.org/v1"

FIA_TREE_URL = (
    "https://apps.fs.usda.gov/fia/datamart/CSV/NC_TREE.csv"
)

PAGE_SIZE = 300
REQUEST_PAUSE_SECONDS = 0.03

MAX_COORDINATE_UNCERTAINTY_M = 10_000
THINNING_RESOLUTION = 0.05
EFFORT_GRID_RESOLUTION = 0.50
MIN_TARGET_EFFORT_PER_CELL_PERIOD = 10

MIN_INAT_RECORDS_PER_SPECIES_PERIOD = 30
MIN_INAT_CELLS_PER_SPECIES_PERIOD = 3
MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD = 15

MIN_FINAL_BENCHMARK_SPECIES = 12
MIN_MATCHED_FIA_PHYSICAL_PLOTS = 500
MIN_STABLE_EFFORT_CELLS = 10
MIN_EFFORT_FRAME_TAXA = 20

TREE_CHUNK_SIZE = 250_000

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_DIR = (
    BASE_DIR / "derived" / "third_region_selection_audit"
)
PILOT_DIR = (
    BASE_DIR / "derived" / "gbif_inaturalist_real_pilot"
)
PA_AUDIT_DIR = (
    BASE_DIR / "derived" / "pa_expanded_tree_taxon_audit"
)

OUT_DIR = (
    BASE_DIR / "derived" / "nc_final_confirmation_dataset"
)
TASK_CACHE_DIR = OUT_DIR / "occurrence_task_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATE_COVERAGE_PATH = (
    SELECTION_DIR / "candidate_state_taxon_coverage.csv"
)
BENCHMARK_TAXONOMY_PATH = (
    PILOT_DIR / "gbif_taxonomy_resolution.csv"
)
EFFORT_TAXA_PATH = (
    PA_AUDIT_DIR / "selected_expanded_tree_taxa.csv"
)

NC_PLOT_CSV = RAW_DIR / "NC_PLOT.csv"
NC_TREE_CSV = RAW_DIR / "NC_TREE.csv"

RAW_OCCURRENCE_PATH = (
    OUT_DIR / "nc_raw_occurrences.parquet"
)
CLEAN_OCCURRENCE_PATH = (
    OUT_DIR / "nc_clean_occurrences.parquet"
)
THINNED_OCCURRENCE_PATH = (
    OUT_DIR / "nc_thinned_occurrences.parquet"
)
FIA_REFERENCE_PATH = (
    OUT_DIR / "nc_fia_same_window_reference.parquet"
)
MATCHED_PLOT_PATH = (
    OUT_DIR / "nc_matched_physical_plots.parquet"
)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)
ACCESS_DATE = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d"
)

print("RUN_UTC:", RUN_UTC)
print("STATE:", STATE)
print("EARLY_YEARS:", EARLY_YEARS)
print("LATE_YEARS:", LATE_YEARS)
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")


# ============================================================
# Helpers
# ============================================================
class CellStop(Exception):
    def __init__(self, status, message, next_step):
        super().__init__(message)
        self.status = status
        self.message = message
        self.next_step = next_step


def append_readme(text):
    readme_path = BASE_DIR / "README.md"
    existing = (
        readme_path.read_text(encoding="utf-8")
        if readme_path.exists()
        else "# Temporal Observation Drift — FIA Validation\n"
    )
    readme_path.write_text(
        existing + text,
        encoding="utf-8",
    )


def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
    )


def build_session():
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=16,
        pool_maxsize=16,
    )

    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "TemporalObservationDriftResearch/0.1 "
                "(North Carolina final confirmation preparation)"
            )
        }
    )
    return session


SESSION = build_session()


def api_get(endpoint, params):
    response = SESSION.get(
        f"{GBIF_API}{endpoint}",
        params=params,
        timeout=(30, 240),
    )
    response.raise_for_status()
    time.sleep(REQUEST_PAUSE_SECONDS)
    return response.json()


def download_with_progress(url, output_path):
    if output_path.exists() and output_path.stat().st_size > 0:
        return "reused"

    response = SESSION.get(
        url,
        stream=True,
        timeout=(30, 600),
    )
    response.raise_for_status()

    total_bytes = int(
        response.headers.get("content-length", 0)
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    with temporary_path.open("wb") as handle:
        with tqdm(
            total=total_bytes if total_bytes > 0 else None,
            desc=f"Downloading {output_path.name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))

    temporary_path.replace(output_path)
    return "downloaded"


def build_occurrence_params(
    taxon_key,
    start_year,
    end_year,
):
    return {
        "datasetKey": INAT_DATASET_KEY,
        "country": "US",
        "stateProvince": STATE_NAME,
        "taxonKey": int(taxon_key),
        "year": f"{int(start_year)},{int(end_year)}",
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "occurrenceStatus": "PRESENT",
        "basisOfRecord": "HUMAN_OBSERVATION",
    }


def stringify(value):
    if value is None:
        return None

    if isinstance(value, list):
        return "; ".join(
            str(item)
            for item in value
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value)


def extract_occurrence(
    record,
    taxon_row,
    period,
):
    return {
        "gbifID": record.get("key"),
        "occurrenceID": record.get("occurrenceID"),
        "query_state_code": STATE,
        "query_state_name": STATE_NAME,
        "fia_species_code": int(
            taxon_row["fia_species_code"]
        ),
        "gbif_query_taxon_key": int(
            taxon_row["gbif_taxon_key"]
        ),
        "gbif_accepted_name": taxon_row[
            "gbif_accepted_name"
        ],
        "is_benchmark_taxon": bool(
            taxon_row["is_benchmark_taxon"]
        ),
        "is_effort_frame_taxon": bool(
            taxon_row["is_effort_frame_taxon"]
        ),
        "query_period": period,
        "taxonKey": record.get("taxonKey"),
        "acceptedTaxonKey": record.get(
            "acceptedTaxonKey"
        ),
        "speciesKey": record.get("speciesKey"),
        "scientificName": record.get(
            "scientificName"
        ),
        "acceptedScientificName": record.get(
            "acceptedScientificName"
        ),
        "species": record.get("species"),
        "decimalLatitude": record.get(
            "decimalLatitude"
        ),
        "decimalLongitude": record.get(
            "decimalLongitude"
        ),
        "coordinateUncertaintyInMeters": record.get(
            "coordinateUncertaintyInMeters"
        ),
        "year": record.get("year"),
        "month": record.get("month"),
        "day": record.get("day"),
        "eventDate": record.get("eventDate"),
        "stateProvince": record.get(
            "stateProvince"
        ),
        "countryCode": record.get("countryCode"),
        "basisOfRecord": record.get(
            "basisOfRecord"
        ),
        "occurrenceStatus": record.get(
            "occurrenceStatus"
        ),
        "recordedBy": stringify(
            record.get("recordedBy")
        ),
        "identifiedBy": stringify(
            record.get("identifiedBy")
        ),
        "datasetKey": record.get("datasetKey"),
        "license": record.get("license"),
        "issues": stringify(
            record.get("issues")
        ),
        "lastInterpreted": record.get(
            "lastInterpreted"
        ),
    }


def event_day_from_frame(frame):
    event_time = pd.to_datetime(
        frame["eventDate"],
        errors="coerce",
        utc=True,
    )

    fallback = pd.to_datetime(
        pd.DataFrame(
            {
                "year": pd.to_numeric(
                    frame["year"],
                    errors="coerce",
                ),
                "month": pd.to_numeric(
                    frame["month"],
                    errors="coerce",
                ),
                "day": pd.to_numeric(
                    frame["day"],
                    errors="coerce",
                ),
            }
        ),
        errors="coerce",
        utc=True,
    )

    final_time = event_time.fillna(fallback)

    result = (
        final_time.dt.strftime("%Y-%m-%d")
        .astype("string")
    )

    missing = result.isna()

    result.loc[missing] = (
        "unknown_"
        + frame.loc[
            missing,
            "gbifID",
        ].astype(str)
    )

    return result


def resolve_column(columns, candidates):
    lookup = {
        str(column).upper(): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]

    return None


def normalize_integer_component(series):
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .astype("Int64")
        .astype("string")
    )


def build_physical_plot_id(
    frame,
    state_col,
    unit_col,
    county_col,
    plot_col,
):
    return (
        normalize_integer_component(
            frame[state_col]
        )
        + "_"
        + normalize_integer_component(
            frame[unit_col]
        )
        + "_"
        + normalize_integer_component(
            frame[county_col]
        )
        + "_"
        + normalize_integer_component(
            frame[plot_col]
        )
    )


def closest_measurement_per_plot(
    plot_df,
    years,
    midpoint,
):
    work = plot_df[
        plot_df["year"].isin(years)
    ].copy()

    work["distance_to_midpoint"] = (
        work["year"] - midpoint
    ).abs()

    return (
        work.sort_values(
            [
                "physical_plot_id",
                "distance_to_midpoint",
                "year",
                "plot_key",
            ]
        )
        .groupby(
            "physical_plot_id",
            as_index=False,
        )
        .first()
    )


def shift_km_decade(
    early_latitude,
    late_latitude,
    early_year,
    late_year,
):
    values = [
        early_latitude,
        late_latitude,
        early_year,
        late_year,
    ]

    if not all(
        np.isfinite(value)
        for value in values
    ):
        return np.nan

    interval = late_year - early_year

    if interval <= 0:
        return np.nan

    return float(
        (
            late_latitude
            - early_latitude
        )
        * 111.32
        * 10
        / interval
    )


def print_stop(status, error, next_step):
    append_readme(
        f"""

## NC final confirmation preparation stopped — {RUN_UTC}
- Status: **{status}**
- Reason: `{error}`
- Next step: {next_step}
"""
    )

    print(
        "\n========== RUN SUMMARY =========="
    )
    print(f"CELL_STATUS: {status}")
    print(f"ERROR: {error}")
    print(f"OUT_DIR: {OUT_DIR}")
    print("PASSED: False")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")


# ============================================================
# Main
# ============================================================
try:
    # --------------------------------------------------------
    # 1. Validate prior outputs and download NC TREE if needed
    # --------------------------------------------------------
    required_paths = [
        STATE_COVERAGE_PATH,
        BENCHMARK_TAXONOMY_PATH,
        EFFORT_TAXA_PATH,
        NC_PLOT_CSV,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing prior outputs: "
            + " | ".join(missing_paths)
        )

    tree_download_status = download_with_progress(
        FIA_TREE_URL,
        NC_TREE_CSV,
    )

    # --------------------------------------------------------
    # 2. Build the locked NC taxon plan
    # --------------------------------------------------------
    coverage_df = pd.read_csv(
        STATE_COVERAGE_PATH
    )

    for column in [
        "is_benchmark_taxon",
        "is_effort_frame_taxon",
        "window_coverage_pass",
    ]:
        coverage_df[column] = to_bool(
            coverage_df[column]
        )

    coverage_df["gbif_taxon_key"] = pd.to_numeric(
        coverage_df["gbif_taxon_key"],
        errors="coerce",
    )

    nc_coverage_df = coverage_df[
        coverage_df["state_code"].eq(STATE)
        & coverage_df["window_coverage_pass"]
        & (
            coverage_df["is_benchmark_taxon"]
            | coverage_df["is_effort_frame_taxon"]
        )
    ].dropna(
        subset=["gbif_taxon_key"]
    ).copy()

    nc_coverage_df["gbif_taxon_key"] = (
        nc_coverage_df["gbif_taxon_key"].astype(int)
    )

    benchmark_map_df = pd.read_csv(
        BENCHMARK_TAXONOMY_PATH
    )

    benchmark_map_df["gbif_taxon_key"] = pd.to_numeric(
        benchmark_map_df["gbif_taxon_key"],
        errors="coerce",
    )
    benchmark_map_df["fia_species_code"] = pd.to_numeric(
        benchmark_map_df["fia_species_code"],
        errors="coerce",
    )

    benchmark_map_df = benchmark_map_df.dropna(
        subset=[
            "gbif_taxon_key",
            "fia_species_code",
        ]
    ).copy()

    benchmark_map_df["gbif_taxon_key"] = (
        benchmark_map_df["gbif_taxon_key"].astype(int)
    )
    benchmark_map_df["fia_species_code"] = (
        benchmark_map_df["fia_species_code"].astype(int)
    )

    benchmark_map_df = (
        benchmark_map_df[
            [
                "gbif_taxon_key",
                "fia_species_code",
                "original_scientific_name",
                "gbif_accepted_name",
            ]
        ]
        .rename(
            columns={
                "original_scientific_name": "scientific_name",
            }
        )
        .drop_duplicates("gbif_taxon_key")
    )

    effort_map_df = pd.read_csv(
        EFFORT_TAXA_PATH
    )

    effort_map_df["gbif_taxon_key"] = pd.to_numeric(
        effort_map_df["gbif_taxon_key"],
        errors="coerce",
    )
    effort_map_df["species_code"] = pd.to_numeric(
        effort_map_df["species_code"],
        errors="coerce",
    )

    effort_map_df = effort_map_df.dropna(
        subset=[
            "gbif_taxon_key",
            "species_code",
        ]
    ).copy()

    effort_map_df["gbif_taxon_key"] = (
        effort_map_df["gbif_taxon_key"].astype(int)
    )
    effort_map_df["species_code"] = (
        effort_map_df["species_code"].astype(int)
    )

    effort_map_df = (
        effort_map_df[
            [
                "gbif_taxon_key",
                "species_code",
                "fia_canonical_name",
                "gbif_accepted_name",
            ]
        ]
        .rename(
            columns={
                "species_code": "fia_species_code",
                "fia_canonical_name": "scientific_name",
            }
        )
        .drop_duplicates("gbif_taxon_key")
    )

    mapping_df = pd.concat(
        [
            benchmark_map_df.assign(mapping_priority=0),
            effort_map_df.assign(mapping_priority=1),
        ],
        ignore_index=True,
    )

    conflict_df = (
        mapping_df.groupby("gbif_taxon_key")[
            "fia_species_code"
        ]
        .nunique()
    )

    conflicting_keys = conflict_df[
        conflict_df > 1
    ].index.tolist()

    if conflicting_keys:
        raise ValueError(
            "Conflicting FIA species codes for GBIF keys: "
            + " | ".join(map(str, conflicting_keys))
        )

    mapping_df = (
        mapping_df.sort_values("mapping_priority")
        .drop_duplicates(
            "gbif_taxon_key",
            keep="first",
        )
        .drop(columns=["mapping_priority"])
    )

    selected_taxa_df = nc_coverage_df.merge(
        mapping_df,
        on="gbif_taxon_key",
        how="left",
        suffixes=("_coverage", "_mapping"),
        validate="one_to_one",
    )

    selected_taxa_df["gbif_accepted_name"] = (
        selected_taxa_df[
            "gbif_accepted_name_mapping"
        ].fillna(
            selected_taxa_df[
                "gbif_accepted_name_coverage"
            ]
        )
    )

    selected_taxa_df = selected_taxa_df.dropna(
        subset=["fia_species_code"]
    ).copy()

    selected_taxa_df["fia_species_code"] = (
        selected_taxa_df["fia_species_code"].astype(int)
    )

    selected_taxa_df = (
        selected_taxa_df[
            [
                "gbif_taxon_key",
                "fia_species_code",
                "scientific_name",
                "gbif_accepted_name",
                "is_benchmark_taxon",
                "is_effort_frame_taxon",
                "early",
                "late",
                "minimum_window_count",
                "total_window_count",
            ]
        ]
        .drop_duplicates("gbif_taxon_key")
        .sort_values(
            [
                "is_benchmark_taxon",
                "is_effort_frame_taxon",
                "minimum_window_count",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    selected_taxa_df.to_csv(
        OUT_DIR / "nc_locked_selected_taxa.csv",
        index=False,
    )

    benchmark_taxa_df = selected_taxa_df[
        selected_taxa_df["is_benchmark_taxon"]
    ].copy()

    effort_taxa_df = selected_taxa_df[
        selected_taxa_df["is_effort_frame_taxon"]
    ].copy()

    if len(benchmark_taxa_df) < MIN_FINAL_BENCHMARK_SPECIES:
        raise CellStop(
            "NC_LOCKED_BENCHMARK_PLAN_TOO_SMALL",
            (
                f"Only {len(benchmark_taxa_df)} "
                "eligible benchmark taxa were mapped."
            ),
            "Review the locked taxon mapping before downloading coordinates.",
        )

    print("\nLOCKED NORTH CAROLINA TAXON PLAN")
    display(selected_taxa_df)

    # --------------------------------------------------------
    # 3. Download occurrence records with per-task caching
    # --------------------------------------------------------
    tasks = []

    for taxon_row in selected_taxa_df.to_dict("records"):
        for period, years, expected_column in [
            ("early", EARLY_YEARS, "early"),
            ("late", LATE_YEARS, "late"),
        ]:
            expected = int(taxon_row[expected_column])

            tasks.append(
                {
                    **taxon_row,
                    "period": period,
                    "start_year": min(years),
                    "end_year": max(years),
                    "expected": expected,
                    "expected_pages": int(
                        math.ceil(expected / PAGE_SIZE)
                    ),
                }
            )

    planned_records = int(
        sum(task["expected"] for task in tasks)
    )
    planned_pages = int(
        sum(task["expected_pages"] for task in tasks)
    )

    download_manifest_rows = []

    page_progress = tqdm(
        total=planned_pages,
        desc="Downloading/reusing NC GBIF pages",
        unit="page",
    )

    for task in tasks:
        cache_path = (
            TASK_CACHE_DIR
            / (
                f"taxon_{int(task['gbif_taxon_key'])}"
                f"_{task['period']}.parquet"
            )
        )

        if cache_path.exists():
            cached_df = pd.read_parquet(cache_path)
            retrieved = len(cached_df)
            page_progress.update(
                task["expected_pages"]
            )
            source_status = "reused"
            del cached_df
        else:
            rows = []
            offset = 0

            params = build_occurrence_params(
                taxon_key=task["gbif_taxon_key"],
                start_year=task["start_year"],
                end_year=task["end_year"],
            )

            pages_used = 0

            while offset < task["expected"]:
                request_params = dict(params)
                request_params["limit"] = min(
                    PAGE_SIZE,
                    task["expected"] - offset,
                )
                request_params["offset"] = offset

                payload = api_get(
                    "/occurrence/search",
                    request_params,
                )

                results = payload.get("results", [])
                pages_used += 1
                page_progress.update(1)

                if not results:
                    break

                rows.extend(
                    extract_occurrence(
                        record,
                        task,
                        task["period"],
                    )
                    for record in results
                )

                offset += len(results)

                if payload.get("endOfRecords", False):
                    break

            if pages_used < task["expected_pages"]:
                page_progress.update(
                    task["expected_pages"] - pages_used
                )

            task_df = pd.DataFrame(rows)
            task_df.to_parquet(
                cache_path,
                index=False,
            )

            retrieved = len(task_df)
            source_status = "downloaded"

            del task_df
            del rows
            gc.collect()

        download_manifest_rows.append(
            {
                "gbif_taxon_key": int(
                    task["gbif_taxon_key"]
                ),
                "fia_species_code": int(
                    task["fia_species_code"]
                ),
                "period": task["period"],
                "expected": int(task["expected"]),
                "retrieved": int(retrieved),
                "cache_status": source_status,
                "cache_path": str(cache_path),
            }
        )

    page_progress.close()

    download_manifest_df = pd.DataFrame(
        download_manifest_rows
    )

    download_manifest_df.to_csv(
        OUT_DIR / "nc_download_manifest.csv",
        index=False,
    )

    task_paths = [
        Path(path)
        for path in download_manifest_df["cache_path"]
    ]

    raw_parts = []

    for task_path in tqdm(
        task_paths,
        desc="Combining NC occurrence task caches",
        unit="task",
    ):
        raw_parts.append(
            pd.read_parquet(task_path)
        )

    raw_df = pd.concat(
        raw_parts,
        ignore_index=True,
    )

    del raw_parts
    gc.collect()

    raw_df.to_parquet(
        RAW_OCCURRENCE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 4. Apply identical quality filters and thinning
    # --------------------------------------------------------
    for column in [
        "gbifID",
        "fia_species_code",
        "gbif_query_taxon_key",
        "decimalLatitude",
        "decimalLongitude",
        "coordinateUncertaintyInMeters",
        "year",
        "month",
        "day",
    ]:
        if column == "gbifID":
            raw_df[column] = raw_df[column].astype(
                "string"
            )
        else:
            raw_df[column] = pd.to_numeric(
                raw_df[column],
                errors="coerce",
            )

    raw_rows_before_dedup = len(raw_df)

    raw_df = (
        raw_df.sort_values("gbifID")
        .drop_duplicates("gbifID")
        .reset_index(drop=True)
    )

    duplicate_gbif_ids_removed = (
        raw_rows_before_dedup
        - len(raw_df)
    )

    valid = (
        raw_df["year"].isin(
            EARLY_YEARS + LATE_YEARS
        )
        & raw_df["decimalLatitude"].between(
            NC_LAT_MIN,
            NC_LAT_MAX,
        )
        & raw_df["decimalLongitude"].between(
            NC_LON_MIN,
            NC_LON_MAX,
        )
        & (
            raw_df[
                "coordinateUncertaintyInMeters"
            ].isna()
            | raw_df[
                "coordinateUncertaintyInMeters"
            ].le(
                MAX_COORDINATE_UNCERTAINTY_M
            )
        )
        & ~(
            raw_df["decimalLatitude"].eq(0)
            & raw_df["decimalLongitude"].eq(0)
        )
    )

    clean_df = raw_df.loc[valid].copy()

    clean_df["fia_species_code"] = (
        clean_df["fia_species_code"].astype(int)
    )
    clean_df["year"] = (
        clean_df["year"].astype(int)
    )

    clean_df["event_day"] = event_day_from_frame(
        clean_df
    )

    clean_df["thin_lat_index"] = np.floor(
        clean_df["decimalLatitude"]
        / THINNING_RESOLUTION
    ).astype("int32")

    clean_df["thin_lon_index"] = np.floor(
        clean_df["decimalLongitude"]
        / THINNING_RESOLUTION
    ).astype("int32")

    clean_df["uncertainty_sort"] = (
        clean_df[
            "coordinateUncertaintyInMeters"
        ].fillna(np.inf)
    )

    clean_df.to_parquet(
        CLEAN_OCCURRENCE_PATH,
        index=False,
    )

    thinned_df = (
        clean_df.sort_values(
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

    thinned_df["period"] = np.where(
        thinned_df["year"].isin(EARLY_YEARS),
        "early",
        "late",
    )

    thinned_df.to_parquet(
        THINNED_OCCURRENCE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 5. Audit iNaturalist benchmark and effort-frame coverage
    # --------------------------------------------------------
    benchmark_codes = set(
        benchmark_taxa_df["fia_species_code"]
        .astype(int)
        .tolist()
    )

    effort_codes = set(
        effort_taxa_df["fia_species_code"]
        .astype(int)
        .tolist()
    )

    benchmark_occurrence_df = thinned_df[
        thinned_df["fia_species_code"].isin(
            benchmark_codes
        )
    ].copy()

    effort_occurrence_df = thinned_df[
        thinned_df["fia_species_code"].isin(
            effort_codes
        )
    ].copy()

    for frame in [
        benchmark_occurrence_df,
        effort_occurrence_df,
    ]:
        frame["effort_grid_lat_index"] = np.floor(
            frame["decimalLatitude"]
            / EFFORT_GRID_RESOLUTION
        ).astype("int32")

        frame["effort_grid_lon_index"] = np.floor(
            frame["decimalLongitude"]
            / EFFORT_GRID_RESOLUTION
        ).astype("int32")

        frame["effort_grid_id"] = (
            frame["effort_grid_lat_index"].astype(str)
            + "_"
            + frame["effort_grid_lon_index"].astype(str)
        )

    benchmark_coverage_df = (
        benchmark_occurrence_df.groupby(
            [
                "fia_species_code",
                "period",
            ],
            as_index=False,
        )
        .agg(
            inat_records=("gbifID", "size"),
            inat_cells=("effort_grid_id", "nunique"),
        )
    )

    benchmark_coverage_wide_df = (
        benchmark_coverage_df.pivot(
            index="fia_species_code",
            columns="period",
            values=[
                "inat_records",
                "inat_cells",
            ],
        )
    )

    benchmark_coverage_wide_df.columns = [
        f"{metric}_{period}"
        for metric, period
        in benchmark_coverage_wide_df.columns
    ]

    benchmark_coverage_wide_df = (
        benchmark_coverage_wide_df.reset_index()
    )

    effort_cell_df = (
        effort_occurrence_df.groupby(
            [
                "period",
                "effort_grid_id",
            ],
            as_index=False,
        )
        .agg(
            target_effort_records=("gbifID", "size"),
        )
    )

    effort_cell_wide_df = (
        effort_cell_df.pivot(
            index="effort_grid_id",
            columns="period",
            values="target_effort_records",
        )
        .fillna(0)
    )

    for period in ["early", "late"]:
        if period not in effort_cell_wide_df.columns:
            effort_cell_wide_df[period] = 0

    stable_effort_cells = set(
        effort_cell_wide_df[
            (
                effort_cell_wide_df["early"]
                >= MIN_TARGET_EFFORT_PER_CELL_PERIOD
            )
            & (
                effort_cell_wide_df["late"]
                >= MIN_TARGET_EFFORT_PER_CELL_PERIOD
            )
        ].index
    )

    effort_taxa_retained = int(
        effort_occurrence_df[
            "fia_species_code"
        ].nunique()
    )

    effort_period_summary_df = (
        effort_occurrence_df.groupby(
            "period",
            as_index=False,
        )
        .agg(
            target_records=("gbifID", "size"),
            target_taxa=(
                "fia_species_code",
                "nunique",
            ),
            target_cells=(
                "effort_grid_id",
                "nunique",
            ),
        )
    )

    effort_period_summary_df.to_csv(
        OUT_DIR / "nc_effort_frame_period_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Build exact same-window repeated FIA reference
    # --------------------------------------------------------
    plot_columns = pd.read_csv(
        NC_PLOT_CSV,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    plot_key_col = resolve_column(
        plot_columns,
        ["CN", "PLOT_CN"],
    )
    year_col = resolve_column(
        plot_columns,
        ["INVYR", "INVENTORY_YEAR"],
    )
    latitude_col = resolve_column(
        plot_columns,
        ["LAT", "LATITUDE"],
    )
    longitude_col = resolve_column(
        plot_columns,
        ["LON", "LONGITUDE"],
    )
    plot_status_col = resolve_column(
        plot_columns,
        ["PLOT_STATUS_CD", "PLOT_STATUS"],
    )
    state_code_col = resolve_column(
        plot_columns,
        ["STATECD", "STATE_CODE"],
    )
    unit_code_col = resolve_column(
        plot_columns,
        ["UNITCD", "UNIT_CODE"],
    )
    county_code_col = resolve_column(
        plot_columns,
        ["COUNTYCD", "COUNTY_CODE"],
    )
    plot_number_col = resolve_column(
        plot_columns,
        ["PLOT", "PLOT_NUMBER", "PLOT_NBR"],
    )

    required_plot_fields = [
        plot_key_col,
        year_col,
        latitude_col,
        longitude_col,
        state_code_col,
        unit_code_col,
        county_code_col,
        plot_number_col,
    ]

    if not all(required_plot_fields):
        raise ValueError(
            "Required NC PLOT identity fields were not resolved."
        )

    plot_usecols = list(
        dict.fromkeys(
            [
                plot_key_col,
                year_col,
                latitude_col,
                longitude_col,
                state_code_col,
                unit_code_col,
                county_code_col,
                plot_number_col,
            ]
            + (
                [plot_status_col]
                if plot_status_col is not None
                else []
            )
        )
    )

    plot_df = pd.read_csv(
        NC_PLOT_CSV,
        usecols=plot_usecols,
        low_memory=False,
        encoding_errors="replace",
        dtype={plot_key_col: "string"},
    )

    plot_df["physical_plot_id"] = (
        build_physical_plot_id(
            plot_df,
            state_code_col,
            unit_code_col,
            county_code_col,
            plot_number_col,
        )
    )

    rename_map = {
        plot_key_col: "plot_key",
        year_col: "year",
        latitude_col: "latitude",
        longitude_col: "longitude",
    }

    if plot_status_col is not None:
        rename_map[plot_status_col] = "plot_status"

    plot_df = plot_df.rename(
        columns=rename_map
    )

    if "plot_status" not in plot_df.columns:
        plot_df["plot_status"] = np.nan

    plot_df["plot_key"] = (
        plot_df["plot_key"]
        .astype("string")
        .str.strip()
    )

    for column in [
        "year",
        "latitude",
        "longitude",
        "plot_status",
    ]:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        )

    plot_valid = (
        plot_df["year"].isin(
            EARLY_YEARS + LATE_YEARS
        )
        & plot_df["latitude"].between(
            NC_LAT_MIN,
            NC_LAT_MAX,
        )
        & plot_df["longitude"].between(
            NC_LON_MIN,
            NC_LON_MAX,
        )
        & plot_df["plot_key"].notna()
        & plot_df["physical_plot_id"].notna()
    )

    if plot_status_col is not None:
        plot_valid &= plot_df["plot_status"].eq(1)

    plot_df = (
        plot_df.loc[
            plot_valid,
            [
                "plot_key",
                "physical_plot_id",
                "year",
                "latitude",
                "longitude",
            ],
        ]
        .drop_duplicates("plot_key")
        .copy()
    )

    plot_df["year"] = plot_df["year"].astype(int)

    early_measurements_df = (
        closest_measurement_per_plot(
            plot_df,
            EARLY_YEARS,
            EARLY_MID_YEAR,
        )
    )

    late_measurements_df = (
        closest_measurement_per_plot(
            plot_df,
            LATE_YEARS,
            LATE_MID_YEAR,
        )
    )

    matched_plot_df = (
        early_measurements_df[
            [
                "physical_plot_id",
                "plot_key",
                "year",
                "latitude",
                "longitude",
            ]
        ]
        .rename(
            columns={
                "plot_key": "early_plot_key",
                "year": "early_year",
                "latitude": "early_latitude",
                "longitude": "early_longitude",
            }
        )
        .merge(
            late_measurements_df[
                [
                    "physical_plot_id",
                    "plot_key",
                    "year",
                    "latitude",
                    "longitude",
                ]
            ].rename(
                columns={
                    "plot_key": "late_plot_key",
                    "year": "late_year",
                    "latitude": "late_latitude",
                    "longitude": "late_longitude",
                }
            ),
            on="physical_plot_id",
            how="inner",
            validate="one_to_one",
        )
    )

    matched_plot_df = matched_plot_df[
        matched_plot_df["late_year"]
        > matched_plot_df["early_year"]
    ].copy()

    matched_plot_df["stable_latitude"] = (
        matched_plot_df[
            [
                "early_latitude",
                "late_latitude",
            ]
        ].mean(axis=1)
    )

    matched_plot_df["stable_longitude"] = (
        matched_plot_df[
            [
                "early_longitude",
                "late_longitude",
            ]
        ].mean(axis=1)
    )

    candidate_plot_keys = set(
        matched_plot_df[
            "early_plot_key"
        ].astype(str)
    ) | set(
        matched_plot_df[
            "late_plot_key"
        ].astype(str)
    )

    tree_columns = pd.read_csv(
        NC_TREE_CSV,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    tree_plot_key_col = resolve_column(
        tree_columns,
        ["PLT_CN", "PLOT_CN"],
    )
    tree_species_col = resolve_column(
        tree_columns,
        ["SPCD", "SPECIES_CODE"],
    )
    tree_status_col = resolve_column(
        tree_columns,
        ["STATUSCD", "STATUS_CODE"],
    )

    if not all(
        [
            tree_plot_key_col,
            tree_species_col,
        ]
    ):
        raise ValueError(
            "Required NC TREE fields were not resolved."
        )

    tree_usecols = [
        tree_plot_key_col,
        tree_species_col,
    ]

    if tree_status_col is not None:
        tree_usecols.append(tree_status_col)

    tree_usecols = list(
        dict.fromkeys(tree_usecols)
    )

    live_measurement_keys = set()
    selected_presence_parts = []
    tree_rows_scanned = 0

    tree_reader = pd.read_csv(
        NC_TREE_CSV,
        usecols=tree_usecols,
        chunksize=TREE_CHUNK_SIZE,
        low_memory=False,
        encoding_errors="replace",
        dtype={
            tree_plot_key_col: "string"
        },
    )

    for chunk in tqdm(
        tree_reader,
        desc="Scanning NC TREE for final confirmation reference",
        unit="chunk",
    ):
        tree_rows_scanned += len(chunk)

        rename_tree = {
            tree_plot_key_col: "plot_key",
            tree_species_col: "species_code",
        }

        if tree_status_col is not None:
            rename_tree[tree_status_col] = "status_code"

        chunk = chunk.rename(
            columns=rename_tree
        )

        chunk["plot_key"] = (
            chunk["plot_key"]
            .astype("string")
            .str.strip()
        )

        chunk["species_code"] = pd.to_numeric(
            chunk["species_code"],
            errors="coerce",
        )

        keep = chunk["plot_key"].isin(
            candidate_plot_keys
        )

        if "status_code" in chunk.columns:
            chunk["status_code"] = pd.to_numeric(
                chunk["status_code"],
                errors="coerce",
            )
            keep &= chunk["status_code"].eq(1)

        live_work = chunk.loc[
            keep,
            [
                "plot_key",
                "species_code",
            ],
        ].dropna(
            subset=["plot_key"]
        )

        if live_work.empty:
            continue

        live_measurement_keys.update(
            live_work["plot_key"].astype(str)
        )

        selected_work = live_work[
            live_work["species_code"].isin(
                benchmark_codes
            )
        ].dropna(
            subset=["species_code"]
        )

        if not selected_work.empty:
            selected_work["species_code"] = (
                selected_work["species_code"].astype(int)
            )

            selected_presence_parts.append(
                selected_work.drop_duplicates(
                    [
                        "plot_key",
                        "species_code",
                    ]
                )
            )

    if not selected_presence_parts:
        raise ValueError(
            "No NC benchmark species were found on matched plots."
        )

    fia_presence_df = (
        pd.concat(
            selected_presence_parts,
            ignore_index=True,
        )
        .drop_duplicates(
            [
                "plot_key",
                "species_code",
            ]
        )
        .reset_index(drop=True)
    )

    del selected_presence_parts
    gc.collect()

    matched_plot_df = matched_plot_df[
        matched_plot_df[
            "early_plot_key"
        ].astype(str).isin(
            live_measurement_keys
        )
        & matched_plot_df[
            "late_plot_key"
        ].astype(str).isin(
            live_measurement_keys
        )
    ].copy()

    matched_plot_df = matched_plot_df.reset_index(
        drop=True
    )

    matched_plot_df.to_parquet(
        MATCHED_PLOT_PATH,
        index=False,
    )

    early_lookup_df = (
        matched_plot_df[
            [
                "physical_plot_id",
                "early_plot_key",
                "early_year",
                "stable_latitude",
                "stable_longitude",
            ]
        ]
        .rename(
            columns={
                "early_plot_key": "plot_key",
                "early_year": "year",
            }
        )
    )
    early_lookup_df["period"] = "early"

    late_lookup_df = (
        matched_plot_df[
            [
                "physical_plot_id",
                "late_plot_key",
                "late_year",
                "stable_latitude",
                "stable_longitude",
            ]
        ]
        .rename(
            columns={
                "late_plot_key": "plot_key",
                "late_year": "year",
            }
        )
    )
    late_lookup_df["period"] = "late"

    plot_period_lookup_df = pd.concat(
        [
            early_lookup_df,
            late_lookup_df,
        ],
        ignore_index=True,
    )

    plot_period_lookup_df["plot_key"] = (
        plot_period_lookup_df["plot_key"]
        .astype("string")
        .str.strip()
    )

    fia_species_period_df = (
        fia_presence_df.merge(
            plot_period_lookup_df,
            on="plot_key",
            how="inner",
            validate="many_to_one",
        )
        .drop_duplicates(
            [
                "species_code",
                "physical_plot_id",
                "period",
            ]
        )
        .groupby(
            [
                "species_code",
                "period",
            ],
            as_index=False,
        )
        .agg(
            fia_n_occupied_plots=(
                "physical_plot_id",
                "nunique",
            ),
            fia_mean_latitude=(
                "stable_latitude",
                "mean",
            ),
            fia_mean_year=("year", "mean"),
        )
    )

    fia_wide_df = fia_species_period_df.pivot(
        index="species_code",
        columns="period",
        values=[
            "fia_n_occupied_plots",
            "fia_mean_latitude",
            "fia_mean_year",
        ],
    )

    fia_wide_df.columns = [
        f"{metric}_{period}"
        for metric, period
        in fia_wide_df.columns
    ]

    fia_wide_df = fia_wide_df.reset_index()

    fia_wide_df["fia_shift_km_decade"] = [
        shift_km_decade(
            row.get(
                "fia_mean_latitude_early",
                np.nan,
            ),
            row.get(
                "fia_mean_latitude_late",
                np.nan,
            ),
            row.get(
                "fia_mean_year_early",
                np.nan,
            ),
            row.get(
                "fia_mean_year_late",
                np.nan,
            ),
        )
        for _, row in fia_wide_df.iterrows()
    ]

    fia_wide_df = (
        benchmark_taxa_df[
            [
                "fia_species_code",
                "scientific_name",
                "gbif_accepted_name",
            ]
        ]
        .drop_duplicates("fia_species_code")
        .merge(
            fia_wide_df,
            left_on="fia_species_code",
            right_on="species_code",
            how="left",
        )
        .drop(
            columns=["species_code"],
            errors="ignore",
        )
    )

    fia_wide_df.to_parquet(
        FIA_REFERENCE_PATH,
        index=False,
    )

    print("\nNORTH CAROLINA FIA SAME-WINDOW REFERENCE")
    display(fia_wide_df)

    # --------------------------------------------------------
    # 7. Final coverage decision
    # --------------------------------------------------------
    comparison_coverage_df = (
        benchmark_taxa_df[
            [
                "fia_species_code",
                "scientific_name",
                "gbif_accepted_name",
            ]
        ]
        .drop_duplicates("fia_species_code")
        .merge(
            benchmark_coverage_wide_df,
            on="fia_species_code",
            how="left",
        )
        .merge(
            fia_wide_df[
                [
                    "fia_species_code",
                    "fia_n_occupied_plots_early",
                    "fia_n_occupied_plots_late",
                    "fia_shift_km_decade",
                ]
            ],
            on="fia_species_code",
            how="left",
        )
    )

    for column in [
        "inat_records_early",
        "inat_records_late",
        "inat_cells_early",
        "inat_cells_late",
        "fia_n_occupied_plots_early",
        "fia_n_occupied_plots_late",
    ]:
        if column not in comparison_coverage_df.columns:
            comparison_coverage_df[column] = np.nan

    comparison_coverage_df[
        "confirmation_coverage_pass"
    ] = (
        comparison_coverage_df[
            "inat_records_early"
        ].ge(
            MIN_INAT_RECORDS_PER_SPECIES_PERIOD
        )
        & comparison_coverage_df[
            "inat_records_late"
        ].ge(
            MIN_INAT_RECORDS_PER_SPECIES_PERIOD
        )
        & comparison_coverage_df[
            "inat_cells_early"
        ].ge(
            MIN_INAT_CELLS_PER_SPECIES_PERIOD
        )
        & comparison_coverage_df[
            "inat_cells_late"
        ].ge(
            MIN_INAT_CELLS_PER_SPECIES_PERIOD
        )
        & comparison_coverage_df[
            "fia_n_occupied_plots_early"
        ].ge(
            MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD
        )
        & comparison_coverage_df[
            "fia_n_occupied_plots_late"
        ].ge(
            MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD
        )
        & comparison_coverage_df[
            "fia_shift_km_decade"
        ].notna()
    )

    comparison_coverage_df.to_csv(
        OUT_DIR / "nc_confirmation_species_coverage.csv",
        index=False,
    )

    print("\nNORTH CAROLINA CONFIRMATION SPECIES COVERAGE")
    display(comparison_coverage_df)

    n_confirmation_species = int(
        comparison_coverage_df[
            "confirmation_coverage_pass"
        ].sum()
    )

    n_matched_physical_plots = len(
        matched_plot_df
    )

    n_stable_effort_cells = len(
        stable_effort_cells
    )

    passed = bool(
        n_confirmation_species
        >= MIN_FINAL_BENCHMARK_SPECIES
        and n_matched_physical_plots
        >= MIN_MATCHED_FIA_PHYSICAL_PLOTS
        and n_stable_effort_cells
        >= MIN_STABLE_EFFORT_CELLS
        and effort_taxa_retained
        >= MIN_EFFORT_FRAME_TAXA
    )

    if passed:
        status = (
            "GO_FOR_FINAL_NC_CANDIDATE_CONFIRMATION"
        )
        next_step = (
            "Run the final locked confirmation of cell_jaccard and "
            "abs_log_observer_growth using the unchanged Virginia "
            "criteria. If neither confirms, permanently close the "
            "sensitivity-index route."
        )
    else:
        status = (
            "NC_FINAL_CONFIRMATION_DATASET_INSUFFICIENT"
        )
        next_step = (
            "Do not test the PA candidates. The final third-region "
            "dataset failed the locked coverage requirements; close "
            "the sensitivity-index route without weakening thresholds."
        )

    # --------------------------------------------------------
    # 8. Visual previews
    # --------------------------------------------------------
    plot_df = comparison_coverage_df.sort_values(
        "inat_records_early"
    ).reset_index(drop=True)

    y = np.arange(len(plot_df))

    plt.figure(
        figsize=(
            11,
            max(6, 0.5 * len(plot_df)),
        )
    )
    plt.scatter(
        plot_df["inat_records_early"],
        y,
        label="iNaturalist early",
    )
    plt.scatter(
        plot_df["inat_records_late"],
        y,
        label="iNaturalist late",
    )
    plt.yticks(
        y,
        plot_df["scientific_name"],
    )
    plt.axvline(
        MIN_INAT_RECORDS_PER_SPECIES_PERIOD,
        linestyle="--",
    )
    plt.xlabel("Thinned iNaturalist records")
    plt.ylabel("Benchmark species")
    plt.title(
        "North Carolina final-confirmation occurrence coverage"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    fia_plot_df = comparison_coverage_df.sort_values(
        "fia_n_occupied_plots_early"
    ).reset_index(drop=True)

    y = np.arange(len(fia_plot_df))

    plt.figure(
        figsize=(
            11,
            max(6, 0.5 * len(fia_plot_df)),
        )
    )
    plt.scatter(
        fia_plot_df[
            "fia_n_occupied_plots_early"
        ],
        y,
        label="FIA early",
    )
    plt.scatter(
        fia_plot_df[
            "fia_n_occupied_plots_late"
        ],
        y,
        label="FIA late",
    )
    plt.yticks(
        y,
        fia_plot_df["scientific_name"],
    )
    plt.axvline(
        MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD,
        linestyle="--",
    )
    plt.xlabel("Occupied repeated FIA plots")
    plt.ylabel("Benchmark species")
    plt.title(
        "North Carolina repeated-plot reference coverage"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 9. README
    # --------------------------------------------------------
    retained_species_compact = (
        comparison_coverage_df[
            comparison_coverage_df[
                "confirmation_coverage_pass"
            ]
        ][
            [
                "fia_species_code",
                "scientific_name",
                "inat_records_early",
                "inat_records_late",
                "inat_cells_early",
                "inat_cells_late",
                "fia_n_occupied_plots_early",
                "fia_n_occupied_plots_late",
            ]
        ]
        .to_dict("records")
    )

    append_readme(
        f"""

## North Carolina final independent dataset — {RUN_UTC}

### Purpose
Prepare the last allowed independent region for confirming the two locked
Pennsylvania correction-transportability candidates.

### Locked candidates for the next cell
1. cell_jaccard
2. abs_log_observer_growth

### Data
- State: North Carolina
- Early window: {EARLY_YEARS}
- Late window: {LATE_YEARS}
- Selected union taxa: {len(selected_taxa_df)}
- Planned benchmark taxa: {len(benchmark_taxa_df)}
- Planned effort-frame taxa: {len(effort_taxa_df)}
- Planned raw records: {planned_records:,}
- Planned API pages: {planned_pages:,}
- Raw occurrence records: {len(raw_df):,}
- Clean records: {len(clean_df):,}
- Thinned records: {len(thinned_df):,}
- Duplicate GBIF IDs removed: {duplicate_gbif_ids_removed:,}
- Dataset key: {INAT_DATASET_KEY}
- Dataset DOI: {INAT_DATASET_DOI}
- Access date: {ACCESS_DATE}

### Processing
- Coordinate uncertainty:
  <= {MAX_COORDINATE_UNCERTAINTY_M:,} m or missing
- Daily-spatial thinning:
  species × date × {THINNING_RESOLUTION} degree cell
- Effort grid:
  {EFFORT_GRID_RESOLUTION} degrees
- Stable effort cell:
  >= {MIN_TARGET_EFFORT_PER_CELL_PERIOD} target-frame records in each period
- Tree abundance weighting: not used
- GPU: not used

### FIA reference
- NC TREE download status: {tree_download_status}
- Exact physical plot:
  STATECD + UNITCD + COUNTYCD + PLOT
- Matched live-tree physical plots:
  {n_matched_physical_plots:,}
- TREE rows scanned:
  {tree_rows_scanned:,}

### Preparation results
- Effort taxa retained:
  {effort_taxa_retained}
- Stable effort cells:
  {n_stable_effort_cells}
- Final confirmation species:
  {n_confirmation_species}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Final stop rule
If neither locked candidate confirms in the next cell, no additional regions,
windows, thresholds or features will be searched for the sensitivity index.
"""
    )

    # --------------------------------------------------------
    # 10. Compact output
    # --------------------------------------------------------
    print(
        "\n========== RUN SUMMARY =========="
    )
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATE: {STATE}")
    print(
        f"NC_TREE_DOWNLOAD_STATUS: "
        f"{tree_download_status}"
    )
    print(
        f"N_SELECTED_UNION_TAXA: "
        f"{len(selected_taxa_df)}"
    )
    print(
        f"N_PLANNED_BENCHMARK_TAXA: "
        f"{len(benchmark_taxa_df)}"
    )
    print(
        f"N_PLANNED_EFFORT_TAXA: "
        f"{len(effort_taxa_df)}"
    )
    print(
        f"PLANNED_RECORDS: "
        f"{planned_records:,}"
    )
    print(
        f"PLANNED_API_PAGES: "
        f"{planned_pages:,}"
    )
    print(
        f"RAW_RECORDS: "
        f"{len(raw_df):,}"
    )
    print(
        f"DUPLICATE_GBIF_IDS_REMOVED: "
        f"{duplicate_gbif_ids_removed:,}"
    )
    print(
        f"CLEAN_RECORDS: "
        f"{len(clean_df):,}"
    )
    print(
        f"THINNED_RECORDS: "
        f"{len(thinned_df):,}"
    )
    print(
        f"N_EFFORT_TAXA_RETAINED: "
        f"{effort_taxa_retained}"
    )
    print(
        f"N_STABLE_EFFORT_CELLS: "
        f"{n_stable_effort_cells}"
    )
    print(
        f"N_MATCHED_FIA_PHYSICAL_PLOTS: "
        f"{n_matched_physical_plots:,}"
    )
    print(
        f"NC_TREE_ROWS_SCANNED: "
        f"{tree_rows_scanned:,}"
    )
    print(
        f"N_CONFIRMATION_SPECIES: "
        f"{n_confirmation_species}"
    )
    print(
        "CONFIRMATION_SPECIES: "
        + json.dumps(
            retained_species_compact,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_CRITERIA: >=12 final benchmark species, "
        ">=500 matched FIA plots, >=10 stable effort cells, "
        "and >=20 retained effort-frame taxa"
    )
    print(
        "FINAL_STOP_RULE: If neither locked candidate confirms "
        "in the next cell, permanently close the "
        "sensitivity-index route."
    )
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except CellStop as exc:
    print_stop(
        exc.status,
        exc.message,
        exc.next_step,
    )

except requests.HTTPError as exc:
    response = exc.response
    status_code = (
        response.status_code
        if response is not None
        else "UNKNOWN"
    )
    url = (
        response.url
        if response is not None
        else "UNKNOWN"
    )

    print_stop(
        "HTTP_FAILED",
        f"HTTP {status_code} for {url}",
        (
            "Retry the cell. Completed per-taxon occurrence caches "
            "and the NC TREE file will be reused."
        ),
    )

except requests.RequestException as exc:
    print_stop(
        "CONNECTION_FAILED",
        repr(exc),
        (
            "Confirm Kaggle Internet is enabled and rerun. "
            "Completed per-taxon caches will be reused."
        ),
    )

except Exception as exc:
    print_stop(
        "CELL16B_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed table. "
            "Do not run the final candidate test yet."
        ),
    )
