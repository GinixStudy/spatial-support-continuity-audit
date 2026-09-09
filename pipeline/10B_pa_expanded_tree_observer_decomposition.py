
# Cell 10B — Expanded PA tree observer decomposition
#
# Goal
# ----
# Use the 33 coverage-qualified PA tree taxa from Cell 10A to test whether the
# observer mechanism becomes identifiable without lowering any repeated-unit
# threshold.
#
# Resource-saving design
# ----------------------
# - Reuse all selected taxa already present in Cell 08A.
# - Download only selected taxa that are missing from the existing cache.
# - Use only the predeclared early and late windows:
#       early: 2013–2018
#       late : 2020–2025
#
# Success criteria remain unchanged from Cell 09A:
# - >=100 matched observer labels
# - >=300 matched observer–species pairs
# - >=100 observer labels eligible for fixed effects
#
# If these criteria still fail, the tree-only personal-observer trajectory
# branch should be closed rather than expanded again.

from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import math
import shutil
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
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 300)

# ============================================================
# Configuration
# ============================================================
STATE = "PA"
STATE_NAME = "Pennsylvania"

EARLY_YEARS = list(range(2013, 2019))
LATE_YEARS = list(range(2020, 2026))
ANALYSIS_YEARS = EARLY_YEARS + LATE_YEARS

INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
INAT_DATASET_DOI = "10.15468/ab3s5x"
GBIF_API = "https://api.gbif.org/v1"

PAGE_SIZE = 300
BUFFER_ROWS = 12_000
REQUEST_PAUSE_SECONDS = 0.03

MAX_COORDINATE_UNCERTAINTY_M = 10_000
THINNING_RESOLUTION = 0.05
FOOTPRINT_GRID_RESOLUTION = 0.50

MIN_OBSERVER_RECORDS_PER_PERIOD = 3
MIN_PAIR_RECORDS_PER_PERIOD = 2
MIN_OBSERVER_YEARS_FOR_FE = 3
MIN_OBSERVER_SPAN_FOR_FE = 4

MIN_BRIDGE_OBSERVERS = 100
MIN_BRIDGE_PAIRS = 300
MIN_FE_OBSERVERS = 100

MATERIAL_SHIFT_KM_DECADE = 5.0
SMALL_WITHIN_SHIFT_KM_DECADE = 3.0

N_BOOTSTRAP = 500
N_SIGN_FLIP = 500
RANDOM_SEED = 20260710

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
PILOT_DIR = BASE_DIR / "derived" / "gbif_inaturalist_real_pilot"
AUDIT_DIR = BASE_DIR / "derived" / "pa_expanded_tree_taxon_audit"
OUT_DIR = BASE_DIR / "derived" / "pa_expanded_tree_observer_decomposition"
RAW_PART_DIR = OUT_DIR / "missing_taxa_raw_parts"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_TAXA_PATH = AUDIT_DIR / "selected_expanded_tree_taxa.csv"
EXISTING_THINNED_PATH = PILOT_DIR / "thinned_occurrences.parquet"

MISSING_RAW_PATH = OUT_DIR / "missing_taxa_raw_occurrences.parquet"
MISSING_THINNED_PATH = OUT_DIR / "missing_taxa_thinned_occurrences.parquet"
EXPANDED_THINNED_PATH = OUT_DIR / "expanded_pa_tree_thinned_occurrences.parquet"

RUN_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
ACCESS_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

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
    readme_path.write_text(existing + text, encoding="utf-8")


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
                "(expanded PA tree observer decomposition)"
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


def build_occurrence_params(taxon_key, start_year, end_year):
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
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_record(record, taxon_row, period):
    return {
        "gbifID": record.get("key"),
        "occurrenceID": record.get("occurrenceID"),
        "query_state_code": STATE,
        "query_state_name": STATE_NAME,
        "fia_species_code": int(taxon_row["species_code"]),
        "query_scientific_name": taxon_row["fia_canonical_name"],
        "gbif_query_taxon_key": int(taxon_row["gbif_taxon_key"]),
        "gbif_accepted_name": taxon_row["gbif_accepted_name"],
        "query_period": period,
        "taxonKey": record.get("taxonKey"),
        "acceptedTaxonKey": record.get("acceptedTaxonKey"),
        "speciesKey": record.get("speciesKey"),
        "scientificName": record.get("scientificName"),
        "acceptedScientificName": record.get("acceptedScientificName"),
        "species": record.get("species"),
        "decimalLatitude": record.get("decimalLatitude"),
        "decimalLongitude": record.get("decimalLongitude"),
        "coordinateUncertaintyInMeters": record.get(
            "coordinateUncertaintyInMeters"
        ),
        "year": record.get("year"),
        "month": record.get("month"),
        "day": record.get("day"),
        "eventDate": record.get("eventDate"),
        "stateProvince": record.get("stateProvince"),
        "countryCode": record.get("countryCode"),
        "basisOfRecord": record.get("basisOfRecord"),
        "occurrenceStatus": record.get("occurrenceStatus"),
        "recordedBy": stringify(record.get("recordedBy")),
        "identifiedBy": stringify(record.get("identifiedBy")),
        "datasetKey": record.get("datasetKey"),
        "license": record.get("license"),
        "issues": stringify(record.get("issues")),
        "lastInterpreted": record.get("lastInterpreted"),
    }


def write_buffer(buffer, part_index):
    if not buffer:
        return part_index

    part_path = RAW_PART_DIR / f"missing_part_{part_index:05d}.parquet"
    pd.DataFrame(buffer).to_parquet(part_path, index=False)
    buffer.clear()
    return part_index + 1


def event_day_from_frame(frame):
    event_time = pd.to_datetime(
        frame["eventDate"],
        errors="coerce",
        utc=True,
    )

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

    final_time = event_time.fillna(fallback)
    result = final_time.dt.strftime("%Y-%m-%d").astype("string")

    missing = result.isna()
    result.loc[missing] = (
        "unknown_" + frame.loc[missing, "gbifID"].astype(str)
    )

    return result


def normalize_observer_label(series):
    normalized = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    blocked = {
        "",
        "nan",
        "none",
        "unknown",
        "anonymous",
        "not recorded",
        "not provided",
    }

    return normalized.mask(normalized.isin(blocked))


def safe_shift_km_decade(
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

    if not all(np.isfinite(value) for value in values):
        return np.nan

    interval = late_year - early_year

    if interval <= 0:
        return np.nan

    return float(
        (late_latitude - early_latitude)
        * 111.32
        * 10
        / interval
    )


def paired_shift_table(frame, id_columns, minimum_records):
    grouped = (
        frame.groupby(
            id_columns + ["period"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_records=("gbifID", "size"),
            mean_latitude=("decimalLatitude", "mean"),
            median_latitude=("decimalLatitude", "median"),
            mean_year=("year", "mean"),
            n_species=("fia_species_code", "nunique"),
            n_cells=("footprint_cell", "nunique"),
        )
    )

    grouped = grouped[
        grouped["n_records"] >= minimum_records
    ].copy()

    wide = grouped.pivot(
        index=id_columns,
        columns="period",
        values=[
            "n_records",
            "mean_latitude",
            "median_latitude",
            "mean_year",
            "n_species",
            "n_cells",
        ],
    )

    wide.columns = [
        f"{metric}_{period}"
        for metric, period in wide.columns
    ]
    wide = wide.reset_index()

    required_columns = [
        "mean_latitude_early",
        "mean_latitude_late",
        "mean_year_early",
        "mean_year_late",
    ]

    for column in required_columns:
        if column not in wide.columns:
            wide[column] = np.nan

    wide = wide.dropna(subset=required_columns).copy()

    wide["mean_shift_km_decade"] = [
        safe_shift_km_decade(
            row["mean_latitude_early"],
            row["mean_latitude_late"],
            row["mean_year_early"],
            row["mean_year_late"],
        )
        for _, row in wide.iterrows()
    ]

    wide["median_shift_km_decade"] = [
        safe_shift_km_decade(
            row.get("median_latitude_early", np.nan),
            row.get("median_latitude_late", np.nan),
            row["mean_year_early"],
            row["mean_year_late"],
        )
        for _, row in wide.iterrows()
    ]

    return wide


def summarize_shift_distribution(
    shifts,
    rng,
    n_bootstrap,
    n_sign_flip,
    label,
):
    values = pd.to_numeric(
        pd.Series(shifts),
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(values) == 0:
        return {
            "label": label,
            "n_units": 0,
            "mean_shift_km_decade": np.nan,
            "median_shift_km_decade": np.nan,
            "bootstrap_mean_low": np.nan,
            "bootstrap_mean_high": np.nan,
            "bootstrap_median_low": np.nan,
            "bootstrap_median_high": np.nan,
            "sign_flip_p_mean": np.nan,
            "sign_flip_p_median": np.nan,
        }

    bootstrap_means = np.empty(n_bootstrap, dtype=float)
    bootstrap_medians = np.empty(n_bootstrap, dtype=float)

    for index in tqdm(
        range(n_bootstrap),
        desc=f"Bootstrap {label}",
        unit="replicate",
        leave=False,
    ):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        bootstrap_means[index] = np.mean(sample)
        bootstrap_medians[index] = np.median(sample)

    observed_mean = float(np.mean(values))
    observed_median = float(np.median(values))

    exceed_mean = 0
    exceed_median = 0

    for _ in tqdm(
        range(n_sign_flip),
        desc=f"Sign-flip {label}",
        unit="replicate",
        leave=False,
    ):
        signs = rng.choice(
            [-1.0, 1.0],
            size=len(values),
            replace=True,
        )
        permuted = values * signs

        exceed_mean += int(
            abs(np.mean(permuted)) >= abs(observed_mean)
        )
        exceed_median += int(
            abs(np.median(permuted)) >= abs(observed_median)
        )

    return {
        "label": label,
        "n_units": len(values),
        "mean_shift_km_decade": observed_mean,
        "median_shift_km_decade": observed_median,
        "bootstrap_mean_low": float(
            np.quantile(bootstrap_means, 0.025)
        ),
        "bootstrap_mean_high": float(
            np.quantile(bootstrap_means, 0.975)
        ),
        "bootstrap_median_low": float(
            np.quantile(bootstrap_medians, 0.025)
        ),
        "bootstrap_median_high": float(
            np.quantile(bootstrap_medians, 0.975)
        ),
        "sign_flip_p_mean": float(
            (1 + exceed_mean) / (n_sign_flip + 1)
        ),
        "sign_flip_p_median": float(
            (1 + exceed_median) / (n_sign_flip + 1)
        ),
    }


def fixed_effect_sufficient_statistics(observer_year_df):
    rows = []

    for observer_label, group in tqdm(
        observer_year_df.groupby("observer_label"),
        desc="Building observer FE statistics",
        unit="observer",
    ):
        group = group.sort_values("year")

        if group["year"].nunique() < MIN_OBSERVER_YEARS_FOR_FE:
            continue

        if (
            group["year"].max() - group["year"].min()
            < MIN_OBSERVER_SPAN_FOR_FE
        ):
            continue

        x = group["year"].to_numpy(dtype=float)
        y = group["observer_mean_latitude"].to_numpy(dtype=float)

        x_centered = x - x.mean()
        y_centered = y - y.mean()

        denominator = float(np.dot(x_centered, x_centered))

        if denominator <= 0:
            continue

        rows.append(
            {
                "observer_label": observer_label,
                "n_years": int(group["year"].nunique()),
                "first_year": int(group["year"].min()),
                "last_year": int(group["year"].max()),
                "numerator": float(
                    np.dot(x_centered, y_centered)
                ),
                "denominator": denominator,
            }
        )

    return pd.DataFrame(rows)


def period_shift(frame, latitude_column, year_column):
    indexed = frame.set_index("period")

    return safe_shift_km_decade(
        indexed.loc["early", latitude_column],
        indexed.loc["late", latitude_column],
        indexed.loc["early", year_column],
        indexed.loc["late", year_column],
    )


def print_stop(status, error, next_step):
    append_readme(
        f"""

## Expanded PA tree observer decomposition stopped — {RUN_UTC}
- Status: **{status}**
- Reason: `{error}`
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


# ============================================================
# Main
# ============================================================
try:
    # --------------------------------------------------------
    # 1. Load selected taxa and existing cache
    # --------------------------------------------------------
    required_paths = [
        SELECTED_TAXA_PATH,
        EXISTING_THINNED_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing prior outputs: " + " | ".join(missing_paths)
        )

    selected_taxa_df = pd.read_csv(SELECTED_TAXA_PATH)

    required_selected_columns = [
        "species_code",
        "fia_canonical_name",
        "gbif_accepted_name",
        "gbif_taxon_key",
        "early_count",
        "late_count",
    ]

    missing_columns = [
        column
        for column in required_selected_columns
        if column not in selected_taxa_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Selected taxa table is missing columns: "
            + " | ".join(missing_columns)
        )

    selected_taxa_df["species_code"] = pd.to_numeric(
        selected_taxa_df["species_code"],
        errors="coerce",
    )
    selected_taxa_df["gbif_taxon_key"] = pd.to_numeric(
        selected_taxa_df["gbif_taxon_key"],
        errors="coerce",
    )

    selected_taxa_df = selected_taxa_df.dropna(
        subset=["species_code", "gbif_taxon_key"]
    ).copy()

    selected_taxa_df["species_code"] = (
        selected_taxa_df["species_code"].astype(int)
    )
    selected_taxa_df["gbif_taxon_key"] = (
        selected_taxa_df["gbif_taxon_key"].astype(int)
    )

    selected_species_codes = set(
        selected_taxa_df["species_code"].tolist()
    )

    existing_df = pd.read_parquet(EXISTING_THINNED_PATH)

    existing_df["fia_species_code"] = pd.to_numeric(
        existing_df["fia_species_code"],
        errors="coerce",
    )
    existing_df["year"] = pd.to_numeric(
        existing_df["year"],
        errors="coerce",
    )

    existing_selected_df = existing_df[
        existing_df["query_state_code"].eq(STATE)
        & existing_df["fia_species_code"].isin(selected_species_codes)
        & existing_df["year"].isin(ANALYSIS_YEARS)
    ].copy()

    existing_selected_df["fia_species_code"] = (
        existing_selected_df["fia_species_code"].astype(int)
    )
    existing_selected_df["year"] = (
        existing_selected_df["year"].astype(int)
    )

    existing_species_codes = set(
        existing_selected_df["fia_species_code"].unique().tolist()
    )

    missing_taxa_df = selected_taxa_df[
        ~selected_taxa_df["species_code"].isin(existing_species_codes)
    ].copy()

    reuse_summary_df = pd.DataFrame(
        [
            {
                "n_selected_taxa": len(selected_taxa_df),
                "n_taxa_reused_from_cell08": len(
                    selected_species_codes & existing_species_codes
                ),
                "n_taxa_to_download": len(missing_taxa_df),
                "existing_reused_records": len(existing_selected_df),
                "estimated_missing_records": int(
                    missing_taxa_df[
                        ["early_count", "late_count"]
                    ].sum().sum()
                ),
            }
        ]
    )

    print("\nREUSE PLAN")
    display(reuse_summary_df)

    # --------------------------------------------------------
    # 2. Download only missing taxa, with caching
    # --------------------------------------------------------
    if MISSING_THINNED_PATH.exists():
        print(
            "\nReusing cached missing-taxa thinned file:",
            MISSING_THINNED_PATH,
        )
        missing_thinned_df = pd.read_parquet(MISSING_THINNED_PATH)

    elif missing_taxa_df.empty:
        missing_thinned_df = pd.DataFrame(
            columns=existing_selected_df.columns
        )

    else:
        planned_pages = int(
            np.ceil(missing_taxa_df["early_count"] / PAGE_SIZE).sum()
            + np.ceil(missing_taxa_df["late_count"] / PAGE_SIZE).sum()
        )

        if MISSING_RAW_PATH.exists():
            print(
                "\nReusing cached missing-taxa raw file:",
                MISSING_RAW_PATH,
            )
            missing_raw_df = pd.read_parquet(MISSING_RAW_PATH)

        else:
            if RAW_PART_DIR.exists():
                shutil.rmtree(RAW_PART_DIR)

            RAW_PART_DIR.mkdir(parents=True, exist_ok=True)

            buffer = []
            part_index = 0

            page_progress = tqdm(
                total=planned_pages,
                desc="Downloading missing-taxa GBIF pages",
                unit="page",
            )

            for taxon_row in missing_taxa_df.to_dict("records"):
                for period, start_year, end_year, expected_column in [
                    (
                        "early",
                        min(EARLY_YEARS),
                        max(EARLY_YEARS),
                        "early_count",
                    ),
                    (
                        "late",
                        min(LATE_YEARS),
                        max(LATE_YEARS),
                        "late_count",
                    ),
                ]:
                    expected = int(taxon_row[expected_column])
                    offset = 0

                    params = build_occurrence_params(
                        taxon_row["gbif_taxon_key"],
                        start_year,
                        end_year,
                    )

                    while offset < expected:
                        request_params = dict(params)
                        request_params["limit"] = min(
                            PAGE_SIZE,
                            expected - offset,
                        )
                        request_params["offset"] = offset

                        payload = api_get(
                            "/occurrence/search",
                            request_params,
                        )

                        results = payload.get("results", [])

                        if not results:
                            page_progress.update(1)
                            break

                        for record in results:
                            buffer.append(
                                extract_record(
                                    record,
                                    taxon_row,
                                    period,
                                )
                            )

                        offset += len(results)
                        page_progress.update(1)

                        if len(buffer) >= BUFFER_ROWS:
                            part_index = write_buffer(
                                buffer,
                                part_index,
                            )

                        if payload.get("endOfRecords", False):
                            break

            part_index = write_buffer(buffer, part_index)
            page_progress.close()

            part_paths = sorted(
                RAW_PART_DIR.glob("missing_part_*.parquet")
            )

            if not part_paths:
                raise ValueError(
                    "No missing-taxa GBIF parts were created."
                )

            raw_parts = []

            for part_path in tqdm(
                part_paths,
                desc="Combining missing-taxa parts",
                unit="part",
            ):
                raw_parts.append(pd.read_parquet(part_path))

            missing_raw_df = pd.concat(
                raw_parts,
                ignore_index=True,
            )

            del raw_parts
            gc.collect()

            missing_raw_df.to_parquet(
                MISSING_RAW_PATH,
                index=False,
            )

        # ----------------------------------------------------
        # 3. Apply the same quality filters and thinning as Cell 08A
        # ----------------------------------------------------
        for column in [
            "decimalLatitude",
            "decimalLongitude",
            "coordinateUncertaintyInMeters",
            "year",
            "month",
            "day",
            "fia_species_code",
            "gbif_query_taxon_key",
        ]:
            missing_raw_df[column] = pd.to_numeric(
                missing_raw_df[column],
                errors="coerce",
            )

        missing_raw_df["gbifID"] = (
            missing_raw_df["gbifID"].astype("string")
        )

        missing_raw_df = (
            missing_raw_df.sort_values("gbifID")
            .drop_duplicates("gbifID")
            .reset_index(drop=True)
        )

        valid = (
            missing_raw_df["year"].isin(ANALYSIS_YEARS)
            & missing_raw_df["decimalLatitude"].between(39.5, 42.6)
            & missing_raw_df["decimalLongitude"].between(-81.0, -74.4)
            & (
                missing_raw_df[
                    "coordinateUncertaintyInMeters"
                ].isna()
                | missing_raw_df[
                    "coordinateUncertaintyInMeters"
                ].le(MAX_COORDINATE_UNCERTAINTY_M)
            )
            & ~(
                missing_raw_df["decimalLatitude"].eq(0)
                & missing_raw_df["decimalLongitude"].eq(0)
            )
        )

        missing_clean_df = missing_raw_df.loc[valid].copy()

        missing_clean_df["fia_species_code"] = (
            missing_clean_df["fia_species_code"].astype(int)
        )
        missing_clean_df["year"] = (
            missing_clean_df["year"].astype(int)
        )

        missing_clean_df["event_day"] = event_day_from_frame(
            missing_clean_df
        )

        missing_clean_df["thin_lat_index"] = np.floor(
            missing_clean_df["decimalLatitude"]
            / THINNING_RESOLUTION
        ).astype("int32")

        missing_clean_df["thin_lon_index"] = np.floor(
            missing_clean_df["decimalLongitude"]
            / THINNING_RESOLUTION
        ).astype("int32")

        missing_clean_df["uncertainty_sort"] = (
            missing_clean_df[
                "coordinateUncertaintyInMeters"
            ].fillna(np.inf)
        )

        missing_thinned_df = (
            missing_clean_df.sort_values(
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

        missing_thinned_df.to_parquet(
            MISSING_THINNED_PATH,
            index=False,
        )

    # --------------------------------------------------------
    # 4. Merge reused and newly downloaded records
    # --------------------------------------------------------
    common_columns = sorted(
        set(existing_selected_df.columns)
        | set(missing_thinned_df.columns)
    )

    existing_selected_df = existing_selected_df.reindex(
        columns=common_columns
    )
    missing_thinned_df = missing_thinned_df.reindex(
        columns=common_columns
    )

    expanded_df = pd.concat(
        [
            existing_selected_df,
            missing_thinned_df,
        ],
        ignore_index=True,
    )

    expanded_df["gbifID"] = expanded_df["gbifID"].astype("string")
    expanded_df["fia_species_code"] = pd.to_numeric(
        expanded_df["fia_species_code"],
        errors="coerce",
    )
    expanded_df["year"] = pd.to_numeric(
        expanded_df["year"],
        errors="coerce",
    )
    expanded_df["decimalLatitude"] = pd.to_numeric(
        expanded_df["decimalLatitude"],
        errors="coerce",
    )
    expanded_df["decimalLongitude"] = pd.to_numeric(
        expanded_df["decimalLongitude"],
        errors="coerce",
    )

    expanded_df = expanded_df.dropna(
        subset=[
            "gbifID",
            "fia_species_code",
            "year",
            "decimalLatitude",
            "decimalLongitude",
        ]
    ).copy()

    expanded_df["fia_species_code"] = (
        expanded_df["fia_species_code"].astype(int)
    )
    expanded_df["year"] = expanded_df["year"].astype(int)

    expanded_df = (
        expanded_df.sort_values("gbifID")
        .drop_duplicates("gbifID")
        .reset_index(drop=True)
    )

    # Reapply thinning after merging to make the combined set internally exact.
    if "event_day" not in expanded_df.columns:
        expanded_df["event_day"] = event_day_from_frame(expanded_df)
    else:
        missing_event_day = expanded_df["event_day"].isna()
        if missing_event_day.any():
            expanded_df.loc[
                missing_event_day,
                "event_day",
            ] = event_day_from_frame(
                expanded_df.loc[missing_event_day]
            )

    expanded_df["thin_lat_index"] = np.floor(
        expanded_df["decimalLatitude"]
        / THINNING_RESOLUTION
    ).astype("int32")

    expanded_df["thin_lon_index"] = np.floor(
        expanded_df["decimalLongitude"]
        / THINNING_RESOLUTION
    ).astype("int32")

    expanded_df = (
        expanded_df.sort_values(
            [
                "fia_species_code",
                "event_day",
                "thin_lat_index",
                "thin_lon_index",
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
        .reset_index(drop=True)
    )

    expanded_df.to_parquet(
        EXPANDED_THINNED_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 5. Prepare observer decomposition frame
    # --------------------------------------------------------
    expanded_df["period"] = np.where(
        expanded_df["year"].isin(EARLY_YEARS),
        "early",
        "late",
    )

    expanded_df["observer_label"] = normalize_observer_label(
        expanded_df["recordedBy"]
    )

    observer_missing_records = int(
        expanded_df["observer_label"].isna().sum()
    )

    analysis_df = expanded_df.dropna(
        subset=["observer_label"]
    ).copy()

    analysis_df["footprint_lat_index"] = np.floor(
        analysis_df["decimalLatitude"]
        / FOOTPRINT_GRID_RESOLUTION
    ).astype("int32")

    analysis_df["footprint_lon_index"] = np.floor(
        analysis_df["decimalLongitude"]
        / FOOTPRINT_GRID_RESOLUTION
    ).astype("int32")

    analysis_df["footprint_cell"] = (
        analysis_df["footprint_lat_index"].astype(str)
        + "_"
        + analysis_df["footprint_lon_index"].astype(str)
    )

    # --------------------------------------------------------
    # 6. Activity concentration and observer turnover audit
    # --------------------------------------------------------
    observer_period_activity_df = (
        analysis_df.groupby(
            ["period", "observer_label"],
            as_index=False,
        )
        .agg(
            n_records=("gbifID", "size"),
            n_species=("fia_species_code", "nunique"),
            n_cells=("footprint_cell", "nunique"),
        )
    )

    concentration_rows = []

    observer_sets = {}

    for period, group in observer_period_activity_df.groupby("period"):
        group = group.sort_values("n_records", ascending=False)
        total = group["n_records"].sum()
        observer_sets[period] = set(group["observer_label"])

        concentration_rows.append(
            {
                "period": period,
                "n_observers": len(group),
                "total_records": int(total),
                "top1_record_fraction": float(
                    group["n_records"].head(1).sum() / total
                ),
                "top10_record_fraction": float(
                    group["n_records"].head(10).sum() / total
                ),
                "median_records_per_observer": float(
                    group["n_records"].median()
                ),
            }
        )

    observer_concentration_df = pd.DataFrame(concentration_rows)

    early_observers = observer_sets.get("early", set())
    late_observers = observer_sets.get("late", set())
    observer_union = early_observers | late_observers
    observer_intersection = early_observers & late_observers

    observer_jaccard = (
        len(observer_intersection) / len(observer_union)
        if observer_union else np.nan
    )

    observer_concentration_df.to_csv(
        OUT_DIR / "observer_concentration_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. Aggregate estimands
    # --------------------------------------------------------
    period_summary_df = (
        analysis_df.groupby("period", as_index=False)
        .agg(
            n_records=("gbifID", "size"),
            n_observers=("observer_label", "nunique"),
            n_species=("fia_species_code", "nunique"),
            n_cells=("footprint_cell", "nunique"),
            pooled_mean_latitude=("decimalLatitude", "mean"),
            mean_year=("year", "mean"),
        )
    )

    cell_period_df = (
        analysis_df.groupby(
            ["period", "footprint_cell"],
            as_index=False,
        )
        .agg(
            cell_mean_latitude=("decimalLatitude", "mean"),
            cell_mean_year=("year", "mean"),
        )
    )

    equal_cell_period_df = (
        cell_period_df.groupby("period", as_index=False)
        .agg(
            equal_cell_mean_latitude=("cell_mean_latitude", "mean"),
            equal_cell_mean_year=("cell_mean_year", "mean"),
            n_cells=("footprint_cell", "nunique"),
        )
    )

    species_period_df = (
        analysis_df.groupby(
            ["period", "fia_species_code"],
            as_index=False,
        )
        .agg(
            species_mean_latitude=("decimalLatitude", "mean"),
            species_mean_year=("year", "mean"),
            n_records=("gbifID", "size"),
        )
    )

    equal_species_period_df = (
        species_period_df.groupby("period", as_index=False)
        .agg(
            equal_species_mean_latitude=("species_mean_latitude", "mean"),
            equal_species_mean_year=("species_mean_year", "mean"),
            n_species=("fia_species_code", "nunique"),
        )
    )

    observer_period_df = (
        analysis_df.groupby(
            ["period", "observer_label"],
            as_index=False,
        )
        .agg(
            observer_mean_latitude=("decimalLatitude", "mean"),
            observer_mean_year=("year", "mean"),
            n_records=("gbifID", "size"),
        )
    )

    equal_observer_period_df = (
        observer_period_df.groupby("period", as_index=False)
        .agg(
            equal_observer_mean_latitude=("observer_mean_latitude", "mean"),
            equal_observer_mean_year=("observer_mean_year", "mean"),
            n_observers=("observer_label", "nunique"),
        )
    )

    aggregate_rows = [
        {
            "estimand": "pooled_records",
            "shift_km_decade": period_shift(
                period_summary_df,
                "pooled_mean_latitude",
                "mean_year",
            ),
        },
        {
            "estimand": "equal_spatial_cells",
            "shift_km_decade": period_shift(
                equal_cell_period_df,
                "equal_cell_mean_latitude",
                "equal_cell_mean_year",
            ),
        },
        {
            "estimand": "equal_species_all",
            "shift_km_decade": period_shift(
                equal_species_period_df,
                "equal_species_mean_latitude",
                "equal_species_mean_year",
            ),
        },
        {
            "estimand": "equal_observers_all",
            "shift_km_decade": period_shift(
                equal_observer_period_df,
                "equal_observer_mean_latitude",
                "equal_observer_mean_year",
            ),
        },
    ]

    # --------------------------------------------------------
    # 8. Matched observers and matched observer-species pairs
    # --------------------------------------------------------
    matched_observer_df = paired_shift_table(
        analysis_df,
        ["observer_label"],
        MIN_OBSERVER_RECORDS_PER_PERIOD,
    )

    matched_pair_df = paired_shift_table(
        analysis_df,
        ["observer_label", "fia_species_code"],
        MIN_PAIR_RECORDS_PER_PERIOD,
    )

    matched_observer_df.to_parquet(
        OUT_DIR / "matched_observer_shifts.parquet",
        index=False,
    )
    matched_pair_df.to_parquet(
        OUT_DIR / "matched_observer_species_shifts.parquet",
        index=False,
    )

    rng = np.random.default_rng(RANDOM_SEED)

    observer_inference = summarize_shift_distribution(
        matched_observer_df["mean_shift_km_decade"],
        rng,
        N_BOOTSTRAP,
        N_SIGN_FLIP,
        "matched observers",
    )

    pair_inference = summarize_shift_distribution(
        matched_pair_df["mean_shift_km_decade"],
        rng,
        N_BOOTSTRAP,
        N_SIGN_FLIP,
        "matched observer-species pairs",
    )

    aggregate_rows.extend(
        [
            {
                "estimand": "matched_observers_equal_weight",
                "shift_km_decade": observer_inference[
                    "mean_shift_km_decade"
                ],
            },
            {
                "estimand": "matched_observer_species_equal_weight",
                "shift_km_decade": pair_inference[
                    "mean_shift_km_decade"
                ],
            },
        ]
    )

    # --------------------------------------------------------
    # 9. Observer fixed-effect annual trend
    # --------------------------------------------------------
    observer_year_df = (
        analysis_df.groupby(
            ["observer_label", "year"],
            as_index=False,
        )
        .agg(
            observer_mean_latitude=("decimalLatitude", "mean"),
            n_records=("gbifID", "size"),
            n_species=("fia_species_code", "nunique"),
            n_cells=("footprint_cell", "nunique"),
        )
    )

    fe_stats_df = fixed_effect_sufficient_statistics(observer_year_df)

    if fe_stats_df.empty:
        fe_slope = np.nan
        fe_bootstrap_low = np.nan
        fe_bootstrap_high = np.nan
    else:
        fe_slope = float(
            fe_stats_df["numerator"].sum()
            / fe_stats_df["denominator"].sum()
            * 111.32
            * 10
        )

        fe_bootstrap = np.empty(N_BOOTSTRAP, dtype=float)

        for index in tqdm(
            range(N_BOOTSTRAP),
            desc="Bootstrap observer fixed effect",
            unit="replicate",
        ):
            sampled = fe_stats_df.iloc[
                rng.integers(
                    0,
                    len(fe_stats_df),
                    size=len(fe_stats_df),
                )
            ]

            denominator = sampled["denominator"].sum()

            fe_bootstrap[index] = (
                sampled["numerator"].sum()
                / denominator
                * 111.32
                * 10
                if denominator > 0
                else np.nan
            )

        fe_bootstrap_low = float(
            np.nanquantile(fe_bootstrap, 0.025)
        )
        fe_bootstrap_high = float(
            np.nanquantile(fe_bootstrap, 0.975)
        )

    aggregate_rows.append(
        {
            "estimand": "observer_fixed_effect_annual",
            "shift_km_decade": fe_slope,
        }
    )

    aggregate_df = pd.DataFrame(aggregate_rows)

    inference_df = pd.DataFrame(
        [
            observer_inference,
            pair_inference,
            {
                "label": "observer fixed-effect annual",
                "n_units": len(fe_stats_df),
                "mean_shift_km_decade": fe_slope,
                "median_shift_km_decade": np.nan,
                "bootstrap_mean_low": fe_bootstrap_low,
                "bootstrap_mean_high": fe_bootstrap_high,
                "bootstrap_median_low": np.nan,
                "bootstrap_median_high": np.nan,
                "sign_flip_p_mean": np.nan,
                "sign_flip_p_median": np.nan,
            },
        ]
    )

    aggregate_df.to_csv(
        OUT_DIR / "expanded_mechanism_estimands.csv",
        index=False,
    )
    inference_df.to_csv(
        OUT_DIR / "expanded_mechanism_inference.csv",
        index=False,
    )
    fe_stats_df.to_csv(
        OUT_DIR / "observer_fixed_effect_statistics.csv",
        index=False,
    )

    print("\nEXPANDED MECHANISM ESTIMANDS")
    display(aggregate_df)

    print("\nEXPANDED INFERENCE SUMMARY")
    display(inference_df)

    print("\nOBSERVER CONCENTRATION")
    display(observer_concentration_df)

    # --------------------------------------------------------
    # 10. Decision
    # --------------------------------------------------------
    aggregate_lookup = aggregate_df.set_index(
        "estimand"
    )["shift_km_decade"].to_dict()

    core_footprint_shift = max(
        abs(
            aggregate_lookup.get(
                "equal_spatial_cells",
                np.nan,
            )
        ),
        abs(
            aggregate_lookup.get(
                "equal_species_all",
                np.nan,
            )
        ),
    )

    matched_observer_shift = abs(
        aggregate_lookup.get(
            "matched_observers_equal_weight",
            np.nan,
        )
    )
    matched_pair_shift = abs(
        aggregate_lookup.get(
            "matched_observer_species_equal_weight",
            np.nan,
        )
    )
    fe_shift_abs = abs(fe_slope)

    coverage_pass = bool(
        len(matched_observer_df) >= MIN_BRIDGE_OBSERVERS
        and len(matched_pair_df) >= MIN_BRIDGE_PAIRS
        and len(fe_stats_df) >= MIN_FE_OBSERVERS
    )

    within_observer_supported = bool(
        coverage_pass
        and (
            matched_observer_shift >= MATERIAL_SHIFT_KM_DECADE
            or matched_pair_shift >= MATERIAL_SHIFT_KM_DECADE
            or fe_shift_abs >= MATERIAL_SHIFT_KM_DECADE
        )
        and (
            observer_inference["sign_flip_p_mean"] <= 0.05
            or pair_inference["sign_flip_p_mean"] <= 0.05
            or (
                np.isfinite(fe_bootstrap_low)
                and np.isfinite(fe_bootstrap_high)
                and (
                    fe_bootstrap_low > 0
                    or fe_bootstrap_high < 0
                )
            )
        )
    )

    turnover_activity_dominates = bool(
        coverage_pass
        and core_footprint_shift >= MATERIAL_SHIFT_KM_DECADE
        and matched_observer_shift < SMALL_WITHIN_SHIFT_KM_DECADE
        and matched_pair_shift < SMALL_WITHIN_SHIFT_KM_DECADE
        and fe_shift_abs < SMALL_WITHIN_SHIFT_KM_DECADE
    )

    if not coverage_pass:
        status = "TREE_ONLY_OBSERVER_TRAJECTORY_NOT_IDENTIFIABLE"
        passed = False
        next_step = (
            "Close the personal-observer trajectory branch for PA trees. "
            "Do not expand the tree set again. Continue with aggregate "
            "observer turnover/activity weighting, or switch the validation "
            "group to a much denser taxonomic frame."
        )
    elif within_observer_supported:
        status = "WITHIN_OBSERVER_MOVEMENT_SUPPORTED_EXPANDED"
        passed = True
        next_step = (
            "The expanded tree frame supports material movement of the "
            "same observer labels. Run species-level raw, equal-cell and "
            "observer-adjusted trend comparisons."
        )
    elif turnover_activity_dominates:
        status = "OBSERVER_TURNOVER_ACTIVITY_DOMINATES_EXPANDED"
        passed = True
        next_step = (
            "The observation footprint moves, but stable observer units do "
            "not. Build the correction around turnover and activity weights."
        )
    else:
        status = "MIXED_OBSERVATION_SYSTEM_DRIFT_EXPANDED"
        passed = True
        next_step = (
            "Coverage is now adequate but the mechanism is mixed. Run both "
            "equal-observer and equal-cell species-level sensitivity analyses."
        )

    # --------------------------------------------------------
    # 11. Visual previews
    # --------------------------------------------------------
    plt.figure(
        figsize=(
            10,
            max(5, 0.55 * len(aggregate_df)),
        )
    )
    plt.barh(
        aggregate_df["estimand"],
        aggregate_df["shift_km_decade"],
    )
    plt.axvline(0)
    plt.xlabel("Latitude shift (km/decade)")
    plt.ylabel("Estimand")
    plt.title(
        "PA expanded tree observation-system decomposition"
    )
    plt.tight_layout()
    plt.show()

    activity_plot_df = observer_concentration_df.sort_values("period")

    plt.figure(figsize=(8, 5))
    x = np.arange(len(activity_plot_df))
    plt.bar(
        x,
        activity_plot_df["top10_record_fraction"],
    )
    plt.xticks(x, activity_plot_df["period"])
    plt.ylabel("Top-10 observer share of records")
    plt.xlabel("Period")
    plt.title("Observer activity concentration")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 12. README
    # --------------------------------------------------------
    append_readme(
        f"""

## Expanded PA tree observer decomposition — {RUN_UTC}

### Purpose
Test whether expanding from 20 benchmark trees to the 33 coverage-qualified PA
tree taxa makes repeated observer trajectories identifiable without lowering
the Cell 09A thresholds.

### Data
- Selected taxa: {len(selected_taxa_df)}
- Taxa reused from Cell 08A:
  {len(selected_species_codes & existing_species_codes)}
- Taxa newly downloaded: {len(missing_taxa_df)}
- Existing reused records: {len(existing_selected_df):,}
- Newly retained thinned records: {len(missing_thinned_df):,}
- Final expanded thinned records: {len(expanded_df):,}
- Early years: {EARLY_YEARS}
- Late years: {LATE_YEARS}
- GBIF dataset key: {INAT_DATASET_KEY}
- Dataset DOI: {INAT_DATASET_DOI}
- Access date: {ACCESS_DATE}
- GPU: not used

### Observer coverage
- Exact normalized observer labels:
  {analysis_df['observer_label'].nunique():,}
- Matched observers:
  {len(matched_observer_df):,}
- Matched observer-species pairs:
  {len(matched_pair_df):,}
- Fixed-effect observers:
  {len(fe_stats_df):,}
- Observer early/late Jaccard:
  {observer_jaccard}

### Locked coverage criteria
- Matched observers >= {MIN_BRIDGE_OBSERVERS}
- Matched observer-species pairs >= {MIN_BRIDGE_PAIRS}
- Fixed-effect observers >= {MIN_FE_OBSERVERS}
- Coverage passed: {coverage_pass}

### Decision
- Within-observer movement supported:
  {within_observer_supported}
- Turnover/activity weighting dominates:
  {turnover_activity_dominates}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
recordedBy remains an exact normalized contributor label rather than a verified
permanent person identifier. The expanded set is used only for the observer
mechanism; the original 20-species benchmark remains locked for calibrated
ecological sensitivity.
"""
    )

    # --------------------------------------------------------
    # 13. Compact output
    # --------------------------------------------------------
    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATE: {STATE}")
    print(f"N_SELECTED_TAXA: {len(selected_taxa_df)}")
    print(
        f"N_REUSED_TAXA: "
        f"{len(selected_species_codes & existing_species_codes)}"
    )
    print(f"N_DOWNLOADED_TAXA: {len(missing_taxa_df)}")
    print(
        f"N_EXISTING_REUSED_RECORDS: "
        f"{len(existing_selected_df):,}"
    )
    print(
        f"N_NEW_THINNED_RECORDS: "
        f"{len(missing_thinned_df):,}"
    )
    print(
        f"N_EXPANDED_THINNED_RECORDS: "
        f"{len(expanded_df):,}"
    )
    print(
        f"N_RECORDS_WITH_OBSERVER_LABEL: "
        f"{len(analysis_df):,}"
    )
    print(
        f"N_UNIQUE_OBSERVER_LABELS: "
        f"{analysis_df['observer_label'].nunique():,}"
    )
    print(
        f"N_MATCHED_OBSERVERS: "
        f"{len(matched_observer_df):,}"
    )
    print(
        f"N_MATCHED_OBSERVER_SPECIES_PAIRS: "
        f"{len(matched_pair_df):,}"
    )
    print(f"N_FE_OBSERVERS: {len(fe_stats_df):,}")
    print(
        f"OBSERVER_EARLY_LATE_JACCARD: "
        f"{observer_jaccard}"
    )
    print(
        "OBSERVER_CONCENTRATION: "
        + json.dumps(
            observer_concentration_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "MECHANISM_ESTIMANDS: "
        + json.dumps(
            aggregate_df.round(6).to_dict("records"),
            ensure_ascii=False,
        )
    )
    print(
        "INFERENCE_SUMMARY: "
        + json.dumps(
            inference_df.round(6).to_dict("records"),
            ensure_ascii=False,
        )
    )
    print(f"COVERAGE_PASSED: {coverage_pass}")
    print(
        f"WITHIN_OBSERVER_SUPPORTED: "
        f"{within_observer_supported}"
    )
    print(
        f"TURNOVER_ACTIVITY_DOMINATES: "
        f"{turnover_activity_dominates}"
    )
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except CellStop as exc:
    print_stop(exc.status, exc.message, exc.next_step)

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
        "GBIF_HTTP_FAILED",
        f"HTTP {status_code} for {url}",
        (
            "Retry once. Completed cache files will be reused."
        ),
    )

except requests.RequestException as exc:
    print_stop(
        "GBIF_CONNECTION_FAILED",
        repr(exc),
        (
            "Confirm Kaggle Internet is enabled and retry. "
            "Completed cache files will be reused."
        ),
    )

except Exception as exc:
    print_stop(
        "CELL10B_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Save this diagnostic summary and the last displayed output. "
            "Do not rerun earlier FIA or GBIF cells."
        ),
    )
