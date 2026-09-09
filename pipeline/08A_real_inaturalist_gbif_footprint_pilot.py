
# Cell 08A — Real iNaturalist/GBIF pilot: data coverage and observation-footprint drift
#
# Current hypothesis
# ------------------
# In opportunity-based iNaturalist records, the spatial center of the observation
# footprint may move over time. If the realized movement reaches the range tested
# in the semi-synthetic benchmark, species migration estimates may be materially
# sensitive to observation drift.
#
# Minimal pilot
# -------------
# - States: West Virginia and Pennsylvania
# - Species: the same 20 benchmark tree species
# - Years: 2010–2025 (exclude incomplete 2026)
# - Source: iNaturalist Research-grade Observations through the GBIF API
# - Core outputs:
#     1) taxonomic reconciliation;
#     2) species-state record counts;
#     3) paged coordinate download;
#     4) quality filtering and conservative daily-spatial thinning;
#     5) annual pooled, equal-cell, species-balanced and observer-balanced centers;
#     6) trend estimates in km/decade;
#     7) comparison with the calibrated semi-synthetic sensitivity range.
#
# Important boundary
# ------------------
# This cell measures a target-group observation footprint, not pure observer
# movement. The final paper must use a registered GBIF occurrence download with
# a DOI; this search-API extraction is only a trend pilot.

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
from scipy.stats import linregress, theilslopes
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 180)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 300)

# ============================================================
# Configuration
# ============================================================
START_YEAR = 2010
END_YEAR = 2025

STATE_CONFIG = {
    "WV": {
        "name": "West Virginia",
        "alternate": "WV",
        "lat_min": 37.0,
        "lat_max": 41.0,
        "lon_min": -83.0,
        "lon_max": -77.0,
    },
    "PA": {
        "name": "Pennsylvania",
        "alternate": "PA",
        "lat_min": 39.5,
        "lat_max": 42.6,
        "lon_min": -81.0,
        "lon_max": -74.4,
    },
}

INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
INAT_DATASET_DOI = "10.15468/ab3s5x"

GBIF_API = "https://api.gbif.org/v1"
PAGE_SIZE = 300
GBIF_QUERY_CAP = 99_000
MAX_PILOT_RECORDS = 600_000
BUFFER_ROWS = 12_000

MAX_COORDINATE_UNCERTAINTY_M = 10_000
THINNING_RESOLUTION = 0.05
FOOTPRINT_GRID_RESOLUTION = 0.50

MIN_RECORDS_PER_SPECIES_YEAR = 20
MIN_QUALIFIED_YEARS_PER_SPECIES = 8

MIN_ANNUAL_TARGET_RECORDS = 100
MIN_ANNUAL_TARGET_SPECIES = 5
MIN_ANNUAL_TARGET_CELLS = 5
MIN_TREND_YEARS = 8

STABLE_CELL_YEAR_FRACTION = 0.60
MIN_STABLE_CELLS = 10
MIN_ADEQUATE_SPECIES_PER_STATE = 10
MIN_MATERIAL_CENTER_DRIFT_KM_DECADE = 5.0

REUSE_FINAL_RAW_FILE = True
REQUEST_PAUSE_SECONDS = 0.03

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
BENCHMARK_DIR = BASE_DIR / "derived" / "semi_synthetic_observation_drift"
CALIBRATION_DIR = BASE_DIR / "derived" / "observation_drift_calibration"

SELECTED_SPECIES_PATH = (
    BENCHMARK_DIR / "selected_benchmark_species.csv"
)
SENSITIVITY_PATH = (
    CALIBRATION_DIR / "calibrated_species_sensitivity.parquet"
)

OUT_DIR = BASE_DIR / "derived" / "gbif_inaturalist_real_pilot"
RAW_PART_DIR = OUT_DIR / "raw_parts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_OCCURRENCE_PATH = OUT_DIR / "raw_occurrences.parquet"
CLEAN_OCCURRENCE_PATH = OUT_DIR / "clean_occurrences.parquet"
THINNED_OCCURRENCE_PATH = OUT_DIR / "thinned_occurrences.parquet"

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)
ACCESS_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

print("RUN_UTC:", RUN_UTC)
print("YEARS:", START_YEAR, "-", END_YEAR)
print("STATES:", list(STATE_CONFIG))
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")


# ============================================================
# Helpers
# ============================================================
class PilotStop(Exception):
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
        pool_connections=20,
        pool_maxsize=20,
    )
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "TemporalObservationDriftResearch/0.1 "
                "(GBIF trend pilot; contact via notebook owner)"
            )
        }
    )
    return session


SESSION = build_session()


def api_get(endpoint, params):
    url = f"{GBIF_API}{endpoint}"
    response = SESSION.get(
        url,
        params=params,
        timeout=(30, 240),
    )
    response.raise_for_status()
    time.sleep(REQUEST_PAUSE_SECONDS)
    return response.json()


def stringify(value):
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def safe_fit(years, values):
    frame = pd.DataFrame(
        {
            "year": pd.to_numeric(years, errors="coerce"),
            "value": pd.to_numeric(values, errors="coerce"),
        }
    ).dropna()

    frame = (
        frame.drop_duplicates("year")
        .sort_values("year")
        .reset_index(drop=True)
    )

    if len(frame) < MIN_TREND_YEARS:
        return {
            "n_years": len(frame),
            "first_year": (
                int(frame["year"].min())
                if len(frame)
                else np.nan
            ),
            "last_year": (
                int(frame["year"].max())
                if len(frame)
                else np.nan
            ),
            "ols_slope_per_year": np.nan,
            "ols_p": np.nan,
            "theil_sen_slope_per_year": np.nan,
            "trend_km_decade": np.nan,
        }

    ols = linregress(
        frame["year"],
        frame["value"],
    )
    robust_slope = theilslopes(
        frame["value"],
        frame["year"],
    )[0]

    return {
        "n_years": len(frame),
        "first_year": int(frame["year"].min()),
        "last_year": int(frame["year"].max()),
        "ols_slope_per_year": float(ols.slope),
        "ols_p": float(ols.pvalue),
        "theil_sen_slope_per_year": float(
            robust_slope
        ),
        "trend_km_decade": float(
            robust_slope * 111.32 * 10
        ),
    }


def event_day_from_frame(frame):
    event_time = pd.to_datetime(
        frame["eventDate"],
        errors="coerce",
        utc=True,
    )

    fallback_parts = pd.DataFrame(
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
    )

    fallback_date = pd.to_datetime(
        fallback_parts,
        errors="coerce",
        utc=True,
    )

    final_time = event_time.fillna(fallback_date)
    event_day = final_time.dt.strftime("%Y-%m-%d")

    missing = event_day.isna()

    event_day = event_day.astype("string")
    event_day.loc[missing] = (
        "unknown_" + frame.loc[missing, "gbifID"].astype(str)
    )

    return event_day


def extract_occurrence(
    record,
    query_row,
):
    return {
        "gbifID": record.get("key"),
        "occurrenceID": record.get("occurrenceID"),
        "query_state_code": query_row["state_code"],
        "query_state_name": query_row["state_name"],
        "query_state_filter": query_row["state_filter"],
        "fia_species_code": query_row["fia_species_code"],
        "query_scientific_name": (
            query_row["original_scientific_name"]
        ),
        "gbif_query_taxon_key": (
            query_row["gbif_taxon_key"]
        ),
        "gbif_accepted_name": (
            query_row["gbif_accepted_name"]
        ),
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
        "publishingOrgKey": record.get(
            "publishingOrgKey"
        ),
        "institutionCode": record.get(
            "institutionCode"
        ),
        "catalogNumber": record.get(
            "catalogNumber"
        ),
        "license": record.get("license"),
        "issues": stringify(record.get("issues")),
        "lastInterpreted": record.get(
            "lastInterpreted"
        ),
    }


def write_buffer(buffer, part_index):
    if not buffer:
        return part_index

    frame = pd.DataFrame(buffer)
    output_path = (
        RAW_PART_DIR
        / f"occurrence_part_{part_index:05d}.parquet"
    )
    frame.to_parquet(output_path, index=False)
    buffer.clear()
    return part_index + 1


def query_count(params):
    count_params = dict(params)
    count_params["limit"] = 0
    payload = api_get(
        "/occurrence/search",
        count_params,
    )
    return int(payload.get("count", 0))


def build_occurrence_params(
    taxon_key,
    state_filter,
    start_year,
    end_year,
):
    return {
        "datasetKey": INAT_DATASET_KEY,
        "country": "US",
        "stateProvince": state_filter,
        "taxonKey": int(taxon_key),
        "year": f"{int(start_year)},{int(end_year)}",
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "occurrenceStatus": "PRESENT",
        "basisOfRecord": "HUMAN_OBSERVATION",
    }


def print_stop_block(status, error, next_step):
    append_readme(
        f"""

## Real GBIF pilot stopped — {RUN_UTC}
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
    # 1. Load the fixed benchmark species list and sensitivity
    # --------------------------------------------------------
    if not SELECTED_SPECIES_PATH.exists():
        raise FileNotFoundError(
            "Missing selected species from Cell 07A: "
            f"{SELECTED_SPECIES_PATH}"
        )

    if not SENSITIVITY_PATH.exists():
        raise FileNotFoundError(
            "Missing calibrated sensitivity output from "
            f"Cell 07B: {SENSITIVITY_PATH}"
        )

    selected_species_df = pd.read_csv(
        SELECTED_SPECIES_PATH
    )

    selected_species_df["species_code"] = (
        pd.to_numeric(
            selected_species_df["species_code"],
            errors="coerce",
        )
    )

    selected_species_df = (
        selected_species_df
        .dropna(
            subset=[
                "species_code",
                "scientific_name",
            ]
        )
        .copy()
    )

    selected_species_df["species_code"] = (
        selected_species_df["species_code"].astype(int)
    )

    sensitivity_df = pd.read_parquet(
        SENSITIVITY_PATH
    )

    median_calibrated_sensitivity = float(
        sensitivity_df[
            "raw_error_per_observer_drift_km"
        ].median()
    )

    print("\nFIXED SPECIES SET")
    display(
        selected_species_df[
            [
                "species_code",
                "scientific_name",
                "common_name",
                "minimum_across_states",
            ]
        ]
    )

    # --------------------------------------------------------
    # 2. Resolve names against the current GBIF taxonomy
    # --------------------------------------------------------
    taxonomy_rows = []

    for row in tqdm(
        selected_species_df.itertuples(index=False),
        total=len(selected_species_df),
        desc="Resolving species in GBIF taxonomy",
        unit="species",
    ):
        payload = api_get(
            "/species/match",
            {
                "name": row.scientific_name,
                "kingdom": "Plantae",
                "verbose": "true",
            },
        )

        usage_key = payload.get("usageKey")
        accepted_key = payload.get(
            "acceptedUsageKey"
        )
        selected_key = (
            accepted_key
            if accepted_key is not None
            else usage_key
        )

        if selected_key is None:
            taxonomy_rows.append(
                {
                    "fia_species_code": int(
                        row.species_code
                    ),
                    "original_scientific_name": (
                        row.scientific_name
                    ),
                    "gbif_taxon_key": np.nan,
                    "gbif_accepted_name": None,
                    "gbif_match_type": payload.get(
                        "matchType"
                    ),
                    "gbif_status": payload.get(
                        "status"
                    ),
                    "gbif_rank": payload.get("rank"),
                    "gbif_confidence": payload.get(
                        "confidence"
                    ),
                    "taxon_resolution_pass": False,
                    "taxon_note": payload.get("note"),
                }
            )
            continue

        accepted_name = (
            payload.get("acceptedScientificName")
            or payload.get("scientificName")
            or payload.get("canonicalName")
        )

        match_type = payload.get("matchType")
        confidence = payload.get("confidence")

        resolution_pass = bool(
            match_type in {
                "EXACT",
                "HIGHERRANK",
                "FUZZY",
            }
            and (
                confidence is None
                or float(confidence) >= 80
            )
        )

        taxonomy_rows.append(
            {
                "fia_species_code": int(
                    row.species_code
                ),
                "original_scientific_name": (
                    row.scientific_name
                ),
                "gbif_taxon_key": int(
                    selected_key
                ),
                "gbif_accepted_name": (
                    accepted_name
                ),
                "gbif_match_type": match_type,
                "gbif_status": payload.get(
                    "status"
                ),
                "gbif_rank": payload.get("rank"),
                "gbif_confidence": confidence,
                "taxon_resolution_pass": (
                    resolution_pass
                ),
                "taxon_note": payload.get("note"),
            }
        )

    taxonomy_df = pd.DataFrame(taxonomy_rows)

    taxonomy_df.to_csv(
        OUT_DIR / "gbif_taxonomy_resolution.csv",
        index=False,
    )

    print("\nGBIF TAXONOMY RESOLUTION")
    display(taxonomy_df)

    unresolved_df = taxonomy_df[
        ~taxonomy_df["taxon_resolution_pass"]
        | taxonomy_df["gbif_taxon_key"].isna()
    ]

    if len(unresolved_df) > 2:
        raise PilotStop(
            "GBIF_TAXONOMY_REVIEW_REQUIRED",
            (
                f"{len(unresolved_df)} species did not resolve "
                "reliably in the GBIF taxonomy."
            ),
            (
                "Review the displayed taxonomy table before "
                "downloading occurrences."
            ),
        )

    resolved_taxonomy_df = taxonomy_df[
        taxonomy_df["taxon_resolution_pass"]
        & taxonomy_df["gbif_taxon_key"].notna()
    ].copy()

    resolved_taxonomy_df["gbif_taxon_key"] = (
        resolved_taxonomy_df[
            "gbif_taxon_key"
        ].astype(int)
    )

    duplicate_key_count = int(
        resolved_taxonomy_df[
            "gbif_taxon_key"
        ].duplicated().sum()
    )

    if duplicate_key_count:
        print(
            f"Warning: {duplicate_key_count} duplicate accepted "
            "GBIF taxon keys were found."
        )

    # --------------------------------------------------------
    # 3. Count records and choose the stateProvince filter
    # --------------------------------------------------------
    count_rows = []

    count_tasks = [
        (taxon_row, state_code)
        for taxon_row in resolved_taxonomy_df.to_dict(
            "records"
        )
        for state_code in STATE_CONFIG
    ]

    for taxon_row, state_code in tqdm(
        count_tasks,
        total=len(count_tasks),
        desc="Counting iNaturalist records",
        unit="species-state",
    ):
        state_info = STATE_CONFIG[state_code]

        full_params = build_occurrence_params(
            taxon_key=taxon_row[
                "gbif_taxon_key"
            ],
            state_filter=state_info["name"],
            start_year=START_YEAR,
            end_year=END_YEAR,
        )

        full_count = query_count(full_params)

        alternate_count = 0
        chosen_filter = state_info["name"]
        chosen_count = full_count

        if full_count == 0:
            alternate_params = build_occurrence_params(
                taxon_key=taxon_row[
                    "gbif_taxon_key"
                ],
                state_filter=state_info[
                    "alternate"
                ],
                start_year=START_YEAR,
                end_year=END_YEAR,
            )
            alternate_count = query_count(
                alternate_params
            )

            if alternate_count > full_count:
                chosen_filter = state_info[
                    "alternate"
                ]
                chosen_count = alternate_count

        count_rows.append(
            {
                "state_code": state_code,
                "state_name": state_info["name"],
                "state_filter": chosen_filter,
                "fia_species_code": taxon_row[
                    "fia_species_code"
                ],
                "original_scientific_name": (
                    taxon_row[
                        "original_scientific_name"
                    ]
                ),
                "gbif_taxon_key": taxon_row[
                    "gbif_taxon_key"
                ],
                "gbif_accepted_name": taxon_row[
                    "gbif_accepted_name"
                ],
                "full_name_count": full_count,
                "alternate_count": alternate_count,
                "query_count": chosen_count,
            }
        )

    count_df = pd.DataFrame(count_rows)

    count_df.to_csv(
        OUT_DIR / "gbif_species_state_counts.csv",
        index=False,
    )

    print("\nSPECIES-STATE RECORD COUNTS")
    display(
        count_df.sort_values(
            ["state_code", "query_count"],
            ascending=[True, False],
        )
    )

    total_query_records = int(
        count_df["query_count"].sum()
    )

    if total_query_records < 500:
        raise PilotStop(
            "GBIF_STATE_FILTER_RETURNED_TOO_FEW",
            (
                f"Only {total_query_records} total records were "
                "returned by the state filters."
            ),
            (
                "Inspect the state filter and taxonomy counts "
                "before downloading."
            ),
        )

    if total_query_records > MAX_PILOT_RECORDS:
        raise PilotStop(
            "GBIF_PILOT_TOO_LARGE",
            (
                f"The pilot query contains {total_query_records:,} "
                f"records, above the limit of "
                f"{MAX_PILOT_RECORDS:,}."
            ),
            (
                "Use an authenticated asynchronous GBIF download "
                "with a DOI, or reduce the pilot species set."
            ),
        )

    # --------------------------------------------------------
    # 4. Build safe query segments under the 100k search cap
    # --------------------------------------------------------
    segment_rows = []

    for row in tqdm(
        count_df.to_dict("records"),
        total=len(count_df),
        desc="Building safe GBIF query segments",
        unit="species-state",
    ):
        total = int(row["query_count"])

        if total == 0:
            continue

        if total <= GBIF_QUERY_CAP:
            segment_rows.append(
                {
                    **row,
                    "segment_start_year": START_YEAR,
                    "segment_end_year": END_YEAR,
                    "segment_count": total,
                }
            )
            continue

        for year in range(
            START_YEAR,
            END_YEAR + 1,
        ):
            params = build_occurrence_params(
                taxon_key=row[
                    "gbif_taxon_key"
                ],
                state_filter=row["state_filter"],
                start_year=year,
                end_year=year,
            )
            year_count = query_count(params)

            if year_count > GBIF_QUERY_CAP:
                raise PilotStop(
                    "GBIF_YEAR_STRATUM_EXCEEDS_SEARCH_CAP",
                    (
                        f"{row['state_code']} "
                        f"{row['gbif_accepted_name']} "
                        f"{year} contains {year_count:,} records."
                    ),
                    (
                        "Use the authenticated GBIF download "
                        "service for this taxon-year."
                    ),
                )

            if year_count > 0:
                segment_rows.append(
                    {
                        **row,
                        "segment_start_year": year,
                        "segment_end_year": year,
                        "segment_count": year_count,
                    }
                )

    segment_df = pd.DataFrame(segment_rows)

    segment_df.to_csv(
        OUT_DIR / "gbif_query_segments.csv",
        index=False,
    )

    planned_records = int(
        segment_df["segment_count"].sum()
    )

    planned_pages = int(
        sum(
            math.ceil(
                count / PAGE_SIZE
            )
            for count in segment_df[
                "segment_count"
            ]
        )
    )

    print("\nGBIF QUERY PLAN")
    display(
        segment_df[
            [
                "state_code",
                "original_scientific_name",
                "state_filter",
                "segment_start_year",
                "segment_end_year",
                "segment_count",
            ]
        ]
    )

    print(
        f"Planned records: {planned_records:,}; "
        f"planned API pages: {planned_pages:,}"
    )

    # --------------------------------------------------------
    # 5. Download paged occurrence records or reuse final cache
    # --------------------------------------------------------
    download_warnings = []

    if (
        REUSE_FINAL_RAW_FILE
        and RAW_OCCURRENCE_PATH.exists()
    ):
        print(
            f"\nReusing existing raw occurrence file: "
            f"{RAW_OCCURRENCE_PATH}"
        )
        raw_df = pd.read_parquet(
            RAW_OCCURRENCE_PATH
        )
    else:
        if RAW_PART_DIR.exists():
            shutil.rmtree(RAW_PART_DIR)

        RAW_PART_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        buffer = []
        part_index = 0

        page_progress = tqdm(
            total=planned_pages,
            desc="Downloading GBIF occurrence pages",
            unit="page",
        )

        for query_row in segment_df.to_dict(
            "records"
        ):
            params = build_occurrence_params(
                taxon_key=query_row[
                    "gbif_taxon_key"
                ],
                state_filter=query_row[
                    "state_filter"
                ],
                start_year=query_row[
                    "segment_start_year"
                ],
                end_year=query_row[
                    "segment_end_year"
                ],
            )

            expected = int(
                query_row["segment_count"]
            )
            offset = 0
            retrieved = 0

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

                results = payload.get(
                    "results",
                    [],
                )

                if not results:
                    download_warnings.append(
                        {
                            "state_code": query_row[
                                "state_code"
                            ],
                            "species": query_row[
                                "gbif_accepted_name"
                            ],
                            "offset": offset,
                            "expected": expected,
                            "warning": (
                                "API returned an empty page "
                                "before the expected count."
                            ),
                        }
                    )
                    page_progress.update(1)
                    break

                for record in results:
                    buffer.append(
                        extract_occurrence(
                            record,
                            query_row,
                        )
                    )

                retrieved += len(results)
                offset += len(results)
                page_progress.update(1)

                if len(buffer) >= BUFFER_ROWS:
                    part_index = write_buffer(
                        buffer,
                        part_index,
                    )

                if payload.get(
                    "endOfRecords",
                    False,
                ):
                    break

            if retrieved != expected:
                download_warnings.append(
                    {
                        "state_code": query_row[
                            "state_code"
                        ],
                        "species": query_row[
                            "gbif_accepted_name"
                        ],
                        "retrieved": retrieved,
                        "expected": expected,
                        "warning": (
                            "Retrieved record count differs "
                            "from the planning count."
                        ),
                    }
                )

        part_index = write_buffer(
            buffer,
            part_index,
        )
        page_progress.close()

        part_paths = sorted(
            RAW_PART_DIR.glob(
                "occurrence_part_*.parquet"
            )
        )

        if not part_paths:
            raise ValueError(
                "No occurrence parquet parts were created."
            )

        raw_parts = []

        for part_path in tqdm(
            part_paths,
            desc="Combining downloaded parquet parts",
            unit="part",
        ):
            raw_parts.append(
                pd.read_parquet(part_path)
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

    if download_warnings:
        pd.DataFrame(
            download_warnings
        ).to_csv(
            OUT_DIR / "download_warnings.csv",
            index=False,
        )

    print(
        f"\nRAW OCCURRENCES: {len(raw_df):,}"
    )

    # --------------------------------------------------------
    # 6. Quality filtering
    # --------------------------------------------------------
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
        raw_df[column] = pd.to_numeric(
            raw_df[column],
            errors="coerce",
        )

    raw_df["gbifID"] = (
        raw_df["gbifID"].astype("string")
    )

    raw_rows_before_dedup = len(raw_df)

    raw_df = (
        raw_df.sort_values("gbifID")
        .drop_duplicates("gbifID")
        .reset_index(drop=True)
    )

    duplicate_gbif_ids_removed = (
        raw_rows_before_dedup - len(raw_df)
    )

    raw_df["coordinate_uncertainty_known"] = (
        raw_df[
            "coordinateUncertaintyInMeters"
        ].notna()
    )

    raw_df["coordinate_uncertainty_acceptable"] = (
        raw_df[
            "coordinateUncertaintyInMeters"
        ].isna()
        | raw_df[
            "coordinateUncertaintyInMeters"
        ].le(
            MAX_COORDINATE_UNCERTAINTY_M
        )
    )

    basic_valid = (
        raw_df["year"].between(
            START_YEAR,
            END_YEAR,
        )
        & raw_df["decimalLatitude"].between(
            -90,
            90,
        )
        & raw_df["decimalLongitude"].between(
            -180,
            180,
        )
        & ~(
            raw_df["decimalLatitude"].eq(0)
            & raw_df["decimalLongitude"].eq(0)
        )
        & raw_df[
            "coordinate_uncertainty_acceptable"
        ]
    )

    state_bounds_valid = np.zeros(
        len(raw_df),
        dtype=bool,
    )

    for state_code, info in STATE_CONFIG.items():
        state_mask = raw_df[
            "query_state_code"
        ].eq(state_code)

        state_bounds_valid |= (
            state_mask
            & raw_df["decimalLatitude"].between(
                info["lat_min"],
                info["lat_max"],
            )
            & raw_df["decimalLongitude"].between(
                info["lon_min"],
                info["lon_max"],
            )
        ).to_numpy()

    clean_df = raw_df.loc[
        basic_valid.to_numpy()
        & state_bounds_valid
    ].copy()

    clean_df["fia_species_code"] = (
        clean_df["fia_species_code"].astype(int)
    )
    clean_df["year"] = clean_df["year"].astype(int)

    clean_df["event_day"] = (
        event_day_from_frame(clean_df)
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

    clean_df = (
        clean_df.sort_values(
            [
                "query_state_code",
                "fia_species_code",
                "event_day",
                "thin_lat_index",
                "thin_lon_index",
                "uncertainty_sort",
                "gbifID",
            ]
        )
        .reset_index(drop=True)
    )

    clean_df.to_parquet(
        CLEAN_OCCURRENCE_PATH,
        index=False,
    )

    thinned_df = (
        clean_df.drop_duplicates(
            [
                "query_state_code",
                "fia_species_code",
                "event_day",
                "thin_lat_index",
                "thin_lon_index",
            ],
            keep="first",
        )
        .drop(
            columns=["uncertainty_sort"]
        )
        .reset_index(drop=True)
    )

    thinned_df.to_parquet(
        THINNED_OCCURRENCE_PATH,
        index=False,
    )

    records_removed_by_quality = (
        len(raw_df) - len(clean_df)
    )
    records_removed_by_thinning = (
        len(clean_df) - len(thinned_df)
    )

    # --------------------------------------------------------
    # 7. Licence and field audit
    # --------------------------------------------------------
    licence_df = (
        raw_df.groupby(
            ["license"],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_records"})
        .sort_values(
            "n_records",
            ascending=False,
        )
    )

    licence_df.to_csv(
        OUT_DIR / "record_licence_summary.csv",
        index=False,
    )

    field_audit_rows = []

    for state_code, group in thinned_df.groupby(
        "query_state_code"
    ):
        recorded_present = (
            group["recordedBy"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )

        uncertainty_known = group[
            "coordinateUncertaintyInMeters"
        ].notna()

        field_audit_rows.append(
            {
                "state_code": state_code,
                "n_thinned_records": len(group),
                "recordedBy_completeness": float(
                    recorded_present.mean()
                ),
                "n_unique_recordedBy": int(
                    group.loc[
                        recorded_present,
                        "recordedBy",
                    ].nunique()
                ),
                "coordinate_uncertainty_completeness": float(
                    uncertainty_known.mean()
                ),
                "median_known_uncertainty_m": (
                    float(
                        group.loc[
                            uncertainty_known,
                            "coordinateUncertaintyInMeters",
                        ].median()
                    )
                    if uncertainty_known.any()
                    else np.nan
                ),
            }
        )

    field_audit_df = pd.DataFrame(
        field_audit_rows
    )

    field_audit_df.to_csv(
        OUT_DIR / "field_completeness_audit.csv",
        index=False,
    )

    print("\nFIELD COMPLETENESS AUDIT")
    display(field_audit_df)

    print("\nRECORD LICENCES")
    display(licence_df)

    # --------------------------------------------------------
    # 8. Species-year coverage
    # --------------------------------------------------------
    species_year_df = (
        thinned_df.groupby(
            [
                "query_state_code",
                "fia_species_code",
                "gbif_accepted_name",
                "year",
            ],
            as_index=False,
        )
        .agg(
            n_records=("gbifID", "size"),
            n_thin_cells=(
                "thin_lat_index",
                "nunique",
            ),
        )
    )

    species_year_df["qualified_year"] = (
        species_year_df["n_records"]
        >= MIN_RECORDS_PER_SPECIES_YEAR
    )

    species_year_df.to_parquet(
        OUT_DIR / "species_year_coverage.parquet",
        index=False,
    )

    coverage_rows = []

    coverage_groups = species_year_df.groupby(
        [
            "query_state_code",
            "fia_species_code",
            "gbif_accepted_name",
        ],
        dropna=False,
    )

    for (
        state_code,
        species_code,
        species_name,
    ), group in tqdm(
        coverage_groups,
        total=coverage_groups.ngroups,
        desc="Summarizing species coverage",
        unit="species-state",
    ):
        qualified = group[
            group["qualified_year"]
        ]

        coverage_rows.append(
            {
                "state_code": state_code,
                "fia_species_code": int(
                    species_code
                ),
                "gbif_accepted_name": species_name,
                "total_records": int(
                    group["n_records"].sum()
                ),
                "first_year": int(
                    group["year"].min()
                ),
                "last_year": int(
                    group["year"].max()
                ),
                "n_years_any": int(
                    group["year"].nunique()
                ),
                "n_qualified_years": int(
                    qualified["year"].nunique()
                ),
                "median_records_qualified_year": (
                    float(
                        qualified[
                            "n_records"
                        ].median()
                    )
                    if len(qualified)
                    else np.nan
                ),
                "adequate_coverage": bool(
                    qualified["year"].nunique()
                    >= MIN_QUALIFIED_YEARS_PER_SPECIES
                ),
            }
        )

    species_coverage_df = pd.DataFrame(
        coverage_rows
    ).sort_values(
        [
            "state_code",
            "adequate_coverage",
            "n_qualified_years",
            "total_records",
        ],
        ascending=[True, False, False, False],
    )

    species_coverage_df.to_csv(
        OUT_DIR / "species_state_coverage_summary.csv",
        index=False,
    )

    print("\nSPECIES-STATE COVERAGE")
    display(species_coverage_df)

    # --------------------------------------------------------
    # 9. Annual target-group observation footprint
    # --------------------------------------------------------
    thinned_df["footprint_lat_index"] = np.floor(
        thinned_df["decimalLatitude"]
        / FOOTPRINT_GRID_RESOLUTION
    ).astype("int32")

    thinned_df["footprint_lon_index"] = np.floor(
        thinned_df["decimalLongitude"]
        / FOOTPRINT_GRID_RESOLUTION
    ).astype("int32")

    thinned_df["footprint_cell"] = (
        thinned_df[
            "footprint_lat_index"
        ].astype(str)
        + "_"
        + thinned_df[
            "footprint_lon_index"
        ].astype(str)
    )

    # Species-year centers for equal-species annual balancing.
    species_center_df = (
        thinned_df.groupby(
            [
                "query_state_code",
                "fia_species_code",
                "year",
            ],
            as_index=False,
        )
        .agg(
            species_year_records=("gbifID", "size"),
            species_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            species_mean_longitude=(
                "decimalLongitude",
                "mean",
            ),
        )
    )

    species_center_df = species_center_df[
        species_center_df[
            "species_year_records"
        ]
        >= MIN_RECORDS_PER_SPECIES_YEAR
    ].copy()

    species_balanced_df = (
        species_center_df.groupby(
            ["query_state_code", "year"],
            as_index=False,
        )
        .agg(
            species_balanced_mean_latitude=(
                "species_mean_latitude",
                "mean",
            ),
            species_balanced_mean_longitude=(
                "species_mean_longitude",
                "mean",
            ),
            n_species_balanced=(
                "fia_species_code",
                "nunique",
            ),
        )
    )

    # Equal-cell footprint center.
    cell_year_df = (
        thinned_df.groupby(
            [
                "query_state_code",
                "year",
                "footprint_cell",
            ],
            as_index=False,
        )
        .agg(
            cell_records=("gbifID", "size"),
            cell_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            cell_mean_longitude=(
                "decimalLongitude",
                "mean",
            ),
        )
    )

    equal_cell_df = (
        cell_year_df.groupby(
            ["query_state_code", "year"],
            as_index=False,
        )
        .agg(
            equal_cell_mean_latitude=(
                "cell_mean_latitude",
                "mean",
            ),
            equal_cell_mean_longitude=(
                "cell_mean_longitude",
                "mean",
            ),
            n_footprint_cells=(
                "footprint_cell",
                "nunique",
            ),
        )
    )

    # Observer-balanced diagnostic. recordedBy is audited, not assumed stable.
    observer_source_df = thinned_df[
        thinned_df["recordedBy"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    observer_year_df = (
        observer_source_df.groupby(
            [
                "query_state_code",
                "year",
                "recordedBy",
            ],
            as_index=False,
        )
        .agg(
            observer_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            observer_mean_longitude=(
                "decimalLongitude",
                "mean",
            ),
            observer_records=("gbifID", "size"),
        )
    )

    observer_balanced_df = (
        observer_year_df.groupby(
            ["query_state_code", "year"],
            as_index=False,
        )
        .agg(
            observer_balanced_mean_latitude=(
                "observer_mean_latitude",
                "mean",
            ),
            observer_balanced_mean_longitude=(
                "observer_mean_longitude",
                "mean",
            ),
            n_observers=("recordedBy", "nunique"),
        )
    )

    annual_df = (
        thinned_df.groupby(
            ["query_state_code", "year"],
            as_index=False,
        )
        .agg(
            n_records=("gbifID", "size"),
            n_species=(
                "fia_species_code",
                "nunique",
            ),
            pooled_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            pooled_mean_longitude=(
                "decimalLongitude",
                "mean",
            ),
            median_coordinate_uncertainty_m=(
                "coordinateUncertaintyInMeters",
                "median",
            ),
        )
        .merge(
            equal_cell_df,
            on=[
                "query_state_code",
                "year",
            ],
            how="left",
        )
        .merge(
            species_balanced_df,
            on=[
                "query_state_code",
                "year",
            ],
            how="left",
        )
        .merge(
            observer_balanced_df,
            on=[
                "query_state_code",
                "year",
            ],
            how="left",
        )
    )

    annual_df["annual_target_usable"] = (
        annual_df["n_records"].ge(
            MIN_ANNUAL_TARGET_RECORDS
        )
        & annual_df["n_species"].ge(
            MIN_ANNUAL_TARGET_SPECIES
        )
        & annual_df["n_footprint_cells"].ge(
            MIN_ANNUAL_TARGET_CELLS
        )
    )

    annual_df.to_parquet(
        OUT_DIR / "annual_observation_footprint.parquet",
        index=False,
    )
    annual_df.to_csv(
        OUT_DIR / "annual_observation_footprint.csv",
        index=False,
    )

    print("\nANNUAL OBSERVATION FOOTPRINT")
    display(annual_df)

    # --------------------------------------------------------
    # 10. Stable target-group cells
    # --------------------------------------------------------
    stable_cell_rows = []

    for state_code, state_annual in annual_df.groupby(
        "query_state_code"
    ):
        usable_years = sorted(
            state_annual.loc[
                state_annual[
                    "annual_target_usable"
                ],
                "year",
            ].unique()
        )

        state_cell_year = cell_year_df[
            (cell_year_df["query_state_code"] == state_code)
            & cell_year_df["year"].isin(
                usable_years
            )
        ].copy()

        if not usable_years:
            stable_cell_rows.append(
                {
                    "state_code": state_code,
                    "n_usable_years": 0,
                    "n_all_cells": 0,
                    "n_stable_cells": 0,
                    "minimum_stable_years": 0,
                }
            )
            continue

        minimum_stable_years = math.ceil(
            STABLE_CELL_YEAR_FRACTION
            * len(usable_years)
        )

        early_years = set(usable_years[:4])
        late_years = set(usable_years[-4:])

        coverage = (
            state_cell_year.groupby(
                "footprint_cell"
            )["year"]
            .nunique()
            .rename("n_sampled_years")
        )

        early = (
            state_cell_year[
                state_cell_year["year"].isin(
                    early_years
                )
            ]
            .groupby("footprint_cell")["year"]
            .nunique()
            .rename("n_early_years")
        )

        late = (
            state_cell_year[
                state_cell_year["year"].isin(
                    late_years
                )
            ]
            .groupby("footprint_cell")["year"]
            .nunique()
            .rename("n_late_years")
        )

        cell_coverage = pd.concat(
            [coverage, early, late],
            axis=1,
        ).fillna(0)

        cell_coverage["stable"] = (
            cell_coverage[
                "n_sampled_years"
            ].ge(minimum_stable_years)
            & cell_coverage[
                "n_early_years"
            ].ge(1)
            & cell_coverage[
                "n_late_years"
            ].ge(1)
        )

        cell_coverage = (
            cell_coverage.reset_index()
        )
        cell_coverage["state_code"] = (
            state_code
        )

        cell_coverage.to_csv(
            OUT_DIR
            / f"{state_code}_stable_cell_coverage.csv",
            index=False,
        )

        stable_cell_rows.append(
            {
                "state_code": state_code,
                "n_usable_years": len(
                    usable_years
                ),
                "n_all_cells": int(
                    cell_coverage[
                        "footprint_cell"
                    ].nunique()
                ),
                "n_stable_cells": int(
                    cell_coverage["stable"].sum()
                ),
                "minimum_stable_years": (
                    minimum_stable_years
                ),
            }
        )

    stable_cell_summary_df = pd.DataFrame(
        stable_cell_rows
    )

    stable_cell_summary_df.to_csv(
        OUT_DIR / "stable_cell_summary.csv",
        index=False,
    )

    print("\nSTABLE CELL SUMMARY")
    display(stable_cell_summary_df)

    # --------------------------------------------------------
    # 11. Trend estimation
    # --------------------------------------------------------
    metric_map = {
        "pooled_mean_latitude": (
            "Pooled occurrence center"
        ),
        "equal_cell_mean_latitude": (
            "Equal-cell observation footprint"
        ),
        "species_balanced_mean_latitude": (
            "Species-balanced center"
        ),
        "observer_balanced_mean_latitude": (
            "Observer-balanced diagnostic"
        ),
    }

    trend_rows = []

    for state_code, state_group in annual_df.groupby(
        "query_state_code"
    ):
        usable = state_group[
            state_group["annual_target_usable"]
        ].copy()

        for metric, label in metric_map.items():
            fit = safe_fit(
                usable["year"],
                usable[metric],
            )

            trend_rows.append(
                {
                    "state_code": state_code,
                    "metric": metric,
                    "metric_label": label,
                    **fit,
                    "projected_median_species_error_km_decade": (
                        abs(
                            fit[
                                "trend_km_decade"
                            ]
                        )
                        * median_calibrated_sensitivity
                        if np.isfinite(
                            fit[
                                "trend_km_decade"
                            ]
                        )
                        else np.nan
                    ),
                }
            )

    trend_df = pd.DataFrame(
        trend_rows
    )

    trend_df.to_csv(
        OUT_DIR / "observation_footprint_trends.csv",
        index=False,
    )

    print("\nOBSERVATION FOOTPRINT TRENDS")
    display(trend_df)

    # --------------------------------------------------------
    # 12. Go / No-Go decision
    # --------------------------------------------------------
    adequate_species_by_state = (
        species_coverage_df.groupby(
            "state_code"
        )["adequate_coverage"]
        .sum()
        .to_dict()
    )

    usable_years_by_state = (
        annual_df.groupby(
            "query_state_code"
        )["annual_target_usable"]
        .sum()
        .to_dict()
    )

    stable_cells_by_state = (
        stable_cell_summary_df.set_index(
            "state_code"
        )["n_stable_cells"]
        .to_dict()
    )

    coverage_pass = all(
        adequate_species_by_state.get(
            state_code,
            0,
        )
        >= MIN_ADEQUATE_SPECIES_PER_STATE
        and usable_years_by_state.get(
            state_code,
            0,
        )
        >= MIN_TREND_YEARS
        and stable_cells_by_state.get(
            state_code,
            0,
        )
        >= MIN_STABLE_CELLS
        for state_code in STATE_CONFIG
    )

    core_trend_df = trend_df[
        trend_df["metric"].isin(
            [
                "equal_cell_mean_latitude",
                "species_balanced_mean_latitude",
            ]
        )
    ].copy()

    maximum_core_drift = (
        float(
            core_trend_df[
                "trend_km_decade"
            ].abs().max()
        )
        if not core_trend_df.empty
        else np.nan
    )

    drift_signal_pass = bool(
        np.isfinite(maximum_core_drift)
        and maximum_core_drift
        >= MIN_MATERIAL_CENTER_DRIFT_KM_DECADE
    )

    observer_field_eligible = bool(
        (
            field_audit_df[
                "recordedBy_completeness"
            ]
            >= 0.60
        ).all()
        and (
            field_audit_df[
                "n_unique_recordedBy"
            ]
            >= 100
        ).all()
    )

    if coverage_pass and drift_signal_pass:
        status = (
            "GO_FOR_REAL_GBIF_CORRECTION_AND_FIA_COMPARISON"
        )
        passed = True
        next_step = (
            "Apply the locked 0.5-degree correction to species "
            "trends and compare raw versus corrected iNaturalist "
            "estimates against repeated-plot FIA reference shifts."
        )
    elif coverage_pass:
        status = "REAL_OBSERVATION_DRIFT_WEAK_IN_PILOT"
        passed = False
        next_step = (
            "The data are sufficient, but target-group center "
            "movement is below the calibrated material range. "
            "Test another region or class of observations before "
            "building a full correction model."
        )
    else:
        status = "REAL_GBIF_PILOT_COVERAGE_INSUFFICIENT"
        passed = False
        next_step = (
            "Do not fit correction models yet. Expand the time "
            "range, species set, or geographic region based on the "
            "coverage table."
        )

    # --------------------------------------------------------
    # 13. Visual previews
    # --------------------------------------------------------
    for state_code in STATE_CONFIG:
        plot_df = annual_df[
            (annual_df["query_state_code"] == state_code)
            & annual_df["annual_target_usable"]
        ].sort_values("year")

        if plot_df.empty:
            continue

        plt.figure(figsize=(10, 5))
        plt.plot(
            plot_df["year"],
            plot_df[
                "equal_cell_mean_latitude"
            ],
            marker="o",
            label="Equal-cell footprint",
        )
        plt.plot(
            plot_df["year"],
            plot_df[
                "species_balanced_mean_latitude"
            ],
            marker="o",
            label="Species-balanced",
        )
        plt.plot(
            plot_df["year"],
            plot_df[
                "pooled_mean_latitude"
            ],
            marker="o",
            label="Pooled occurrences",
        )
        plt.xlabel("Year")
        plt.ylabel("Latitude center")
        plt.title(
            f"{state_code}: iNaturalist target-group "
            "observation footprint"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    coverage_plot_df = (
        species_coverage_df[
            species_coverage_df[
                "adequate_coverage"
            ]
        ]
        .groupby(
            [
                "fia_species_code",
                "gbif_accepted_name",
            ],
            as_index=False,
        )
        .agg(
            minimum_qualified_years=(
                "n_qualified_years",
                "min",
            ),
            total_records=(
                "total_records",
                "sum",
            ),
        )
        .sort_values(
            "total_records",
            ascending=False,
        )
        .head(20)
        .sort_values("total_records")
    )

    if not coverage_plot_df.empty:
        labels = coverage_plot_df[
            "gbif_accepted_name"
        ].fillna(
            coverage_plot_df[
                "fia_species_code"
            ].astype(str)
        )

        plt.figure(figsize=(10, 8))
        plt.barh(
            labels,
            coverage_plot_df[
                "total_records"
            ],
        )
        plt.xlabel(
            "Thinned iNaturalist records in WV + PA"
        )
        plt.ylabel("Species")
        plt.title(
            "Real-data pilot species coverage"
        )
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 14. README
    # --------------------------------------------------------
    trend_compact = (
        trend_df[
            [
                "state_code",
                "metric",
                "n_years",
                "trend_km_decade",
                "projected_median_species_error_km_decade",
            ]
        ]
        .round(4)
        .to_dict("records")
    )

    append_readme(
        f"""

## Real iNaturalist/GBIF observation-footprint pilot — {RUN_UTC}

### Hypothesis
The spatial center of opportunity-based observations for the fixed tree target
group may move over time, potentially biasing apparent species range shifts.

### Data source
- Dataset: iNaturalist Research-grade Observations
- GBIF dataset key: `{INAT_DATASET_KEY}`
- Dataset DOI: `{INAT_DATASET_DOI}`
- Access date: {ACCESS_DATE}
- Access method: GBIF occurrence search API
- Years: {START_YEAR}–{END_YEAR}
- States: {list(STATE_CONFIG)}
- Taxa: {len(resolved_taxonomy_df)}
- Record-level licence field retained: Yes

### Query filters
- country = US
- datasetKey = {INAT_DATASET_KEY}
- hasCoordinate = true
- hasGeospatialIssue = false
- occurrenceStatus = PRESENT
- basisOfRecord = HUMAN_OBSERVATION
- stateProvince selected separately for WV and PA
- taxonKey resolved through the current GBIF species matching service

### Data processing
- Raw records downloaded: {len(raw_df):,}
- Duplicate GBIF IDs removed: {duplicate_gbif_ids_removed:,}
- Records removed by coordinate/year/uncertainty checks:
  {records_removed_by_quality:,}
- Daily-spatial thinning resolution: {THINNING_RESOLUTION} degrees
- Records removed by thinning: {records_removed_by_thinning:,}
- Final thinned records: {len(thinned_df):,}
- Coordinate uncertainty threshold:
  <= {MAX_COORDINATE_UNCERTAINTY_M:,} m, with missing values retained and flagged
- Footprint grid: {FOOTPRINT_GRID_RESOLUTION} degrees
- GPU: not used

### Calibrated benchmark linkage
- Median species error per 1 km/decade observation-center drift:
  {median_calibrated_sensitivity:.4f}
- Maximum observed core target-group center drift:
  {maximum_core_drift} km/decade
- Material benchmark threshold:
  {MIN_MATERIAL_CENTER_DRIFT_KM_DECADE} km/decade

### Decision
- Coverage passed: {coverage_pass}
- Material center-drift signal passed: {drift_signal_pass}
- Observer field suitable for observer-level follow-up:
  {observer_field_eligible}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
The pooled, equal-cell and species-balanced centers describe the spatial footprint
of records for this target group. They do not isolate observer movement from real
species redistribution, species-composition changes, platform growth or changes
in identification behaviour. recordedBy values are not assumed to be stable
person identifiers. This search-API pilot does not provide a citable occurrence
download DOI; final analysis must be repeated using a registered GBIF download.
"""
    )

    # --------------------------------------------------------
    # 15. Compact output block
    # --------------------------------------------------------
    coverage_state_compact = []

    for state_code in STATE_CONFIG:
        coverage_state_compact.append(
            {
                "state": state_code,
                "adequate_species": int(
                    adequate_species_by_state.get(
                        state_code,
                        0,
                    )
                ),
                "usable_years": int(
                    usable_years_by_state.get(
                        state_code,
                        0,
                    )
                ),
                "stable_cells": int(
                    stable_cells_by_state.get(
                        state_code,
                        0,
                    )
                ),
            }
        )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(
        f"GBIF_DATASET_KEY: "
        f"{INAT_DATASET_KEY}"
    )
    print(
        f"GBIF_DATASET_DOI: "
        f"{INAT_DATASET_DOI}"
    )
    print(
        f"ACCESS_DATE: {ACCESS_DATE}"
    )
    print(
        f"N_RESOLVED_SPECIES: "
        f"{len(resolved_taxonomy_df)}"
    )
    print(
        f"TOTAL_QUERY_COUNT: "
        f"{total_query_records:,}"
    )
    print(
        f"PLANNED_QUERY_SEGMENTS: "
        f"{len(segment_df)}"
    )
    print(
        f"PLANNED_API_PAGES: "
        f"{planned_pages:,}"
    )
    print(
        f"RAW_RECORDS: {len(raw_df):,}"
    )
    print(
        f"DUPLICATE_GBIF_IDS_REMOVED: "
        f"{duplicate_gbif_ids_removed:,}"
    )
    print(
        f"QUALITY_FILTERED_RECORDS: "
        f"{len(clean_df):,}"
    )
    print(
        f"THINNED_RECORDS: "
        f"{len(thinned_df):,}"
    )
    print(
        f"RECORDS_REMOVED_BY_QUALITY: "
        f"{records_removed_by_quality:,}"
    )
    print(
        f"RECORDS_REMOVED_BY_THINNING: "
        f"{records_removed_by_thinning:,}"
    )
    print(
        "STATE_COVERAGE: "
        + json.dumps(
            coverage_state_compact,
            ensure_ascii=False,
        )
    )
    print(
        "FIELD_AUDIT: "
        + json.dumps(
            field_audit_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "FOOTPRINT_TRENDS: "
        + json.dumps(
            trend_compact,
            ensure_ascii=False,
        )
    )
    print(
        f"MEDIAN_CALIBRATED_SENSITIVITY: "
        f"{median_calibrated_sensitivity:.6f}"
    )
    print(
        f"MAX_CORE_CENTER_DRIFT_KM_DECADE: "
        f"{maximum_core_drift}"
    )
    print(
        f"COVERAGE_PASSED: "
        f"{coverage_pass}"
    )
    print(
        f"DRIFT_SIGNAL_PASSED: "
        f"{drift_signal_pass}"
    )
    print(
        f"OBSERVER_FIELD_ELIGIBLE: "
        f"{observer_field_eligible}"
    )
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except PilotStop as exc:
    print_stop_block(
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

    print_stop_block(
        "GBIF_HTTP_FAILED",
        f"HTTP {status_code} for {url}",
        (
            "Retry once. If it repeats, save this diagnostic summary "
            "without deleting cached outputs."
        ),
    )

except requests.RequestException as exc:
    print_stop_block(
        "GBIF_CONNECTION_FAILED",
        repr(exc),
        (
            "Confirm Kaggle Internet is enabled and retry. "
            "Completed final raw output will be reused."
        ),
    )

except Exception as exc:
    print_stop_block(
        "CELL08A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Save this diagnostic summary and the last displayed "
            "table. Do not rerun earlier FIA cells."
        ),
    )
