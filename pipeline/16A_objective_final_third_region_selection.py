
# Cell 16A — Objective selection of one final third confirmation region
#
# Current evidence
# ----------------
# Pennsylvania discovery:
# - cell_jaccard rho = 0.718
# - abs_log_observer_growth rho = 0.654
#
# Virginia independent confirmation:
# - directions remained positive;
# - effect sizes were weak;
# - neither candidate met the locked confirmation criteria.
#
# Because Virginia retained only 11 species, one final independent region is
# justified. This cell selects that region using coverage only. It does not
# calculate correction gains or inspect the two candidate-feature associations.
#
# Candidate eastern-forest states:
# - North Carolina
# - Tennessee
# - Ohio
# - New York
# - Maryland
#
# Final stop rule
# ---------------
# After the selected third-region confirmation:
# - confirmed candidate(s): proceed to cross-region synthesis;
# - no confirmed candidate: permanently close the sensitivity-index route.
#
# This cell downloads only FIA PLOT files when missing and queries GBIF counts.
# It does not download occurrence coordinates or FIA TREE files.

from pathlib import Path
from datetime import datetime, timezone
import json
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
pd.set_option("display.max_columns", 180)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 300)

# ============================================================
# Locked configuration
# ============================================================
CANDIDATE_STATES = {
    "NC": "North Carolina",
    "TN": "Tennessee",
    "OH": "Ohio",
    "NY": "New York",
    "MD": "Maryland",
}

EARLY_YEARS = list(range(2013, 2019))
LATE_YEARS = list(range(2020, 2026))
EARLY_MID_YEAR = float(np.mean(EARLY_YEARS))
LATE_MID_YEAR = float(np.mean(LATE_YEARS))

MIN_RECORDS_PER_TAXON_WINDOW = 30
MIN_BENCHMARK_ELIGIBLE_TAXA = 12
MIN_EFFORT_ELIGIBLE_TAXA = 20
MIN_MATCHED_FIA_PLOTS = 500
MAX_ESTIMATED_DOWNLOAD_RECORDS = 300_000

PAGE_SIZE = 300
REQUEST_PAUSE_SECONDS = 0.03
REUSE_COUNT_CACHE = True

INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
INAT_DATASET_DOI = "10.15468/ab3s5x"
GBIF_API = "https://api.gbif.org/v1"

FIA_PLOT_URL_TEMPLATE = (
    "https://apps.fs.usda.gov/fia/datamart/CSV/{state}_PLOT.csv"
)

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PILOT_DIR = BASE_DIR / "derived" / "gbif_inaturalist_real_pilot"
PA_AUDIT_DIR = BASE_DIR / "derived" / "pa_expanded_tree_taxon_audit"

OUT_DIR = BASE_DIR / "derived" / "third_region_selection_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_TAXONOMY_PATH = (
    PILOT_DIR / "gbif_taxonomy_resolution.csv"
)
EFFORT_TAXA_PATH = (
    PA_AUDIT_DIR / "selected_expanded_tree_taxa.csv"
)
COUNT_CACHE_PATH = (
    OUT_DIR / "candidate_state_taxon_counts.csv"
)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)
ACCESS_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

print("RUN_UTC:", RUN_UTC)
print("CANDIDATE_STATES:", CANDIDATE_STATES)
print("EARLY_YEARS:", EARLY_YEARS)
print("LATE_YEARS:", LATE_YEARS)
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")


# ============================================================
# Helpers
# ============================================================
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
        pool_connections=12,
        pool_maxsize=12,
    )

    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "TemporalObservationDriftResearch/0.1 "
                "(final third-region coverage selection)"
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


def occurrence_count(
    state_name,
    taxon_key,
    start_year,
    end_year,
):
    payload = api_get(
        "/occurrence/search",
        {
            "datasetKey": INAT_DATASET_KEY,
            "country": "US",
            "stateProvince": state_name,
            "taxonKey": int(taxon_key),
            "year": f"{int(start_year)},{int(end_year)}",
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "occurrenceStatus": "PRESENT",
            "basisOfRecord": "HUMAN_OBSERVATION",
            "limit": 0,
        },
    )
    return int(payload.get("count", 0))


def download_with_progress(url, output_path):
    if output_path.exists() and output_path.stat().st_size > 0:
        return "reused"

    response = SESSION.get(
        url,
        stream=True,
        timeout=(30, 300),
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
        pd.to_numeric(series, errors="coerce")
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
        normalize_integer_component(frame[state_col])
        + "_"
        + normalize_integer_component(frame[unit_col])
        + "_"
        + normalize_integer_component(frame[county_col])
        + "_"
        + normalize_integer_component(frame[plot_col])
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


def audit_fia_plot_pairs(state_code):
    plot_path = RAW_DIR / f"{state_code}_PLOT.csv"

    columns = pd.read_csv(
        plot_path,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    plot_key_col = resolve_column(
        columns,
        ["CN", "PLOT_CN"],
    )
    year_col = resolve_column(
        columns,
        ["INVYR", "INVENTORY_YEAR"],
    )
    latitude_col = resolve_column(
        columns,
        ["LAT", "LATITUDE"],
    )
    longitude_col = resolve_column(
        columns,
        ["LON", "LONGITUDE"],
    )
    status_col = resolve_column(
        columns,
        ["PLOT_STATUS_CD", "PLOT_STATUS"],
    )
    state_col = resolve_column(
        columns,
        ["STATECD", "STATE_CODE"],
    )
    unit_col = resolve_column(
        columns,
        ["UNITCD", "UNIT_CODE"],
    )
    county_col = resolve_column(
        columns,
        ["COUNTYCD", "COUNTY_CODE"],
    )
    plot_number_col = resolve_column(
        columns,
        ["PLOT", "PLOT_NUMBER", "PLOT_NBR"],
    )

    required = [
        plot_key_col,
        year_col,
        latitude_col,
        longitude_col,
        state_col,
        unit_col,
        county_col,
        plot_number_col,
    ]

    if not all(required):
        raise ValueError(
            f"{state_code}: required PLOT identity fields were not resolved."
        )

    usecols = list(
        dict.fromkeys(
            [
                plot_key_col,
                year_col,
                latitude_col,
                longitude_col,
                state_col,
                unit_col,
                county_col,
                plot_number_col,
            ]
            + ([status_col] if status_col else [])
        )
    )

    frame = pd.read_csv(
        plot_path,
        usecols=usecols,
        low_memory=False,
        encoding_errors="replace",
        dtype={plot_key_col: "string"},
    )

    frame["physical_plot_id"] = build_physical_plot_id(
        frame,
        state_col,
        unit_col,
        county_col,
        plot_number_col,
    )

    rename_map = {
        plot_key_col: "plot_key",
        year_col: "year",
        latitude_col: "latitude",
        longitude_col: "longitude",
    }

    if status_col:
        rename_map[status_col] = "plot_status"

    frame = frame.rename(columns=rename_map)

    if "plot_status" not in frame.columns:
        frame["plot_status"] = np.nan

    for column in [
        "year",
        "latitude",
        "longitude",
        "plot_status",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["plot_key"] = (
        frame["plot_key"]
        .astype("string")
        .str.strip()
    )

    valid = (
        frame["year"].isin(
            EARLY_YEARS + LATE_YEARS
        )
        & frame["latitude"].between(24, 50)
        & frame["longitude"].between(-130, -60)
        & frame["plot_key"].notna()
        & frame["physical_plot_id"].notna()
    )

    if status_col:
        valid &= frame["plot_status"].eq(1)

    frame = (
        frame.loc[
            valid,
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

    frame["year"] = frame["year"].astype(int)

    early = closest_measurement_per_plot(
        frame,
        EARLY_YEARS,
        EARLY_MID_YEAR,
    )
    late = closest_measurement_per_plot(
        frame,
        LATE_YEARS,
        LATE_MID_YEAR,
    )

    pairs = (
        early[
            [
                "physical_plot_id",
                "plot_key",
                "year",
            ]
        ]
        .rename(
            columns={
                "plot_key": "early_plot_key",
                "year": "early_year",
            }
        )
        .merge(
            late[
                [
                    "physical_plot_id",
                    "plot_key",
                    "year",
                ]
            ].rename(
                columns={
                    "plot_key": "late_plot_key",
                    "year": "late_year",
                }
            ),
            on="physical_plot_id",
            how="inner",
            validate="one_to_one",
        )
    )

    pairs = pairs[
        pairs["late_year"] > pairs["early_year"]
    ].copy()

    return {
        "state_code": state_code,
        "fia_plot_rows_in_windows": len(frame),
        "fia_early_physical_plots": int(
            early["physical_plot_id"].nunique()
        ),
        "fia_late_physical_plots": int(
            late["physical_plot_id"].nunique()
        ),
        "fia_matched_physical_plots": len(pairs),
        "fia_median_interval_years": (
            float(
                (
                    pairs["late_year"]
                    - pairs["early_year"]
                ).median()
            )
            if len(pairs)
            else np.nan
        ),
    }


def print_failure(status, error, next_step):
    append_readme(
        f"""

## Final third-region selection failure — {RUN_UTC}
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


# ============================================================
# Main
# ============================================================
try:
    # --------------------------------------------------------
    # 1. Load locked taxon sets
    # --------------------------------------------------------
    required_paths = [
        BENCHMARK_TAXONOMY_PATH,
        EFFORT_TAXA_PATH,
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

    benchmark_df = pd.read_csv(
        BENCHMARK_TAXONOMY_PATH
    )
    effort_df = pd.read_csv(
        EFFORT_TAXA_PATH
    )

    benchmark_required = [
        "fia_species_code",
        "original_scientific_name",
        "gbif_taxon_key",
        "gbif_accepted_name",
        "taxon_resolution_pass",
    ]
    effort_required = [
        "species_code",
        "fia_canonical_name",
        "gbif_taxon_key",
        "gbif_accepted_name",
    ]

    missing_benchmark_columns = [
        column
        for column in benchmark_required
        if column not in benchmark_df.columns
    ]
    missing_effort_columns = [
        column
        for column in effort_required
        if column not in effort_df.columns
    ]

    if missing_benchmark_columns:
        raise KeyError(
            "Benchmark taxonomy is missing: "
            + " | ".join(missing_benchmark_columns)
        )

    if missing_effort_columns:
        raise KeyError(
            "Effort-frame taxa are missing: "
            + " | ".join(missing_effort_columns)
        )

    benchmark_df["gbif_taxon_key"] = pd.to_numeric(
        benchmark_df["gbif_taxon_key"],
        errors="coerce",
    )

    benchmark_pass = (
        benchmark_df["taxon_resolution_pass"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    benchmark_df = benchmark_df[
        benchmark_pass
        & benchmark_df["gbif_taxon_key"].notna()
    ].copy()

    benchmark_df["gbif_taxon_key"] = (
        benchmark_df["gbif_taxon_key"].astype(int)
    )

    effort_df["gbif_taxon_key"] = pd.to_numeric(
        effort_df["gbif_taxon_key"],
        errors="coerce",
    )

    effort_df = effort_df.dropna(
        subset=["gbif_taxon_key"]
    ).copy()

    effort_df["gbif_taxon_key"] = (
        effort_df["gbif_taxon_key"].astype(int)
    )

    benchmark_taxon_keys = set(
        benchmark_df["gbif_taxon_key"].tolist()
    )
    effort_taxon_keys = set(
        effort_df["gbif_taxon_key"].tolist()
    )

    master_rows = []

    for taxon_key in sorted(
        benchmark_taxon_keys | effort_taxon_keys
    ):
        benchmark_matches = benchmark_df[
            benchmark_df["gbif_taxon_key"] == taxon_key
        ]
        effort_matches = effort_df[
            effort_df["gbif_taxon_key"] == taxon_key
        ]

        accepted_name = None

        if not benchmark_matches.empty:
            accepted_name = benchmark_matches.iloc[0][
                "gbif_accepted_name"
            ]

        if (
            (accepted_name is None or pd.isna(accepted_name))
            and not effort_matches.empty
        ):
            accepted_name = effort_matches.iloc[0][
                "gbif_accepted_name"
            ]

        master_rows.append(
            {
                "gbif_taxon_key": int(taxon_key),
                "gbif_accepted_name": accepted_name,
                "is_benchmark_taxon": (
                    taxon_key in benchmark_taxon_keys
                ),
                "is_effort_frame_taxon": (
                    taxon_key in effort_taxon_keys
                ),
            }
        )

    master_taxa_df = pd.DataFrame(master_rows)

    master_taxa_df.to_csv(
        OUT_DIR / "locked_master_taxa.csv",
        index=False,
    )

    print("\nLOCKED TAXON SETS")
    display(
        pd.DataFrame(
            [
                {
                    "benchmark_taxa": len(
                        benchmark_taxon_keys
                    ),
                    "effort_frame_taxa": len(
                        effort_taxon_keys
                    ),
                    "unique_union_taxa": len(
                        master_taxa_df
                    ),
                }
            ]
        )
    )

    # --------------------------------------------------------
    # 2. Download missing candidate-state FIA PLOT files
    # --------------------------------------------------------
    plot_download_rows = []

    for state_code in tqdm(
        CANDIDATE_STATES,
        desc="Preparing candidate-state FIA PLOT files",
        unit="state",
    ):
        output_path = RAW_DIR / f"{state_code}_PLOT.csv"
        url = FIA_PLOT_URL_TEMPLATE.format(
            state=state_code
        )

        download_status = download_with_progress(
            url,
            output_path,
        )

        plot_download_rows.append(
            {
                "state_code": state_code,
                "path": str(output_path),
                "status": download_status,
                "size_bytes": output_path.stat().st_size,
            }
        )

    plot_download_df = pd.DataFrame(
        plot_download_rows
    )

    plot_download_df.to_csv(
        OUT_DIR / "candidate_plot_downloads.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Incremental GBIF count cache
    # --------------------------------------------------------
    if (
        REUSE_COUNT_CACHE
        and COUNT_CACHE_PATH.exists()
    ):
        count_df = pd.read_csv(
            COUNT_CACHE_PATH
        )
    else:
        count_df = pd.DataFrame(
            columns=[
                "state_code",
                "state_name",
                "gbif_taxon_key",
                "gbif_accepted_name",
                "period",
                "start_year",
                "end_year",
                "count",
            ]
        )

    completed_keys = set()

    if not count_df.empty:
        count_df["gbif_taxon_key"] = pd.to_numeric(
            count_df["gbif_taxon_key"],
            errors="coerce",
        )

        count_df = count_df.dropna(
            subset=[
                "state_code",
                "gbif_taxon_key",
                "period",
            ]
        ).copy()

        count_df["gbif_taxon_key"] = (
            count_df["gbif_taxon_key"].astype(int)
        )

        completed_keys = set(
            zip(
                count_df["state_code"],
                count_df["gbif_taxon_key"],
                count_df["period"],
            )
        )

    tasks = []

    for state_code, state_name in CANDIDATE_STATES.items():
        for row in master_taxa_df.to_dict("records"):
            for period, years in [
                ("early", EARLY_YEARS),
                ("late", LATE_YEARS),
            ]:
                key = (
                    state_code,
                    int(row["gbif_taxon_key"]),
                    period,
                )

                if key not in completed_keys:
                    tasks.append(
                        {
                            "state_code": state_code,
                            "state_name": state_name,
                            "gbif_taxon_key": int(
                                row["gbif_taxon_key"]
                            ),
                            "gbif_accepted_name": row[
                                "gbif_accepted_name"
                            ],
                            "period": period,
                            "start_year": min(years),
                            "end_year": max(years),
                        }
                    )

    new_rows = []

    for task_index, task in enumerate(
        tqdm(
            tasks,
            desc="Counting third-region iNaturalist records",
            unit="state-taxon-period",
        ),
        start=1,
    ):
        value = occurrence_count(
            state_name=task["state_name"],
            taxon_key=task["gbif_taxon_key"],
            start_year=task["start_year"],
            end_year=task["end_year"],
        )

        new_rows.append(
            {
                **task,
                "count": value,
            }
        )

        if (
            task_index % 20 == 0
            or task_index == len(tasks)
        ):
            count_df = pd.concat(
                [
                    count_df,
                    pd.DataFrame(new_rows),
                ],
                ignore_index=True,
            )

            count_df = count_df.drop_duplicates(
                [
                    "state_code",
                    "gbif_taxon_key",
                    "period",
                ],
                keep="last",
            )

            count_df.to_csv(
                COUNT_CACHE_PATH,
                index=False,
            )

            new_rows.clear()

    count_df["count"] = pd.to_numeric(
        count_df["count"],
        errors="coerce",
    ).fillna(0).astype(int)

    # --------------------------------------------------------
    # 4. State-taxon early/late coverage
    # --------------------------------------------------------
    count_wide_df = (
        count_df.pivot_table(
            index=[
                "state_code",
                "state_name",
                "gbif_taxon_key",
                "gbif_accepted_name",
            ],
            columns="period",
            values="count",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    for period in ["early", "late"]:
        if period not in count_wide_df.columns:
            count_wide_df[period] = 0

    count_wide_df = count_wide_df.merge(
        master_taxa_df,
        on=[
            "gbif_taxon_key",
            "gbif_accepted_name",
        ],
        how="left",
    )

    count_wide_df["minimum_window_count"] = (
        count_wide_df[
            ["early", "late"]
        ].min(axis=1)
    )

    count_wide_df["total_window_count"] = (
        count_wide_df["early"]
        + count_wide_df["late"]
    )

    count_wide_df["window_coverage_pass"] = (
        count_wide_df["early"]
        >= MIN_RECORDS_PER_TAXON_WINDOW
    ) & (
        count_wide_df["late"]
        >= MIN_RECORDS_PER_TAXON_WINDOW
    )

    count_wide_df.to_csv(
        OUT_DIR / "candidate_state_taxon_coverage.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Exact repeated FIA physical-plot coverage
    # --------------------------------------------------------
    fia_rows = []

    for state_code in tqdm(
        CANDIDATE_STATES,
        desc="Auditing candidate-state FIA repeated plots",
        unit="state",
    ):
        fia_rows.append(
            audit_fia_plot_pairs(
                state_code
            )
        )

    fia_summary_df = pd.DataFrame(
        fia_rows
    )

    fia_summary_df.to_csv(
        OUT_DIR / "candidate_state_fia_plot_summary.csv",
        index=False,
    )

    print("\nCANDIDATE-STATE FIA REPEATED PLOTS")
    display(fia_summary_df)

    # --------------------------------------------------------
    # 6. Objective recommendation
    # --------------------------------------------------------
    state_rows = []

    for state_code, state_name in CANDIDATE_STATES.items():
        state_counts = count_wide_df[
            count_wide_df["state_code"] == state_code
        ].copy()

        benchmark_counts = state_counts[
            state_counts["is_benchmark_taxon"].fillna(False)
        ].copy()

        effort_counts = state_counts[
            state_counts["is_effort_frame_taxon"].fillna(False)
        ].copy()

        benchmark_eligible = int(
            benchmark_counts[
                "window_coverage_pass"
            ].sum()
        )
        effort_eligible = int(
            effort_counts[
                "window_coverage_pass"
            ].sum()
        )

        estimated_records = int(
            effort_counts.loc[
                effort_counts["window_coverage_pass"],
                "total_window_count",
            ].sum()
        )

        estimated_pages = int(
            np.ceil(
                effort_counts.loc[
                    effort_counts["window_coverage_pass"],
                    "early",
                ]
                / PAGE_SIZE
            ).sum()
            + np.ceil(
                effort_counts.loc[
                    effort_counts["window_coverage_pass"],
                    "late",
                ]
                / PAGE_SIZE
            ).sum()
        )

        fia_row = fia_summary_df[
            fia_summary_df["state_code"] == state_code
        ].iloc[0]

        criteria_pass = bool(
            benchmark_eligible
            >= MIN_BENCHMARK_ELIGIBLE_TAXA
            and effort_eligible
            >= MIN_EFFORT_ELIGIBLE_TAXA
            and int(
                fia_row[
                    "fia_matched_physical_plots"
                ]
            )
            >= MIN_MATCHED_FIA_PLOTS
            and estimated_records
            <= MAX_ESTIMATED_DOWNLOAD_RECORDS
        )

        state_rows.append(
            {
                "state_code": state_code,
                "state_name": state_name,
                "benchmark_eligible_taxa": benchmark_eligible,
                "effort_frame_eligible_taxa": effort_eligible,
                "estimated_effort_records": estimated_records,
                "estimated_api_pages": estimated_pages,
                "fia_matched_physical_plots": int(
                    fia_row[
                        "fia_matched_physical_plots"
                    ]
                ),
                "fia_median_interval_years": (
                    fia_row[
                        "fia_median_interval_years"
                    ]
                ),
                "criteria_pass": criteria_pass,
            }
        )

    state_summary_df = pd.DataFrame(
        state_rows
    )

    state_summary_df = state_summary_df.sort_values(
        [
            "criteria_pass",
            "benchmark_eligible_taxa",
            "effort_frame_eligible_taxa",
            "fia_matched_physical_plots",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    state_summary_df["selection_rank"] = np.arange(
        1,
        len(state_summary_df) + 1,
    )

    recommended_row = state_summary_df.iloc[0]
    recommended_state = recommended_row["state_code"]

    state_summary_df.to_csv(
        OUT_DIR / "third_region_selection_summary.csv",
        index=False,
    )

    print("\nFINAL THIRD-REGION SELECTION SUMMARY")
    display(state_summary_df)

    recommended_species_df = count_wide_df[
        (count_wide_df["state_code"] == recommended_state)
        & count_wide_df["is_benchmark_taxon"].fillna(False)
    ].sort_values(
        [
            "window_coverage_pass",
            "minimum_window_count",
        ],
        ascending=[
            False,
            False,
        ],
    )

    recommended_species_df.to_csv(
        OUT_DIR
        / f"{recommended_state}_benchmark_taxon_coverage.csv",
        index=False,
    )

    print(
        f"\n{recommended_state} LOCKED BENCHMARK SPECIES COVERAGE"
    )
    display(recommended_species_df)

    passed = bool(
        recommended_row["criteria_pass"]
    )

    if passed:
        status = "GO_FOR_FINAL_THIRD_REGION_CONFIRMATION"
        next_step = (
            f"Use {recommended_state} as the final independent region. "
            "Download only its eligible locked taxa, build the same-window "
            "FIA reference, and test only cell_jaccard and "
            "abs_log_observer_growth with the unchanged VA criteria. "
            "If neither confirms, permanently close the index route."
        )
    else:
        status = "NO_FINAL_THIRD_REGION_MEETS_POWER_CRITERIA"
        next_step = (
            "No candidate region has enough locked coverage for a final "
            "independent confirmation. Do not weaken thresholds. Close the "
            "sensitivity-index route and write the correction-heterogeneity "
            "and failed-transportability paper."
        )

    # --------------------------------------------------------
    # 7. Visual previews
    # --------------------------------------------------------
    plot_df = state_summary_df.sort_values(
        "benchmark_eligible_taxa"
    )

    plt.figure(figsize=(9, 5))
    plt.barh(
        plot_df["state_code"],
        plot_df["benchmark_eligible_taxa"],
    )
    plt.axvline(
        MIN_BENCHMARK_ELIGIBLE_TAXA,
        linestyle="--",
        label="Required benchmark taxa",
    )
    plt.xlabel(
        "Benchmark species with >=30 records in both windows"
    )
    plt.ylabel("Candidate state")
    plt.title(
        "Coverage for the final independent confirmation"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    fia_plot_df = state_summary_df.sort_values(
        "fia_matched_physical_plots"
    )

    plt.figure(figsize=(9, 5))
    plt.barh(
        fia_plot_df["state_code"],
        fia_plot_df["fia_matched_physical_plots"],
    )
    plt.axvline(
        MIN_MATCHED_FIA_PLOTS,
        linestyle="--",
        label="Required repeated plots",
    )
    plt.xlabel(
        "Same-window matched FIA physical plots"
    )
    plt.ylabel("Candidate state")
    plt.title(
        "Standardized reference coverage"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 8. README
    # --------------------------------------------------------
    append_readme(
        f"""

## Final third-region selection audit — {RUN_UTC}

### Reason
Virginia preserved the Pennsylvania candidate directions but did not confirm
either candidate. Because only 11 Virginia species were available, one final
independent region is allowed before the sensitivity-index route is closed.

### Candidate states
- North Carolina
- Tennessee
- Ohio
- New York
- Maryland

### Locked selection criteria
- Benchmark taxa with >= {MIN_RECORDS_PER_TAXON_WINDOW} records in both windows:
  >= {MIN_BENCHMARK_ELIGIBLE_TAXA}
- Effort-frame taxa meeting the same threshold:
  >= {MIN_EFFORT_ELIGIBLE_TAXA}
- Same-window matched FIA physical plots:
  >= {MIN_MATCHED_FIA_PLOTS}
- Estimated coordinate records:
  <= {MAX_ESTIMATED_DOWNLOAD_RECORDS:,}

### Windows
- Early: {EARLY_YEARS}
- Late: {LATE_YEARS}

### Data
- iNaturalist Research-grade observations through GBIF
- Dataset key: {INAT_DATASET_KEY}
- Dataset DOI: {INAT_DATASET_DOI}
- Access date: {ACCESS_DATE}
- Occurrence coordinates downloaded in this cell: No
- FIA TREE files downloaded in this cell: No
- GPU: not used

### Result
- Recommended state: {recommended_state}
- Criteria passed: {passed}
- Status: **{status}**
- Next step: {next_step}

### Final stop rule
If neither locked candidate confirms in the selected third region, no additional
regions, windows or features will be searched for the sensitivity index.
"""
    )

    # --------------------------------------------------------
    # 9. Compact output
    # --------------------------------------------------------
    compact_states = (
        state_summary_df.round(6).to_dict(
            "records"
        )
    )

    recommended_compact = (
        recommended_species_df[
            [
                "gbif_taxon_key",
                "gbif_accepted_name",
                "early",
                "late",
                "minimum_window_count",
                "window_coverage_pass",
            ]
        ]
        .to_dict("records")
    )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(
        f"LOCKED_BENCHMARK_TAXA: "
        f"{len(benchmark_taxon_keys)}"
    )
    print(
        f"LOCKED_EFFORT_FRAME_TAXA: "
        f"{len(effort_taxon_keys)}"
    )
    print(
        "STATE_SELECTION_RESULTS: "
        + json.dumps(
            compact_states,
            ensure_ascii=False,
        )
    )
    print(
        f"RECOMMENDED_STATE: "
        f"{recommended_state}"
    )
    print(
        "RECOMMENDED_STATE_BENCHMARK_COVERAGE: "
        + json.dumps(
            recommended_compact,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_CRITERIA: >=12 benchmark taxa, >=20 effort-frame "
        "taxa, >=500 same-window FIA physical plots, and estimated "
        "coordinate records <=300,000"
    )
    print(
        "FINAL_STOP_RULE: If neither locked candidate confirms in "
        "the selected third region, permanently close the "
        "sensitivity-index route."
    )
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

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

    print_failure(
        "HTTP_FAILED",
        f"HTTP {status_code} for {url}",
        (
            "Retry once. Existing PLOT files and completed count queries "
            "will be reused."
        ),
    )

except requests.RequestException as exc:
    print_failure(
        "CONNECTION_FAILED",
        repr(exc),
        (
            "Confirm Kaggle Internet is enabled and rerun. Completed "
            "downloads and count queries will be reused."
        ),
    )

except Exception as exc:
    print_failure(
        "CELL16A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Save this diagnostic summary and the last displayed table. "
            "Do not download occurrence coordinates yet."
        ),
    )
