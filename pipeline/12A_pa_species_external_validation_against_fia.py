
# Cell 12A — PA species-level external validation against same-window repeated FIA plots
#
# Core question
# -------------
# Does the locked 0.5-degree observation-effort correction make real
# iNaturalist tree-shift estimates more consistent with standardized FIA
# repeated-physical-plot estimates over the same early and late windows?
#
# Locked ecological species set:
# - the original 20 benchmark species from the semi-synthetic experiment
#
# Observation-effort frame:
# - all 33 expanded PA tree taxa from Cell 10B
#
# Time windows:
# - early: 2013–2018
# - late : 2020–2025
#
# iNaturalist estimands:
# 1. raw record-weighted center shift
# 2. equal occupied-cell center shift
# 3. target-group-effort standardized shift on a fixed cell frame (PRIMARY)
# 4. equal-observer center shift (secondary diagnostic)
#
# FIA reference:
# - exact same physical plots sampled in both windows
# - one species presence per physical plot and period
# - no tree-count abundance weighting
#
# Success:
# - enough species for comparison;
# - primary correction reduces median absolute error versus FIA by >=20%;
# - bootstrap probability of lower median error >=90%;
# - direction agreement or rank correlation is not worsened.
#
# This is a trend validation, not climate attribution.

from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 220)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 320)

# ============================================================
# Configuration
# ============================================================
STATE = "PA"

EARLY_YEARS = list(range(2013, 2019))
LATE_YEARS = list(range(2020, 2026))
EARLY_MID_YEAR = float(np.mean(EARLY_YEARS))
LATE_MID_YEAR = float(np.mean(LATE_YEARS))

PRIMARY_GRID_RESOLUTION = 0.50
SENSITIVITY_GRID_RESOLUTIONS = [0.25, 0.50, 1.00]

MIN_TARGET_EFFORT_PER_CELL_PERIOD = 10
MIN_INAT_RECORDS_PER_SPECIES_PERIOD = 30
MIN_INAT_POSITIVE_CELLS_PER_SPECIES_PERIOD = 3
MIN_INAT_OBSERVERS_PER_SPECIES_PERIOD = 10

MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD = 15
MIN_MATCHED_PHYSICAL_PLOTS = 500

MIN_EVALUATION_SPECIES = 10
TARGET_MEDIAN_ERROR_REDUCTION = 0.20
TARGET_BOOTSTRAP_IMPROVEMENT_PROBABILITY = 0.90

TREE_CHUNK_SIZE = 250_000
N_SPECIES_BOOTSTRAP = 1000
RANDOM_SEED = 20260710

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
RAW_DIR = BASE_DIR / "raw"
BENCHMARK_DIR = BASE_DIR / "derived" / "semi_synthetic_observation_drift"
EXPANDED_DIR = (
    BASE_DIR
    / "derived"
    / "pa_expanded_tree_observer_decomposition"
)
OUT_DIR = (
    BASE_DIR
    / "derived"
    / "pa_species_external_validation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PA_PLOT_CSV = RAW_DIR / "PA_PLOT.csv"
PA_TREE_CSV = RAW_DIR / "PA_TREE.csv"

BENCHMARK_SPECIES_CSV = (
    BENCHMARK_DIR / "selected_benchmark_species.csv"
)
EXPANDED_INAT_PATH = (
    EXPANDED_DIR
    / "expanded_pa_tree_thinned_occurrences.parquet"
)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("STATE:", STATE)
print("EARLY_YEARS:", EARLY_YEARS)
print("LATE_YEARS:", LATE_YEARS)
print("PRIMARY_GRID_RESOLUTION:", PRIMARY_GRID_RESOLUTION)
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


def resolve_column(columns, candidates):
    lookup = {str(column).upper(): str(column) for column in columns}
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


def safe_spearman(x, y):
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if (
        len(frame) < 3
        or frame["x"].nunique() < 2
        or frame["y"].nunique() < 2
    ):
        return np.nan

    return float(
        spearmanr(frame["x"], frame["y"]).statistic
    )


def direction_agreement(estimate, reference, minimum_reference=1.0):
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)

    valid = (
        np.isfinite(estimate)
        & np.isfinite(reference)
        & (np.abs(reference) >= minimum_reference)
    )

    if not valid.any():
        return np.nan

    return float(
        np.mean(
            np.sign(estimate[valid])
            == np.sign(reference[valid])
        )
    )


def closest_measurement_per_plot(
    plot_df,
    years,
    midpoint,
):
    subset = plot_df[
        plot_df["year"].isin(years)
    ].copy()

    subset["distance_to_midpoint"] = (
        subset["year"] - midpoint
    ).abs()

    subset = (
        subset.sort_values(
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

    return subset


def build_inat_estimands(
    all_tree_df,
    benchmark_species_df,
    resolution,
):
    work = all_tree_df.copy()

    work["grid_lat_index"] = np.floor(
        work["decimalLatitude"] / resolution
    ).astype("int32")
    work["grid_lon_index"] = np.floor(
        work["decimalLongitude"] / resolution
    ).astype("int32")
    work["grid_id"] = (
        work["grid_lat_index"].astype(str)
        + "_"
        + work["grid_lon_index"].astype(str)
    )

    # Fixed target-group effort frame.
    effort_df = (
        work.groupby(
            ["period", "grid_id"],
            as_index=False,
        )
        .agg(
            target_effort_records=("gbifID", "size"),
            cell_latitude=("decimalLatitude", "mean"),
        )
    )

    effort_wide = (
        effort_df.pivot(
            index="grid_id",
            columns="period",
            values="target_effort_records",
        )
        .fillna(0)
    )

    for period in ["early", "late"]:
        if period not in effort_wide.columns:
            effort_wide[period] = 0

    stable_cells = set(
        effort_wide[
            (effort_wide["early"] >= MIN_TARGET_EFFORT_PER_CELL_PERIOD)
            & (effort_wide["late"] >= MIN_TARGET_EFFORT_PER_CELL_PERIOD)
        ].index
    )

    pooled_cell_latitude = (
        work[
            work["grid_id"].isin(stable_cells)
        ]
        .groupby("grid_id")["decimalLatitude"]
        .mean()
        .to_dict()
    )

    period_total_records = (
        work.groupby("period")["gbifID"].size()
    )

    stable_record_fraction = {}

    for period in ["early", "late"]:
        period_rows = work["period"].eq(period)
        stable_rows = (
            period_rows
            & work["grid_id"].isin(stable_cells)
        )

        stable_record_fraction[period] = (
            float(stable_rows.sum() / period_rows.sum())
            if period_rows.sum() > 0
            else np.nan
        )

    benchmark_codes = set(
        benchmark_species_df["species_code"]
        .astype(int)
        .tolist()
    )

    species_df = work[
        work["fia_species_code"].isin(
            benchmark_codes
        )
    ].copy()

    raw_period_df = (
        species_df.groupby(
            ["fia_species_code", "period"],
            as_index=False,
        )
        .agg(
            raw_n_records=("gbifID", "size"),
            raw_mean_latitude=("decimalLatitude", "mean"),
            raw_n_cells=("grid_id", "nunique"),
            raw_n_observers=("observer_label", "nunique"),
        )
    )

    raw_wide = raw_period_df.pivot(
        index="fia_species_code",
        columns="period",
        values=[
            "raw_n_records",
            "raw_mean_latitude",
            "raw_n_cells",
            "raw_n_observers",
        ],
    )
    raw_wide.columns = [
        f"{metric}_{period}"
        for metric, period in raw_wide.columns
    ]
    raw_wide = raw_wide.reset_index()

    # Equal occupied-cell center.
    species_cell_df = (
        species_df.groupby(
            [
                "fia_species_code",
                "period",
                "grid_id",
            ],
            as_index=False,
        )
        .agg(
            species_cell_records=("gbifID", "size"),
            species_cell_latitude=("decimalLatitude", "mean"),
        )
    )

    equal_cell_period_df = (
        species_cell_df.groupby(
            ["fia_species_code", "period"],
            as_index=False,
        )
        .agg(
            equal_cell_mean_latitude=(
                "species_cell_latitude",
                "mean",
            ),
            equal_cell_positive_cells=(
                "grid_id",
                "nunique",
            ),
        )
    )

    equal_cell_wide = equal_cell_period_df.pivot(
        index="fia_species_code",
        columns="period",
        values=[
            "equal_cell_mean_latitude",
            "equal_cell_positive_cells",
        ],
    )
    equal_cell_wide.columns = [
        f"{metric}_{period}"
        for metric, period in equal_cell_wide.columns
    ]
    equal_cell_wide = equal_cell_wide.reset_index()

    # Equal-observer center.
    observer_period_df = (
        species_df.dropna(
            subset=["observer_label"]
        )
        .groupby(
            [
                "fia_species_code",
                "period",
                "observer_label",
            ],
            as_index=False,
        )
        .agg(
            observer_species_latitude=(
                "decimalLatitude",
                "mean",
            ),
        )
    )

    observer_center_df = (
        observer_period_df.groupby(
            ["fia_species_code", "period"],
            as_index=False,
        )
        .agg(
            observer_balanced_mean_latitude=(
                "observer_species_latitude",
                "mean",
            ),
            observer_balanced_n_observers=(
                "observer_label",
                "nunique",
            ),
        )
    )

    observer_wide = observer_center_df.pivot(
        index="fia_species_code",
        columns="period",
        values=[
            "observer_balanced_mean_latitude",
            "observer_balanced_n_observers",
        ],
    )
    observer_wide.columns = [
        f"{metric}_{period}"
        for metric, period in observer_wide.columns
    ]
    observer_wide = observer_wide.reset_index()

    # Target-group effort-standardized cell rates on a fixed frame.
    effort_lookup = effort_df[
        effort_df["grid_id"].isin(
            stable_cells
        )
    ][
        [
            "period",
            "grid_id",
            "target_effort_records",
        ]
    ].copy()

    standardized_rows = []

    for species_code in sorted(benchmark_codes):
        species_counts = (
            species_df[
                species_df["fia_species_code"]
                == species_code
            ]
            .groupby(
                ["period", "grid_id"]
            )["gbifID"]
            .size()
            .rename("species_records")
            .reset_index()
        )

        merged = effort_lookup.merge(
            species_counts,
            on=["period", "grid_id"],
            how="left",
        )

        merged["species_records"] = (
            merged["species_records"]
            .fillna(0)
            .astype(float)
        )

        merged["relative_occurrence_rate"] = (
            merged["species_records"]
            / merged["target_effort_records"]
        )

        merged["cell_latitude"] = (
            merged["grid_id"].map(
                pooled_cell_latitude
            )
        )

        for period in ["early", "late"]:
            period_frame = merged[
                merged["period"] == period
            ].copy()

            total_species_records = float(
                period_frame["species_records"].sum()
            )
            positive_cells = int(
                (
                    period_frame[
                        "species_records"
                    ] > 0
                ).sum()
            )
            rate_sum = float(
                period_frame[
                    "relative_occurrence_rate"
                ].sum()
            )

            standardized_latitude = (
                float(
                    np.average(
                        period_frame[
                            "cell_latitude"
                        ],
                        weights=period_frame[
                            "relative_occurrence_rate"
                        ],
                    )
                )
                if (
                    rate_sum > 0
                    and period_frame[
                        "cell_latitude"
                    ].notna().all()
                )
                else np.nan
            )

            standardized_rows.append(
                {
                    "fia_species_code": species_code,
                    "period": period,
                    "effort_standardized_latitude": (
                        standardized_latitude
                    ),
                    "effort_standardized_species_records": (
                        total_species_records
                    ),
                    "effort_standardized_positive_cells": (
                        positive_cells
                    ),
                }
            )

    standardized_df = pd.DataFrame(
        standardized_rows
    )

    standardized_wide = standardized_df.pivot(
        index="fia_species_code",
        columns="period",
        values=[
            "effort_standardized_latitude",
            "effort_standardized_species_records",
            "effort_standardized_positive_cells",
        ],
    )
    standardized_wide.columns = [
        f"{metric}_{period}"
        for metric, period in standardized_wide.columns
    ]
    standardized_wide = standardized_wide.reset_index()

    combined = (
        benchmark_species_df[
            [
                "species_code",
                "scientific_name",
                "common_name",
            ]
        ]
        .rename(
            columns={
                "species_code": "fia_species_code",
            }
        )
        .merge(
            raw_wide,
            on="fia_species_code",
            how="left",
        )
        .merge(
            equal_cell_wide,
            on="fia_species_code",
            how="left",
        )
        .merge(
            standardized_wide,
            on="fia_species_code",
            how="left",
        )
        .merge(
            observer_wide,
            on="fia_species_code",
            how="left",
        )
    )

    combined["raw_shift_km_decade"] = [
        shift_km_decade(
            row.get(
                "raw_mean_latitude_early",
                np.nan,
            ),
            row.get(
                "raw_mean_latitude_late",
                np.nan,
            ),
            EARLY_MID_YEAR,
            LATE_MID_YEAR,
        )
        for _, row in combined.iterrows()
    ]

    combined["equal_cell_shift_km_decade"] = [
        shift_km_decade(
            row.get(
                "equal_cell_mean_latitude_early",
                np.nan,
            ),
            row.get(
                "equal_cell_mean_latitude_late",
                np.nan,
            ),
            EARLY_MID_YEAR,
            LATE_MID_YEAR,
        )
        for _, row in combined.iterrows()
    ]

    combined["effort_standardized_shift_km_decade"] = [
        shift_km_decade(
            row.get(
                "effort_standardized_latitude_early",
                np.nan,
            ),
            row.get(
                "effort_standardized_latitude_late",
                np.nan,
            ),
            EARLY_MID_YEAR,
            LATE_MID_YEAR,
        )
        for _, row in combined.iterrows()
    ]

    combined["observer_balanced_shift_km_decade"] = [
        shift_km_decade(
            row.get(
                "observer_balanced_mean_latitude_early",
                np.nan,
            ),
            row.get(
                "observer_balanced_mean_latitude_late",
                np.nan,
            ),
            EARLY_MID_YEAR,
            LATE_MID_YEAR,
        )
        for _, row in combined.iterrows()
    ]

    frame_summary = {
        "grid_resolution": resolution,
        "n_all_cells": int(
            work["grid_id"].nunique()
        ),
        "n_stable_cells": len(stable_cells),
        "early_stable_record_fraction": (
            stable_record_fraction["early"]
        ),
        "late_stable_record_fraction": (
            stable_record_fraction["late"]
        ),
        "early_target_records": int(
            period_total_records.get(
                "early",
                0,
            )
        ),
        "late_target_records": int(
            period_total_records.get(
                "late",
                0,
            )
        ),
    }

    return combined, frame_summary


def bootstrap_method_comparison(
    evaluation_df,
    correction_column,
    rng,
):
    raw_error = evaluation_df[
        "raw_abs_error"
    ].to_numpy(dtype=float)

    correction_error = evaluation_df[
        correction_column
    ].to_numpy(dtype=float)

    n_species = len(evaluation_df)

    reduction_values = np.empty(
        N_SPECIES_BOOTSTRAP,
        dtype=float,
    )
    lower_count = 0

    for replicate in tqdm(
        range(N_SPECIES_BOOTSTRAP),
        desc=f"Bootstrap {correction_column}",
        unit="replicate",
        leave=False,
    ):
        indices = rng.integers(
            0,
            n_species,
            size=n_species,
        )

        raw_median = float(
            np.median(raw_error[indices])
        )
        correction_median = float(
            np.median(
                correction_error[indices]
            )
        )

        reduction_values[replicate] = (
            1 - correction_median / raw_median
            if raw_median > 0
            else np.nan
        )

        lower_count += int(
            correction_median < raw_median
        )

    return {
        "bootstrap_reduction_low": float(
            np.nanquantile(
                reduction_values,
                0.025,
            )
        ),
        "bootstrap_reduction_high": float(
            np.nanquantile(
                reduction_values,
                0.975,
            )
        ),
        "bootstrap_probability_lower_error": float(
            lower_count
            / N_SPECIES_BOOTSTRAP
        ),
    }


def print_failure(status, error, next_step):
    append_readme(
        f"""

## PA species external validation failure — {RUN_UTC}
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
    # 1. Validate inputs and load locked ecological species
    # --------------------------------------------------------
    required_paths = [
        PA_PLOT_CSV,
        PA_TREE_CSV,
        BENCHMARK_SPECIES_CSV,
        EXPANDED_INAT_PATH,
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

    benchmark_species_df = pd.read_csv(
        BENCHMARK_SPECIES_CSV
    )

    benchmark_species_df["species_code"] = (
        pd.to_numeric(
            benchmark_species_df["species_code"],
            errors="coerce",
        )
    )

    benchmark_species_df = (
        benchmark_species_df
        .dropna(
            subset=[
                "species_code",
                "scientific_name",
            ]
        )
        .copy()
    )

    benchmark_species_df["species_code"] = (
        benchmark_species_df[
            "species_code"
        ].astype(int)
    )

    benchmark_species_codes = set(
        benchmark_species_df[
            "species_code"
        ].tolist()
    )

    print("\nLOCKED BENCHMARK SPECIES")
    display(
        benchmark_species_df[
            [
                "species_code",
                "scientific_name",
                "common_name",
            ]
        ]
    )

    # --------------------------------------------------------
    # 2. Build same-window repeated physical FIA plot pairs
    # --------------------------------------------------------
    plot_columns = pd.read_csv(
        PA_PLOT_CSV,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    plot_key_col = resolve_column(
        plot_columns,
        ["CN", "PLOT_CN"],
    )
    inventory_year_col = resolve_column(
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
        inventory_year_col,
        latitude_col,
        longitude_col,
        state_code_col,
        unit_code_col,
        county_code_col,
        plot_number_col,
    ]

    if not all(required_plot_fields):
        raise ValueError(
            "Required PA PLOT identity fields were not resolved."
        )

    plot_usecols = list(
        dict.fromkeys(
            [
                plot_key_col,
                inventory_year_col,
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
        PA_PLOT_CSV,
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
        inventory_year_col: "year",
        latitude_col: "latitude",
        longitude_col: "longitude",
    }

    if plot_status_col is not None:
        rename_map[
            plot_status_col
        ] = "plot_status"

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

    valid_plot = (
        plot_df["year"].isin(
            EARLY_YEARS + LATE_YEARS
        )
        & plot_df["latitude"].between(
            39.5,
            42.6,
        )
        & plot_df["longitude"].between(
            -81.0,
            -74.4,
        )
        & plot_df["plot_key"].notna()
        & plot_df[
            "physical_plot_id"
        ].notna()
    )

    if plot_status_col is not None:
        valid_plot &= plot_df[
            "plot_status"
        ].eq(1)

    plot_df = (
        plot_df.loc[
            valid_plot,
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

    plot_df["year"] = (
        plot_df["year"].astype(int)
    )

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

    # --------------------------------------------------------
    # 3. Scan PA TREE for forested measurement keys and species presence
    # --------------------------------------------------------
    tree_columns = pd.read_csv(
        PA_TREE_CSV,
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
            "Required PA TREE fields were not resolved."
        )

    tree_usecols = [
        tree_plot_key_col,
        tree_species_col,
    ]

    if tree_status_col is not None:
        tree_usecols.append(
            tree_status_col
        )

    tree_usecols = list(
        dict.fromkeys(tree_usecols)
    )

    live_measurement_keys = set()
    selected_presence_parts = []
    tree_rows_scanned = 0

    tree_reader = pd.read_csv(
        PA_TREE_CSV,
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
        desc="Scanning PA TREE for same-window FIA reference",
        unit="chunk",
    ):
        tree_rows_scanned += len(chunk)

        chunk = chunk.rename(
            columns={
                tree_plot_key_col: "plot_key",
                tree_species_col: "species_code",
                **(
                    {
                        tree_status_col: "status_code"
                    }
                    if tree_status_col is not None
                    else {}
                ),
            }
        )

        chunk["plot_key"] = (
            chunk["plot_key"]
            .astype("string")
            .str.strip()
        )
        chunk["species_code"] = (
            pd.to_numeric(
                chunk["species_code"],
                errors="coerce",
            )
        )

        keep = chunk[
            "plot_key"
        ].isin(candidate_plot_keys)

        if "status_code" in chunk.columns:
            chunk["status_code"] = (
                pd.to_numeric(
                    chunk["status_code"],
                    errors="coerce",
                )
            )
            keep &= chunk[
                "status_code"
            ].eq(1)

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
            live_work[
                "plot_key"
            ].astype(str)
        )

        selected_work = live_work[
            live_work["species_code"].isin(
                benchmark_species_codes
            )
        ].dropna(
            subset=["species_code"]
        )

        if not selected_work.empty:
            selected_work[
                "species_code"
            ] = selected_work[
                "species_code"
            ].astype(int)

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
            "No benchmark species were found on matched PA plots."
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

    n_matched_physical_plots = len(
        matched_plot_df
    )

    if (
        n_matched_physical_plots
        < MIN_MATCHED_PHYSICAL_PLOTS
    ):
        raise ValueError(
            f"Only {n_matched_physical_plots} "
            "same-window matched FIA plots were available."
        )

    # --------------------------------------------------------
    # 4. Compute FIA presence shifts on exact matched plots
    # --------------------------------------------------------
    early_key_map = (
        matched_plot_df[
            [
                "physical_plot_id",
                "early_plot_key",
                "early_year",
                "stable_latitude",
            ]
        ]
        .rename(
            columns={
                "early_plot_key": "plot_key",
                "early_year": "year",
            }
        )
    )
    early_key_map["period"] = "early"

    late_key_map = (
        matched_plot_df[
            [
                "physical_plot_id",
                "late_plot_key",
                "late_year",
                "stable_latitude",
            ]
        ]
        .rename(
            columns={
                "late_plot_key": "plot_key",
                "late_year": "year",
            }
        )
    )
    late_key_map["period"] = "late"

    plot_period_lookup_df = pd.concat(
        [
            early_key_map,
            late_key_map,
        ],
        ignore_index=True,
    )

    plot_period_lookup_df[
        "plot_key"
    ] = (
        plot_period_lookup_df[
            "plot_key"
        ]
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
            fia_mean_year=(
                "year",
                "mean",
            ),
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
        for metric, period in fia_wide_df.columns
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

    fia_wide_df.to_parquet(
        OUT_DIR
        / "fia_same_window_species_reference.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Load real iNaturalist records and compute all resolutions
    # --------------------------------------------------------
    inat_df = pd.read_parquet(
        EXPANDED_INAT_PATH
    )

    for column in [
        "year",
        "fia_species_code",
        "decimalLatitude",
        "decimalLongitude",
    ]:
        inat_df[column] = pd.to_numeric(
            inat_df[column],
            errors="coerce",
        )

    inat_df = inat_df[
        inat_df["year"].isin(
            EARLY_YEARS + LATE_YEARS
        )
        & inat_df[
            "fia_species_code"
        ].notna()
        & inat_df[
            "decimalLatitude"
        ].notna()
        & inat_df[
            "decimalLongitude"
        ].notna()
    ].copy()

    inat_df["year"] = (
        inat_df["year"].astype(int)
    )
    inat_df["fia_species_code"] = (
        inat_df[
            "fia_species_code"
        ].astype(int)
    )

    inat_df["period"] = np.where(
        inat_df["year"].isin(
            EARLY_YEARS
        ),
        "early",
        "late",
    )

    inat_df["observer_label"] = (
        normalize_observer_label(
            inat_df["recordedBy"]
        )
    )

    resolution_tables = {}
    frame_rows = []

    for resolution in tqdm(
        SENSITIVITY_GRID_RESOLUTIONS,
        desc="Computing real-data grid estimands",
        unit="resolution",
    ):
        estimand_df, frame_summary = (
            build_inat_estimands(
                inat_df,
                benchmark_species_df,
                resolution,
            )
        )

        resolution_tables[resolution] = (
            estimand_df
        )
        frame_rows.append(
            frame_summary
        )

        tag = str(resolution).replace(
            ".",
            "p",
        )

        estimand_df.to_parquet(
            OUT_DIR
            / f"inat_species_estimands_grid_{tag}.parquet",
            index=False,
        )

    frame_summary_df = pd.DataFrame(
        frame_rows
    )

    frame_summary_df.to_csv(
        OUT_DIR / "grid_frame_summary.csv",
        index=False,
    )

    print("\nREAL-DATA GRID FRAME SUMMARY")
    display(frame_summary_df)

    primary_inat_df = resolution_tables[
        PRIMARY_GRID_RESOLUTION
    ].copy()

    # --------------------------------------------------------
    # 6. Merge iNaturalist estimates with FIA reference
    # --------------------------------------------------------
    comparison_df = (
        primary_inat_df.merge(
            fia_wide_df,
            left_on="fia_species_code",
            right_on="species_code",
            how="left",
        )
    )

    comparison_df[
        "raw_coverage_pass"
    ] = (
        comparison_df[
            "raw_n_records_early"
        ].ge(
            MIN_INAT_RECORDS_PER_SPECIES_PERIOD
        )
        & comparison_df[
            "raw_n_records_late"
        ].ge(
            MIN_INAT_RECORDS_PER_SPECIES_PERIOD
        )
    )

    comparison_df[
        "effort_standardized_coverage_pass"
    ] = (
        comparison_df[
            "effort_standardized_species_records_early"
        ].ge(
            MIN_INAT_RECORDS_PER_SPECIES_PERIOD
        )
        & comparison_df[
            "effort_standardized_species_records_late"
        ].ge(
            MIN_INAT_RECORDS_PER_SPECIES_PERIOD
        )
        & comparison_df[
            "effort_standardized_positive_cells_early"
        ].ge(
            MIN_INAT_POSITIVE_CELLS_PER_SPECIES_PERIOD
        )
        & comparison_df[
            "effort_standardized_positive_cells_late"
        ].ge(
            MIN_INAT_POSITIVE_CELLS_PER_SPECIES_PERIOD
        )
    )

    comparison_df[
        "observer_coverage_pass"
    ] = (
        comparison_df[
            "observer_balanced_n_observers_early"
        ].ge(
            MIN_INAT_OBSERVERS_PER_SPECIES_PERIOD
        )
        & comparison_df[
            "observer_balanced_n_observers_late"
        ].ge(
            MIN_INAT_OBSERVERS_PER_SPECIES_PERIOD
        )
    )

    comparison_df[
        "fia_coverage_pass"
    ] = (
        comparison_df[
            "fia_n_occupied_plots_early"
        ].ge(
            MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD
        )
        & comparison_df[
            "fia_n_occupied_plots_late"
        ].ge(
            MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD
        )
    )

    comparison_df[
        "primary_evaluation_species"
    ] = (
        comparison_df["raw_coverage_pass"]
        & comparison_df[
            "effort_standardized_coverage_pass"
        ]
        & comparison_df[
            "fia_coverage_pass"
        ]
        & comparison_df[
            [
                "raw_shift_km_decade",
                "effort_standardized_shift_km_decade",
                "fia_shift_km_decade",
            ]
        ].notna().all(axis=1)
    )

    evaluation_df = comparison_df[
        comparison_df[
            "primary_evaluation_species"
        ]
    ].copy()

    # Errors relative to FIA reference.
    method_columns = {
        "raw": "raw_shift_km_decade",
        "equal_cell": "equal_cell_shift_km_decade",
        "effort_standardized": (
            "effort_standardized_shift_km_decade"
        ),
        "observer_balanced": (
            "observer_balanced_shift_km_decade"
        ),
    }

    for method, column in method_columns.items():
        comparison_df[
            f"{method}_abs_error"
        ] = (
            comparison_df[column]
            - comparison_df[
                "fia_shift_km_decade"
            ]
        ).abs()

    evaluation_df = comparison_df[
        comparison_df[
            "primary_evaluation_species"
        ]
    ].copy()

    evaluation_df.to_csv(
        OUT_DIR / "species_external_validation.csv",
        index=False,
    )
    evaluation_df.to_parquet(
        OUT_DIR / "species_external_validation.parquet",
        index=False,
    )

    print("\nSPECIES EXTERNAL VALIDATION")
    display(
        evaluation_df[
            [
                "fia_species_code",
                "scientific_name",
                "raw_shift_km_decade",
                "equal_cell_shift_km_decade",
                "effort_standardized_shift_km_decade",
                "observer_balanced_shift_km_decade",
                "fia_shift_km_decade",
                "raw_abs_error",
                "equal_cell_abs_error",
                "effort_standardized_abs_error",
                "observer_balanced_abs_error",
                "raw_n_records_early",
                "raw_n_records_late",
                "fia_n_occupied_plots_early",
                "fia_n_occupied_plots_late",
            ]
        ]
    )

    # --------------------------------------------------------
    # 7. Method-level external validation metrics
    # --------------------------------------------------------
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    method_rows = []

    for method, estimate_column in method_columns.items():
        error_column = f"{method}_abs_error"

        method_eval = evaluation_df[
            [
                estimate_column,
                "fia_shift_km_decade",
                "raw_abs_error",
                error_column,
            ]
        ].dropna()

        if method_eval.empty:
            continue

        median_abs_error = float(
            method_eval[
                error_column
            ].median()
        )
        mean_abs_error = float(
            method_eval[
                error_column
            ].mean()
        )

        raw_median_abs_error = float(
            method_eval[
                "raw_abs_error"
            ].median()
        )

        median_error_reduction = (
            1
            - median_abs_error
            / raw_median_abs_error
            if raw_median_abs_error > 0
            else np.nan
        )

        bootstrap = (
            bootstrap_method_comparison(
                method_eval.rename(
                    columns={
                        error_column: (
                            "method_abs_error"
                        )
                    }
                ),
                "method_abs_error",
                rng,
            )
            if method != "raw"
            else {
                "bootstrap_reduction_low": 0.0,
                "bootstrap_reduction_high": 0.0,
                "bootstrap_probability_lower_error": 0.0,
            }
        )

        paired_error_difference = (
            method_eval[
                "raw_abs_error"
            ]
            - method_eval[
                error_column
            ]
        )

        if (
            method != "raw"
            and paired_error_difference.abs().sum() > 0
        ):
            try:
                wilcoxon_p = float(
                    wilcoxon(
                        method_eval[
                            "raw_abs_error"
                        ],
                        method_eval[
                            error_column
                        ],
                        alternative="greater",
                        zero_method="wilcox",
                    ).pvalue
                )
            except Exception:
                wilcoxon_p = np.nan
        else:
            wilcoxon_p = np.nan

        method_rows.append(
            {
                "method": method,
                "n_species": len(
                    method_eval
                ),
                "median_abs_error_km_decade": (
                    median_abs_error
                ),
                "mean_abs_error_km_decade": (
                    mean_abs_error
                ),
                "median_error_reduction_vs_raw": (
                    median_error_reduction
                ),
                "direction_agreement": (
                    direction_agreement(
                        method_eval[
                            estimate_column
                        ],
                        method_eval[
                            "fia_shift_km_decade"
                        ],
                    )
                ),
                "spearman_with_fia": (
                    safe_spearman(
                        method_eval[
                            estimate_column
                        ],
                        method_eval[
                            "fia_shift_km_decade"
                        ],
                    )
                ),
                "wilcoxon_p_lower_error_vs_raw": (
                    wilcoxon_p
                ),
                **bootstrap,
            }
        )

    method_summary_df = pd.DataFrame(
        method_rows
    )

    method_summary_df.to_csv(
        OUT_DIR / "method_external_validation_summary.csv",
        index=False,
    )

    print("\nMETHOD EXTERNAL VALIDATION SUMMARY")
    display(method_summary_df)

    # --------------------------------------------------------
    # 8. Grid-resolution sensitivity for primary correction
    # --------------------------------------------------------
    grid_sensitivity_rows = []

    for resolution, estimand_df in resolution_tables.items():
        merged = (
            estimand_df.merge(
                fia_wide_df,
                left_on="fia_species_code",
                right_on="species_code",
                how="inner",
            )
        )

        valid = (
            merged[
                "raw_n_records_early"
            ].ge(
                MIN_INAT_RECORDS_PER_SPECIES_PERIOD
            )
            & merged[
                "raw_n_records_late"
            ].ge(
                MIN_INAT_RECORDS_PER_SPECIES_PERIOD
            )
            & merged[
                "effort_standardized_species_records_early"
            ].ge(
                MIN_INAT_RECORDS_PER_SPECIES_PERIOD
            )
            & merged[
                "effort_standardized_species_records_late"
            ].ge(
                MIN_INAT_RECORDS_PER_SPECIES_PERIOD
            )
            & merged[
                "effort_standardized_positive_cells_early"
            ].ge(
                MIN_INAT_POSITIVE_CELLS_PER_SPECIES_PERIOD
            )
            & merged[
                "effort_standardized_positive_cells_late"
            ].ge(
                MIN_INAT_POSITIVE_CELLS_PER_SPECIES_PERIOD
            )
            & merged[
                "fia_n_occupied_plots_early"
            ].ge(
                MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD
            )
            & merged[
                "fia_n_occupied_plots_late"
            ].ge(
                MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD
            )
        )

        valid_df = merged.loc[
            valid
        ].dropna(
            subset=[
                "raw_shift_km_decade",
                "effort_standardized_shift_km_decade",
                "fia_shift_km_decade",
            ]
        )

        if valid_df.empty:
            continue

        raw_error = (
            valid_df[
                "raw_shift_km_decade"
            ]
            - valid_df[
                "fia_shift_km_decade"
            ]
        ).abs()

        corrected_error = (
            valid_df[
                "effort_standardized_shift_km_decade"
            ]
            - valid_df[
                "fia_shift_km_decade"
            ]
        ).abs()

        raw_median = float(
            raw_error.median()
        )
        corrected_median = float(
            corrected_error.median()
        )

        grid_sensitivity_rows.append(
            {
                "grid_resolution": resolution,
                "n_species": len(
                    valid_df
                ),
                "raw_median_abs_error": (
                    raw_median
                ),
                "corrected_median_abs_error": (
                    corrected_median
                ),
                "median_error_reduction": (
                    1
                    - corrected_median
                    / raw_median
                    if raw_median > 0
                    else np.nan
                ),
                "direction_agreement": (
                    direction_agreement(
                        valid_df[
                            "effort_standardized_shift_km_decade"
                        ],
                        valid_df[
                            "fia_shift_km_decade"
                        ],
                    )
                ),
                "spearman_with_fia": (
                    safe_spearman(
                        valid_df[
                            "effort_standardized_shift_km_decade"
                        ],
                        valid_df[
                            "fia_shift_km_decade"
                        ],
                    )
                ),
            }
        )

    grid_sensitivity_df = pd.DataFrame(
        grid_sensitivity_rows
    )

    grid_sensitivity_df.to_csv(
        OUT_DIR / "grid_resolution_external_validation.csv",
        index=False,
    )

    print("\nGRID-RESOLUTION EXTERNAL VALIDATION")
    display(grid_sensitivity_df)

    # --------------------------------------------------------
    # 9. Decision
    # --------------------------------------------------------
    n_evaluation_species = len(
        evaluation_df
    )

    primary_rows = method_summary_df[
        method_summary_df["method"]
        == "effort_standardized"
    ]

    raw_rows = method_summary_df[
        method_summary_df["method"]
        == "raw"
    ]

    if primary_rows.empty or raw_rows.empty:
        primary_row = None
        raw_row = None
    else:
        primary_row = primary_rows.iloc[0]
        raw_row = raw_rows.iloc[0]

    coverage_pass = bool(
        n_matched_physical_plots
        >= MIN_MATCHED_PHYSICAL_PLOTS
        and n_evaluation_species
        >= MIN_EVALUATION_SPECIES
    )

    if primary_row is not None:
        error_reduction_pass = bool(
            primary_row[
                "median_error_reduction_vs_raw"
            ]
            >= TARGET_MEDIAN_ERROR_REDUCTION
        )

        bootstrap_pass = bool(
            primary_row[
                "bootstrap_probability_lower_error"
            ]
            >= TARGET_BOOTSTRAP_IMPROVEMENT_PROBABILITY
        )

        direction_not_worse = bool(
            (
                pd.isna(
                    raw_row[
                        "direction_agreement"
                    ]
                )
                or pd.isna(
                    primary_row[
                        "direction_agreement"
                    ]
                )
                or primary_row[
                    "direction_agreement"
                ]
                >= raw_row[
                    "direction_agreement"
                ]
            )
        )

        rank_not_worse = bool(
            (
                pd.isna(
                    raw_row[
                        "spearman_with_fia"
                    ]
                )
                or pd.isna(
                    primary_row[
                        "spearman_with_fia"
                    ]
                )
                or primary_row[
                    "spearman_with_fia"
                ]
                >= raw_row[
                    "spearman_with_fia"
                ]
            )
        )

        agreement_pass = bool(
            direction_not_worse
            or rank_not_worse
        )
    else:
        error_reduction_pass = False
        bootstrap_pass = False
        agreement_pass = False

    passed = bool(
        coverage_pass
        and error_reduction_pass
        and bootstrap_pass
        and agreement_pass
    )

    if passed:
        status = (
            "GO_FOR_FULL_PA_CORRECTION_CONFIRMATION"
        )
        next_step = (
            "The locked 0.5-degree target-group effort correction "
            "improves agreement with same-window repeated FIA plots. "
            "Next run annual-window sensitivity, observer-cluster "
            "bootstrap and registered GBIF download confirmation."
        )
    elif coverage_pass and error_reduction_pass:
        status = (
            "REAL_CORRECTION_PARTIAL_EXTERNAL_VALIDATION"
        )
        next_step = (
            "The correction reduces point-estimate error but the "
            "across-species evidence is not yet stable. Inspect which "
            "species drive the gain and run window sensitivity before "
            "any full-scale expansion."
        )
    elif coverage_pass:
        status = (
            "REAL_CORRECTION_NOT_VALIDATED_AGAINST_FIA"
        )
        next_step = (
            "The semi-synthetic correction does not improve real PA "
            "agreement with FIA. Do not scale the method; analyze the "
            "failure conditions and consider a sensitivity-index paper."
        )
    else:
        status = (
            "EXTERNAL_VALIDATION_COVERAGE_INSUFFICIENT"
        )
        next_step = (
            "Too few species or repeated FIA plots support a fair "
            "comparison. Do not interpret correction performance."
        )

    # --------------------------------------------------------
    # 10. Visual previews
    # --------------------------------------------------------
    if not evaluation_df.empty:
        plt.figure(figsize=(8, 7))
        plt.scatter(
            evaluation_df[
                "fia_shift_km_decade"
            ],
            evaluation_df[
                "raw_shift_km_decade"
            ],
            label="Raw iNaturalist",
        )
        plt.scatter(
            evaluation_df[
                "fia_shift_km_decade"
            ],
            evaluation_df[
                "effort_standardized_shift_km_decade"
            ],
            label="Effort-standardized iNaturalist",
        )

        combined_values = pd.concat(
            [
                evaluation_df[
                    "fia_shift_km_decade"
                ],
                evaluation_df[
                    "raw_shift_km_decade"
                ],
                evaluation_df[
                    "effort_standardized_shift_km_decade"
                ],
            ]
        ).dropna()

        if len(combined_values):
            lower = float(
                combined_values.min()
            )
            upper = float(
                combined_values.max()
            )
            plt.plot(
                [lower, upper],
                [lower, upper],
                linestyle="--",
                label="1:1",
            )

        plt.xlabel(
            "FIA repeated-plot shift (km/decade)"
        )
        plt.ylabel(
            "iNaturalist shift (km/decade)"
        )
        plt.title(
            "PA species shifts: iNaturalist versus FIA"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

        error_plot_df = (
            evaluation_df[
                [
                    "scientific_name",
                    "raw_abs_error",
                    "effort_standardized_abs_error",
                ]
            ]
            .sort_values(
                "raw_abs_error",
                ascending=False,
            )
        )

        y = np.arange(
            len(error_plot_df)
        )

        plt.figure(
            figsize=(
                11,
                max(
                    6,
                    0.55 * len(
                        error_plot_df
                    ),
                ),
            )
        )
        plt.scatter(
            error_plot_df[
                "raw_abs_error"
            ],
            y,
            label="Raw error",
        )
        plt.scatter(
            error_plot_df[
                "effort_standardized_abs_error"
            ],
            y,
            label="Corrected error",
        )

        for index, row in error_plot_df.reset_index(
            drop=True
        ).iterrows():
            plt.plot(
                [
                    row["raw_abs_error"],
                    row[
                        "effort_standardized_abs_error"
                    ],
                ],
                [index, index],
            )

        plt.yticks(
            y,
            error_plot_df[
                "scientific_name"
            ],
        )
        plt.xlabel(
            "Absolute error versus FIA (km/decade)"
        )
        plt.ylabel("Species")
        plt.title(
            "Effect of the locked 0.5-degree correction"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 11. README
    # --------------------------------------------------------
    compact_methods = (
        method_summary_df.round(6).to_dict(
            "records"
        )
    )

    compact_grids = (
        grid_sensitivity_df.round(6).to_dict(
            "records"
        )
    )

    top_species_comparison = (
        evaluation_df[
            [
                "fia_species_code",
                "scientific_name",
                "raw_shift_km_decade",
                "effort_standardized_shift_km_decade",
                "fia_shift_km_decade",
                "raw_abs_error",
                "effort_standardized_abs_error",
            ]
        ]
        .sort_values(
            "raw_abs_error",
            ascending=False,
        )
        .head(15)
        .round(4)
        .to_dict("records")
    )

    append_readme(
        f"""

## PA species external validation against repeated FIA plots — {RUN_UTC}

### Core hypothesis
The locked 0.5-degree target-group effort correction should make real
iNaturalist species-shift estimates more consistent with same-window repeated
FIA physical plots.

### Locked design
- Ecological species set:
  {len(benchmark_species_df)} original benchmark species
- Observation-effort frame:
  all expanded PA tree taxa from Cell 10B
- Early window: {EARLY_YEARS}
- Late window: {LATE_YEARS}
- Primary correction grid:
  {PRIMARY_GRID_RESOLUTION} degrees
- Grid sensitivities:
  {SENSITIVITY_GRID_RESOLUTIONS}
- Target effort threshold:
  {MIN_TARGET_EFFORT_PER_CELL_PERIOD} records per cell-period
- Tree abundance weighting: not used
- Random seed: {RANDOM_SEED}
- GPU: not used

### FIA reference
- Same physical plot identifier:
  STATECD + UNITCD + COUNTYCD + PLOT
- Same-window matched physical plots:
  {n_matched_physical_plots:,}
- PA TREE rows scanned:
  {tree_rows_scanned:,}
- One species presence per plot-period

### External validation
- Evaluation species:
  {n_evaluation_species}
- Coverage passed:
  {coverage_pass}
- Primary median-error reduction passed:
  {error_reduction_pass}
- Bootstrap improvement passed:
  {bootstrap_pass}
- Direction/rank agreement not worsened:
  {agreement_pass}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
FIA repeated-plot occurrence shifts are a standardized external reference, not
a perfect biological truth. iNaturalist target-group effort is used as a proxy
for opportunity-based sampling effort. Zero species records in a stable
cell-period are treated as zero relative occurrence within the selected tree
target group, not confirmed ecological absence. No climate attribution is made.
"""
    )

    # --------------------------------------------------------
    # 12. Compact output block
    # --------------------------------------------------------
    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATE: {STATE}")
    print(f"EARLY_YEARS: {EARLY_YEARS}")
    print(f"LATE_YEARS: {LATE_YEARS}")
    print(
        f"N_LOCKED_BENCHMARK_SPECIES: "
        f"{len(benchmark_species_df)}"
    )
    print(
        f"N_MATCHED_FIA_PHYSICAL_PLOTS: "
        f"{n_matched_physical_plots:,}"
    )
    print(
        f"PA_TREE_ROWS_SCANNED: "
        f"{tree_rows_scanned:,}"
    )
    print(
        "GRID_FRAME_SUMMARY: "
        + json.dumps(
            frame_summary_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        f"N_EVALUATION_SPECIES: "
        f"{n_evaluation_species}"
    )
    print(
        "METHOD_VALIDATION: "
        + json.dumps(
            compact_methods,
            ensure_ascii=False,
        )
    )
    print(
        "GRID_SENSITIVITY: "
        + json.dumps(
            compact_grids,
            ensure_ascii=False,
        )
    )
    print(
        "TOP_SPECIES_COMPARISON: "
        + json.dumps(
            top_species_comparison,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_CRITERIA: >=10 species; primary 0.5-degree "
        "effort correction reduces median absolute error by >=20%; "
        "bootstrap probability of lower median error >=0.90; "
        "direction agreement or rank correlation not worsened"
    )
    print(f"COVERAGE_PASSED: {coverage_pass}")
    print(
        f"ERROR_REDUCTION_PASSED: "
        f"{error_reduction_pass}"
    )
    print(
        f"BOOTSTRAP_IMPROVEMENT_PASSED: "
        f"{bootstrap_pass}"
    )
    print(
        f"AGREEMENT_PASSED: "
        f"{agreement_pass}"
    )
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except Exception as exc:
    print_failure(
        "CELL12A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed output. "
            "Do not download new data or rerun prior observer analyses."
        ),
    )
