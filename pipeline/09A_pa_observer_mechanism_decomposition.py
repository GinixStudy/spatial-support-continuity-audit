
# Cell 09A — Pennsylvania observer-turnover and within-observer decomposition
#
# Current hypothesis
# ------------------
# Pennsylvania shows a northward target-group footprint:
# - equal-cell center ≈ 12.2 km/decade
# - species-balanced center ≈ 8.8 km/decade
# - observer-balanced center ≈ 0.3 km/decade
#
# This suggests that apparent movement may arise from observer turnover,
# observer activity weighting, or species composition rather than movement
# of the same observer labels.
#
# Minimal mechanism test
# ----------------------
# 1. Use the existing thinned PA iNaturalist records.
# 2. Compare early and late windows using:
#    - pooled records
#    - equal spatial cells
#    - equal species
#    - equal observer labels
#    - the same observer labels in both windows
#    - the same observer–species pairs in both windows
# 3. Estimate an observer fixed-effect annual trend using observer-year centers.
# 4. Bootstrap observers/pairs and run paired sign-flip placebos.
#
# Important boundary
# ------------------
# recordedBy is treated as an exact normalized contributor label, not a verified
# permanent person identifier. No alias merging is attempted.

from pathlib import Path
from datetime import datetime, timezone
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 180)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 280)

# ============================================================
# Configuration
# ============================================================
STATE = "PA"

WINDOW_YEARS = 4
FOOTPRINT_GRID_RESOLUTION = 0.50

MIN_OBSERVER_RECORDS_PER_PERIOD = 3
MIN_PAIR_RECORDS_PER_PERIOD = 2
MIN_OBSERVER_YEARS_FOR_FE = 3
MIN_OBSERVER_SPAN_FOR_FE = 4

N_BOOTSTRAP = 500
N_SIGN_FLIP = 500
RANDOM_SEED = 20260710

MATERIAL_SHIFT_KM_DECADE = 5.0
SMALL_WITHIN_SHIFT_KM_DECADE = 3.0
MIN_BRIDGE_OBSERVERS = 100
MIN_BRIDGE_PAIRS = 300

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
PILOT_DIR = (
    BASE_DIR / "derived" / "gbif_inaturalist_real_pilot"
)
OUT_DIR = (
    BASE_DIR / "derived" / "pa_observer_mechanism_decomposition"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

THINNED_PATH = PILOT_DIR / "thinned_occurrences.parquet"
ANNUAL_PATH = PILOT_DIR / "annual_observation_footprint.parquet"

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("STATE:", STATE)
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


def normalize_observer_label(series):
    result = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    generic = {
        "",
        "nan",
        "none",
        "unknown",
        "anonymous",
        "not recorded",
        "not provided",
    }

    return result.mask(result.isin(generic))


def safe_shift_km_decade(
    early_lat,
    late_lat,
    early_year,
    late_year,
):
    values = [
        early_lat,
        late_lat,
        early_year,
        late_year,
    ]

    if not all(np.isfinite(value) for value in values):
        return np.nan

    interval = late_year - early_year

    if interval <= 0:
        return np.nan

    return float(
        (late_lat - early_lat)
        * 111.32
        * 10
        / interval
    )


def paired_shift_table(
    frame,
    id_columns,
    minimum_records,
):
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

    required = [
        "mean_latitude_early",
        "mean_latitude_late",
        "mean_year_early",
        "mean_year_late",
    ]

    for column in required:
        if column not in wide.columns:
            wide[column] = np.nan

    wide = wide.dropna(subset=required).copy()

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
            row["median_latitude_early"],
            row["median_latitude_late"],
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

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )
    bootstrap_medians = np.empty(
        n_bootstrap,
        dtype=float,
    )

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
            abs(np.mean(permuted))
            >= abs(observed_mean)
        )
        exceed_median += int(
            abs(np.median(permuted))
            >= abs(observed_median)
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
            (1 + exceed_mean)
            / (n_sign_flip + 1)
        ),
        "sign_flip_p_median": float(
            (1 + exceed_median)
            / (n_sign_flip + 1)
        ),
    }


def annual_slope_km_decade(
    frame,
    year_column,
    value_column,
):
    work = frame[
        [year_column, value_column]
    ].dropna().drop_duplicates(
        year_column
    ).sort_values(year_column)

    if len(work) < 3:
        return np.nan

    x = work[year_column].to_numpy(dtype=float)
    y = work[value_column].to_numpy(dtype=float)

    x_centered = x - x.mean()
    denominator = np.dot(
        x_centered,
        x_centered,
    )

    if denominator == 0:
        return np.nan

    slope = np.dot(
        x_centered,
        y - y.mean(),
    ) / denominator

    return float(slope * 111.32 * 10)


def fixed_effect_sufficient_statistics(
    observer_year_df,
):
    rows = []

    for observer_label, group in tqdm(
        observer_year_df.groupby(
            "observer_label"
        ),
        desc="Building observer FE statistics",
        unit="observer",
    ):
        group = group.sort_values("year")

        if (
            group["year"].nunique()
            < MIN_OBSERVER_YEARS_FOR_FE
        ):
            continue

        if (
            group["year"].max()
            - group["year"].min()
            < MIN_OBSERVER_SPAN_FOR_FE
        ):
            continue

        x = group["year"].to_numpy(dtype=float)
        y = group[
            "observer_mean_latitude"
        ].to_numpy(dtype=float)

        x_centered = x - x.mean()
        y_centered = y - y.mean()

        denominator = float(
            np.dot(x_centered, x_centered)
        )

        if denominator <= 0:
            continue

        rows.append(
            {
                "observer_label": observer_label,
                "n_years": int(
                    group["year"].nunique()
                ),
                "first_year": int(
                    group["year"].min()
                ),
                "last_year": int(
                    group["year"].max()
                ),
                "numerator": float(
                    np.dot(
                        x_centered,
                        y_centered,
                    )
                ),
                "denominator": denominator,
            }
        )

    return pd.DataFrame(rows)


def print_failure(status, error, next_step):
    append_readme(
        f"""

## PA observer decomposition failure — {RUN_UTC}
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
    # 1. Load existing pilot data
    # --------------------------------------------------------
    required_paths = [
        THINNED_PATH,
        ANNUAL_PATH,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Cell 08A outputs: "
            + " | ".join(missing)
        )

    occurrence_df = pd.read_parquet(
        THINNED_PATH
    )
    annual_input_df = pd.read_parquet(
        ANNUAL_PATH
    )

    occurrence_df = occurrence_df[
        occurrence_df[
            "query_state_code"
        ].eq(STATE)
    ].copy()

    annual_state_df = annual_input_df[
        annual_input_df[
            "query_state_code"
        ].eq(STATE)
        & annual_input_df[
            "annual_target_usable"
        ].fillna(False)
    ].copy()

    usable_years = sorted(
        annual_state_df["year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if len(usable_years) < 2 * WINDOW_YEARS:
        raise ValueError(
            f"Only {len(usable_years)} usable PA years."
        )

    early_years = usable_years[
        :WINDOW_YEARS
    ]
    late_years = usable_years[
        -WINDOW_YEARS:
    ]

    occurrence_df["year"] = pd.to_numeric(
        occurrence_df["year"],
        errors="coerce",
    )
    occurrence_df[
        "decimalLatitude"
    ] = pd.to_numeric(
        occurrence_df["decimalLatitude"],
        errors="coerce",
    )
    occurrence_df[
        "decimalLongitude"
    ] = pd.to_numeric(
        occurrence_df["decimalLongitude"],
        errors="coerce",
    )

    occurrence_df = occurrence_df[
        occurrence_df["year"].isin(
            early_years + late_years
        )
        & occurrence_df[
            "decimalLatitude"
        ].notna()
        & occurrence_df[
            "decimalLongitude"
        ].notna()
    ].copy()

    occurrence_df["year"] = (
        occurrence_df["year"].astype(int)
    )

    occurrence_df["period"] = np.where(
        occurrence_df["year"].isin(
            early_years
        ),
        "early",
        "late",
    )

    occurrence_df["observer_label"] = (
        normalize_observer_label(
            occurrence_df["recordedBy"]
        )
    )

    occurrence_df = occurrence_df.dropna(
        subset=["observer_label"]
    ).copy()

    occurrence_df["footprint_lat_index"] = (
        np.floor(
            occurrence_df[
                "decimalLatitude"
            ]
            / FOOTPRINT_GRID_RESOLUTION
        ).astype("int32")
    )
    occurrence_df["footprint_lon_index"] = (
        np.floor(
            occurrence_df[
                "decimalLongitude"
            ]
            / FOOTPRINT_GRID_RESOLUTION
        ).astype("int32")
    )
    occurrence_df["footprint_cell"] = (
        occurrence_df[
            "footprint_lat_index"
        ].astype(str)
        + "_"
        + occurrence_df[
            "footprint_lon_index"
        ].astype(str)
    )

    if len(occurrence_df) < 1000:
        raise ValueError(
            "Too few PA early/late records after filtering."
        )

    # --------------------------------------------------------
    # 2. Observer activity audit
    # --------------------------------------------------------
    observer_activity_df = (
        occurrence_df.groupby(
            "observer_label",
            as_index=False,
        )
        .agg(
            n_records=("gbifID", "size"),
            n_years=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_species=(
                "fia_species_code",
                "nunique",
            ),
            n_cells=(
                "footprint_cell",
                "nunique",
            ),
        )
    )

    observer_activity_df[
        "year_span"
    ] = (
        observer_activity_df["last_year"]
        - observer_activity_df["first_year"]
    )

    observer_activity_df = (
        observer_activity_df.sort_values(
            "n_records",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    observer_activity_df.to_csv(
        OUT_DIR / "observer_activity_audit.csv",
        index=False,
    )

    print("\nOBSERVER ACTIVITY AUDIT")
    display(observer_activity_df.head(50))

    # --------------------------------------------------------
    # 3. Early/late aggregate estimands
    # --------------------------------------------------------
    period_summary_df = (
        occurrence_df.groupby(
            "period",
            as_index=False,
        )
        .agg(
            n_records=("gbifID", "size"),
            n_observers=(
                "observer_label",
                "nunique",
            ),
            n_species=(
                "fia_species_code",
                "nunique",
            ),
            n_cells=(
                "footprint_cell",
                "nunique",
            ),
            pooled_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            pooled_median_latitude=(
                "decimalLatitude",
                "median",
            ),
            mean_year=("year", "mean"),
        )
    )

    observer_period_df = (
        occurrence_df.groupby(
            ["observer_label", "period"],
            as_index=False,
        )
        .agg(
            n_records=("gbifID", "size"),
            observer_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            observer_median_latitude=(
                "decimalLatitude",
                "median",
            ),
            observer_mean_year=("year", "mean"),
        )
    )

    equal_observer_period_df = (
        observer_period_df.groupby(
            "period",
            as_index=False,
        )
        .agg(
            equal_observer_mean_latitude=(
                "observer_mean_latitude",
                "mean",
            ),
            equal_observer_median_latitude=(
                "observer_median_latitude",
                "median",
            ),
            equal_observer_mean_year=(
                "observer_mean_year",
                "mean",
            ),
            n_observers=(
                "observer_label",
                "nunique",
            ),
        )
    )

    species_period_df = (
        occurrence_df.groupby(
            ["fia_species_code", "period"],
            as_index=False,
        )
        .agg(
            n_records=("gbifID", "size"),
            species_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            species_mean_year=("year", "mean"),
        )
    )

    equal_species_period_df = (
        species_period_df.groupby(
            "period",
            as_index=False,
        )
        .agg(
            equal_species_mean_latitude=(
                "species_mean_latitude",
                "mean",
            ),
            equal_species_mean_year=(
                "species_mean_year",
                "mean",
            ),
            n_species=(
                "fia_species_code",
                "nunique",
            ),
        )
    )

    cell_period_df = (
        occurrence_df.groupby(
            ["footprint_cell", "period"],
            as_index=False,
        )
        .agg(
            n_records=("gbifID", "size"),
            cell_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            cell_mean_year=("year", "mean"),
        )
    )

    equal_cell_period_df = (
        cell_period_df.groupby(
            "period",
            as_index=False,
        )
        .agg(
            equal_cell_mean_latitude=(
                "cell_mean_latitude",
                "mean",
            ),
            equal_cell_mean_year=(
                "cell_mean_year",
                "mean",
            ),
            n_cells=(
                "footprint_cell",
                "nunique",
            ),
        )
    )

    def two_period_shift(
        frame,
        latitude_column,
        year_column,
    ):
        indexed = frame.set_index("period")

        return safe_shift_km_decade(
            indexed.loc[
                "early",
                latitude_column,
            ],
            indexed.loc[
                "late",
                latitude_column,
            ],
            indexed.loc[
                "early",
                year_column,
            ],
            indexed.loc[
                "late",
                year_column,
            ],
        )

    aggregate_rows = [
        {
            "estimand": "pooled_records",
            "shift_km_decade": two_period_shift(
                period_summary_df,
                "pooled_mean_latitude",
                "mean_year",
            ),
        },
        {
            "estimand": "equal_spatial_cells",
            "shift_km_decade": two_period_shift(
                equal_cell_period_df,
                "equal_cell_mean_latitude",
                "equal_cell_mean_year",
            ),
        },
        {
            "estimand": "equal_species_all",
            "shift_km_decade": two_period_shift(
                equal_species_period_df,
                "equal_species_mean_latitude",
                "equal_species_mean_year",
            ),
        },
        {
            "estimand": "equal_observers_all",
            "shift_km_decade": two_period_shift(
                equal_observer_period_df,
                "equal_observer_mean_latitude",
                "equal_observer_mean_year",
            ),
        },
    ]

    # --------------------------------------------------------
    # 4. Matched observers and matched observer-species pairs
    # --------------------------------------------------------
    matched_observer_df = paired_shift_table(
        occurrence_df,
        ["observer_label"],
        MIN_OBSERVER_RECORDS_PER_PERIOD,
    )

    matched_pair_df = paired_shift_table(
        occurrence_df,
        [
            "observer_label",
            "fia_species_code",
        ],
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

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    observer_shift_summary = (
        summarize_shift_distribution(
            matched_observer_df[
                "mean_shift_km_decade"
            ],
            rng,
            N_BOOTSTRAP,
            N_SIGN_FLIP,
            "matched observers",
        )
    )

    pair_shift_summary = (
        summarize_shift_distribution(
            matched_pair_df[
                "mean_shift_km_decade"
            ],
            rng,
            N_BOOTSTRAP,
            N_SIGN_FLIP,
            "matched observer-species pairs",
        )
    )

    aggregate_rows.extend(
        [
            {
                "estimand": "matched_observers_equal_weight",
                "shift_km_decade": (
                    observer_shift_summary[
                        "mean_shift_km_decade"
                    ]
                ),
            },
            {
                "estimand": (
                    "matched_observer_species_equal_weight"
                ),
                "shift_km_decade": (
                    pair_shift_summary[
                        "mean_shift_km_decade"
                    ]
                ),
            },
        ]
    )

    aggregate_df = pd.DataFrame(
        aggregate_rows
    )

    # --------------------------------------------------------
    # 5. Observer-year fixed-effect trend
    # --------------------------------------------------------
    observer_year_df = (
        occurrence_df.groupby(
            ["observer_label", "year"],
            as_index=False,
        )
        .agg(
            observer_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            n_records=("gbifID", "size"),
            n_species=(
                "fia_species_code",
                "nunique",
            ),
            n_cells=(
                "footprint_cell",
                "nunique",
            ),
        )
    )

    fe_stats_df = (
        fixed_effect_sufficient_statistics(
            observer_year_df
        )
    )

    fe_stats_df.to_csv(
        OUT_DIR / "observer_fixed_effect_statistics.csv",
        index=False,
    )

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

        fe_bootstrap = np.empty(
            N_BOOTSTRAP,
            dtype=float,
        )

        for index in tqdm(
            range(N_BOOTSTRAP),
            desc="Bootstrap observer fixed effect",
            unit="replicate",
        ):
            sampled_indices = rng.integers(
                0,
                len(fe_stats_df),
                size=len(fe_stats_df),
            )

            sampled = fe_stats_df.iloc[
                sampled_indices
            ]

            denominator = sampled[
                "denominator"
            ].sum()

            fe_bootstrap[index] = (
                sampled["numerator"].sum()
                / denominator
                * 111.32
                * 10
                if denominator > 0
                else np.nan
            )

        fe_bootstrap_low = float(
            np.nanquantile(
                fe_bootstrap,
                0.025,
            )
        )
        fe_bootstrap_high = float(
            np.nanquantile(
                fe_bootstrap,
                0.975,
            )
        )

    aggregate_df = pd.concat(
        [
            aggregate_df,
            pd.DataFrame(
                [
                    {
                        "estimand": (
                            "observer_fixed_effect_annual"
                        ),
                        "shift_km_decade": fe_slope,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    aggregate_df.to_csv(
        OUT_DIR / "mechanism_estimands.csv",
        index=False,
    )

    inferential_df = pd.DataFrame(
        [
            observer_shift_summary,
            pair_shift_summary,
            {
                "label": "observer fixed-effect annual",
                "n_units": len(fe_stats_df),
                "mean_shift_km_decade": fe_slope,
                "median_shift_km_decade": np.nan,
                "bootstrap_mean_low": (
                    fe_bootstrap_low
                ),
                "bootstrap_mean_high": (
                    fe_bootstrap_high
                ),
                "bootstrap_median_low": np.nan,
                "bootstrap_median_high": np.nan,
                "sign_flip_p_mean": np.nan,
                "sign_flip_p_median": np.nan,
            },
        ]
    )

    inferential_df.to_csv(
        OUT_DIR / "mechanism_inference_summary.csv",
        index=False,
    )

    print("\nMECHANISM ESTIMANDS")
    display(aggregate_df)

    print("\nMATCHED AND FIXED-EFFECT INFERENCE")
    display(inferential_df)

    # --------------------------------------------------------
    # 6. Annual trend comparison
    # --------------------------------------------------------
    annual_observer_center_df = (
        observer_year_df.groupby(
            "year",
            as_index=False,
        )
        .agg(
            equal_observer_mean_latitude=(
                "observer_mean_latitude",
                "mean",
            ),
            n_observers=(
                "observer_label",
                "nunique",
            ),
        )
    )

    annual_pair_df = (
        occurrence_df.groupby(
            [
                "observer_label",
                "fia_species_code",
                "year",
            ],
            as_index=False,
        )
        .agg(
            pair_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            pair_records=("gbifID", "size"),
        )
    )

    annual_equal_pair_df = (
        annual_pair_df.groupby(
            "year",
            as_index=False,
        )
        .agg(
            equal_observer_species_latitude=(
                "pair_mean_latitude",
                "mean",
            ),
            n_observer_species_pairs=(
                "pair_mean_latitude",
                "size",
            ),
        )
    )

    annual_compare_df = (
        annual_state_df[
            [
                "year",
                "pooled_mean_latitude",
                "equal_cell_mean_latitude",
                "species_balanced_mean_latitude",
                "observer_balanced_mean_latitude",
            ]
        ]
        .merge(
            annual_observer_center_df,
            on="year",
            how="left",
        )
        .merge(
            annual_equal_pair_df,
            on="year",
            how="left",
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    annual_compare_df.to_csv(
        OUT_DIR / "annual_mechanism_centers.csv",
        index=False,
    )

    annual_trend_rows = []

    for column in [
        "pooled_mean_latitude",
        "equal_cell_mean_latitude",
        "species_balanced_mean_latitude",
        "observer_balanced_mean_latitude",
        "equal_observer_mean_latitude",
        "equal_observer_species_latitude",
    ]:
        annual_trend_rows.append(
            {
                "metric": column,
                "trend_km_decade": (
                    annual_slope_km_decade(
                        annual_compare_df,
                        "year",
                        column,
                    )
                ),
            }
        )

    annual_trend_df = pd.DataFrame(
        annual_trend_rows
    )

    annual_trend_df.to_csv(
        OUT_DIR / "annual_mechanism_trends.csv",
        index=False,
    )

    print("\nANNUAL MECHANISM TRENDS")
    display(annual_trend_df)

    # --------------------------------------------------------
    # 7. Decision
    # --------------------------------------------------------
    aggregate_lookup = (
        aggregate_df.set_index(
            "estimand"
        )["shift_km_decade"]
        .to_dict()
    )

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
        len(matched_observer_df)
        >= MIN_BRIDGE_OBSERVERS
        and len(matched_pair_df)
        >= MIN_BRIDGE_PAIRS
        and len(fe_stats_df)
        >= MIN_BRIDGE_OBSERVERS
    )

    within_observer_supported = bool(
        coverage_pass
        and (
            matched_observer_shift
            >= MATERIAL_SHIFT_KM_DECADE
            or matched_pair_shift
            >= MATERIAL_SHIFT_KM_DECADE
            or fe_shift_abs
            >= MATERIAL_SHIFT_KM_DECADE
        )
        and (
            observer_shift_summary[
                "sign_flip_p_mean"
            ]
            <= 0.05
            or pair_shift_summary[
                "sign_flip_p_mean"
            ]
            <= 0.05
            or (
                np.isfinite(
                    fe_bootstrap_low
                )
                and np.isfinite(
                    fe_bootstrap_high
                )
                and (
                    fe_bootstrap_low > 0
                    or fe_bootstrap_high < 0
                )
            )
        )
    )

    turnover_activity_dominates = bool(
        coverage_pass
        and core_footprint_shift
        >= MATERIAL_SHIFT_KM_DECADE
        and matched_observer_shift
        < SMALL_WITHIN_SHIFT_KM_DECADE
        and matched_pair_shift
        < SMALL_WITHIN_SHIFT_KM_DECADE
        and fe_shift_abs
        < SMALL_WITHIN_SHIFT_KM_DECADE
    )

    if within_observer_supported:
        status = "WITHIN_OBSERVER_MOVEMENT_SUPPORTED"
        passed = True
        next_step = (
            "The same observer labels show material movement. "
            "Proceed to species-level raw-versus-0.5-degree-corrected "
            "trends and compare them with observer-label movement."
        )
    elif turnover_activity_dominates:
        status = (
            "OBSERVER_TURNOVER_OR_ACTIVITY_WEIGHTING_DOMINATES"
        )
        passed = True
        next_step = (
            "The target-group footprint moves, but the same observer "
            "labels and observer-species pairs do not. Build the real "
            "correction around observer turnover and activity weighting, "
            "not personal movement."
        )
    elif coverage_pass:
        status = "MIXED_OBSERVATION_SYSTEM_DRIFT"
        passed = True
        next_step = (
            "Both observer composition and within-observer movement "
            "may contribute. Run species-level correction with both "
            "equal-observer and equal-cell sensitivity analyses."
        )
    else:
        status = "OBSERVER_DECOMPOSITION_COVERAGE_INSUFFICIENT"
        passed = False
        next_step = (
            "The observer labels do not provide enough repeated units. "
            "Expand the PA time range or target species before inference."
        )

    # --------------------------------------------------------
    # 8. Visual previews
    # --------------------------------------------------------
    plot_estimands_df = aggregate_df.dropna(
        subset=["shift_km_decade"]
    ).copy()

    plt.figure(
        figsize=(
            10,
            max(
                5,
                0.55 * len(
                    plot_estimands_df
                ),
            ),
        )
    )
    plt.barh(
        plot_estimands_df["estimand"],
        plot_estimands_df[
            "shift_km_decade"
        ],
    )
    plt.axvline(0)
    plt.xlabel("Latitude shift (km/decade)")
    plt.ylabel("Estimand")
    plt.title(
        "Pennsylvania observation-system decomposition"
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for column, label in [
        (
            "equal_cell_mean_latitude",
            "Equal cells",
        ),
        (
            "species_balanced_mean_latitude",
            "Equal species",
        ),
        (
            "equal_observer_mean_latitude",
            "Equal observers",
        ),
        (
            "equal_observer_species_latitude",
            "Equal observer-species pairs",
        ),
    ]:
        plt.plot(
            annual_compare_df["year"],
            annual_compare_df[column],
            marker="o",
            label=label,
        )

    plt.xlabel("Year")
    plt.ylabel("Latitude center")
    plt.title(
        "PA annual centers under different weighting systems"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 9. README
    # --------------------------------------------------------
    append_readme(
        f"""

## Pennsylvania observer mechanism decomposition — {RUN_UTC}

### Hypothesis
The PA target-group footprint may move because of observer turnover, observer
activity weighting, or species composition rather than movement of the same
observer labels.

### Data
- Source: existing Cell 08A thinned iNaturalist/GBIF records
- State: {STATE}
- Usable years: {usable_years}
- Early window: {early_years}
- Late window: {late_years}
- Early/late records used: {len(occurrence_df):,}
- Exact normalized observer labels: {occurrence_df['observer_label'].nunique():,}
- No observer alias merging performed

### Repeated-unit coverage
- Matched observer labels: {len(matched_observer_df):,}
- Matched observer-species pairs: {len(matched_pair_df):,}
- Observer labels eligible for annual fixed effects: {len(fe_stats_df):,}

### Key estimands
- Pooled records:
  {aggregate_lookup.get('pooled_records')} km/decade
- Equal spatial cells:
  {aggregate_lookup.get('equal_spatial_cells')} km/decade
- Equal species:
  {aggregate_lookup.get('equal_species_all')} km/decade
- Equal all observer labels:
  {aggregate_lookup.get('equal_observers_all')} km/decade
- Matched observer labels:
  {aggregate_lookup.get('matched_observers_equal_weight')} km/decade
- Matched observer-species pairs:
  {aggregate_lookup.get('matched_observer_species_equal_weight')} km/decade
- Observer fixed-effect annual trend:
  {fe_slope} km/decade
- Observer fixed-effect 95% bootstrap interval:
  [{fe_bootstrap_low}, {fe_bootstrap_high}]

### Decision
- Coverage passed: {coverage_pass}
- Within-observer movement supported: {within_observer_supported}
- Turnover/activity weighting dominates: {turnover_activity_dominates}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
recordedBy is an exact contributor label and is not verified as a permanent
person identifier. Changes may reflect username formatting, joint observers,
or platform metadata conventions. The decomposition concerns the selected tree
target group, not all iNaturalist activity in Pennsylvania.
"""
    )

    # --------------------------------------------------------
    # 10. Compact output
    # --------------------------------------------------------
    compact_estimands = (
        aggregate_df.round(6).to_dict(
            "records"
        )
    )

    compact_inference = (
        inferential_df.round(6).to_dict(
            "records"
        )
    )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATE: {STATE}")
    print(f"USABLE_YEARS: {usable_years}")
    print(f"EARLY_YEARS: {early_years}")
    print(f"LATE_YEARS: {late_years}")
    print(
        f"N_RECORDS_EARLY_LATE: "
        f"{len(occurrence_df):,}"
    )
    print(
        f"N_UNIQUE_OBSERVER_LABELS: "
        f"{occurrence_df['observer_label'].nunique():,}"
    )
    print(
        f"N_MATCHED_OBSERVERS: "
        f"{len(matched_observer_df):,}"
    )
    print(
        f"N_MATCHED_OBSERVER_SPECIES_PAIRS: "
        f"{len(matched_pair_df):,}"
    )
    print(
        f"N_FE_OBSERVERS: "
        f"{len(fe_stats_df):,}"
    )
    print(
        "MECHANISM_ESTIMANDS: "
        + json.dumps(
            compact_estimands,
            ensure_ascii=False,
        )
    )
    print(
        "INFERENCE_SUMMARY: "
        + json.dumps(
            compact_inference,
            ensure_ascii=False,
        )
    )
    print(
        "ANNUAL_TRENDS: "
        + json.dumps(
            annual_trend_df.round(6).to_dict(
                "records"
            ),
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

except Exception as exc:
    print_failure(
        "CELL09A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        "Send this COPY block and the last displayed output. "
        "Do not rerun Cell 08A.",
    )
