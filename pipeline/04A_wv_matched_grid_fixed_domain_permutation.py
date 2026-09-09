
# Cell 04A — WV matched-grid, fixed-domain, and year-permutation test
#
# Hypothesis
# The raw species latitude-center trend remains materially different after:
# 1) equal weighting of occupied spatial grid cells;
# 2) restriction to a stable, repeatedly sampled spatial domain.
#
# Negative control
# Globally shuffle year order while preserving paired raw/fixed annual centers.
# The chronological result should exceed the shuffled-year null distribution.
#
# This is still a trend validation, not the final paper model.

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display
from scipy.stats import linregress, theilslopes

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 140)
pd.set_option("display.max_colwidth", 220)
pd.set_option("display.width", 240)

# ============================================================
# Configuration
# ============================================================
STATE = "WV"
MIN_YEAR = 2000

GRID_RESOLUTIONS = [0.50, 0.25]
PRIMARY_RESOLUTION = 0.50

STABLE_YEAR_FRACTION = 0.60
MIN_EARLY_WINDOW_YEARS = 2
MIN_LATE_WINDOW_YEARS = 2
WINDOW_SIZE = 5

MIN_OCCUPIED_PLOTS_RAW_YEAR = 10
MIN_OCCUPIED_CELLS_YEAR = 3
MIN_PAIRED_YEARS = 15
MIN_PRIMARY_STABLE_CELLS = 8
MIN_PRIMARY_SPECIES = 20

MATERIAL_DIFF_KM_DECADE = 2.0
MEANINGFUL_REVERSAL_MIN_MAG_KM_DECADE = 1.0
TARGET_REVERSAL_FRACTION = 0.10

N_PERMUTATIONS = 300
RANDOM_SEED = 20260710

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
RAW_DIR = BASE_DIR / "raw"
CENTER_DIR = BASE_DIR / "derived" / "wv_center_screen"
OUT_DIR = BASE_DIR / "derived" / "wv_fixed_domain_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_CSV = RAW_DIR / f"{STATE}_PLOT.csv"
PRESENCE_PARQUET = CENTER_DIR / "species_plot_year_presence.parquet"
RAW_CENTER_PARQUET = CENTER_DIR / "species_annual_centers.parquet"
SELECTED_SPECIES_CSV = CENTER_DIR / "selected_species.csv"

RUN_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

print("RUN_UTC:", RUN_UTC)
print("OUT_DIR:", OUT_DIR)
print("PRIMARY_RESOLUTION:", PRIMARY_RESOLUTION)
print("N_PERMUTATIONS:", N_PERMUTATIONS)


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


def add_grid_columns(df, resolution):
    result = df.copy()
    result["grid_lat_index"] = np.floor(
        result["latitude"] / resolution
    ).astype("int32")
    result["grid_lon_index"] = np.floor(
        result["longitude"] / resolution
    ).astype("int32")
    result["grid_id"] = (
        result["grid_lat_index"].astype(str)
        + "_"
        + result["grid_lon_index"].astype(str)
    )
    return result


def slope_summary(years, values):
    data = pd.DataFrame(
        {
            "year": pd.to_numeric(years, errors="coerce"),
            "value": pd.to_numeric(values, errors="coerce"),
        }
    ).dropna()

    data = data.drop_duplicates("year").sort_values("year")

    if len(data) < MIN_PAIRED_YEARS:
        return {
            "n_years": len(data),
            "ols_slope": np.nan,
            "ols_p": np.nan,
            "theil_sen_slope": np.nan,
        }

    ols = linregress(data["year"], data["value"])
    robust = theilslopes(data["value"], data["year"])[0]

    return {
        "n_years": len(data),
        "ols_slope": float(ols.slope),
        "ols_p": float(ols.pvalue),
        "theil_sen_slope": float(robust),
    }


def ols_slope_fast(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < MIN_PAIRED_YEARS:
        return np.nan

    x_centered = x - x.mean()
    denominator = np.dot(x_centered, x_centered)

    if denominator == 0:
        return np.nan

    return float(np.dot(x_centered, y - y.mean()) / denominator)


def meaningful_reversal(raw_value, fixed_value):
    if not np.isfinite(raw_value) or not np.isfinite(fixed_value):
        return False

    return bool(
        np.sign(raw_value) != np.sign(fixed_value)
        and abs(raw_value) >= MEANINGFUL_REVERSAL_MIN_MAG_KM_DECADE
        and abs(fixed_value) >= MEANINGFUL_REVERSAL_MIN_MAG_KM_DECADE
    )


def species_name(row):
    value = row.get("scientific_name")
    if pd.notna(value) and str(value).strip():
        return str(value)
    return f"SPCD {int(row['species_code'])}"


def print_failure(status, error, next_step):
    append_readme(
        f"""

## WV fixed-domain test failure — {RUN_UTC}
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
    # 1. Validate and load previous outputs
    # --------------------------------------------------------
    required_files = [
        PLOT_CSV,
        PRESENCE_PARQUET,
        RAW_CENTER_PARQUET,
        SELECTED_SPECIES_CSV,
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing outputs from prior cells: " + " | ".join(missing)
        )

    presence_df = pd.read_parquet(PRESENCE_PARQUET)
    raw_center_df = pd.read_parquet(RAW_CENTER_PARQUET)
    taxonomy_df = (
        pd.read_csv(SELECTED_SPECIES_CSV)
        [["species_code", "scientific_name", "common_name"]]
        .drop_duplicates("species_code")
    )

    presence_df["year"] = pd.to_numeric(
        presence_df["year"], errors="coerce"
    )
    presence_df["species_code"] = pd.to_numeric(
        presence_df["species_code"], errors="coerce"
    )

    presence_df = presence_df.dropna(
        subset=[
            "species_code", "year", "latitude",
            "longitude", "plot_key"
        ]
    ).copy()

    presence_df["year"] = presence_df["year"].astype(int)
    presence_df["species_code"] = presence_df["species_code"].astype(int)
    presence_df = presence_df[presence_df["year"] >= MIN_YEAR].copy()

    raw_center_df["year"] = pd.to_numeric(
        raw_center_df["year"], errors="coerce"
    )
    raw_center_df["species_code"] = pd.to_numeric(
        raw_center_df["species_code"], errors="coerce"
    )

    raw_center_df = raw_center_df.dropna(
        subset=["species_code", "year", "mean_latitude"]
    ).copy()

    raw_center_df["year"] = raw_center_df["year"].astype(int)
    raw_center_df["species_code"] = raw_center_df[
        "species_code"
    ].astype(int)

    raw_center_df = raw_center_df[
        (raw_center_df["year"] >= MIN_YEAR)
        & (
            raw_center_df["n_occupied_plots"]
            >= MIN_OCCUPIED_PLOTS_RAW_YEAR
        )
    ][
        [
            "species_code", "year", "mean_latitude",
            "mean_elevation", "n_occupied_plots"
        ]
    ].rename(
        columns={
            "mean_latitude": "raw_mean_latitude",
            "mean_elevation": "raw_mean_elevation",
        }
    )

    # --------------------------------------------------------
    # 2. Load the small PLOT table for survey-domain coverage
    # --------------------------------------------------------
    plot_columns = pd.read_csv(
        PLOT_CSV,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    plot_key_col = resolve_column(plot_columns, ["CN", "PLOT_CN"])
    plot_year_col = resolve_column(
        plot_columns, ["INVYR", "INVENTORY_YEAR"]
    )
    lat_col = resolve_column(plot_columns, ["LAT", "LATITUDE"])
    lon_col = resolve_column(plot_columns, ["LON", "LONGITUDE"])
    elev_col = resolve_column(plot_columns, ["ELEV", "ELEVATION"])
    status_col = resolve_column(
        plot_columns, ["PLOT_STATUS_CD", "PLOT_STATUS"]
    )

    if not all([plot_key_col, plot_year_col, lat_col, lon_col]):
        raise ValueError(
            "Required PLOT fields could not be resolved."
        )

    plot_usecols = [
        plot_key_col, plot_year_col, lat_col, lon_col
    ]
    if elev_col:
        plot_usecols.append(elev_col)
    if status_col:
        plot_usecols.append(status_col)

    plot_usecols = list(dict.fromkeys(plot_usecols))

    plot_df = pd.read_csv(
        PLOT_CSV,
        usecols=plot_usecols,
        low_memory=False,
        encoding_errors="replace",
        dtype={plot_key_col: "string"},
    ).rename(
        columns={
            plot_key_col: "plot_key",
            plot_year_col: "year",
            lat_col: "latitude",
            lon_col: "longitude",
        }
    )

    if elev_col:
        plot_df = plot_df.rename(columns={elev_col: "elevation"})
    else:
        plot_df["elevation"] = np.nan

    if status_col:
        plot_df = plot_df.rename(columns={status_col: "plot_status"})
    else:
        plot_df["plot_status"] = np.nan

    for column in [
        "year", "latitude", "longitude",
        "elevation", "plot_status"
    ]:
        plot_df[column] = pd.to_numeric(
            plot_df[column], errors="coerce"
        )

    plot_df["plot_key"] = (
        plot_df["plot_key"].astype("string").str.strip()
    )

    plot_valid = (
        plot_df["year"].ge(MIN_YEAR)
        & plot_df["latitude"].between(24, 50)
        & plot_df["longitude"].between(-130, -60)
        & plot_df["plot_key"].notna()
    )

    if status_col:
        plot_valid &= plot_df["plot_status"].eq(1)

    plot_df = (
        plot_df.loc[
            plot_valid,
            [
                "plot_key", "year", "latitude",
                "longitude", "elevation"
            ],
        ]
        .drop_duplicates("plot_key")
        .copy()
    )

    plot_df["year"] = plot_df["year"].astype(int)

    all_years = sorted(plot_df["year"].unique().tolist())
    n_total_years = len(all_years)

    if n_total_years < MIN_PAIRED_YEARS:
        raise ValueError(
            f"Only {n_total_years} usable PLOT years were found."
        )

    early_years = set(all_years[:WINDOW_SIZE])
    late_years = set(all_years[-WINDOW_SIZE:])
    min_stable_years = math.ceil(
        STABLE_YEAR_FRACTION * n_total_years
    )

    # --------------------------------------------------------
    # 3. Run matched-grid and fixed-domain calculations
    # --------------------------------------------------------
    resolution_summaries = []
    all_slope_tables = {}
    all_center_tables = {}
    stable_cell_tables = {}

    for resolution in tqdm(
        GRID_RESOLUTIONS,
        desc="Testing grid resolutions",
        unit="resolution",
    ):
        tag = str(resolution).replace(".", "p")

        plot_grid_df = add_grid_columns(plot_df, resolution)
        presence_grid_df = add_grid_columns(
            presence_df, resolution
        )

        # Survey coverage by cell-year.
        plot_grid_year_df = (
            plot_grid_df.groupby(
                ["grid_id", "grid_lat_index", "grid_lon_index", "year"],
                as_index=False,
            )
            .agg(
                n_sampled_plots=("plot_key", "nunique"),
                sampled_cell_latitude=("latitude", "mean"),
                sampled_cell_longitude=("longitude", "mean"),
                sampled_cell_elevation=("elevation", "mean"),
            )
        )

        cell_coverage_df = (
            plot_grid_year_df.groupby(
                ["grid_id", "grid_lat_index", "grid_lon_index"],
                as_index=False,
            )
            .agg(
                n_sampled_years=("year", "nunique"),
                first_sampled_year=("year", "min"),
                last_sampled_year=("year", "max"),
                mean_sampled_plots_year=("n_sampled_plots", "mean"),
                cell_latitude=("sampled_cell_latitude", "mean"),
                cell_longitude=("sampled_cell_longitude", "mean"),
            )
        )

        early_counts = (
            plot_grid_year_df[
                plot_grid_year_df["year"].isin(early_years)
            ]
            .groupby("grid_id")["year"]
            .nunique()
            .rename("n_early_years")
        )

        late_counts = (
            plot_grid_year_df[
                plot_grid_year_df["year"].isin(late_years)
            ]
            .groupby("grid_id")["year"]
            .nunique()
            .rename("n_late_years")
        )

        cell_coverage_df = (
            cell_coverage_df
            .merge(early_counts, on="grid_id", how="left")
            .merge(late_counts, on="grid_id", how="left")
        )

        cell_coverage_df[
            ["n_early_years", "n_late_years"]
        ] = cell_coverage_df[
            ["n_early_years", "n_late_years"]
        ].fillna(0)

        cell_coverage_df["stable_domain"] = (
            (cell_coverage_df["n_sampled_years"] >= min_stable_years)
            & (
                cell_coverage_df["n_early_years"]
                >= MIN_EARLY_WINDOW_YEARS
            )
            & (
                cell_coverage_df["n_late_years"]
                >= MIN_LATE_WINDOW_YEARS
            )
        )

        stable_cells_df = cell_coverage_df[
            cell_coverage_df["stable_domain"]
        ].copy()

        stable_ids = set(stable_cells_df["grid_id"])
        stable_cell_tables[resolution] = stable_cells_df

        cell_coverage_df.to_csv(
            OUT_DIR / f"grid_{tag}_cell_coverage.csv",
            index=False,
        )

        stable_cells_df.to_csv(
            OUT_DIR / f"grid_{tag}_stable_cells.csv",
            index=False,
        )

        # One species-cell-year vote regardless of the number of plots.
        species_grid_year_df = (
            presence_grid_df.groupby(
                [
                    "species_code", "year", "grid_id",
                    "grid_lat_index", "grid_lon_index"
                ],
                as_index=False,
            )
            .agg(
                n_occupied_plots=("plot_key", "nunique"),
                occupied_cell_latitude=("latitude", "mean"),
                occupied_cell_longitude=("longitude", "mean"),
                occupied_cell_elevation=("elevation", "mean"),
            )
        )

        matched_center_df = (
            species_grid_year_df.groupby(
                ["species_code", "year"],
                as_index=False,
            )
            .agg(
                matched_n_occupied_cells=("grid_id", "nunique"),
                matched_mean_latitude=(
                    "occupied_cell_latitude", "mean"
                ),
                matched_mean_elevation=(
                    "occupied_cell_elevation", "mean"
                ),
            )
        )

        fixed_species_grid_df = species_grid_year_df[
            species_grid_year_df["grid_id"].isin(stable_ids)
        ].copy()

        fixed_center_df = (
            fixed_species_grid_df.groupby(
                ["species_code", "year"],
                as_index=False,
            )
            .agg(
                fixed_n_occupied_cells=("grid_id", "nunique"),
                fixed_mean_latitude=(
                    "occupied_cell_latitude", "mean"
                ),
                fixed_mean_elevation=(
                    "occupied_cell_elevation", "mean"
                ),
            )
        )

        combined_center_df = (
            raw_center_df
            .merge(
                matched_center_df,
                on=["species_code", "year"],
                how="left",
            )
            .merge(
                fixed_center_df,
                on=["species_code", "year"],
                how="left",
            )
            .merge(
                taxonomy_df,
                on="species_code",
                how="left",
            )
        )

        combined_center_df.to_parquet(
            OUT_DIR / f"grid_{tag}_annual_centers.parquet",
            index=False,
        )

        all_center_tables[resolution] = combined_center_df

        # Use paired years and the same minimum cell requirement.
        slope_rows = []

        species_groups = combined_center_df.groupby(
            "species_code", sort=True
        )

        for species_code, group in tqdm(
            species_groups,
            total=species_groups.ngroups,
            desc=f"Fitting species slopes {resolution:.2f}°",
            unit="species",
            leave=False,
        ):
            paired = group[
                (
                    group["matched_n_occupied_cells"]
                    >= MIN_OCCUPIED_CELLS_YEAR
                )
                & (
                    group["fixed_n_occupied_cells"]
                    >= MIN_OCCUPIED_CELLS_YEAR
                )
            ].copy()

            raw_fit = slope_summary(
                paired["year"],
                paired["raw_mean_latitude"],
            )
            matched_fit = slope_summary(
                paired["year"],
                paired["matched_mean_latitude"],
            )
            fixed_fit = slope_summary(
                paired["year"],
                paired["fixed_mean_latitude"],
            )

            row0 = group.iloc[0]

            raw_km_decade = (
                raw_fit["theil_sen_slope"] * 111.32 * 10
                if np.isfinite(raw_fit["theil_sen_slope"])
                else np.nan
            )

            matched_km_decade = (
                matched_fit["theil_sen_slope"] * 111.32 * 10
                if np.isfinite(matched_fit["theil_sen_slope"])
                else np.nan
            )

            fixed_km_decade = (
                fixed_fit["theil_sen_slope"] * 111.32 * 10
                if np.isfinite(fixed_fit["theil_sen_slope"])
                else np.nan
            )

            raw_ols_km_decade = (
                raw_fit["ols_slope"] * 111.32 * 10
                if np.isfinite(raw_fit["ols_slope"])
                else np.nan
            )

            fixed_ols_km_decade = (
                fixed_fit["ols_slope"] * 111.32 * 10
                if np.isfinite(fixed_fit["ols_slope"])
                else np.nan
            )

            slope_rows.append(
                {
                    "resolution_degrees": resolution,
                    "species_code": int(species_code),
                    "scientific_name": row0["scientific_name"],
                    "common_name": row0["common_name"],
                    "n_paired_years": raw_fit["n_years"],
                    "median_raw_occupied_plots": float(
                        paired["n_occupied_plots"].median()
                    ) if not paired.empty else np.nan,
                    "median_matched_occupied_cells": float(
                        paired[
                            "matched_n_occupied_cells"
                        ].median()
                    ) if not paired.empty else np.nan,
                    "median_fixed_occupied_cells": float(
                        paired[
                            "fixed_n_occupied_cells"
                        ].median()
                    ) if not paired.empty else np.nan,
                    "raw_latitude_km_decade": raw_km_decade,
                    "matched_grid_latitude_km_decade": (
                        matched_km_decade
                    ),
                    "fixed_domain_latitude_km_decade": (
                        fixed_km_decade
                    ),
                    "raw_minus_fixed_km_decade": (
                        raw_km_decade - fixed_km_decade
                        if np.isfinite(raw_km_decade)
                        and np.isfinite(fixed_km_decade)
                        else np.nan
                    ),
                    "raw_ols_km_decade": raw_ols_km_decade,
                    "fixed_ols_km_decade": fixed_ols_km_decade,
                    "raw_minus_fixed_ols_km_decade": (
                        raw_ols_km_decade - fixed_ols_km_decade
                        if np.isfinite(raw_ols_km_decade)
                        and np.isfinite(fixed_ols_km_decade)
                        else np.nan
                    ),
                    "raw_ols_p": raw_fit["ols_p"],
                    "fixed_ols_p": fixed_fit["ols_p"],
                }
            )

        slope_df = pd.DataFrame(slope_rows)

        slope_df["material_difference"] = (
            slope_df["raw_minus_fixed_km_decade"].abs()
            >= MATERIAL_DIFF_KM_DECADE
        )

        slope_df["meaningful_direction_reversal"] = [
            meaningful_reversal(raw, fixed)
            for raw, fixed in zip(
                slope_df["raw_latitude_km_decade"],
                slope_df["fixed_domain_latitude_km_decade"],
            )
        ]

        slope_df = slope_df.sort_values(
            "raw_minus_fixed_km_decade",
            key=lambda series: series.abs(),
            ascending=False,
        ).reset_index(drop=True)

        slope_df.to_parquet(
            OUT_DIR / f"grid_{tag}_species_slopes.parquet",
            index=False,
        )
        slope_df.to_csv(
            OUT_DIR / f"grid_{tag}_species_slopes.csv",
            index=False,
        )

        all_slope_tables[resolution] = slope_df

        strong_df = slope_df[
            slope_df["n_paired_years"] >= MIN_PAIRED_YEARS
        ].copy()

        n_strong = len(strong_df)
        median_abs_difference = (
            float(
                strong_df["raw_minus_fixed_km_decade"]
                .abs()
                .median()
            )
            if n_strong else np.nan
        )

        n_material = int(
            strong_df["material_difference"].sum()
        )
        n_reversal = int(
            strong_df[
                "meaningful_direction_reversal"
            ].sum()
        )

        reversal_fraction = (
            n_reversal / n_strong if n_strong else np.nan
        )

        stable_plot_fraction = (
            plot_grid_df["grid_id"].isin(stable_ids).mean()
            if len(plot_grid_df) else np.nan
        )

        resolution_summaries.append(
            {
                "resolution_degrees": resolution,
                "n_all_grid_cells": cell_coverage_df["grid_id"].nunique(),
                "n_stable_grid_cells": len(stable_cells_df),
                "min_stable_sampled_years": min_stable_years,
                "stable_plot_record_fraction": stable_plot_fraction,
                "n_strong_species": n_strong,
                "median_abs_raw_fixed_km_decade": (
                    median_abs_difference
                ),
                "n_material_difference": n_material,
                "n_meaningful_reversal": n_reversal,
                "meaningful_reversal_fraction": reversal_fraction,
            }
        )

    resolution_summary_df = pd.DataFrame(resolution_summaries)
    resolution_summary_df.to_csv(
        OUT_DIR / "resolution_summary.csv",
        index=False,
    )

    print("\nRESOLUTION SUMMARY")
    display(resolution_summary_df)

    # --------------------------------------------------------
    # 4. Cross-resolution consistency
    # --------------------------------------------------------
    primary_slopes_df = all_slope_tables[PRIMARY_RESOLUTION].copy()

    other_resolutions = [
        value for value in GRID_RESOLUTIONS
        if value != PRIMARY_RESOLUTION
    ]

    cross_resolution_df = primary_slopes_df[
        [
            "species_code", "scientific_name",
            "n_paired_years",
            "raw_minus_fixed_km_decade",
            "meaningful_direction_reversal",
        ]
    ].rename(
        columns={
            "raw_minus_fixed_km_decade": (
                "primary_raw_minus_fixed_km_decade"
            ),
            "meaningful_direction_reversal": (
                "primary_meaningful_reversal"
            ),
        }
    )

    for resolution in other_resolutions:
        tag = str(resolution).replace(".", "p")
        other = all_slope_tables[resolution][
            [
                "species_code",
                "raw_minus_fixed_km_decade",
                "meaningful_direction_reversal",
            ]
        ].rename(
            columns={
                "raw_minus_fixed_km_decade": (
                    f"resolution_{tag}_raw_minus_fixed_km_decade"
                ),
                "meaningful_direction_reversal": (
                    f"resolution_{tag}_meaningful_reversal"
                ),
            }
        )

        cross_resolution_df = cross_resolution_df.merge(
            other,
            on="species_code",
            how="left",
        )

    consistency_values = []

    if other_resolutions:
        comparison_resolution = other_resolutions[0]
        comparison_tag = str(comparison_resolution).replace(".", "p")
        comparison_col = (
            f"resolution_{comparison_tag}"
            "_raw_minus_fixed_km_decade"
        )

        valid_consistency = cross_resolution_df[
            cross_resolution_df[
                [
                    "primary_raw_minus_fixed_km_decade",
                    comparison_col,
                ]
            ].notna().all(axis=1)
        ].copy()

        if not valid_consistency.empty:
            valid_consistency["same_correction_direction"] = (
                np.sign(
                    valid_consistency[
                        "primary_raw_minus_fixed_km_decade"
                    ]
                )
                == np.sign(valid_consistency[comparison_col])
            )

            correction_direction_consistency = float(
                valid_consistency[
                    "same_correction_direction"
                ].mean()
            )

            correction_magnitude_spearman = float(
                valid_consistency[
                    [
                        "primary_raw_minus_fixed_km_decade",
                        comparison_col,
                    ]
                ].corr(method="spearman").iloc[0, 1]
            )
        else:
            correction_direction_consistency = np.nan
            correction_magnitude_spearman = np.nan
    else:
        correction_direction_consistency = np.nan
        correction_magnitude_spearman = np.nan

    cross_resolution_df.to_csv(
        OUT_DIR / "cross_resolution_consistency.csv",
        index=False,
    )

    print("\nCROSS-RESOLUTION CONSISTENCY")
    display(cross_resolution_df.head(50))

    # --------------------------------------------------------
    # 5. Global year-order permutation placebo
    # --------------------------------------------------------
    primary_centers_df = all_center_tables[
        PRIMARY_RESOLUTION
    ].copy()

    primary_paired_df = primary_centers_df[
        (
            primary_centers_df["n_occupied_plots"]
            >= MIN_OCCUPIED_PLOTS_RAW_YEAR
        )
        & (
            primary_centers_df["fixed_n_occupied_cells"]
            >= MIN_OCCUPIED_CELLS_YEAR
        )
    ][
        [
            "species_code", "year",
            "raw_mean_latitude", "fixed_mean_latitude"
        ]
    ].dropna()

    eligible_counts = (
        primary_paired_df.groupby("species_code")["year"]
        .nunique()
    )
    eligible_species = eligible_counts[
        eligible_counts >= MIN_PAIRED_YEARS
    ].index.tolist()

    primary_paired_df = primary_paired_df[
        primary_paired_df["species_code"].isin(
            eligible_species
        )
    ].copy()

    permutation_groups = {
        int(species_code): group.sort_values("year").copy()
        for species_code, group in primary_paired_df.groupby(
            "species_code"
        )
    }

    observed_ols_rows = []

    for species_code, group in permutation_groups.items():
        raw_slope = (
            ols_slope_fast(
                group["year"], group["raw_mean_latitude"]
            )
            * 111.32 * 10
        )
        fixed_slope = (
            ols_slope_fast(
                group["year"], group["fixed_mean_latitude"]
            )
            * 111.32 * 10
        )

        observed_ols_rows.append(
            {
                "species_code": species_code,
                "raw_ols_km_decade": raw_slope,
                "fixed_ols_km_decade": fixed_slope,
                "difference_ols_km_decade": (
                    raw_slope - fixed_slope
                ),
                "meaningful_reversal": meaningful_reversal(
                    raw_slope, fixed_slope
                ),
            }
        )

    observed_ols_df = pd.DataFrame(observed_ols_rows)
    observed_median_abs_difference = float(
        observed_ols_df[
            "difference_ols_km_decade"
        ].abs().median()
    )
    observed_reversal_fraction = float(
        observed_ols_df["meaningful_reversal"].mean()
    )

    unique_years = np.array(
        sorted(primary_paired_df["year"].unique()),
        dtype=int,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    permutation_rows = []

    for permutation_id in tqdm(
        range(N_PERMUTATIONS),
        desc="Running global year-order permutations",
        unit="permutation",
    ):
        shuffled_years = rng.permutation(unique_years)
        year_map = dict(zip(unique_years, shuffled_years))

        differences = []
        reversals = []

        for species_code, group in permutation_groups.items():
            permuted_x = group["year"].map(year_map).to_numpy()

            raw_slope = (
                ols_slope_fast(
                    permuted_x,
                    group["raw_mean_latitude"].to_numpy(),
                )
                * 111.32 * 10
            )

            fixed_slope = (
                ols_slope_fast(
                    permuted_x,
                    group["fixed_mean_latitude"].to_numpy(),
                )
                * 111.32 * 10
            )

            if np.isfinite(raw_slope) and np.isfinite(fixed_slope):
                differences.append(raw_slope - fixed_slope)
                reversals.append(
                    meaningful_reversal(raw_slope, fixed_slope)
                )

        permutation_rows.append(
            {
                "permutation_id": permutation_id,
                "median_abs_raw_fixed_difference_km_decade": (
                    float(np.median(np.abs(differences)))
                    if differences else np.nan
                ),
                "meaningful_reversal_fraction": (
                    float(np.mean(reversals))
                    if reversals else np.nan
                ),
            }
        )

    permutation_df = pd.DataFrame(permutation_rows)

    permutation_df.to_parquet(
        OUT_DIR / "year_permutation_null.parquet",
        index=False,
    )
    permutation_df.to_csv(
        OUT_DIR / "year_permutation_null.csv",
        index=False,
    )

    p_median_difference = float(
        (
            1
            + (
                permutation_df[
                    "median_abs_raw_fixed_difference_km_decade"
                ]
                >= observed_median_abs_difference
            ).sum()
        )
        / (N_PERMUTATIONS + 1)
    )

    p_reversal_fraction = float(
        (
            1
            + (
                permutation_df[
                    "meaningful_reversal_fraction"
                ]
                >= observed_reversal_fraction
            ).sum()
        )
        / (N_PERMUTATIONS + 1)
    )

    permutation_summary_df = pd.DataFrame(
        [
            {
                "metric": "median_abs_raw_fixed_difference_km_decade",
                "observed": observed_median_abs_difference,
                "null_median": permutation_df[
                    "median_abs_raw_fixed_difference_km_decade"
                ].median(),
                "null_95th_percentile": permutation_df[
                    "median_abs_raw_fixed_difference_km_decade"
                ].quantile(0.95),
                "empirical_p_upper": p_median_difference,
            },
            {
                "metric": "meaningful_reversal_fraction",
                "observed": observed_reversal_fraction,
                "null_median": permutation_df[
                    "meaningful_reversal_fraction"
                ].median(),
                "null_95th_percentile": permutation_df[
                    "meaningful_reversal_fraction"
                ].quantile(0.95),
                "empirical_p_upper": p_reversal_fraction,
            },
        ]
    )

    permutation_summary_df.to_csv(
        OUT_DIR / "year_permutation_summary.csv",
        index=False,
    )

    print("\nYEAR-PERMUTATION SUMMARY")
    display(permutation_summary_df)

    # --------------------------------------------------------
    # 6. Decision
    # --------------------------------------------------------
    primary_summary = resolution_summary_df[
        resolution_summary_df["resolution_degrees"]
        == PRIMARY_RESOLUTION
    ].iloc[0]

    primary_domain_pass = bool(
        primary_summary["n_stable_grid_cells"]
        >= MIN_PRIMARY_STABLE_CELLS
        and primary_summary["n_strong_species"]
        >= MIN_PRIMARY_SPECIES
    )

    primary_material_signal = bool(
        primary_summary[
            "median_abs_raw_fixed_km_decade"
        ] >= MATERIAL_DIFF_KM_DECADE
        or primary_summary[
            "meaningful_reversal_fraction"
        ] >= TARGET_REVERSAL_FRACTION
    )

    permutation_pass = bool(
        p_median_difference <= 0.05
        or p_reversal_fraction <= 0.05
    )

    if (
        primary_domain_pass
        and primary_material_signal
        and permutation_pass
    ):
        status = "GO_ADD_APPALACHIAN_STATES"
        passed = True
        next_step = (
            "Repeat the same fixed-domain test in PA, VA, KY, "
            "then evaluate pooled and state-held-out consistency."
        )
    elif primary_domain_pass and primary_material_signal:
        status = "MATERIAL_EFFECT_BUT_PLACEBO_NOT_PASSED"
        passed = False
        next_step = (
            "Do not expand to full scale yet. Inspect annual panel "
            "structure and use repeated-plot or panel-matched controls."
        )
    elif primary_domain_pass:
        status = "FIXED_DOMAIN_REDUCED_SIGNAL"
        passed = False
        next_step = (
            "The simple centering signal weakened after support control. "
            "Test adjacent states before rejecting the mechanism."
        )
    else:
        status = "STABLE_DOMAIN_INSUFFICIENT"
        passed = False
        next_step = (
            "Use a coarser grid or combine adjacent states; WV alone "
            "does not provide a large enough fixed spatial domain."
        )

    # --------------------------------------------------------
    # 7. Visual previews
    # --------------------------------------------------------
    primary_stable_df = stable_cell_tables[
        PRIMARY_RESOLUTION
    ].copy()
    primary_all_cells = pd.read_csv(
        OUT_DIR / "grid_0p5_cell_coverage.csv"
    )

    plt.figure(figsize=(9, 7))
    plt.scatter(
        primary_all_cells["cell_longitude"],
        primary_all_cells["cell_latitude"],
        s=primary_all_cells["n_sampled_years"] * 3,
        label="All sampled cells",
    )
    plt.scatter(
        primary_stable_df["cell_longitude"],
        primary_stable_df["cell_latitude"],
        s=primary_stable_df["n_sampled_years"] * 3,
        label="Stable-domain cells",
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(
        f"FIA {STATE}: {PRIMARY_RESOLUTION:.2f}° grid support"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    primary_strong_df = primary_slopes_df[
        primary_slopes_df["n_paired_years"]
        >= MIN_PAIRED_YEARS
    ].head(15).copy()

    if not primary_strong_df.empty:
        primary_strong_df = primary_strong_df.sort_values(
            "raw_minus_fixed_km_decade"
        )
        labels = primary_strong_df.apply(
            species_name, axis=1
        )
        y = np.arange(len(primary_strong_df))

        plt.figure(figsize=(11, 8))
        plt.scatter(
            primary_strong_df[
                "raw_latitude_km_decade"
            ],
            y,
            label="Raw plot-level center",
        )
        plt.scatter(
            primary_strong_df[
                "fixed_domain_latitude_km_decade"
            ],
            y,
            label="Stable fixed-domain center",
        )

        for index, row in primary_strong_df.reset_index(
            drop=True
        ).iterrows():
            plt.plot(
                [
                    row["raw_latitude_km_decade"],
                    row["fixed_domain_latitude_km_decade"],
                ],
                [index, index],
            )

        plt.yticks(y, labels)
        plt.axvline(0)
        plt.xlabel("Latitude-center trend (km/decade)")
        plt.ylabel("Species")
        plt.title(
            f"FIA {STATE}: raw vs fixed-domain trends"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(9, 5))
    plt.hist(
        permutation_df[
            "median_abs_raw_fixed_difference_km_decade"
        ],
        bins=30,
    )
    plt.axvline(
        observed_median_abs_difference,
        label="Observed chronological order",
    )
    plt.xlabel(
        "Median absolute raw-fixed difference (km/decade)"
    )
    plt.ylabel("Permutation count")
    plt.title("Global year-order permutation placebo")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 8. README
    # --------------------------------------------------------
    top_changed = (
        primary_slopes_df[
            primary_slopes_df["n_paired_years"]
            >= MIN_PAIRED_YEARS
        ][
            [
                "species_code", "scientific_name",
                "n_paired_years",
                "raw_latitude_km_decade",
                "fixed_domain_latitude_km_decade",
                "raw_minus_fixed_km_decade",
                "meaningful_direction_reversal",
            ]
        ]
        .head(10)
        .round(4)
        .to_dict("records")
    )

    append_readme(
        f"""

## WV matched-grid and fixed-domain test — {RUN_UTC}

### Hypothesis
A material part of the raw species latitude-center trend remains after equal
weighting of occupied spatial cells and restriction to a repeatedly sampled
fixed domain.

### Configuration
- State: {STATE}
- Minimum year: {MIN_YEAR}
- Grid resolutions: {GRID_RESOLUTIONS}
- Primary resolution: {PRIMARY_RESOLUTION}
- Stable-cell requirement: sampled in at least {STABLE_YEAR_FRACTION:.0%}
  of usable years, at least {MIN_EARLY_WINDOW_YEARS} early-window years,
  and at least {MIN_LATE_WINDOW_YEARS} late-window years
- Minimum occupied cells per species-year: {MIN_OCCUPIED_CELLS_YEAR}
- Minimum paired years per species: {MIN_PAIRED_YEARS}
- Permutations: {N_PERMUTATIONS}
- Random seed: {RANDOM_SEED}
- GPU: not used

### Primary-resolution results
- Stable cells: {int(primary_summary['n_stable_grid_cells'])}
- Stable-domain plot-record fraction:
  {primary_summary['stable_plot_record_fraction']:.4f}
- Strong-coverage species:
  {int(primary_summary['n_strong_species'])}
- Median absolute raw-fixed difference:
  {primary_summary['median_abs_raw_fixed_km_decade']:.4f} km/decade
- Material-difference species:
  {int(primary_summary['n_material_difference'])}
- Meaningful reversals:
  {int(primary_summary['n_meaningful_reversal'])}
- Meaningful reversal fraction:
  {primary_summary['meaningful_reversal_fraction']:.4f}

### Resolution sensitivity
- Correction-direction consistency:
  {correction_direction_consistency}
- Correction-magnitude Spearman correlation:
  {correction_magnitude_spearman}

### Year-order placebo
- Observed OLS median absolute raw-fixed difference:
  {observed_median_abs_difference:.4f} km/decade
- Empirical upper-tail p-value:
  {p_median_difference:.4f}
- Observed meaningful reversal fraction:
  {observed_reversal_fraction:.4f}
- Empirical upper-tail p-value:
  {p_reversal_fraction:.4f}

### Decision
- Status: **{status}**
- Next step: {next_step}

### Interpretation boundary
The fixed-domain result controls changing broad spatial support and cell-level
sampling density, but it does not yet control FIA panel identity, repeated-plot
dependence, detectability, forest disturbance, mortality, recruitment, or
climate. Public coordinates are perturbed. Taxonomic names have not yet been
harmonized to current accepted names.
"""
    )

    # --------------------------------------------------------
    # 9. Compact output block
    # --------------------------------------------------------
    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATE: {STATE}")
    print(f"PRIMARY_RESOLUTION_DEGREES: {PRIMARY_RESOLUTION}")
    print(
        f"N_PRIMARY_ALL_GRID_CELLS: "
        f"{int(primary_summary['n_all_grid_cells'])}"
    )
    print(
        f"N_PRIMARY_STABLE_GRID_CELLS: "
        f"{int(primary_summary['n_stable_grid_cells'])}"
    )
    print(
        f"PRIMARY_STABLE_PLOT_RECORD_FRACTION: "
        f"{primary_summary['stable_plot_record_fraction']:.6f}"
    )
    print(
        f"N_PRIMARY_STRONG_SPECIES: "
        f"{int(primary_summary['n_strong_species'])}"
    )
    print(
        f"PRIMARY_MEDIAN_ABS_RAW_FIXED_KM_DECADE: "
        f"{primary_summary['median_abs_raw_fixed_km_decade']:.6f}"
    )
    print(
        f"PRIMARY_N_MATERIAL_DIFFERENCE: "
        f"{int(primary_summary['n_material_difference'])}"
    )
    print(
        f"PRIMARY_N_MEANINGFUL_REVERSAL: "
        f"{int(primary_summary['n_meaningful_reversal'])}"
    )
    print(
        f"PRIMARY_MEANINGFUL_REVERSAL_FRACTION: "
        f"{primary_summary['meaningful_reversal_fraction']:.6f}"
    )
    print(
        f"CROSS_RESOLUTION_DIRECTION_CONSISTENCY: "
        f"{correction_direction_consistency}"
    )
    print(
        f"CROSS_RESOLUTION_MAGNITUDE_SPEARMAN: "
        f"{correction_magnitude_spearman}"
    )
    print(
        f"PERM_OBS_MEDIAN_ABS_DIFF_KM_DECADE: "
        f"{observed_median_abs_difference:.6f}"
    )
    print(
        f"PERM_NULL95_MEDIAN_ABS_DIFF_KM_DECADE: "
        f"{permutation_df['median_abs_raw_fixed_difference_km_decade'].quantile(0.95):.6f}"
    )
    print(
        f"PERM_P_MEDIAN_ABS_DIFF: "
        f"{p_median_difference:.6f}"
    )
    print(
        f"PERM_OBS_REVERSAL_FRACTION: "
        f"{observed_reversal_fraction:.6f}"
    )
    print(
        f"PERM_NULL95_REVERSAL_FRACTION: "
        f"{permutation_df['meaningful_reversal_fraction'].quantile(0.95):.6f}"
    )
    print(
        f"PERM_P_REVERSAL_FRACTION: "
        f"{p_reversal_fraction:.6f}"
    )
    print(
        "TOP_CHANGED_PRIMARY: "
        + json.dumps(top_changed, ensure_ascii=False)
    )
    print(
        "SUCCESS_CRITERIA: primary stable cells >= "
        f"{MIN_PRIMARY_STABLE_CELLS}; strong species >= "
        f"{MIN_PRIMARY_SPECIES}; material fixed-domain effect; "
        "at least one year-permutation empirical p <= 0.05"
    )
    print(f"PRIMARY_DOMAIN_PASSED: {primary_domain_pass}")
    print(
        f"PRIMARY_MATERIAL_SIGNAL_PASSED: "
        f"{primary_material_signal}"
    )
    print(f"PERMUTATION_PASSED: {permutation_pass}")
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except Exception as exc:
    print_failure(
        "CELL04A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        "Send this COPY block and the last displayed output. "
        "Do not rerun earlier cells unless a required file is missing.",
    )
