
# Cell 05A — Four-state Appalachian replication and leave-one-state-out test
#
# States:
#   WV: reuse files downloaded in Cell 02B
#   PA, VA, KY: download current official FIA PLOT and TREE CSV files
#
# Core hypothesis:
# The raw-versus-fixed-domain difference observed in WV is reproducible across
# neighboring Appalachian states and is not dominated by a single state.
#
# Minimal design:
# 1. Use the same 42 WV-screened species in all states.
# 2. Use one species presence per plot-year.
# 3. Use a 0.5-degree grid.
# 4. Define a stable spatial domain independently within each state.
# 5. Estimate raw and fixed-domain latitude-center slopes.
# 6. Run global year-order permutations within each state.
# 7. Combine state-level correction effects with a transparent median meta-summary.
# 8. Test leave-one-state-out stability.
#
# Interpretation boundary:
# This confirms or rejects cross-state reproducibility of the observation-support
# effect. It does not establish climate causality or population migration.

from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import math
import warnings

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display
from scipy.stats import linregress, theilslopes, spearmanr

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 160)
pd.set_option("display.max_colwidth", 220)
pd.set_option("display.width", 260)

# ============================================================
# Configuration
# ============================================================
STATES = ["WV", "PA", "VA", "KY"]
ADDED_STATES = ["PA", "VA", "KY"]

MIN_YEAR = 2000
GRID_RESOLUTION = 0.50

TREE_CHUNK_SIZE = 250_000

STABLE_YEAR_FRACTION = 0.60
WINDOW_SIZE = 5
MIN_EARLY_WINDOW_YEARS = 2
MIN_LATE_WINDOW_YEARS = 2

MIN_RAW_PLOTS_PER_YEAR = 10
MIN_FIXED_CELLS_PER_YEAR = 3
MIN_PAIRED_YEARS = 15
MIN_STABLE_CELLS = 8
MIN_STRONG_SPECIES_STATE = 20

MATERIAL_DIFF_KM_DECADE = 2.0
MEANINGFUL_REVERSAL_MIN_MAG_KM_DECADE = 1.0
TARGET_REVERSAL_FRACTION = 0.10

N_PERMUTATIONS = 200
RANDOM_SEED = 20260710

MIN_META_STATES = 3
MIN_META_SPECIES = 20
TARGET_STATE_MAJORITY_CONSISTENCY = 0.60
TARGET_LOO_SIGN_STABILITY = 0.80
TARGET_NON_WV_MEDIAN_ABS_EFFECT = 2.0

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
RAW_DIR = BASE_DIR / "raw"
CENTER_DIR = BASE_DIR / "derived" / "wv_center_screen"
OUT_DIR = BASE_DIR / "derived" / "appalachian_four_state_replication"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_SPECIES_CSV = CENTER_DIR / "selected_species.csv"

BASE_URL = "https://apps.fs.usda.gov/fia/datamart/CSV"
RUN_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

print("RUN_UTC:", RUN_UTC)
print("STATES:", STATES)
print("GRID_RESOLUTION:", GRID_RESOLUTION)
print("N_PERMUTATIONS_PER_STATE:", N_PERMUTATIONS)
print("OUT_DIR:", OUT_DIR)


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


def human_bytes(value):
    value = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024


def resolve_column(columns, candidates):
    lookup = {str(column).upper(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    return None


def stream_download(url, destination, label):
    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"Reusing {destination.name}: "
            f"{human_bytes(destination.stat().st_size)}"
        )
        return {
            "url": url,
            "path": str(destination),
            "status": "reused",
            "size_bytes": destination.stat().st_size,
        }

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    headers = {"User-Agent": "Mozilla/5.0 FIA-research-replication/1.0"}

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, 900),
            allow_redirects=True,
            headers=headers,
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(temporary, "wb") as handle, tqdm(
                total=total if total > 0 else None,
                desc=label,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
                        progress.update(len(block))

        temporary.replace(destination)

        return {
            "url": url,
            "path": str(destination),
            "status": "downloaded",
            "size_bytes": destination.stat().st_size,
        }

    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def add_grid_columns(frame):
    result = frame.copy()
    result["grid_lat_index"] = np.floor(
        result["latitude"] / GRID_RESOLUTION
    ).astype("int32")
    result["grid_lon_index"] = np.floor(
        result["longitude"] / GRID_RESOLUTION
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


def safe_spearman(x, y):
    data = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan

    return float(spearmanr(data["x"], data["y"]).statistic)


def print_failure(status, error, next_step):
    append_readme(
        f"""

## Four-state replication failure — {RUN_UTC}
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
# Per-state calculation
# ============================================================
def process_state(
    state,
    plot_csv,
    tree_csv,
    selected_species_codes,
    taxonomy_df,
    permutation_seed,
):
    state_dir = OUT_DIR / state
    state_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Resolve and read PLOT
    # --------------------------------------------------------
    plot_columns = pd.read_csv(
        plot_csv,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    plot_key_col = resolve_column(plot_columns, ["CN", "PLOT_CN"])
    plot_year_col = resolve_column(
        plot_columns, ["INVYR", "INVENTORY_YEAR"]
    )
    latitude_col = resolve_column(plot_columns, ["LAT", "LATITUDE"])
    longitude_col = resolve_column(plot_columns, ["LON", "LONGITUDE"])
    elevation_col = resolve_column(plot_columns, ["ELEV", "ELEVATION"])
    plot_status_col = resolve_column(
        plot_columns, ["PLOT_STATUS_CD", "PLOT_STATUS"]
    )

    if not all(
        [plot_key_col, plot_year_col, latitude_col, longitude_col]
    ):
        raise ValueError(
            f"{state}: required PLOT fields were not found."
        )

    plot_usecols = [
        plot_key_col,
        plot_year_col,
        latitude_col,
        longitude_col,
    ]

    if elevation_col:
        plot_usecols.append(elevation_col)
    if plot_status_col:
        plot_usecols.append(plot_status_col)

    plot_usecols = list(dict.fromkeys(plot_usecols))

    plot_df = pd.read_csv(
        plot_csv,
        usecols=plot_usecols,
        low_memory=False,
        encoding_errors="replace",
        dtype={plot_key_col: "string"},
    ).rename(
        columns={
            plot_key_col: "plot_key",
            plot_year_col: "year",
            latitude_col: "latitude",
            longitude_col: "longitude",
        }
    )

    if elevation_col:
        plot_df = plot_df.rename(
            columns={elevation_col: "elevation"}
        )
    else:
        plot_df["elevation"] = np.nan

    if plot_status_col:
        plot_df = plot_df.rename(
            columns={plot_status_col: "plot_status"}
        )
    else:
        plot_df["plot_status"] = np.nan

    for column in [
        "year",
        "latitude",
        "longitude",
        "elevation",
        "plot_status",
    ]:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        )

    plot_df["plot_key"] = (
        plot_df["plot_key"].astype("string").str.strip()
    )

    plot_filter = (
        plot_df["year"].ge(MIN_YEAR)
        & plot_df["latitude"].between(24, 50)
        & plot_df["longitude"].between(-130, -60)
        & plot_df["plot_key"].notna()
    )

    if plot_status_col:
        plot_filter &= plot_df["plot_status"].eq(1)

    plot_df = (
        plot_df.loc[
            plot_filter,
            [
                "plot_key",
                "year",
                "latitude",
                "longitude",
                "elevation",
            ],
        ]
        .drop_duplicates("plot_key")
        .copy()
    )

    plot_df["year"] = plot_df["year"].astype(int)

    usable_years = sorted(plot_df["year"].unique().tolist())
    n_usable_years = len(usable_years)

    if n_usable_years < MIN_PAIRED_YEARS:
        raise ValueError(
            f"{state}: only {n_usable_years} usable PLOT years."
        )

    # --------------------------------------------------------
    # 2. Stable-domain grid cells
    # --------------------------------------------------------
    plot_grid_df = add_grid_columns(plot_df)

    plot_grid_year_df = (
        plot_grid_df.groupby(
            [
                "grid_id",
                "grid_lat_index",
                "grid_lon_index",
                "year",
            ],
            as_index=False,
        )
        .agg(
            n_sampled_plots=("plot_key", "nunique"),
            cell_latitude=("latitude", "mean"),
            cell_longitude=("longitude", "mean"),
        )
    )

    early_years = set(usable_years[:WINDOW_SIZE])
    late_years = set(usable_years[-WINDOW_SIZE:])
    minimum_stable_years = math.ceil(
        STABLE_YEAR_FRACTION * n_usable_years
    )

    cell_coverage_df = (
        plot_grid_year_df.groupby(
            ["grid_id", "grid_lat_index", "grid_lon_index"],
            as_index=False,
        )
        .agg(
            n_sampled_years=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            mean_sampled_plots_year=("n_sampled_plots", "mean"),
            cell_latitude=("cell_latitude", "mean"),
            cell_longitude=("cell_longitude", "mean"),
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
        cell_coverage_df["n_sampled_years"].ge(
            minimum_stable_years
        )
        & cell_coverage_df["n_early_years"].ge(
            MIN_EARLY_WINDOW_YEARS
        )
        & cell_coverage_df["n_late_years"].ge(
            MIN_LATE_WINDOW_YEARS
        )
    )

    stable_cells_df = cell_coverage_df[
        cell_coverage_df["stable_domain"]
    ].copy()

    stable_ids = set(stable_cells_df["grid_id"])

    cell_coverage_df.to_csv(
        state_dir / "grid_cell_coverage.csv",
        index=False,
    )
    stable_cells_df.to_csv(
        state_dir / "stable_grid_cells.csv",
        index=False,
    )

    stable_plot_fraction = float(
        plot_grid_df["grid_id"].isin(stable_ids).mean()
    )

    # --------------------------------------------------------
    # 3. Resolve and scan TREE
    # --------------------------------------------------------
    tree_columns = pd.read_csv(
        tree_csv,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    tree_plot_key_col = resolve_column(
        tree_columns, ["PLT_CN", "PLOT_CN"]
    )
    tree_year_col = resolve_column(
        tree_columns, ["INVYR", "INVENTORY_YEAR"]
    )
    species_col = resolve_column(
        tree_columns, ["SPCD", "SPECIES_CODE"]
    )
    tree_status_col = resolve_column(
        tree_columns, ["STATUSCD", "STATUS_CODE"]
    )

    if not all([tree_plot_key_col, tree_year_col, species_col]):
        raise ValueError(
            f"{state}: required TREE fields were not found."
        )

    tree_usecols = [
        tree_plot_key_col,
        tree_year_col,
        species_col,
    ]

    if tree_status_col:
        tree_usecols.append(tree_status_col)

    tree_usecols = list(dict.fromkeys(tree_usecols))

    plot_lookup = plot_df.rename(
        columns={"year": "plot_year"}
    )[
        [
            "plot_key",
            "plot_year",
            "latitude",
            "longitude",
            "elevation",
        ]
    ]

    presence_parts = []

    tree_rows_scanned = 0
    selected_live_rows = 0
    matched_rows = 0
    unmatched_rows = 0
    year_mismatch_rows = 0

    tree_reader = pd.read_csv(
        tree_csv,
        usecols=tree_usecols,
        chunksize=TREE_CHUNK_SIZE,
        low_memory=False,
        encoding_errors="replace",
        dtype={tree_plot_key_col: "string"},
    )

    for chunk in tqdm(
        tree_reader,
        desc=f"{state}: scanning TREE",
        unit="chunk",
        leave=True,
    ):
        tree_rows_scanned += len(chunk)

        rename_map = {
            tree_plot_key_col: "plot_key",
            tree_year_col: "tree_year",
            species_col: "species_code",
        }

        if tree_status_col:
            rename_map[tree_status_col] = "status_code"

        chunk = chunk.rename(columns=rename_map)

        chunk["plot_key"] = (
            chunk["plot_key"].astype("string").str.strip()
        )
        chunk["tree_year"] = pd.to_numeric(
            chunk["tree_year"],
            errors="coerce",
        )
        chunk["species_code"] = pd.to_numeric(
            chunk["species_code"],
            errors="coerce",
        )

        keep = (
            chunk["tree_year"].ge(MIN_YEAR)
            & chunk["species_code"].isin(selected_species_codes)
            & chunk["plot_key"].notna()
        )

        if "status_code" in chunk.columns:
            chunk["status_code"] = pd.to_numeric(
                chunk["status_code"],
                errors="coerce",
            )
            keep &= chunk["status_code"].eq(1)

        work = chunk.loc[
            keep,
            ["plot_key", "tree_year", "species_code"],
        ].copy()

        selected_live_rows += len(work)

        if work.empty:
            continue

        work["tree_year"] = work["tree_year"].astype(int)
        work["species_code"] = work["species_code"].astype(int)

        joined = work.merge(
            plot_lookup,
            on="plot_key",
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        matched_mask = joined["_merge"].eq("both")
        matched_rows += int(matched_mask.sum())
        unmatched_rows += int((~matched_mask).sum())

        joined = joined.loc[matched_mask].drop(columns="_merge")

        year_mismatch = (
            joined["tree_year"] != joined["plot_year"]
        )
        year_mismatch_rows += int(year_mismatch.sum())

        joined = joined.loc[~year_mismatch].copy()
        joined = joined.rename(columns={"tree_year": "year"})

        # One species presence per plot-year.
        joined = joined[
            [
                "species_code",
                "plot_key",
                "year",
                "latitude",
                "longitude",
                "elevation",
            ]
        ].drop_duplicates(
            ["species_code", "plot_key", "year"]
        )

        presence_parts.append(joined)

    if not presence_parts:
        raise ValueError(
            f"{state}: no selected species matched usable plots."
        )

    presence_df = (
        pd.concat(presence_parts, ignore_index=True)
        .drop_duplicates(
            ["species_code", "plot_key", "year"]
        )
        .reset_index(drop=True)
    )

    del presence_parts
    gc.collect()

    presence_df.to_parquet(
        state_dir / "species_plot_year_presence.parquet",
        index=False,
    )

    join_match_rate = (
        matched_rows / selected_live_rows
        if selected_live_rows else 0.0
    )
    year_mismatch_rate = (
        year_mismatch_rows / matched_rows
        if matched_rows else np.nan
    )

    # --------------------------------------------------------
    # 4. Raw and fixed-domain annual centers
    # --------------------------------------------------------
    raw_center_df = (
        presence_df.groupby(
            ["species_code", "year"],
            as_index=False,
        )
        .agg(
            n_occupied_plots=("plot_key", "nunique"),
            raw_mean_latitude=("latitude", "mean"),
        )
    )

    presence_grid_df = add_grid_columns(presence_df)

    species_grid_year_df = (
        presence_grid_df.groupby(
            [
                "species_code",
                "year",
                "grid_id",
                "grid_lat_index",
                "grid_lon_index",
            ],
            as_index=False,
        )
        .agg(
            n_occupied_plots_cell=("plot_key", "nunique"),
            occupied_cell_latitude=("latitude", "mean"),
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
                "occupied_cell_latitude",
                "mean",
            ),
        )
    )

    fixed_grid_df = species_grid_year_df[
        species_grid_year_df["grid_id"].isin(stable_ids)
    ].copy()

    fixed_center_df = (
        fixed_grid_df.groupby(
            ["species_code", "year"],
            as_index=False,
        )
        .agg(
            fixed_n_occupied_cells=("grid_id", "nunique"),
            fixed_mean_latitude=(
                "occupied_cell_latitude",
                "mean",
            ),
        )
    )

    annual_center_df = (
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

    annual_center_df.to_parquet(
        state_dir / "annual_species_centers.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # 5. State-level slopes
    # --------------------------------------------------------
    slope_rows = []

    species_groups = annual_center_df.groupby(
        "species_code",
        sort=True,
    )

    for species_code, group in tqdm(
        species_groups,
        total=species_groups.ngroups,
        desc=f"{state}: fitting species slopes",
        unit="species",
        leave=True,
    ):
        paired = group[
            group["n_occupied_plots"].ge(
                MIN_RAW_PLOTS_PER_YEAR
            )
            & group["fixed_n_occupied_cells"].ge(
                MIN_FIXED_CELLS_PER_YEAR
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

        first = group.iloc[0]

        raw_robust = (
            raw_fit["theil_sen_slope"] * 111.32 * 10
            if np.isfinite(raw_fit["theil_sen_slope"])
            else np.nan
        )
        matched_robust = (
            matched_fit["theil_sen_slope"] * 111.32 * 10
            if np.isfinite(matched_fit["theil_sen_slope"])
            else np.nan
        )
        fixed_robust = (
            fixed_fit["theil_sen_slope"] * 111.32 * 10
            if np.isfinite(fixed_fit["theil_sen_slope"])
            else np.nan
        )

        raw_ols = (
            raw_fit["ols_slope"] * 111.32 * 10
            if np.isfinite(raw_fit["ols_slope"])
            else np.nan
        )
        fixed_ols = (
            fixed_fit["ols_slope"] * 111.32 * 10
            if np.isfinite(fixed_fit["ols_slope"])
            else np.nan
        )

        slope_rows.append(
            {
                "state": state,
                "species_code": int(species_code),
                "scientific_name": first["scientific_name"],
                "common_name": first["common_name"],
                "n_paired_years": raw_fit["n_years"],
                "median_occupied_plots_year": (
                    float(paired["n_occupied_plots"].median())
                    if not paired.empty else np.nan
                ),
                "median_fixed_cells_year": (
                    float(
                        paired[
                            "fixed_n_occupied_cells"
                        ].median()
                    )
                    if not paired.empty else np.nan
                ),
                "raw_latitude_km_decade": raw_robust,
                "matched_grid_latitude_km_decade": (
                    matched_robust
                ),
                "fixed_domain_latitude_km_decade": (
                    fixed_robust
                ),
                "correction_effect_km_decade": (
                    raw_robust - fixed_robust
                    if np.isfinite(raw_robust)
                    and np.isfinite(fixed_robust)
                    else np.nan
                ),
                "raw_ols_km_decade": raw_ols,
                "fixed_ols_km_decade": fixed_ols,
                "correction_effect_ols_km_decade": (
                    raw_ols - fixed_ols
                    if np.isfinite(raw_ols)
                    and np.isfinite(fixed_ols)
                    else np.nan
                ),
                "raw_ols_p": raw_fit["ols_p"],
                "fixed_ols_p": fixed_fit["ols_p"],
            }
        )

    slopes_df = pd.DataFrame(slope_rows)

    slopes_df["material_difference"] = (
        slopes_df["correction_effect_km_decade"].abs()
        >= MATERIAL_DIFF_KM_DECADE
    )

    slopes_df["meaningful_direction_reversal"] = [
        meaningful_reversal(raw, fixed)
        for raw, fixed in zip(
            slopes_df["raw_latitude_km_decade"],
            slopes_df["fixed_domain_latitude_km_decade"],
        )
    ]

    slopes_df = slopes_df.sort_values(
        "correction_effect_km_decade",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)

    slopes_df.to_parquet(
        state_dir / "species_slopes.parquet",
        index=False,
    )
    slopes_df.to_csv(
        state_dir / "species_slopes.csv",
        index=False,
    )

    strong_df = slopes_df[
        slopes_df["n_paired_years"] >= MIN_PAIRED_YEARS
    ].copy()

    n_strong_species = len(strong_df)
    median_abs_effect = (
        float(
            strong_df["correction_effect_km_decade"]
            .abs()
            .median()
        )
        if n_strong_species else np.nan
    )
    n_material = int(strong_df["material_difference"].sum())
    n_reversal = int(
        strong_df["meaningful_direction_reversal"].sum()
    )
    reversal_fraction = (
        n_reversal / n_strong_species
        if n_strong_species else np.nan
    )

    # --------------------------------------------------------
    # 6. Global year-order permutation within state
    # --------------------------------------------------------
    permutation_input_df = annual_center_df[
        annual_center_df["n_occupied_plots"].ge(
            MIN_RAW_PLOTS_PER_YEAR
        )
        & annual_center_df["fixed_n_occupied_cells"].ge(
            MIN_FIXED_CELLS_PER_YEAR
        )
    ][
        [
            "species_code",
            "year",
            "raw_mean_latitude",
            "fixed_mean_latitude",
        ]
    ].dropna()

    eligible_counts = (
        permutation_input_df.groupby(
            "species_code"
        )["year"].nunique()
    )
    eligible_species = eligible_counts[
        eligible_counts >= MIN_PAIRED_YEARS
    ].index.tolist()

    permutation_input_df = permutation_input_df[
        permutation_input_df["species_code"].isin(
            eligible_species
        )
    ].copy()

    permutation_groups = {
        int(species_code): group.sort_values("year").copy()
        for species_code, group in permutation_input_df.groupby(
            "species_code"
        )
    }

    observed_rows = []

    for species_code, group in permutation_groups.items():
        raw_slope = (
            ols_slope_fast(
                group["year"],
                group["raw_mean_latitude"],
            )
            * 111.32 * 10
        )
        fixed_slope = (
            ols_slope_fast(
                group["year"],
                group["fixed_mean_latitude"],
            )
            * 111.32 * 10
        )

        observed_rows.append(
            {
                "species_code": species_code,
                "difference": raw_slope - fixed_slope,
                "reversal": meaningful_reversal(
                    raw_slope,
                    fixed_slope,
                ),
            }
        )

    observed_df = pd.DataFrame(observed_rows)

    observed_median_abs = float(
        observed_df["difference"].abs().median()
    )
    observed_reversal_fraction = float(
        observed_df["reversal"].mean()
    )

    unique_years = np.array(
        sorted(permutation_input_df["year"].unique()),
        dtype=int,
    )

    rng = np.random.default_rng(permutation_seed)
    permutation_rows = []

    for permutation_id in tqdm(
        range(N_PERMUTATIONS),
        desc=f"{state}: year permutations",
        unit="permutation",
        leave=True,
    ):
        shuffled_years = rng.permutation(unique_years)
        year_map = dict(zip(unique_years, shuffled_years))

        differences = []
        reversals = []

        for species_code, group in permutation_groups.items():
            permuted_years = group["year"].map(
                year_map
            ).to_numpy()

            raw_slope = (
                ols_slope_fast(
                    permuted_years,
                    group["raw_mean_latitude"].to_numpy(),
                )
                * 111.32 * 10
            )

            fixed_slope = (
                ols_slope_fast(
                    permuted_years,
                    group[
                        "fixed_mean_latitude"
                    ].to_numpy(),
                )
                * 111.32 * 10
            )

            if np.isfinite(raw_slope) and np.isfinite(fixed_slope):
                differences.append(raw_slope - fixed_slope)
                reversals.append(
                    meaningful_reversal(
                        raw_slope,
                        fixed_slope,
                    )
                )

        permutation_rows.append(
            {
                "state": state,
                "permutation_id": permutation_id,
                "median_abs_difference": (
                    float(np.median(np.abs(differences)))
                    if differences else np.nan
                ),
                "reversal_fraction": (
                    float(np.mean(reversals))
                    if reversals else np.nan
                ),
            }
        )

    permutation_df = pd.DataFrame(permutation_rows)

    p_median = float(
        (
            1
            + (
                permutation_df["median_abs_difference"]
                >= observed_median_abs
            ).sum()
        )
        / (N_PERMUTATIONS + 1)
    )

    p_reversal = float(
        (
            1
            + (
                permutation_df["reversal_fraction"]
                >= observed_reversal_fraction
            ).sum()
        )
        / (N_PERMUTATIONS + 1)
    )

    permutation_df.to_parquet(
        state_dir / "year_permutation_null.parquet",
        index=False,
    )

    domain_pass = bool(
        len(stable_cells_df) >= MIN_STABLE_CELLS
        and n_strong_species >= MIN_STRONG_SPECIES_STATE
        and join_match_rate >= 0.95
    )

    material_signal = bool(
        (
            np.isfinite(median_abs_effect)
            and median_abs_effect >= MATERIAL_DIFF_KM_DECADE
        )
        or (
            np.isfinite(reversal_fraction)
            and reversal_fraction >= TARGET_REVERSAL_FRACTION
        )
    )

    permutation_pass = bool(
        p_median <= 0.05 or p_reversal <= 0.05
    )

    state_replication_pass = bool(
        domain_pass
        and material_signal
        and permutation_pass
    )

    summary = {
        "state": state,
        "plot_rows": len(plot_df),
        "tree_rows_scanned": tree_rows_scanned,
        "selected_live_tree_rows": selected_live_rows,
        "join_match_rate": join_match_rate,
        "year_mismatch_rate": year_mismatch_rate,
        "presence_rows": len(presence_df),
        "n_usable_years": n_usable_years,
        "n_all_grid_cells": int(
            cell_coverage_df["grid_id"].nunique()
        ),
        "n_stable_grid_cells": len(stable_cells_df),
        "stable_plot_fraction": stable_plot_fraction,
        "n_strong_species": n_strong_species,
        "median_abs_correction_km_decade": median_abs_effect,
        "n_material_difference": n_material,
        "n_meaningful_reversal": n_reversal,
        "meaningful_reversal_fraction": reversal_fraction,
        "permutation_observed_median_abs": observed_median_abs,
        "permutation_null95_median_abs": float(
            permutation_df[
                "median_abs_difference"
            ].quantile(0.95)
        ),
        "permutation_p_median_abs": p_median,
        "permutation_observed_reversal_fraction": (
            observed_reversal_fraction
        ),
        "permutation_null95_reversal_fraction": float(
            permutation_df[
                "reversal_fraction"
            ].quantile(0.95)
        ),
        "permutation_p_reversal": p_reversal,
        "domain_pass": domain_pass,
        "material_signal": material_signal,
        "permutation_pass": permutation_pass,
        "state_replication_pass": state_replication_pass,
    }

    del (
        plot_df,
        plot_grid_df,
        plot_grid_year_df,
        presence_df,
        presence_grid_df,
        species_grid_year_df,
        fixed_grid_df,
        annual_center_df,
        permutation_input_df,
    )
    gc.collect()

    return slopes_df, summary


# ============================================================
# Main
# ============================================================
try:
    # --------------------------------------------------------
    # 1. Load the fixed WV species set
    # --------------------------------------------------------
    if not SELECTED_SPECIES_CSV.exists():
        raise FileNotFoundError(
            f"Missing selected species file: {SELECTED_SPECIES_CSV}"
        )

    selected_species_df = pd.read_csv(
        SELECTED_SPECIES_CSV
    )

    taxonomy_df = (
        selected_species_df[
            ["species_code", "scientific_name", "common_name"]
        ]
        .drop_duplicates("species_code")
        .copy()
    )

    taxonomy_df["species_code"] = pd.to_numeric(
        taxonomy_df["species_code"],
        errors="coerce",
    )
    taxonomy_df = taxonomy_df.dropna(
        subset=["species_code"]
    ).copy()
    taxonomy_df["species_code"] = taxonomy_df[
        "species_code"
    ].astype(int)

    selected_species_codes = set(
        taxonomy_df["species_code"].tolist()
    )

    if len(selected_species_codes) < 20:
        raise ValueError(
            "The prior WV selected-species set is unexpectedly small."
        )

    print("\nFIXED CROSS-STATE SPECIES SET")
    display(taxonomy_df)

    # --------------------------------------------------------
    # 2. Download PA, VA, KY; reuse WV
    # --------------------------------------------------------
    manifest_rows = []

    for state in tqdm(
        STATES,
        desc="Preparing state files",
        unit="state",
    ):
        for table_name in ["PLOT", "TREE"]:
            url = f"{BASE_URL}/{state}_{table_name}.csv"
            destination = RAW_DIR / f"{state}_{table_name}.csv"

            if state == "WV" and destination.exists():
                manifest_rows.append(
                    {
                        "state": state,
                        "table": table_name,
                        "url": url,
                        "path": str(destination),
                        "status": "reused",
                        "size_bytes": destination.stat().st_size,
                    }
                )
            else:
                result = stream_download(
                    url,
                    destination,
                    f"Downloading {state}_{table_name}",
                )
                manifest_rows.append(
                    {
                        "state": state,
                        "table": table_name,
                        **result,
                    }
                )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df["size"] = manifest_df["size_bytes"].map(
        human_bytes
    )
    manifest_df["download_date_utc"] = RUN_UTC

    manifest_df.to_csv(
        OUT_DIR / "download_manifest.csv",
        index=False,
    )

    print("\nDOWNLOAD MANIFEST")
    display(
        manifest_df[
            [
                "state",
                "table",
                "status",
                "size",
                "url",
            ]
        ]
    )

    # --------------------------------------------------------
    # 3. Process all four states with the same method
    # --------------------------------------------------------
    state_slope_tables = []
    state_summary_rows = []

    for state_index, state in enumerate(STATES):
        print("\n" + "=" * 72)
        print(f"PROCESSING STATE: {state}")
        print("=" * 72)

        plot_csv = RAW_DIR / f"{state}_PLOT.csv"
        tree_csv = RAW_DIR / f"{state}_TREE.csv"

        if not plot_csv.exists() or not tree_csv.exists():
            raise FileNotFoundError(
                f"{state}: PLOT or TREE file is missing."
            )

        slopes_df, summary = process_state(
            state=state,
            plot_csv=plot_csv,
            tree_csv=tree_csv,
            selected_species_codes=selected_species_codes,
            taxonomy_df=taxonomy_df,
            permutation_seed=(
                RANDOM_SEED + state_index * 10_000
            ),
        )

        state_slope_tables.append(slopes_df)
        state_summary_rows.append(summary)

        print(f"\n{state} SUMMARY")
        display(pd.DataFrame([summary]))

    all_state_slopes_df = pd.concat(
        state_slope_tables,
        ignore_index=True,
    )

    state_summary_df = pd.DataFrame(
        state_summary_rows
    ).sort_values("state").reset_index(drop=True)

    all_state_slopes_df.to_parquet(
        OUT_DIR / "all_state_species_slopes.parquet",
        index=False,
    )
    all_state_slopes_df.to_csv(
        OUT_DIR / "all_state_species_slopes.csv",
        index=False,
    )
    state_summary_df.to_csv(
        OUT_DIR / "state_replication_summary.csv",
        index=False,
    )

    print("\nFOUR-STATE SUMMARY")
    display(state_summary_df)

    # --------------------------------------------------------
    # 4. Transparent species-level meta-summary
    # --------------------------------------------------------
    eligible_state_effects_df = all_state_slopes_df[
        all_state_slopes_df["n_paired_years"]
        >= MIN_PAIRED_YEARS
    ].dropna(
        subset=["correction_effect_km_decade"]
    ).copy()

    meta_rows = []

    meta_groups = eligible_state_effects_df.groupby(
        "species_code",
        sort=True,
    )

    for species_code, group in tqdm(
        meta_groups,
        total=meta_groups.ngroups,
        desc="Building species meta-summary",
        unit="species",
    ):
        effects = group[
            "correction_effect_km_decade"
        ].to_numpy(dtype=float)

        n_states = group["state"].nunique()
        positive_count = int((effects > 0).sum())
        negative_count = int((effects < 0).sum())
        nonzero_count = positive_count + negative_count

        majority_fraction = (
            max(positive_count, negative_count) / nonzero_count
            if nonzero_count else np.nan
        )

        non_wv_effects = group.loc[
            group["state"] != "WV",
            "correction_effect_km_decade",
        ]

        first = group.iloc[0]

        meta_rows.append(
            {
                "species_code": int(species_code),
                "scientific_name": first["scientific_name"],
                "common_name": first["common_name"],
                "n_states": int(n_states),
                "median_correction_km_decade": float(
                    np.median(effects)
                ),
                "median_abs_correction_km_decade": float(
                    np.median(np.abs(effects))
                ),
                "minimum_correction_km_decade": float(
                    np.min(effects)
                ),
                "maximum_correction_km_decade": float(
                    np.max(effects)
                ),
                "positive_state_count": positive_count,
                "negative_state_count": negative_count,
                "majority_sign_fraction": majority_fraction,
                "non_wv_state_count": int(
                    non_wv_effects.notna().sum()
                ),
                "non_wv_median_correction_km_decade": (
                    float(non_wv_effects.median())
                    if non_wv_effects.notna().any()
                    else np.nan
                ),
                "non_wv_median_abs_correction_km_decade": (
                    float(non_wv_effects.abs().median())
                    if non_wv_effects.notna().any()
                    else np.nan
                ),
            }
        )

    species_meta_df = pd.DataFrame(meta_rows)

    species_meta_df["eligible_meta_species"] = (
        species_meta_df["n_states"] >= MIN_META_STATES
    )

    species_meta_df = species_meta_df.sort_values(
        "median_abs_correction_km_decade",
        ascending=False,
    ).reset_index(drop=True)

    species_meta_df.to_parquet(
        OUT_DIR / "species_meta_summary.parquet",
        index=False,
    )
    species_meta_df.to_csv(
        OUT_DIR / "species_meta_summary.csv",
        index=False,
    )

    print("\nSPECIES META-SUMMARY")
    display(species_meta_df.head(50))

    eligible_meta_df = species_meta_df[
        species_meta_df["eligible_meta_species"]
    ].copy()

    n_meta_species = len(eligible_meta_df)

    majority_consistency_fraction = (
        float(
            (
                eligible_meta_df[
                    "majority_sign_fraction"
                ] >= 0.75
            ).mean()
        )
        if n_meta_species else np.nan
    )

    non_wv_median_abs_effect = (
        float(
            eligible_meta_df[
                "non_wv_median_abs_correction_km_decade"
            ].median()
        )
        if n_meta_species else np.nan
    )

    # --------------------------------------------------------
    # 5. Leave-one-state-out meta stability
    # --------------------------------------------------------
    full_meta_lookup = (
        eligible_meta_df.set_index("species_code")[
            "median_correction_km_decade"
        ]
    )

    loo_rows = []
    loo_species_rows = []

    for held_out_state in tqdm(
        STATES,
        desc="Leave-one-state-out analysis",
        unit="state",
    ):
        remaining_df = eligible_state_effects_df[
            eligible_state_effects_df["state"]
            != held_out_state
        ].copy()

        loo_meta = (
            remaining_df.groupby(
                "species_code"
            )["correction_effect_km_decade"]
            .agg(["median", "count"])
            .rename(
                columns={
                    "median": "loo_median_effect",
                    "count": "loo_n_states",
                }
            )
        )

        comparison_df = pd.DataFrame(
            {
                "full_meta_effect": full_meta_lookup,
            }
        ).join(
            loo_meta,
            how="inner",
        )

        comparison_df = comparison_df[
            comparison_df["loo_n_states"] >= 2
        ].copy()

        comparison_df["sign_stable"] = (
            np.sign(comparison_df["full_meta_effect"])
            == np.sign(comparison_df["loo_median_effect"])
        )

        sign_agreement = (
            float(comparison_df["sign_stable"].mean())
            if len(comparison_df) else np.nan
        )

        magnitude_spearman = safe_spearman(
            comparison_df["full_meta_effect"],
            comparison_df["loo_median_effect"],
        )

        loo_rows.append(
            {
                "held_out_state": held_out_state,
                "n_species_compared": len(comparison_df),
                "sign_agreement_with_full_meta": (
                    sign_agreement
                ),
                "magnitude_spearman_with_full_meta": (
                    magnitude_spearman
                ),
                "median_abs_change_from_full_meta": (
                    float(
                        (
                            comparison_df["full_meta_effect"]
                            - comparison_df["loo_median_effect"]
                        ).abs().median()
                    )
                    if len(comparison_df) else np.nan
                ),
            }
        )

        for species_code, row in comparison_df.iterrows():
            loo_species_rows.append(
                {
                    "held_out_state": held_out_state,
                    "species_code": int(species_code),
                    "full_meta_effect": row[
                        "full_meta_effect"
                    ],
                    "loo_meta_effect": row[
                        "loo_median_effect"
                    ],
                    "loo_n_states": int(
                        row["loo_n_states"]
                    ),
                    "sign_stable": bool(
                        row["sign_stable"]
                    ),
                }
            )

    loo_summary_df = pd.DataFrame(loo_rows)
    loo_species_df = pd.DataFrame(loo_species_rows)

    loo_summary_df.to_csv(
        OUT_DIR / "leave_one_state_out_summary.csv",
        index=False,
    )
    loo_species_df.to_parquet(
        OUT_DIR / "leave_one_state_out_species.parquet",
        index=False,
    )

    print("\nLEAVE-ONE-STATE-OUT SUMMARY")
    display(loo_summary_df)

    overall_loo_sign_stability = (
        float(loo_species_df["sign_stable"].mean())
        if len(loo_species_df) else np.nan
    )

    minimum_loo_state_sign_agreement = (
        float(
            loo_summary_df[
                "sign_agreement_with_full_meta"
            ].min()
        )
        if len(loo_summary_df) else np.nan
    )

    # --------------------------------------------------------
    # 6. Replication decision
    # --------------------------------------------------------
    n_domain_states = int(
        state_summary_df["domain_pass"].sum()
    )
    n_material_states = int(
        state_summary_df["material_signal"].sum()
    )
    n_permutation_states = int(
        state_summary_df["permutation_pass"].sum()
    )
    n_replication_states = int(
        state_summary_df["state_replication_pass"].sum()
    )

    n_added_replication_states = int(
        state_summary_df[
            state_summary_df["state"].isin(ADDED_STATES)
        ]["state_replication_pass"].sum()
    )

    state_replication_criterion = bool(
        n_domain_states >= 3
        and n_replication_states >= 3
        and n_added_replication_states >= 2
    )

    meta_coverage_criterion = bool(
        n_meta_species >= MIN_META_SPECIES
    )

    non_wv_effect_criterion = bool(
        np.isfinite(non_wv_median_abs_effect)
        and non_wv_median_abs_effect
        >= TARGET_NON_WV_MEDIAN_ABS_EFFECT
    )

    cross_state_consistency_criterion = bool(
        np.isfinite(majority_consistency_fraction)
        and majority_consistency_fraction
        >= TARGET_STATE_MAJORITY_CONSISTENCY
    )

    loo_criterion = bool(
        np.isfinite(overall_loo_sign_stability)
        and overall_loo_sign_stability
        >= TARGET_LOO_SIGN_STABILITY
    )

    passed = bool(
        state_replication_criterion
        and meta_coverage_criterion
        and non_wv_effect_criterion
        and cross_state_consistency_criterion
        and loo_criterion
    )

    if passed:
        status = "GO_FOR_OBSERVATION_DATA_COMPARISON"
        next_step = (
            "The FIA mechanism replicated across states. "
            "Next obtain opportunity-based GBIF/iNaturalist records "
            "for the same species and compare bias sensitivity."
        )
    elif (
        state_replication_criterion
        and meta_coverage_criterion
        and non_wv_effect_criterion
    ):
        status = "CROSS_STATE_EFFECT_WITH_HETEROGENEITY"
        next_step = (
            "The effect replicated but direction varies by state. "
            "Stratify by range position, topography, and species occupancy "
            "before moving to opportunity-based records."
        )
    elif n_added_replication_states >= 1:
        status = "PARTIAL_CROSS_STATE_REPLICATION"
        next_step = (
            "The WV result is not fully general. Inspect which added state "
            "replicated and whether inventory-panel structure explains failure."
        )
    else:
        status = "WV_SIGNAL_NOT_REPLICATED"
        next_step = (
            "Do not scale to GBIF yet. Treat the WV result as state-specific "
            "until panel-matched or repeated-plot controls clarify it."
        )

    # --------------------------------------------------------
    # 7. Visual previews
    # --------------------------------------------------------
    plot_state_summary = state_summary_df.sort_values(
        "state"
    ).copy()

    x = np.arange(len(plot_state_summary))

    plt.figure(figsize=(9, 5))
    plt.bar(
        x,
        plot_state_summary[
            "median_abs_correction_km_decade"
        ],
    )
    plt.axhline(
        MATERIAL_DIFF_KM_DECADE,
        linestyle="--",
        label="Material threshold",
    )
    plt.xticks(x, plot_state_summary["state"])
    plt.ylabel(
        "Median absolute raw-fixed difference (km/decade)"
    )
    plt.xlabel("State")
    plt.title("Appalachian state replication")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if not eligible_meta_df.empty:
        meta_plot_df = (
            eligible_meta_df.head(20)
            .sort_values("median_correction_km_decade")
            .copy()
        )

        labels = meta_plot_df[
            "scientific_name"
        ].fillna(
            meta_plot_df["species_code"].astype(str)
        )

        plt.figure(figsize=(10, 8))
        plt.barh(
            labels,
            meta_plot_df[
                "median_correction_km_decade"
            ],
        )
        plt.axvline(0)
        plt.xlabel(
            "Median state correction effect "
            "(raw minus fixed, km/decade)"
        )
        plt.ylabel("Species")
        plt.title("Cross-state species correction effects")
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(9, 5))
    plt.bar(
        loo_summary_df["held_out_state"],
        loo_summary_df[
            "sign_agreement_with_full_meta"
        ],
    )
    plt.axhline(
        TARGET_LOO_SIGN_STABILITY,
        linestyle="--",
        label="Target",
    )
    plt.ylim(0, 1.05)
    plt.xlabel("Held-out state")
    plt.ylabel("Sign agreement with full meta")
    plt.title("Leave-one-state-out stability")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 8. README
    # --------------------------------------------------------
    top_meta_species = (
        eligible_meta_df[
            [
                "species_code",
                "scientific_name",
                "n_states",
                "median_correction_km_decade",
                "median_abs_correction_km_decade",
                "majority_sign_fraction",
                "non_wv_median_correction_km_decade",
            ]
        ]
        .head(10)
        .round(4)
        .to_dict("records")
    )

    append_readme(
        f"""

## Four-state Appalachian replication — {RUN_UTC}

### Hypothesis
The WV raw-versus-fixed-domain difference should reproduce in neighboring
Appalachian states and remain stable when any one state is removed.

### Data
- States: {STATES}
- Source: official FIA DataMart state PLOT and TREE CSV files
- Download/audit date: {RUN_UTC}
- Fixed species set: {len(selected_species_codes)} species selected in WV

### Configuration
- Minimum year: {MIN_YEAR}
- Grid resolution: {GRID_RESOLUTION} degrees
- Stable-domain fraction: {STABLE_YEAR_FRACTION}
- Minimum paired years: {MIN_PAIRED_YEARS}
- Minimum raw occupied plots per species-year:
  {MIN_RAW_PLOTS_PER_YEAR}
- Minimum fixed cells per species-year:
  {MIN_FIXED_CELLS_PER_YEAR}
- Permutations per state: {N_PERMUTATIONS}
- Random seed: {RANDOM_SEED}
- GPU: not used

### State-level results
- States with adequate domains: {n_domain_states}/4
- States with material signals: {n_material_states}/4
- States passing year permutation: {n_permutation_states}/4
- Full state replication passes: {n_replication_states}/4
- Added-state replication passes:
  {n_added_replication_states}/3

### Cross-state meta results
- Species represented in at least {MIN_META_STATES} states:
  {n_meta_species}
- Fraction with at least 75% state-sign agreement:
  {majority_consistency_fraction}
- Median non-WV absolute correction:
  {non_wv_median_abs_effect} km/decade

### Leave-one-state-out results
- Overall sign stability: {overall_loo_sign_stability}
- Minimum held-out-state sign agreement:
  {minimum_loo_state_sign_agreement}

### Decision
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
The result measures sensitivity to changing sampled spatial support in FIA.
It does not identify climate causality, biological migration, abundance change,
recruitment, mortality, detectability, or disturbance mechanisms. Public FIA
coordinates are perturbed, and inventory panel identity is not yet explicitly
modeled.
"""
    )

    # --------------------------------------------------------
    # 9. Compact output block
    # --------------------------------------------------------
    state_compact = (
        state_summary_df[
            [
                "state",
                "n_usable_years",
                "n_stable_grid_cells",
                "n_strong_species",
                "median_abs_correction_km_decade",
                "meaningful_reversal_fraction",
                "permutation_p_median_abs",
                "permutation_p_reversal",
                "state_replication_pass",
            ]
        ]
        .round(6)
        .to_dict("records")
    )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATES: {STATES}")
    print(f"N_FIXED_SPECIES: {len(selected_species_codes)}")
    print(
        "STATE_RESULTS: "
        + json.dumps(state_compact, ensure_ascii=False)
    )
    print(f"N_DOMAIN_PASS_STATES: {n_domain_states}")
    print(f"N_MATERIAL_SIGNAL_STATES: {n_material_states}")
    print(
        f"N_PERMUTATION_PASS_STATES: "
        f"{n_permutation_states}"
    )
    print(
        f"N_FULL_REPLICATION_STATES: "
        f"{n_replication_states}"
    )
    print(
        f"N_ADDED_REPLICATION_STATES: "
        f"{n_added_replication_states}"
    )
    print(f"N_META_SPECIES: {n_meta_species}")
    print(
        f"STATE_MAJORITY_SIGN_CONSISTENCY_FRACTION: "
        f"{majority_consistency_fraction}"
    )
    print(
        f"NON_WV_MEDIAN_ABS_EFFECT_KM_DECADE: "
        f"{non_wv_median_abs_effect}"
    )
    print(
        f"OVERALL_LOO_SIGN_STABILITY: "
        f"{overall_loo_sign_stability}"
    )
    print(
        f"MIN_LOO_STATE_SIGN_AGREEMENT: "
        f"{minimum_loo_state_sign_agreement}"
    )
    print(
        "LOO_SUMMARY: "
        + json.dumps(
            loo_summary_df.round(6).to_dict("records"),
            ensure_ascii=False,
        )
    )
    print(
        "TOP_META_SPECIES: "
        + json.dumps(top_meta_species, ensure_ascii=False)
    )
    print(
        "SUCCESS_CRITERIA: >=3 state replication passes including "
        ">=2 added states; >=20 meta species; non-WV median absolute "
        "effect >=2 km/decade; cross-state majority consistency >=0.60; "
        "leave-one-state-out sign stability >=0.80"
    )
    print(
        f"STATE_REPLICATION_CRITERION: "
        f"{state_replication_criterion}"
    )
    print(
        f"META_COVERAGE_CRITERION: "
        f"{meta_coverage_criterion}"
    )
    print(
        f"NON_WV_EFFECT_CRITERION: "
        f"{non_wv_effect_criterion}"
    )
    print(
        f"CROSS_STATE_CONSISTENCY_CRITERION: "
        f"{cross_state_consistency_criterion}"
    )
    print(f"LOO_CRITERION: {loo_criterion}")
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
    failing_url = (
        response.url
        if response is not None
        else "UNKNOWN"
    )

    print_failure(
        "DOWNLOAD_HTTP_FAILED",
        f"HTTP {status_code} for {failing_url}",
        "Send this COPY block back. Do not rerun completed downloads.",
    )

except requests.RequestException as exc:
    print_failure(
        "DOWNLOAD_CONNECTION_FAILED",
        repr(exc),
        "Retry once with Internet enabled. Existing completed files will be reused.",
    )

except Exception as exc:
    print_failure(
        "CELL05A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        "Send this COPY block and the last displayed state summary. "
        "Completed state downloads and outputs will be reused.",
    )
