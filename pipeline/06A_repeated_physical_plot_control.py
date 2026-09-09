
# Cell 06A — Repeated physical-plot early/late control across four states
#
# Why this step:
# The four-state fixed-grid effect did not replicate as a broad state-level
# phenomenon. FIA uses rotating inventory panels, so annual spatial support can
# vary even when the broad grid domain appears stable.
#
# Hypothesis:
# If the apparent range-shift sensitivity is truly caused by temporal observation
# support, raw early-to-late shifts should differ from shifts estimated on the
# exact same physical plots measured in both an early and a late window.
#
# Minimal design:
# - Same 42 species as prior cells.
# - Same four states: WV, PA, VA, KY.
# - Identify physical plots using FIA STATECD + UNITCD + COUNTYCD + PLOT.
# - Select one early and one late measurement per physical plot.
# - Retain only plots with live-tree sampling in both windows.
# - Compare:
#     raw shift: all usable early plots vs all usable late plots
#     matched shift: the same repeated physical plots in both windows
# - Use plot-level presence, not individual-tree abundance.
# - Run global within-plot early/late swap permutations.
#
# This is a mechanism-control experiment, not a final ecological model.

from pathlib import Path
from datetime import datetime, timezone
import gc
import json
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
STATES = ["WV", "PA", "VA", "KY"]
ADDED_STATES = ["PA", "VA", "KY"]

MIN_YEAR = 2000
WINDOW_SIZE = 6
TREE_CHUNK_SIZE = 250_000

MIN_RAW_OCCUPIED_PLOTS_WINDOW = 10
MIN_MATCHED_OCCUPIED_PLOTS_WINDOW = 10
MIN_MATCHED_PHYSICAL_PLOTS_STATE = 200
MIN_STRONG_SPECIES_STATE = 20

MATERIAL_CORRECTION_KM_DECADE = 1.50
MEANINGFUL_REVERSAL_MIN_MAG_KM_DECADE = 1.00
TARGET_REVERSAL_FRACTION = 0.10

N_PERMUTATIONS = 300
RANDOM_SEED = 20260710

MIN_META_STATES = 3
MIN_META_SPECIES = 20
TARGET_NON_WV_MEDIAN_ABS_CORRECTION = 1.50
TARGET_META_SIGN_CONSISTENCY = 0.60
TARGET_LOO_SIGN_STABILITY = 0.75

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
RAW_DIR = BASE_DIR / "raw"
CENTER_DIR = BASE_DIR / "derived" / "wv_center_screen"
OUT_DIR = BASE_DIR / "derived" / "repeated_physical_plot_control"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_SPECIES_CSV = CENTER_DIR / "selected_species.csv"

RUN_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

print("RUN_UTC:", RUN_UTC)
print("STATES:", STATES)
print("WINDOW_SIZE:", WINDOW_SIZE)
print("N_PERMUTATIONS:", N_PERMUTATIONS)
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


def resolve_column(columns, candidates):
    lookup = {str(column).upper(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    return None


def normalize_integer_component(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64").astype("string")


def build_physical_plot_id(frame, state_col, unit_col, county_col, plot_col):
    return (
        normalize_integer_component(frame[state_col])
        + "_"
        + normalize_integer_component(frame[unit_col])
        + "_"
        + normalize_integer_component(frame[county_col])
        + "_"
        + normalize_integer_component(frame[plot_col])
    )


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

    if interval <= 5:
        return np.nan

    return float(
        (late_latitude - early_latitude)
        * 111.32
        * 10
        / interval
    )


def meaningful_reversal(raw_shift, matched_shift):
    if not np.isfinite(raw_shift) or not np.isfinite(matched_shift):
        return False

    return bool(
        np.sign(raw_shift) != np.sign(matched_shift)
        and abs(raw_shift) >= MEANINGFUL_REVERSAL_MIN_MAG_KM_DECADE
        and abs(matched_shift) >= MEANINGFUL_REVERSAL_MIN_MAG_KM_DECADE
    )


def safe_spearman(x, y):
    values = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if (
        len(values) < 3
        or values["x"].nunique() < 2
        or values["y"].nunique() < 2
    ):
        return np.nan

    return float(values.corr(method="spearman").iloc[0, 1])


def print_failure(status, error, next_step):
    append_readme(
        f"""

## Repeated physical-plot control failure — {RUN_UTC}
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
# Per-state analysis
# ============================================================
def process_state(
    state,
    plot_csv,
    tree_csv,
    species_codes,
    taxonomy_df,
    random_seed,
):
    state_dir = OUT_DIR / state
    state_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Resolve PLOT fields
    # --------------------------------------------------------
    plot_columns = pd.read_csv(
        plot_csv,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    plot_key_col = resolve_column(plot_columns, ["CN", "PLOT_CN"])
    inventory_year_col = resolve_column(
        plot_columns,
        ["INVYR", "INVENTORY_YEAR"],
    )
    latitude_col = resolve_column(plot_columns, ["LAT", "LATITUDE"])
    longitude_col = resolve_column(plot_columns, ["LON", "LONGITUDE"])
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
            f"{state}: physical plot identity fields unavailable. "
            f"Resolved: plot_key={plot_key_col}, year={inventory_year_col}, "
            f"lat={latitude_col}, lon={longitude_col}, state={state_code_col}, "
            f"unit={unit_code_col}, county={county_code_col}, "
            f"plot={plot_number_col}"
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
            + ([plot_status_col] if plot_status_col else [])
        )
    )

    plot_df = pd.read_csv(
        plot_csv,
        usecols=plot_usecols,
        low_memory=False,
        encoding_errors="replace",
        dtype={plot_key_col: "string"},
    )

    plot_df["physical_plot_id"] = build_physical_plot_id(
        plot_df,
        state_code_col,
        unit_code_col,
        county_code_col,
        plot_number_col,
    )

    rename_map = {
        plot_key_col: "plot_key",
        inventory_year_col: "year",
        latitude_col: "latitude",
        longitude_col: "longitude",
    }

    if plot_status_col:
        rename_map[plot_status_col] = "plot_status"

    plot_df = plot_df.rename(columns=rename_map)

    if "plot_status" not in plot_df.columns:
        plot_df["plot_status"] = np.nan

    plot_df["plot_key"] = (
        plot_df["plot_key"].astype("string").str.strip()
    )

    for column in ["year", "latitude", "longitude", "plot_status"]:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        )

    valid_plot = (
        plot_df["year"].ge(MIN_YEAR)
        & plot_df["latitude"].between(24, 50)
        & plot_df["longitude"].between(-130, -60)
        & plot_df["plot_key"].notna()
        & plot_df["physical_plot_id"].notna()
    )

    if plot_status_col:
        valid_plot &= plot_df["plot_status"].eq(1)

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

    plot_df["year"] = plot_df["year"].astype(int)

    usable_years = sorted(plot_df["year"].unique().tolist())

    if len(usable_years) < 2 * WINDOW_SIZE:
        raise ValueError(
            f"{state}: only {len(usable_years)} usable years; "
            f"need at least {2 * WINDOW_SIZE}."
        )

    early_years = usable_years[:WINDOW_SIZE]
    late_years = usable_years[-WINDOW_SIZE:]

    # Choose the earliest usable measurement in the early window and
    # latest usable measurement in the late window for each physical plot.
    early_measurements_df = (
        plot_df[plot_df["year"].isin(early_years)]
        .sort_values(["physical_plot_id", "year", "plot_key"])
        .groupby("physical_plot_id", as_index=False)
        .first()
    )
    early_measurements_df["period"] = "early"

    late_measurements_df = (
        plot_df[plot_df["year"].isin(late_years)]
        .sort_values(
            ["physical_plot_id", "year", "plot_key"],
            ascending=[True, False, True],
        )
        .groupby("physical_plot_id", as_index=False)
        .first()
    )
    late_measurements_df["period"] = "late"

    candidate_measurements_df = pd.concat(
        [early_measurements_df, late_measurements_df],
        ignore_index=True,
    )

    candidate_plot_keys = set(
        candidate_measurements_df["plot_key"]
        .dropna()
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # 2. Resolve and scan TREE
    # --------------------------------------------------------
    tree_columns = pd.read_csv(
        tree_csv,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    tree_plot_key_col = resolve_column(
        tree_columns,
        ["PLT_CN", "PLOT_CN"],
    )
    species_col = resolve_column(
        tree_columns,
        ["SPCD", "SPECIES_CODE"],
    )
    tree_status_col = resolve_column(
        tree_columns,
        ["STATUSCD", "STATUS_CODE"],
    )

    if not all([tree_plot_key_col, species_col]):
        raise ValueError(
            f"{state}: required TREE fields unavailable."
        )

    tree_usecols = [tree_plot_key_col, species_col]

    if tree_status_col:
        tree_usecols.append(tree_status_col)

    tree_usecols = list(dict.fromkeys(tree_usecols))

    live_measurement_keys = set()
    selected_presence_parts = []

    tree_rows_scanned = 0
    candidate_live_tree_rows = 0
    selected_live_tree_rows = 0

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
        desc=f"{state}: scanning TREE for repeated plots",
        unit="chunk",
    ):
        tree_rows_scanned += len(chunk)

        rename_map = {
            tree_plot_key_col: "plot_key",
            species_col: "species_code",
        }

        if tree_status_col:
            rename_map[tree_status_col] = "status_code"

        chunk = chunk.rename(columns=rename_map)

        chunk["plot_key"] = (
            chunk["plot_key"].astype("string").str.strip()
        )
        chunk["species_code"] = pd.to_numeric(
            chunk["species_code"],
            errors="coerce",
        )

        keep = chunk["plot_key"].isin(candidate_plot_keys)

        if "status_code" in chunk.columns:
            chunk["status_code"] = pd.to_numeric(
                chunk["status_code"],
                errors="coerce",
            )
            keep &= chunk["status_code"].eq(1)

        live_work = chunk.loc[
            keep,
            ["plot_key", "species_code"],
        ].copy()

        candidate_live_tree_rows += len(live_work)

        if live_work.empty:
            continue

        live_measurement_keys.update(
            live_work["plot_key"].dropna().astype(str).tolist()
        )

        selected = live_work[
            live_work["species_code"].isin(species_codes)
        ].dropna(subset=["species_code"])

        if not selected.empty:
            selected["species_code"] = selected[
                "species_code"
            ].astype(int)

            selected_presence_parts.append(
                selected.drop_duplicates(
                    ["plot_key", "species_code"]
                )
            )

            selected_live_tree_rows += len(selected)

    if not selected_presence_parts:
        raise ValueError(
            f"{state}: no selected species were found in candidate repeated plots."
        )

    selected_presence_df = (
        pd.concat(selected_presence_parts, ignore_index=True)
        .drop_duplicates(["plot_key", "species_code"])
        .reset_index(drop=True)
    )

    del selected_presence_parts
    gc.collect()

    # --------------------------------------------------------
    # 3. Restrict to forested/live-tree measurements
    # --------------------------------------------------------
    eligible_measurements_df = candidate_measurements_df[
        candidate_measurements_df["plot_key"].isin(
            live_measurement_keys
        )
    ].copy()

    eligible_measurements_df.to_parquet(
        state_dir / "eligible_early_late_measurements.parquet",
        index=False,
    )

    early_eligible_df = eligible_measurements_df[
        eligible_measurements_df["period"] == "early"
    ].copy()

    late_eligible_df = eligible_measurements_df[
        eligible_measurements_df["period"] == "late"
    ].copy()

    matched_pairs_df = early_eligible_df[
        [
            "physical_plot_id",
            "plot_key",
            "year",
            "latitude",
            "longitude",
        ]
    ].rename(
        columns={
            "plot_key": "early_plot_key",
            "year": "early_year",
            "latitude": "early_latitude",
            "longitude": "early_longitude",
        }
    ).merge(
        late_eligible_df[
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

    matched_pairs_df["interval_years"] = (
        matched_pairs_df["late_year"]
        - matched_pairs_df["early_year"]
    )

    matched_pairs_df = matched_pairs_df[
        matched_pairs_df["interval_years"] > 5
    ].copy()

    matched_pairs_df["stable_latitude"] = (
        matched_pairs_df[
            ["early_latitude", "late_latitude"]
        ].mean(axis=1)
    )
    matched_pairs_df["stable_longitude"] = (
        matched_pairs_df[
            ["early_longitude", "late_longitude"]
        ].mean(axis=1)
    )

    matched_pairs_df = matched_pairs_df.sort_values(
        "physical_plot_id"
    ).reset_index(drop=True)

    matched_pairs_df.to_parquet(
        state_dir / "matched_physical_plot_pairs.parquet",
        index=False,
    )

    n_matched_physical_plots = len(matched_pairs_df)

    if n_matched_physical_plots < 20:
        raise ValueError(
            f"{state}: only {n_matched_physical_plots} matched physical plots."
        )

    # --------------------------------------------------------
    # 4. Raw early/late centers using all eligible measurements
    # --------------------------------------------------------
    measurement_lookup_df = eligible_measurements_df[
        [
            "plot_key",
            "physical_plot_id",
            "period",
            "year",
            "latitude",
            "longitude",
        ]
    ]

    raw_presence_df = selected_presence_df.merge(
        measurement_lookup_df,
        on="plot_key",
        how="inner",
        validate="many_to_one",
    )

    raw_presence_df = raw_presence_df.drop_duplicates(
        ["species_code", "physical_plot_id", "period"]
    )

    raw_center_df = (
        raw_presence_df.groupby(
            ["species_code", "period"],
            as_index=False,
        )
        .agg(
            n_occupied_plots=("physical_plot_id", "nunique"),
            mean_latitude=("latitude", "mean"),
            mean_year=("year", "mean"),
        )
    )

    raw_wide_df = raw_center_df.pivot(
        index="species_code",
        columns="period",
        values=[
            "n_occupied_plots",
            "mean_latitude",
            "mean_year",
        ],
    )

    raw_wide_df.columns = [
        f"raw_{metric}_{period}"
        for metric, period in raw_wide_df.columns
    ]
    raw_wide_df = raw_wide_df.reset_index()

    # --------------------------------------------------------
    # 5. Matched physical-plot presence matrices
    # --------------------------------------------------------
    species_list = sorted(species_codes)
    species_to_index = {
        species_code: index
        for index, species_code in enumerate(species_list)
    }

    n_species = len(species_list)
    n_plots = len(matched_pairs_df)

    early_matrix = np.zeros(
        (n_plots, n_species),
        dtype=np.uint8,
    )
    late_matrix = np.zeros(
        (n_plots, n_species),
        dtype=np.uint8,
    )

    early_key_to_row = {
        str(key): index
        for index, key in enumerate(
            matched_pairs_df["early_plot_key"]
        )
    }
    late_key_to_row = {
        str(key): index
        for index, key in enumerate(
            matched_pairs_df["late_plot_key"]
        )
    }

    for plot_key, species_code in tqdm(
        selected_presence_df[
            ["plot_key", "species_code"]
        ].itertuples(index=False, name=None),
        total=len(selected_presence_df),
        desc=f"{state}: building matched presence matrices",
        unit="presence",
    ):
        species_index = species_to_index.get(int(species_code))

        if species_index is None:
            continue

        key = str(plot_key)

        early_index = early_key_to_row.get(key)
        if early_index is not None:
            early_matrix[early_index, species_index] = 1

        late_index = late_key_to_row.get(key)
        if late_index is not None:
            late_matrix[late_index, species_index] = 1

    stable_latitude = matched_pairs_df[
        "stable_latitude"
    ].to_numpy(dtype=float)

    early_year_vector = matched_pairs_df[
        "early_year"
    ].to_numpy(dtype=float)

    late_year_vector = matched_pairs_df[
        "late_year"
    ].to_numpy(dtype=float)

    early_counts = early_matrix.sum(axis=0).astype(float)
    late_counts = late_matrix.sum(axis=0).astype(float)

    early_lat_means = np.divide(
        stable_latitude @ early_matrix,
        early_counts,
        out=np.full(n_species, np.nan),
        where=early_counts > 0,
    )
    late_lat_means = np.divide(
        stable_latitude @ late_matrix,
        late_counts,
        out=np.full(n_species, np.nan),
        where=late_counts > 0,
    )

    early_year_means = np.divide(
        early_year_vector @ early_matrix,
        early_counts,
        out=np.full(n_species, np.nan),
        where=early_counts > 0,
    )
    late_year_means = np.divide(
        late_year_vector @ late_matrix,
        late_counts,
        out=np.full(n_species, np.nan),
        where=late_counts > 0,
    )

    matched_shift_array = np.array(
        [
            safe_shift_km_decade(
                early_lat_means[index],
                late_lat_means[index],
                early_year_means[index],
                late_year_means[index],
            )
            for index in range(n_species)
        ],
        dtype=float,
    )

    matched_summary_df = pd.DataFrame(
        {
            "species_code": species_list,
            "matched_early_occupied_plots": early_counts.astype(int),
            "matched_late_occupied_plots": late_counts.astype(int),
            "matched_early_mean_latitude": early_lat_means,
            "matched_late_mean_latitude": late_lat_means,
            "matched_early_mean_year": early_year_means,
            "matched_late_mean_year": late_year_means,
            "matched_shift_km_decade": matched_shift_array,
        }
    )

    # Occupancy transitions.
    persistent = (
        (early_matrix == 1) & (late_matrix == 1)
    ).sum(axis=0)
    colonized = (
        (early_matrix == 0) & (late_matrix == 1)
    ).sum(axis=0)
    lost = (
        (early_matrix == 1) & (late_matrix == 0)
    ).sum(axis=0)

    matched_summary_df["persistent_plots"] = persistent
    matched_summary_df["colonized_plots"] = colonized
    matched_summary_df["lost_plots"] = lost
    matched_summary_df["net_occupancy_change_plots"] = (
        late_counts - early_counts
    )
    matched_summary_df[
        "net_occupancy_change_fraction"
    ] = (
        late_counts - early_counts
    ) / n_matched_physical_plots

    # --------------------------------------------------------
    # 6. Combine raw and matched estimates
    # --------------------------------------------------------
    result_df = (
        pd.DataFrame({"species_code": species_list})
        .merge(raw_wide_df, on="species_code", how="left")
        .merge(
            matched_summary_df,
            on="species_code",
            how="left",
        )
        .merge(
            taxonomy_df,
            on="species_code",
            how="left",
        )
    )

    result_df["raw_shift_km_decade"] = [
        safe_shift_km_decade(
            row.get("raw_mean_latitude_early", np.nan),
            row.get("raw_mean_latitude_late", np.nan),
            row.get("raw_mean_year_early", np.nan),
            row.get("raw_mean_year_late", np.nan),
        )
        for _, row in result_df.iterrows()
    ]

    result_df["correction_effect_km_decade"] = (
        result_df["raw_shift_km_decade"]
        - result_df["matched_shift_km_decade"]
    )

    result_df["meaningful_direction_reversal"] = [
        meaningful_reversal(raw, matched)
        for raw, matched in zip(
            result_df["raw_shift_km_decade"],
            result_df["matched_shift_km_decade"],
        )
    ]

    result_df["material_correction"] = (
        result_df["correction_effect_km_decade"].abs()
        >= MATERIAL_CORRECTION_KM_DECADE
    )

    result_df["matched_reduces_absolute_shift"] = (
        result_df["matched_shift_km_decade"].abs()
        < result_df["raw_shift_km_decade"].abs()
    )

    strong_mask = (
        result_df[
            "raw_n_occupied_plots_early"
        ].ge(MIN_RAW_OCCUPIED_PLOTS_WINDOW)
        & result_df[
            "raw_n_occupied_plots_late"
        ].ge(MIN_RAW_OCCUPIED_PLOTS_WINDOW)
        & result_df[
            "matched_early_occupied_plots"
        ].ge(MIN_MATCHED_OCCUPIED_PLOTS_WINDOW)
        & result_df[
            "matched_late_occupied_plots"
        ].ge(MIN_MATCHED_OCCUPIED_PLOTS_WINDOW)
        & result_df["raw_shift_km_decade"].notna()
        & result_df["matched_shift_km_decade"].notna()
    )

    result_df["strong_coverage"] = strong_mask

    strong_indices = np.where(strong_mask.to_numpy())[0]

    # --------------------------------------------------------
    # 7. Global within-plot early/late swap permutation
    # --------------------------------------------------------
    rng = np.random.default_rng(random_seed)

    observed_correction = result_df[
        "correction_effect_km_decade"
    ].to_numpy(dtype=float)

    observed_matched_shift = result_df[
        "matched_shift_km_decade"
    ].to_numpy(dtype=float)

    species_matched_exceed_count = np.zeros(
        n_species,
        dtype=int,
    )
    species_correction_exceed_count = np.zeros(
        n_species,
        dtype=int,
    )

    permutation_rows = []

    for permutation_id in tqdm(
        range(N_PERMUTATIONS),
        desc=f"{state}: repeated-plot swap permutations",
        unit="permutation",
    ):
        swap_mask = rng.random(n_plots) < 0.5
        swap_matrix = swap_mask[:, None]

        perm_early = np.where(
            swap_matrix,
            late_matrix,
            early_matrix,
        )
        perm_late = np.where(
            swap_matrix,
            early_matrix,
            late_matrix,
        )

        perm_early_counts = perm_early.sum(axis=0).astype(float)
        perm_late_counts = perm_late.sum(axis=0).astype(float)

        perm_early_lat = np.divide(
            stable_latitude @ perm_early,
            perm_early_counts,
            out=np.full(n_species, np.nan),
            where=perm_early_counts > 0,
        )
        perm_late_lat = np.divide(
            stable_latitude @ perm_late,
            perm_late_counts,
            out=np.full(n_species, np.nan),
            where=perm_late_counts > 0,
        )

        perm_early_year = np.divide(
            early_year_vector @ perm_early,
            perm_early_counts,
            out=np.full(n_species, np.nan),
            where=perm_early_counts > 0,
        )
        perm_late_year = np.divide(
            late_year_vector @ perm_late,
            perm_late_counts,
            out=np.full(n_species, np.nan),
            where=perm_late_counts > 0,
        )

        perm_shift = np.array(
            [
                safe_shift_km_decade(
                    perm_early_lat[index],
                    perm_late_lat[index],
                    perm_early_year[index],
                    perm_late_year[index],
                )
                for index in range(n_species)
            ],
            dtype=float,
        )

        raw_shift_array = result_df[
            "raw_shift_km_decade"
        ].to_numpy(dtype=float)

        perm_correction = raw_shift_array - perm_shift

        valid_species = strong_mask.to_numpy()

        species_matched_exceed_count += (
            valid_species
            & np.isfinite(perm_shift)
            & (
                np.abs(perm_shift)
                >= np.abs(observed_matched_shift)
            )
        )

        species_correction_exceed_count += (
            valid_species
            & np.isfinite(perm_correction)
            & (
                np.abs(perm_correction)
                >= np.abs(observed_correction)
            )
        )

        strong_perm_correction = perm_correction[
            strong_indices
        ]
        strong_perm_raw = raw_shift_array[
            strong_indices
        ]
        strong_perm_matched = perm_shift[
            strong_indices
        ]

        valid_perm = (
            np.isfinite(strong_perm_correction)
            & np.isfinite(strong_perm_raw)
            & np.isfinite(strong_perm_matched)
        )

        if valid_perm.any():
            correction_values = strong_perm_correction[
                valid_perm
            ]
            raw_values = strong_perm_raw[valid_perm]
            matched_values = strong_perm_matched[
                valid_perm
            ]

            reversals = [
                meaningful_reversal(raw, matched)
                for raw, matched in zip(
                    raw_values,
                    matched_values,
                )
            ]

            permutation_rows.append(
                {
                    "state": state,
                    "permutation_id": permutation_id,
                    "median_abs_correction_km_decade": float(
                        np.median(
                            np.abs(correction_values)
                        )
                    ),
                    "meaningful_reversal_fraction": float(
                        np.mean(reversals)
                    ),
                }
            )

    permutation_df = pd.DataFrame(permutation_rows)

    result_df["matched_shift_permutation_p"] = (
        1 + species_matched_exceed_count
    ) / (N_PERMUTATIONS + 1)

    result_df["correction_permutation_p"] = (
        1 + species_correction_exceed_count
    ) / (N_PERMUTATIONS + 1)

    result_df.loc[
        ~strong_mask,
        [
            "matched_shift_permutation_p",
            "correction_permutation_p",
        ],
    ] = np.nan

    strong_df = result_df[
        result_df["strong_coverage"]
    ].copy()

    n_strong_species = len(strong_df)

    observed_median_abs_correction = (
        float(
            strong_df[
                "correction_effect_km_decade"
            ].abs().median()
        )
        if n_strong_species else np.nan
    )

    observed_reversal_fraction = (
        float(
            strong_df[
                "meaningful_direction_reversal"
            ].mean()
        )
        if n_strong_species else np.nan
    )

    attenuation_fraction = (
        float(
            strong_df[
                "matched_reduces_absolute_shift"
            ].mean()
        )
        if n_strong_species else np.nan
    )

    n_material_correction = int(
        strong_df["material_correction"].sum()
    )
    n_reversal = int(
        strong_df[
            "meaningful_direction_reversal"
        ].sum()
    )

    permutation_p_median = float(
        (
            1
            + (
                permutation_df[
                    "median_abs_correction_km_decade"
                ]
                >= observed_median_abs_correction
            ).sum()
        )
        / (N_PERMUTATIONS + 1)
    )

    permutation_p_reversal = float(
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

    result_df = result_df.sort_values(
        "correction_effect_km_decade",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)

    result_df.to_parquet(
        state_dir / "species_repeated_plot_results.parquet",
        index=False,
    )
    result_df.to_csv(
        state_dir / "species_repeated_plot_results.csv",
        index=False,
    )
    permutation_df.to_parquet(
        state_dir / "repeated_plot_permutation_null.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # 8. State decision
    # --------------------------------------------------------
    domain_pass = bool(
        n_matched_physical_plots
        >= MIN_MATCHED_PHYSICAL_PLOTS_STATE
        and n_strong_species
        >= MIN_STRONG_SPECIES_STATE
    )

    material_signal = bool(
        (
            np.isfinite(observed_median_abs_correction)
            and observed_median_abs_correction
            >= MATERIAL_CORRECTION_KM_DECADE
        )
        or (
            np.isfinite(observed_reversal_fraction)
            and observed_reversal_fraction
            >= TARGET_REVERSAL_FRACTION
        )
    )

    permutation_pass = bool(
        permutation_p_median <= 0.05
        or permutation_p_reversal <= 0.05
    )

    state_pass = bool(
        domain_pass
        and material_signal
        and permutation_pass
    )

    summary = {
        "state": state,
        "early_years": ",".join(map(str, early_years)),
        "late_years": ",".join(map(str, late_years)),
        "candidate_measurements": len(candidate_measurements_df),
        "eligible_live_tree_measurements": len(
            eligible_measurements_df
        ),
        "n_matched_physical_plots": n_matched_physical_plots,
        "median_plot_interval_years": float(
            matched_pairs_df["interval_years"].median()
        ),
        "tree_rows_scanned": tree_rows_scanned,
        "candidate_live_tree_rows": candidate_live_tree_rows,
        "selected_live_tree_rows": selected_live_tree_rows,
        "n_strong_species": n_strong_species,
        "median_abs_correction_km_decade": (
            observed_median_abs_correction
        ),
        "n_material_correction": n_material_correction,
        "n_meaningful_reversal": n_reversal,
        "meaningful_reversal_fraction": (
            observed_reversal_fraction
        ),
        "attenuation_fraction": attenuation_fraction,
        "permutation_null95_median_abs_correction": float(
            permutation_df[
                "median_abs_correction_km_decade"
            ].quantile(0.95)
        ),
        "permutation_p_median_abs_correction": (
            permutation_p_median
        ),
        "permutation_null95_reversal_fraction": float(
            permutation_df[
                "meaningful_reversal_fraction"
            ].quantile(0.95)
        ),
        "permutation_p_reversal_fraction": (
            permutation_p_reversal
        ),
        "domain_pass": domain_pass,
        "material_signal": material_signal,
        "permutation_pass": permutation_pass,
        "state_pass": state_pass,
    }

    del (
        plot_df,
        candidate_measurements_df,
        selected_presence_df,
        raw_presence_df,
        early_matrix,
        late_matrix,
    )
    gc.collect()

    return result_df, summary


# ============================================================
# Main
# ============================================================
try:
    # --------------------------------------------------------
    # 1. Load fixed species set
    # --------------------------------------------------------
    if not SELECTED_SPECIES_CSV.exists():
        raise FileNotFoundError(
            f"Missing selected species file: {SELECTED_SPECIES_CSV}"
        )

    taxonomy_df = (
        pd.read_csv(SELECTED_SPECIES_CSV)[
            [
                "species_code",
                "scientific_name",
                "common_name",
            ]
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

    species_codes = set(
        taxonomy_df["species_code"].tolist()
    )

    if len(species_codes) < 20:
        raise ValueError(
            "The selected-species set is unexpectedly small."
        )

    print("\nFIXED SPECIES SET")
    display(taxonomy_df)

    # --------------------------------------------------------
    # 2. Validate state files and process
    # --------------------------------------------------------
    missing_state_files = []

    for state in STATES:
        for table in ["PLOT", "TREE"]:
            path = RAW_DIR / f"{state}_{table}.csv"
            if not path.exists():
                missing_state_files.append(str(path))

    if missing_state_files:
        raise FileNotFoundError(
            "Missing state files from Cell 05A: "
            + " | ".join(missing_state_files)
        )

    state_result_tables = []
    state_summaries = []

    for state_index, state in enumerate(STATES):
        print("\n" + "=" * 78)
        print(f"REPEATED PHYSICAL-PLOT CONTROL: {state}")
        print("=" * 78)

        state_results_df, state_summary = process_state(
            state=state,
            plot_csv=RAW_DIR / f"{state}_PLOT.csv",
            tree_csv=RAW_DIR / f"{state}_TREE.csv",
            species_codes=species_codes,
            taxonomy_df=taxonomy_df,
            random_seed=(
                RANDOM_SEED + state_index * 10000
            ),
        )

        state_result_tables.append(state_results_df)
        state_summaries.append(state_summary)

        print(f"\n{state} SUMMARY")
        display(pd.DataFrame([state_summary]))

    all_results_df = pd.concat(
        state_result_tables,
        ignore_index=True,
    )
    state_summary_df = pd.DataFrame(
        state_summaries
    ).sort_values("state").reset_index(drop=True)

    all_results_df.to_parquet(
        OUT_DIR / "all_state_repeated_plot_results.parquet",
        index=False,
    )
    all_results_df.to_csv(
        OUT_DIR / "all_state_repeated_plot_results.csv",
        index=False,
    )
    state_summary_df.to_csv(
        OUT_DIR / "state_repeated_plot_summary.csv",
        index=False,
    )

    print("\nFOUR-STATE REPEATED-PLOT SUMMARY")
    display(state_summary_df)

    # --------------------------------------------------------
    # 3. Cross-state species meta-summary
    # --------------------------------------------------------
    eligible_df = all_results_df[
        all_results_df["strong_coverage"]
        & all_results_df[
            "correction_effect_km_decade"
        ].notna()
    ].copy()

    meta_rows = []

    groups = eligible_df.groupby(
        "species_code",
        sort=True,
    )

    for species_code, group in tqdm(
        groups,
        total=groups.ngroups,
        desc="Building repeated-plot species meta-summary",
        unit="species",
    ):
        effects = group[
            "correction_effect_km_decade"
        ].to_numpy(dtype=float)

        positive_count = int((effects > 0).sum())
        negative_count = int((effects < 0).sum())
        nonzero_count = positive_count + negative_count

        majority_sign_fraction = (
            max(positive_count, negative_count)
            / nonzero_count
            if nonzero_count else np.nan
        )

        non_wv_group = group[
            group["state"] != "WV"
        ]

        first = group.iloc[0]

        meta_rows.append(
            {
                "species_code": int(species_code),
                "scientific_name": first["scientific_name"],
                "common_name": first["common_name"],
                "n_states": int(group["state"].nunique()),
                "median_correction_km_decade": float(
                    np.median(effects)
                ),
                "median_abs_correction_km_decade": float(
                    np.median(np.abs(effects))
                ),
                "majority_sign_fraction": (
                    majority_sign_fraction
                ),
                "n_material_states": int(
                    group["material_correction"].sum()
                ),
                "n_reversal_states": int(
                    group[
                        "meaningful_direction_reversal"
                    ].sum()
                ),
                "median_matched_shift_permutation_p": float(
                    group[
                        "matched_shift_permutation_p"
                    ].median()
                ),
                "median_correction_permutation_p": float(
                    group[
                        "correction_permutation_p"
                    ].median()
                ),
                "non_wv_state_count": int(
                    non_wv_group["state"].nunique()
                ),
                "non_wv_median_correction_km_decade": (
                    float(
                        non_wv_group[
                            "correction_effect_km_decade"
                        ].median()
                    )
                    if not non_wv_group.empty
                    else np.nan
                ),
                "non_wv_median_abs_correction_km_decade": (
                    float(
                        non_wv_group[
                            "correction_effect_km_decade"
                        ].abs().median()
                    )
                    if not non_wv_group.empty
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
        OUT_DIR / "repeated_plot_species_meta.parquet",
        index=False,
    )
    species_meta_df.to_csv(
        OUT_DIR / "repeated_plot_species_meta.csv",
        index=False,
    )

    print("\nREPEATED-PLOT SPECIES META-SUMMARY")
    display(species_meta_df.head(50))

    eligible_meta_df = species_meta_df[
        species_meta_df["eligible_meta_species"]
    ].copy()

    n_meta_species = len(eligible_meta_df)

    majority_sign_consistency_fraction = (
        float(
            (
                eligible_meta_df[
                    "majority_sign_fraction"
                ] >= 0.75
            ).mean()
        )
        if n_meta_species else np.nan
    )

    non_wv_median_abs_correction = (
        float(
            eligible_meta_df[
                "non_wv_median_abs_correction_km_decade"
            ].median()
        )
        if n_meta_species else np.nan
    )

    # --------------------------------------------------------
    # 4. Leave-one-state-out meta stability
    # --------------------------------------------------------
    full_meta_lookup = eligible_meta_df.set_index(
        "species_code"
    )["median_correction_km_decade"]

    loo_rows = []
    loo_species_rows = []

    for held_out_state in tqdm(
        STATES,
        desc="Repeated-plot leave-one-state-out",
        unit="state",
    ):
        remaining = eligible_df[
            eligible_df["state"] != held_out_state
        ].copy()

        loo_meta = (
            remaining.groupby(
                "species_code"
            )["correction_effect_km_decade"]
            .agg(["median", "count"])
            .rename(
                columns={
                    "median": "loo_effect",
                    "count": "loo_n_states",
                }
            )
        )

        comparison = pd.DataFrame(
            {"full_effect": full_meta_lookup}
        ).join(
            loo_meta,
            how="inner",
        )

        comparison = comparison[
            comparison["loo_n_states"] >= 2
        ].copy()

        comparison["sign_stable"] = (
            np.sign(comparison["full_effect"])
            == np.sign(comparison["loo_effect"])
        )

        sign_agreement = (
            float(comparison["sign_stable"].mean())
            if len(comparison) else np.nan
        )

        magnitude_spearman = safe_spearman(
            comparison["full_effect"],
            comparison["loo_effect"],
        )

        loo_rows.append(
            {
                "held_out_state": held_out_state,
                "n_species_compared": len(comparison),
                "sign_agreement": sign_agreement,
                "magnitude_spearman": magnitude_spearman,
                "median_abs_change": (
                    float(
                        (
                            comparison["full_effect"]
                            - comparison["loo_effect"]
                        ).abs().median()
                    )
                    if len(comparison) else np.nan
                ),
            }
        )

        for species_code, row in comparison.iterrows():
            loo_species_rows.append(
                {
                    "held_out_state": held_out_state,
                    "species_code": int(species_code),
                    "full_effect": row["full_effect"],
                    "loo_effect": row["loo_effect"],
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
        OUT_DIR / "repeated_plot_loo_summary.csv",
        index=False,
    )
    loo_species_df.to_parquet(
        OUT_DIR / "repeated_plot_loo_species.parquet",
        index=False,
    )

    print("\nREPEATED-PLOT LEAVE-ONE-STATE-OUT")
    display(loo_summary_df)

    overall_loo_sign_stability = (
        float(loo_species_df["sign_stable"].mean())
        if len(loo_species_df) else np.nan
    )

    # --------------------------------------------------------
    # 5. Final trend decision
    # --------------------------------------------------------
    n_domain_pass_states = int(
        state_summary_df["domain_pass"].sum()
    )
    n_material_states = int(
        state_summary_df["material_signal"].sum()
    )
    n_permutation_pass_states = int(
        state_summary_df["permutation_pass"].sum()
    )
    n_state_passes = int(
        state_summary_df["state_pass"].sum()
    )
    n_added_state_passes = int(
        state_summary_df[
            state_summary_df["state"].isin(
                ADDED_STATES
            )
        ]["state_pass"].sum()
    )

    domain_criterion = bool(
        n_domain_pass_states >= 3
    )
    state_replication_criterion = bool(
        n_state_passes >= 2
        and n_added_state_passes >= 1
    )
    meta_coverage_criterion = bool(
        n_meta_species >= MIN_META_SPECIES
    )
    non_wv_effect_criterion = bool(
        np.isfinite(non_wv_median_abs_correction)
        and non_wv_median_abs_correction
        >= TARGET_NON_WV_MEDIAN_ABS_CORRECTION
    )
    meta_consistency_criterion = bool(
        np.isfinite(
            majority_sign_consistency_fraction
        )
        and majority_sign_consistency_fraction
        >= TARGET_META_SIGN_CONSISTENCY
    )
    loo_criterion = bool(
        np.isfinite(overall_loo_sign_stability)
        and overall_loo_sign_stability
        >= TARGET_LOO_SIGN_STABILITY
    )

    passed = bool(
        domain_criterion
        and state_replication_criterion
        and meta_coverage_criterion
        and non_wv_effect_criterion
        and meta_consistency_criterion
        and loo_criterion
    )

    if passed:
        status = "GO_FOR_GBIF_OBSERVATION_DRIFT_COMPARISON"
        next_step = (
            "The effect survives repeated-physical-plot control. "
            "Proceed to a small GBIF/iNaturalist comparison for the "
            "same species and regions."
        )
    elif (
        domain_criterion
        and meta_coverage_criterion
        and non_wv_effect_criterion
        and loo_criterion
    ):
        status = "PIVOT_TO_SPECIES_SENSITIVITY_FRAMEWORK"
        next_step = (
            "A universal state-level effect is weak, but stable "
            "species-specific sensitivity remains. Model which species "
            "traits and range structures predict correction magnitude."
        )
    elif domain_criterion and n_state_passes == 0:
        status = "ROTATING_PANEL_ARTIFACT_DOMINATES"
        next_step = (
            "The earlier WV/fixed-grid signal does not survive exact "
            "physical-plot matching. Do not claim general observation "
            "drift from FIA annual panels."
        )
    else:
        status = "REPEATED_PLOT_CONTROL_INCONCLUSIVE"
        next_step = (
            "Repeated-plot coverage is insufficient or heterogeneous. "
            "Inspect state-specific matched-plot counts before changing "
            "the scientific direction."
        )

    # --------------------------------------------------------
    # 6. Visual previews
    # --------------------------------------------------------
    plot_state_df = state_summary_df.sort_values(
        "state"
    ).copy()

    plt.figure(figsize=(9, 5))
    plt.bar(
        plot_state_df["state"],
        plot_state_df[
            "median_abs_correction_km_decade"
        ],
    )
    plt.axhline(
        MATERIAL_CORRECTION_KM_DECADE,
        linestyle="--",
        label="Material threshold",
    )
    plt.xlabel("State")
    plt.ylabel(
        "Median absolute raw-matched correction "
        "(km/decade)"
    )
    plt.title("Repeated physical-plot control")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if not eligible_meta_df.empty:
        top_meta_df = (
            eligible_meta_df.head(20)
            .sort_values(
                "median_correction_km_decade"
            )
            .copy()
        )

        labels = top_meta_df[
            "scientific_name"
        ].fillna(
            top_meta_df["species_code"].astype(str)
        )

        plt.figure(figsize=(10, 8))
        plt.barh(
            labels,
            top_meta_df[
                "median_correction_km_decade"
            ],
        )
        plt.axvline(0)
        plt.xlabel(
            "Median correction effect "
            "(raw minus repeated-plot, km/decade)"
        )
        plt.ylabel("Species")
        plt.title(
            "Cross-state repeated-plot correction effects"
        )
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(9, 5))
    plt.bar(
        loo_summary_df["held_out_state"],
        loo_summary_df["sign_agreement"],
    )
    plt.axhline(
        TARGET_LOO_SIGN_STABILITY,
        linestyle="--",
        label="Target",
    )
    plt.ylim(0, 1.05)
    plt.xlabel("Held-out state")
    plt.ylabel("Sign agreement")
    plt.title(
        "Repeated-plot leave-one-state-out stability"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 7. README
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

## Repeated physical-plot control — {RUN_UTC}

### Hypothesis
Raw early-to-late latitude-center shifts should change when estimated on the
exact same physical FIA plots measured in both early and late windows.

### Data and design
- States: {STATES}
- Species: {len(species_codes)} fixed WV-screened species
- Physical plot identifier:
  STATECD + UNITCD + COUNTYCD + PLOT
- Early/late window size: {WINDOW_SIZE} usable inventory years per state
- Forest-sampling proxy:
  candidate plot measurements containing at least one live TREE record
- Presence unit:
  one species presence per plot measurement
- Tree abundance weighting: not used
- Permutations:
  {N_PERMUTATIONS} global within-plot early/late swaps per state
- Random seed: {RANDOM_SEED}
- GPU: not used

### State-level results
- Adequate repeated-plot domains: {n_domain_pass_states}/4
- Material state signals: {n_material_states}/4
- State permutation passes: {n_permutation_pass_states}/4
- Full state passes: {n_state_passes}/4
- Added-state passes: {n_added_state_passes}/3

### Cross-state results
- Meta species represented in at least {MIN_META_STATES} states:
  {n_meta_species}
- Median non-WV absolute correction:
  {non_wv_median_abs_correction} km/decade
- Fraction with at least 75% state-sign agreement:
  {majority_sign_consistency_fraction}
- Leave-one-state-out sign stability:
  {overall_loo_sign_stability}

### Decision
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
This control removes changing physical plot identity between early and late
windows. It does not model FIA subplot design, seedlings below the TREE size
threshold, forest condition classes, detectability, disturbance, mortality,
recruitment, or climate. A plot with no live TREE records was excluded from the
forested sampling frame.
"""
    )

    # --------------------------------------------------------
    # 8. Compact output
    # --------------------------------------------------------
    state_compact = (
        state_summary_df[
            [
                "state",
                "early_years",
                "late_years",
                "n_matched_physical_plots",
                "median_plot_interval_years",
                "n_strong_species",
                "median_abs_correction_km_decade",
                "meaningful_reversal_fraction",
                "attenuation_fraction",
                "permutation_p_median_abs_correction",
                "permutation_p_reversal_fraction",
                "state_pass",
            ]
        ]
        .round(6)
        .to_dict("records")
    )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATES: {STATES}")
    print(f"N_FIXED_SPECIES: {len(species_codes)}")
    print(
        "STATE_RESULTS: "
        + json.dumps(
            state_compact,
            ensure_ascii=False,
        )
    )
    print(
        f"N_DOMAIN_PASS_STATES: "
        f"{n_domain_pass_states}"
    )
    print(
        f"N_MATERIAL_SIGNAL_STATES: "
        f"{n_material_states}"
    )
    print(
        f"N_PERMUTATION_PASS_STATES: "
        f"{n_permutation_pass_states}"
    )
    print(
        f"N_FULL_STATE_PASSES: "
        f"{n_state_passes}"
    )
    print(
        f"N_ADDED_STATE_PASSES: "
        f"{n_added_state_passes}"
    )
    print(f"N_META_SPECIES: {n_meta_species}")
    print(
        f"NON_WV_MEDIAN_ABS_CORRECTION_KM_DECADE: "
        f"{non_wv_median_abs_correction}"
    )
    print(
        f"META_MAJORITY_SIGN_CONSISTENCY_FRACTION: "
        f"{majority_sign_consistency_fraction}"
    )
    print(
        f"OVERALL_LOO_SIGN_STABILITY: "
        f"{overall_loo_sign_stability}"
    )
    print(
        "LOO_SUMMARY: "
        + json.dumps(
            loo_summary_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "TOP_META_SPECIES: "
        + json.dumps(
            top_meta_species,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_CRITERIA: >=3 adequate state domains; "
        ">=2 state passes including >=1 added state; "
        ">=20 meta species; non-WV median absolute correction "
        ">=1.5 km/decade; meta sign consistency >=0.60; "
        "leave-one-state-out sign stability >=0.75"
    )
    print(
        f"DOMAIN_CRITERION: "
        f"{domain_criterion}"
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
        f"META_CONSISTENCY_CRITERION: "
        f"{meta_consistency_criterion}"
    )
    print(f"LOO_CRITERION: {loo_criterion}")
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except Exception as exc:
    print_failure(
        "CELL06A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        "Save this diagnostic summary and the last displayed state output. "
        "Do not rerun previous downloads unless a file is missing.",
    )
