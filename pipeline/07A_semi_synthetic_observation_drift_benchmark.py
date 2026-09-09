
# Cell 07A — Semi-synthetic observation-drift benchmark on repeated FIA plots
#
# Scientific pivot
# ----------------
# Exact repeated-physical-plot matching showed that the earlier annual-panel
# effect did not survive as a natural cross-state observation-drift signal.
#
# We now use repeated plots as a standardized benchmark and inject controlled
# observation-system drift. This answers a different but stronger causal question:
#
# Under what amount of spatial sampling drift can an apparent species migration
# be created, and can transparent spatial post-stratification recover the true
# repeated-plot trend?
#
# Minimal trend version
# ---------------------
# - States: WV and PA
# - Species: up to 20 species with strong matched-plot coverage in both states
# - Drift mechanism: opposite early/late sampling gradients along latitude
# - Drift severity: 0.0, 0.5, 1.0, 1.5
# - Sampling fraction: 50% of repeated physical plots per period
# - Repetitions: 200 per state and severity
# - Correction: 0.5-degree spatial post-stratification to the full repeated-plot frame
#
# Important boundary
# ------------------
# This benchmark tests causal plausibility and correction performance.
# It does NOT show that real GBIF or iNaturalist observers actually moved this way.

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
STATES = ["WV", "PA"]

MAX_SPECIES = 20
MIN_SELECTION_PLOTS_PER_PERIOD = 20
MIN_OBSERVED_OCCURRENCES = 5

GRID_RESOLUTION = 0.50
SAMPLE_FRACTION = 0.50

DRIFT_SEVERITIES = [0.0, 0.5, 1.0, 1.5]
N_REPLICATES = 200
RANDOM_SEED = 20260710

NULL_MAX_MEDIAN_ERROR_KM_DECADE = 1.50
HIGH_DRIFT_MIN_MEDIAN_ERROR_KM_DECADE = 2.00
MIN_CORRECTION_REDUCTION = 0.30
MIN_FRACTION_CORRECTED_BETTER = 0.60
MIN_SEVERITY_SPEARMAN = 0.80

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
REPEATED_DIR = (
    BASE_DIR / "derived" / "repeated_physical_plot_control"
)
PRESENCE_DIR = (
    BASE_DIR / "derived" / "appalachian_four_state_replication"
)
OUT_DIR = (
    BASE_DIR / "derived" / "semi_synthetic_observation_drift"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("STATES:", STATES)
print("DRIFT_SEVERITIES:", DRIFT_SEVERITIES)
print("N_REPLICATES:", N_REPLICATES)
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

    return float(frame.corr(method="spearman").iloc[0, 1])


def vector_shift_km_decade(
    early_lat,
    late_lat,
    early_year,
    late_year,
):
    early_lat = np.asarray(early_lat, dtype=float)
    late_lat = np.asarray(late_lat, dtype=float)
    early_year = np.asarray(early_year, dtype=float)
    late_year = np.asarray(late_year, dtype=float)

    interval = late_year - early_year

    valid = (
        np.isfinite(early_lat)
        & np.isfinite(late_lat)
        & np.isfinite(early_year)
        & np.isfinite(late_year)
        & (interval > 5)
    )

    result = np.full(
        np.broadcast(
            early_lat,
            late_lat,
            early_year,
            late_year,
        ).shape,
        np.nan,
        dtype=float,
    )

    result[valid] = (
        (late_lat[valid] - early_lat[valid])
        * 111.32
        * 10
        / interval[valid]
    )

    return result


def weighted_without_replacement(
    rng,
    weights,
    sample_size,
):
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 1e-12, None)
    probabilities = weights / weights.sum()

    selected = rng.choice(
        len(weights),
        size=sample_size,
        replace=False,
        p=probabilities,
    )

    mask = np.zeros(len(weights), dtype=bool)
    mask[selected] = True
    return mask


def build_poststratification_weights(
    sample_mask,
    grid_index,
    target_grid_counts,
):
    sample_grid_counts = np.bincount(
        grid_index[sample_mask],
        minlength=len(target_grid_counts),
    ).astype(float)

    available = sample_grid_counts > 0

    cell_weights = np.zeros(
        len(target_grid_counts),
        dtype=float,
    )

    cell_weights[available] = (
        target_grid_counts[available]
        / sample_grid_counts[available]
    )

    plot_weights = np.zeros(
        len(sample_mask),
        dtype=float,
    )

    plot_weights[sample_mask] = cell_weights[
        grid_index[sample_mask]
    ]

    target_mass_covered = (
        target_grid_counts[available].sum()
        / target_grid_counts.sum()
    )

    return plot_weights, float(target_mass_covered)


def matrix_center(
    presence_matrix,
    plot_mask,
    plot_values,
    minimum_occurrences,
):
    selected_matrix = presence_matrix[plot_mask]
    selected_values = plot_values[plot_mask]

    counts = selected_matrix.sum(axis=0).astype(float)

    means = np.divide(
        selected_values @ selected_matrix,
        counts,
        out=np.full(
            presence_matrix.shape[1],
            np.nan,
            dtype=float,
        ),
        where=counts >= minimum_occurrences,
    )

    return means, counts


def weighted_matrix_center(
    presence_matrix,
    plot_weights,
    plot_values,
    minimum_occurrences,
):
    positive = plot_weights > 0
    selected_matrix = presence_matrix[positive]
    selected_weights = plot_weights[positive]
    selected_values = plot_values[positive]

    raw_counts = selected_matrix.sum(axis=0).astype(float)

    weighted_denominator = (
        selected_weights @ selected_matrix
    )

    weighted_numerator = (
        (selected_weights * selected_values)
        @ selected_matrix
    )

    means = np.divide(
        weighted_numerator,
        weighted_denominator,
        out=np.full(
            presence_matrix.shape[1],
            np.nan,
            dtype=float,
        ),
        where=(
            (weighted_denominator > 0)
            & (raw_counts >= minimum_occurrences)
        ),
    )

    return means, raw_counts


def meaningful_direction_error(
    estimate,
    truth,
    minimum_magnitude=1.0,
):
    if not np.isfinite(estimate) or not np.isfinite(truth):
        return False

    if abs(truth) < minimum_magnitude:
        return False

    return bool(
        abs(estimate) >= minimum_magnitude
        and np.sign(estimate) != np.sign(truth)
    )


def print_failure(status, error, next_step):
    append_readme(
        f"""

## Semi-synthetic benchmark failure — {RUN_UTC}
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
# Species selection across both states
# ============================================================
try:
    selection_tables = []

    for state in STATES:
        result_path = (
            REPEATED_DIR
            / state
            / "species_repeated_plot_results.parquet"
        )

        if not result_path.exists():
            raise FileNotFoundError(
                f"Missing Cell 06 output: {result_path}"
            )

        frame = pd.read_parquet(result_path)

        frame["state"] = state

        required_columns = [
            "species_code",
            "scientific_name",
            "common_name",
            "strong_coverage",
            "matched_early_occupied_plots",
            "matched_late_occupied_plots",
        ]

        missing = [
            column
            for column in required_columns
            if column not in frame.columns
        ]

        if missing:
            raise KeyError(
                f"{state}: missing columns: {missing}"
            )

        frame = frame[
            frame["strong_coverage"].fillna(False)
        ].copy()

        frame["minimum_period_plots"] = frame[
            [
                "matched_early_occupied_plots",
                "matched_late_occupied_plots",
            ]
        ].min(axis=1)

        selection_tables.append(
            frame[
                [
                    "state",
                    "species_code",
                    "scientific_name",
                    "common_name",
                    "minimum_period_plots",
                ]
            ]
        )

    selection_long_df = pd.concat(
        selection_tables,
        ignore_index=True,
    )

    selection_wide_df = (
        selection_long_df.pivot_table(
            index=[
                "species_code",
                "scientific_name",
                "common_name",
            ],
            columns="state",
            values="minimum_period_plots",
            aggfunc="max",
        )
        .reset_index()
    )

    for state in STATES:
        if state not in selection_wide_df.columns:
            selection_wide_df[state] = np.nan

    selection_wide_df["minimum_across_states"] = (
        selection_wide_df[STATES].min(axis=1)
    )

    selected_species_df = (
        selection_wide_df[
            selection_wide_df["minimum_across_states"]
            >= MIN_SELECTION_PLOTS_PER_PERIOD
        ]
        .sort_values(
            "minimum_across_states",
            ascending=False,
        )
        .head(MAX_SPECIES)
        .reset_index(drop=True)
    )

    if len(selected_species_df) < 10:
        raise ValueError(
            "Fewer than 10 species met the shared matched-plot "
            "coverage requirement."
        )

    selected_species_codes = (
        selected_species_df["species_code"]
        .astype(int)
        .tolist()
    )

    selected_species_df.to_csv(
        OUT_DIR / "selected_benchmark_species.csv",
        index=False,
    )

    print("\nSELECTED BENCHMARK SPECIES")
    display(selected_species_df)

    # --------------------------------------------------------
    # State benchmark loop
    # --------------------------------------------------------
    all_result_parts = []
    state_frame_rows = []

    for state_index, state in enumerate(STATES):
        print("\n" + "=" * 80)
        print(f"SEMI-SYNTHETIC BENCHMARK: {state}")
        print("=" * 80)

        pair_path = (
            REPEATED_DIR
            / state
            / "matched_physical_plot_pairs.parquet"
        )
        presence_path = (
            PRESENCE_DIR
            / state
            / "species_plot_year_presence.parquet"
        )

        if not pair_path.exists():
            raise FileNotFoundError(
                f"Missing matched pairs: {pair_path}"
            )

        if not presence_path.exists():
            raise FileNotFoundError(
                f"Missing species presence: {presence_path}"
            )

        pairs_df = pd.read_parquet(pair_path)
        presence_df = pd.read_parquet(presence_path)

        required_pair_columns = [
            "physical_plot_id",
            "early_plot_key",
            "late_plot_key",
            "early_year",
            "late_year",
            "stable_latitude",
            "stable_longitude",
        ]

        missing_pair_columns = [
            column
            for column in required_pair_columns
            if column not in pairs_df.columns
        ]

        if missing_pair_columns:
            raise KeyError(
                f"{state}: missing pair columns: "
                f"{missing_pair_columns}"
            )

        pairs_df = (
            pairs_df[required_pair_columns]
            .dropna(
                subset=[
                    "early_plot_key",
                    "late_plot_key",
                    "stable_latitude",
                    "early_year",
                    "late_year",
                ]
            )
            .drop_duplicates("physical_plot_id")
            .reset_index(drop=True)
        )

        for column in [
            "early_plot_key",
            "late_plot_key",
        ]:
            pairs_df[column] = (
                pairs_df[column]
                .astype("string")
                .str.strip()
            )

        for column in [
            "early_year",
            "late_year",
            "stable_latitude",
            "stable_longitude",
        ]:
            pairs_df[column] = pd.to_numeric(
                pairs_df[column],
                errors="coerce",
            )

        presence_df["plot_key"] = (
            presence_df["plot_key"]
            .astype("string")
            .str.strip()
        )
        presence_df["species_code"] = pd.to_numeric(
            presence_df["species_code"],
            errors="coerce",
        )

        presence_df = presence_df[
            presence_df["species_code"].isin(
                selected_species_codes
            )
        ].dropna(
            subset=["plot_key", "species_code"]
        ).copy()

        presence_df["species_code"] = (
            presence_df["species_code"].astype(int)
        )

        n_plots = len(pairs_df)
        n_species = len(selected_species_codes)

        if n_plots < 200:
            raise ValueError(
                f"{state}: only {n_plots} matched plots."
            )

        species_to_column = {
            species_code: index
            for index, species_code
            in enumerate(selected_species_codes)
        }

        early_key_to_row = {
            str(key): index
            for index, key in enumerate(
                pairs_df["early_plot_key"]
            )
        }
        late_key_to_row = {
            str(key): index
            for index, key in enumerate(
                pairs_df["late_plot_key"]
            )
        }

        early_matrix = np.zeros(
            (n_plots, n_species),
            dtype=np.uint8,
        )
        late_matrix = np.zeros(
            (n_plots, n_species),
            dtype=np.uint8,
        )

        for plot_key, species_code in tqdm(
            presence_df[
                ["plot_key", "species_code"]
            ].drop_duplicates().itertuples(
                index=False,
                name=None,
            ),
            total=len(
                presence_df[
                    ["plot_key", "species_code"]
                ].drop_duplicates()
            ),
            desc=f"{state}: building presence matrices",
            unit="presence",
        ):
            column = species_to_column.get(
                int(species_code)
            )

            if column is None:
                continue

            key = str(plot_key)

            early_row = early_key_to_row.get(key)
            if early_row is not None:
                early_matrix[early_row, column] = 1

            late_row = late_key_to_row.get(key)
            if late_row is not None:
                late_matrix[late_row, column] = 1

        stable_latitude = pairs_df[
            "stable_latitude"
        ].to_numpy(dtype=float)

        early_year = pairs_df[
            "early_year"
        ].to_numpy(dtype=float)

        late_year = pairs_df[
            "late_year"
        ].to_numpy(dtype=float)

        # Full repeated-plot truth.
        full_mask = np.ones(n_plots, dtype=bool)

        true_early_lat, true_early_counts = matrix_center(
            early_matrix,
            full_mask,
            stable_latitude,
            MIN_OBSERVED_OCCURRENCES,
        )
        true_late_lat, true_late_counts = matrix_center(
            late_matrix,
            full_mask,
            stable_latitude,
            MIN_OBSERVED_OCCURRENCES,
        )
        true_early_year, _ = matrix_center(
            early_matrix,
            full_mask,
            early_year,
            MIN_OBSERVED_OCCURRENCES,
        )
        true_late_year, _ = matrix_center(
            late_matrix,
            full_mask,
            late_year,
            MIN_OBSERVED_OCCURRENCES,
        )

        true_shift = vector_shift_km_decade(
            true_early_lat,
            true_late_lat,
            true_early_year,
            true_late_year,
        )

        # Stable grid frame for post-stratification.
        grid_lat_index = np.floor(
            stable_latitude / GRID_RESOLUTION
        ).astype(int)
        grid_lon_index = np.floor(
            pairs_df["stable_longitude"].to_numpy(dtype=float)
            / GRID_RESOLUTION
        ).astype(int)

        grid_labels = pd.Series(
            [
                f"{lat}_{lon}"
                for lat, lon in zip(
                    grid_lat_index,
                    grid_lon_index,
                )
            ]
        )

        unique_grids = sorted(
            grid_labels.unique().tolist()
        )
        grid_to_index = {
            grid: index
            for index, grid in enumerate(unique_grids)
        }
        grid_index = grid_labels.map(
            grid_to_index
        ).to_numpy(dtype=int)

        target_grid_counts = np.bincount(
            grid_index,
            minlength=len(unique_grids),
        ).astype(float)

        latitude_z = (
            stable_latitude
            - np.nanmean(stable_latitude)
        ) / np.nanstd(stable_latitude)

        sample_size = max(
            20,
            int(round(SAMPLE_FRACTION * n_plots)),
        )
        sample_size = min(sample_size, n_plots - 1)

        state_frame_rows.append(
            {
                "state": state,
                "n_matched_plots": n_plots,
                "n_grid_cells": len(unique_grids),
                "sample_size_each_period": sample_size,
                "sample_fraction": sample_size / n_plots,
                "n_species": n_species,
            }
        )

        rng = np.random.default_rng(
            RANDOM_SEED + state_index * 100000
        )

        state_result_parts = []

        scenario_iterator = [
            (severity, replicate)
            for severity in DRIFT_SEVERITIES
            for replicate in range(N_REPLICATES)
        ]

        for severity, replicate in tqdm(
            scenario_iterator,
            desc=f"{state}: drift simulations",
            unit="replicate",
        ):
            # Early sampling is biased southward and late sampling northward.
            # Severity 0 gives independent uniform samples.
            early_weights = np.exp(
                -severity * latitude_z
            )
            late_weights = np.exp(
                severity * latitude_z
            )

            early_sample_mask = weighted_without_replacement(
                rng,
                early_weights,
                sample_size,
            )
            late_sample_mask = weighted_without_replacement(
                rng,
                late_weights,
                sample_size,
            )

            raw_early_lat, raw_early_counts = matrix_center(
                early_matrix,
                early_sample_mask,
                stable_latitude,
                MIN_OBSERVED_OCCURRENCES,
            )
            raw_late_lat, raw_late_counts = matrix_center(
                late_matrix,
                late_sample_mask,
                stable_latitude,
                MIN_OBSERVED_OCCURRENCES,
            )
            raw_early_year, _ = matrix_center(
                early_matrix,
                early_sample_mask,
                early_year,
                MIN_OBSERVED_OCCURRENCES,
            )
            raw_late_year, _ = matrix_center(
                late_matrix,
                late_sample_mask,
                late_year,
                MIN_OBSERVED_OCCURRENCES,
            )

            raw_shift = vector_shift_km_decade(
                raw_early_lat,
                raw_late_lat,
                raw_early_year,
                raw_late_year,
            )

            early_post_weights, early_frame_coverage = (
                build_poststratification_weights(
                    early_sample_mask,
                    grid_index,
                    target_grid_counts,
                )
            )
            late_post_weights, late_frame_coverage = (
                build_poststratification_weights(
                    late_sample_mask,
                    grid_index,
                    target_grid_counts,
                )
            )

            corrected_early_lat, _ = weighted_matrix_center(
                early_matrix,
                early_post_weights,
                stable_latitude,
                MIN_OBSERVED_OCCURRENCES,
            )
            corrected_late_lat, _ = weighted_matrix_center(
                late_matrix,
                late_post_weights,
                stable_latitude,
                MIN_OBSERVED_OCCURRENCES,
            )
            corrected_early_year, _ = weighted_matrix_center(
                early_matrix,
                early_post_weights,
                early_year,
                MIN_OBSERVED_OCCURRENCES,
            )
            corrected_late_year, _ = weighted_matrix_center(
                late_matrix,
                late_post_weights,
                late_year,
                MIN_OBSERVED_OCCURRENCES,
            )

            corrected_shift = vector_shift_km_decade(
                corrected_early_lat,
                corrected_late_lat,
                corrected_early_year,
                corrected_late_year,
            )

            valid = (
                np.isfinite(true_shift)
                & np.isfinite(raw_shift)
                & np.isfinite(corrected_shift)
            )

            if not valid.any():
                continue

            part = pd.DataFrame(
                {
                    "state": state,
                    "severity": severity,
                    "replicate": replicate,
                    "species_code": selected_species_codes,
                    "true_shift_km_decade": true_shift,
                    "raw_shift_km_decade": raw_shift,
                    "corrected_shift_km_decade": (
                        corrected_shift
                    ),
                    "raw_early_occurrences": raw_early_counts,
                    "raw_late_occurrences": raw_late_counts,
                    "early_frame_coverage": (
                        early_frame_coverage
                    ),
                    "late_frame_coverage": (
                        late_frame_coverage
                    ),
                }
            )

            part = part.loc[valid].copy()

            part["raw_error_km_decade"] = (
                part["raw_shift_km_decade"]
                - part["true_shift_km_decade"]
            )
            part["corrected_error_km_decade"] = (
                part["corrected_shift_km_decade"]
                - part["true_shift_km_decade"]
            )

            part["raw_abs_error_km_decade"] = (
                part["raw_error_km_decade"].abs()
            )
            part["corrected_abs_error_km_decade"] = (
                part["corrected_error_km_decade"].abs()
            )

            part["correction_improved"] = (
                part["corrected_abs_error_km_decade"]
                < part["raw_abs_error_km_decade"]
            )

            part["raw_direction_error"] = [
                meaningful_direction_error(
                    estimate,
                    truth,
                )
                for estimate, truth in zip(
                    part["raw_shift_km_decade"],
                    part["true_shift_km_decade"],
                )
            ]

            part["corrected_direction_error"] = [
                meaningful_direction_error(
                    estimate,
                    truth,
                )
                for estimate, truth in zip(
                    part["corrected_shift_km_decade"],
                    part["true_shift_km_decade"],
                )
            ]

            state_result_parts.append(part)

        state_results_df = pd.concat(
            state_result_parts,
            ignore_index=True,
        )

        state_results_df.to_parquet(
            OUT_DIR / f"{state}_simulation_results.parquet",
            index=False,
        )

        all_result_parts.append(state_results_df)

        del (
            pairs_df,
            presence_df,
            early_matrix,
            late_matrix,
            state_result_parts,
        )
        gc.collect()

    all_results_df = pd.concat(
        all_result_parts,
        ignore_index=True,
    )

    all_results_df = all_results_df.merge(
        selected_species_df[
            [
                "species_code",
                "scientific_name",
                "common_name",
            ]
        ],
        on="species_code",
        how="left",
    )

    all_results_df.to_parquet(
        OUT_DIR / "all_simulation_results.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # Summary by state and severity
    # --------------------------------------------------------
    summary_rows = []

    groups = all_results_df.groupby(
        ["state", "severity"],
        sort=True,
    )

    for (state, severity), group in tqdm(
        groups,
        total=groups.ngroups,
        desc="Summarizing state-severity results",
        unit="group",
    ):
        raw_median = float(
            group[
                "raw_abs_error_km_decade"
            ].median()
        )
        corrected_median = float(
            group[
                "corrected_abs_error_km_decade"
            ].median()
        )

        reduction = (
            1 - corrected_median / raw_median
            if raw_median > 0
            else np.nan
        )

        summary_rows.append(
            {
                "state": state,
                "severity": float(severity),
                "n_species_replicates": len(group),
                "n_species": group[
                    "species_code"
                ].nunique(),
                "median_abs_raw_error_km_decade": (
                    raw_median
                ),
                "median_abs_corrected_error_km_decade": (
                    corrected_median
                ),
                "median_error_reduction_fraction": (
                    reduction
                ),
                "fraction_corrected_better": float(
                    group[
                        "correction_improved"
                    ].mean()
                ),
                "raw_direction_error_fraction": float(
                    group[
                        "raw_direction_error"
                    ].mean()
                ),
                "corrected_direction_error_fraction": float(
                    group[
                        "corrected_direction_error"
                    ].mean()
                ),
                "median_early_frame_coverage": float(
                    group[
                        "early_frame_coverage"
                    ].median()
                ),
                "median_late_frame_coverage": float(
                    group[
                        "late_frame_coverage"
                    ].median()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUT_DIR / "state_severity_summary.csv",
        index=False,
    )

    print("\nSTATE-SEVERITY SUMMARY")
    display(summary_df)

    # --------------------------------------------------------
    # Preliminary species sensitivity index
    # --------------------------------------------------------
    sensitivity_source_df = all_results_df[
        all_results_df["severity"].isin([1.0, 1.5])
    ].copy()

    species_sensitivity_df = (
        sensitivity_source_df.groupby(
            [
                "species_code",
                "scientific_name",
                "common_name",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_states=("state", "nunique"),
            n_species_replicates=(
                "raw_abs_error_km_decade",
                "size",
            ),
            observation_drift_sensitivity_km_decade=(
                "raw_abs_error_km_decade",
                "median",
            ),
            corrected_residual_error_km_decade=(
                "corrected_abs_error_km_decade",
                "median",
            ),
            fraction_corrected_better=(
                "correction_improved",
                "mean",
            ),
            raw_direction_error_fraction=(
                "raw_direction_error",
                "mean",
            ),
            corrected_direction_error_fraction=(
                "corrected_direction_error",
                "mean",
            ),
        )
        .sort_values(
            "observation_drift_sensitivity_km_decade",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    species_sensitivity_df[
        "correction_gain_fraction"
    ] = (
        1
        - species_sensitivity_df[
            "corrected_residual_error_km_decade"
        ]
        / species_sensitivity_df[
            "observation_drift_sensitivity_km_decade"
        ]
    )

    species_sensitivity_df.to_parquet(
        OUT_DIR / "preliminary_species_sensitivity.parquet",
        index=False,
    )
    species_sensitivity_df.to_csv(
        OUT_DIR / "preliminary_species_sensitivity.csv",
        index=False,
    )

    print("\nPRELIMINARY SPECIES SENSITIVITY")
    display(species_sensitivity_df)

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------
    state_decision_rows = []

    for state in STATES:
        state_summary = summary_df[
            summary_df["state"] == state
        ].sort_values("severity")

        null_row = state_summary[
            state_summary["severity"] == 0.0
        ].iloc[0]

        high_row = state_summary[
            state_summary["severity"] == 1.5
        ].iloc[0]

        severity_spearman = safe_spearman(
            state_summary["severity"],
            state_summary[
                "median_abs_raw_error_km_decade"
            ],
        )

        null_pass = bool(
            null_row[
                "median_abs_raw_error_km_decade"
            ]
            <= NULL_MAX_MEDIAN_ERROR_KM_DECADE
        )

        high_drift_pass = bool(
            high_row[
                "median_abs_raw_error_km_decade"
            ]
            >= HIGH_DRIFT_MIN_MEDIAN_ERROR_KM_DECADE
        )

        correction_pass = bool(
            high_row[
                "median_error_reduction_fraction"
            ]
            >= MIN_CORRECTION_REDUCTION
            and high_row[
                "fraction_corrected_better"
            ]
            >= MIN_FRACTION_CORRECTED_BETTER
        )

        monotonic_pass = bool(
            np.isfinite(severity_spearman)
            and severity_spearman
            >= MIN_SEVERITY_SPEARMAN
        )

        state_pass = bool(
            null_pass
            and high_drift_pass
            and correction_pass
            and monotonic_pass
        )

        state_decision_rows.append(
            {
                "state": state,
                "null_median_error": null_row[
                    "median_abs_raw_error_km_decade"
                ],
                "high_drift_raw_median_error": high_row[
                    "median_abs_raw_error_km_decade"
                ],
                "high_drift_corrected_median_error": high_row[
                    "median_abs_corrected_error_km_decade"
                ],
                "high_drift_error_reduction_fraction": high_row[
                    "median_error_reduction_fraction"
                ],
                "high_drift_fraction_corrected_better": high_row[
                    "fraction_corrected_better"
                ],
                "severity_error_spearman": severity_spearman,
                "null_pass": null_pass,
                "high_drift_pass": high_drift_pass,
                "correction_pass": correction_pass,
                "monotonic_pass": monotonic_pass,
                "state_pass": state_pass,
            }
        )

    decision_df = pd.DataFrame(
        state_decision_rows
    )

    decision_df.to_csv(
        OUT_DIR / "benchmark_decision_summary.csv",
        index=False,
    )

    print("\nBENCHMARK DECISION SUMMARY")
    display(decision_df)

    n_state_passes = int(
        decision_df["state_pass"].sum()
    )

    passed = bool(
        n_state_passes == len(STATES)
    )

    if passed:
        status = "GO_FOR_REAL_GBIF_PILOT"
        next_step = (
            "The benchmark shows that controlled observation drift can "
            "create false migration signals and that transparent spatial "
            "standardization recovers part of the truth. Next obtain a "
            "small GBIF/iNaturalist sample for the same species and test "
            "whether real observation-system drift falls inside the "
            "validated severity range."
        )
    elif n_state_passes == 1:
        status = "BENCHMARK_PARTIAL_REPLICATION"
        next_step = (
            "The mechanism is plausible in one state but not yet stable. "
            "Inspect occupancy breadth and grid support before adding "
            "real GBIF records."
        )
    else:
        status = "BENCHMARK_METHOD_NOT_EFFECTIVE"
        next_step = (
            "The selected drift mechanism or correction is not reliable "
            "enough. Do not proceed to GBIF until the benchmark design "
            "is revised."
        )

    # --------------------------------------------------------
    # Visual previews
    # --------------------------------------------------------
    for state in STATES:
        plot_df = summary_df[
            summary_df["state"] == state
        ].sort_values("severity")

        plt.figure(figsize=(9, 5))
        plt.plot(
            plot_df["severity"],
            plot_df[
                "median_abs_raw_error_km_decade"
            ],
            marker="o",
            label="Raw",
        )
        plt.plot(
            plot_df["severity"],
            plot_df[
                "median_abs_corrected_error_km_decade"
            ],
            marker="o",
            label="Spatially standardized",
        )
        plt.xlabel("Injected observation-drift severity")
        plt.ylabel(
            "Median absolute migration error (km/decade)"
        )
        plt.title(
            f"{state}: semi-synthetic observation-drift benchmark"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    top_sensitivity_df = (
        species_sensitivity_df.head(15)
        .sort_values(
            "observation_drift_sensitivity_km_decade"
        )
    )

    labels = top_sensitivity_df[
        "scientific_name"
    ].fillna(
        top_sensitivity_df["species_code"].astype(str)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        labels,
        top_sensitivity_df[
            "observation_drift_sensitivity_km_decade"
        ],
    )
    plt.xlabel(
        "Preliminary observation-drift sensitivity "
        "(median absolute error, km/decade)"
    )
    plt.ylabel("Species")
    plt.title(
        "Species most sensitive to injected sampling drift"
    )
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    frame_df = pd.DataFrame(state_frame_rows)
    frame_df.to_csv(
        OUT_DIR / "benchmark_frame_summary.csv",
        index=False,
    )

    top_species = (
        species_sensitivity_df[
            [
                "species_code",
                "scientific_name",
                "observation_drift_sensitivity_km_decade",
                "corrected_residual_error_km_decade",
                "correction_gain_fraction",
            ]
        ]
        .head(10)
        .round(4)
        .to_dict("records")
    )

    append_readme(
        f"""

## Semi-synthetic observation-drift benchmark — {RUN_UTC}

### Reason for pivot
Exact repeated-physical-plot matching showed that the earlier annual FIA panel
signal did not survive as a natural observation-drift effect. Repeated plots are
therefore used as a standardized truth benchmark rather than as evidence of real
observer movement.

### Hypothesis
Controlled northward movement of the sampled plot distribution can manufacture
apparent species latitude shifts, and transparent spatial post-stratification
can reduce this error.

### Configuration
- States: {STATES}
- Species: {len(selected_species_codes)}
- Drift severities: {DRIFT_SEVERITIES}
- Replicates per state and severity: {N_REPLICATES}
- Sample fraction per period: {SAMPLE_FRACTION}
- Spatial post-stratification grid: {GRID_RESOLUTION} degrees
- Minimum observed occurrences: {MIN_OBSERVED_OCCURRENCES}
- Random seed: {RANDOM_SEED}
- GPU: not used

### Success criteria
- Severity 0 median raw error <=
  {NULL_MAX_MEDIAN_ERROR_KM_DECADE} km/decade
- Severity 1.5 median raw error >=
  {HIGH_DRIFT_MIN_MEDIAN_ERROR_KM_DECADE} km/decade
- Spatial correction reduces median error by >=
  {MIN_CORRECTION_REDUCTION:.0%}
- Correction improves >=
  {MIN_FRACTION_CORRECTED_BETTER:.0%} of species-replicates
- Error rises monotonically with drift severity
  (Spearman >= {MIN_SEVERITY_SPEARMAN})

### Decision
- State passes: {n_state_passes}/{len(STATES)}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
This is a semi-synthetic causal benchmark. It demonstrates what a specified
sampling drift can do under controlled conditions. It does not estimate the
actual magnitude or direction of GBIF/iNaturalist observer movement.
"""
    )

    # --------------------------------------------------------
    # Compact output
    # --------------------------------------------------------
    compact_summary = (
        summary_df.round(6).to_dict("records")
    )
    compact_decision = (
        decision_df.round(6).to_dict("records")
    )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATES: {STATES}")
    print(
        f"N_SELECTED_SPECIES: "
        f"{len(selected_species_codes)}"
    )
    print(
        "SELECTED_SPECIES: "
        + json.dumps(
            selected_species_df[
                [
                    "species_code",
                    "scientific_name",
                    "minimum_across_states",
                ]
            ].to_dict("records"),
            ensure_ascii=False,
        )
    )
    print(
        "STATE_SEVERITY_RESULTS: "
        + json.dumps(
            compact_summary,
            ensure_ascii=False,
        )
    )
    print(
        "STATE_DECISIONS: "
        + json.dumps(
            compact_decision,
            ensure_ascii=False,
        )
    )
    print(
        "TOP_SENSITIVE_SPECIES: "
        + json.dumps(
            top_species,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_CRITERIA: null error <=1.5; high-drift raw error "
        ">=2.0 km/decade; correction reduction >=30%; correction "
        "better in >=60%; severity-error Spearman >=0.80 in both states"
    )
    print(f"N_STATE_PASSES: {n_state_passes}")
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except Exception as exc:
    print_failure(
        "CELL07A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        "Send this COPY block and the last displayed output. "
        "Do not rerun large downloads.",
    )
