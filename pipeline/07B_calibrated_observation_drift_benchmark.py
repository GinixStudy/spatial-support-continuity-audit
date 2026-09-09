
# Cell 07B — Calibrate observation drift in interpretable km/decade
#
# Why this step
# -------------
# Cell 07A showed a strong severity-response and large correction gains in both
# WV and PA. WV failed only because its severity-0 resampling error (1.61 km/decade)
# narrowly exceeded an arbitrary fixed threshold of 1.50.
#
# This cell removes that arbitrary decision rule.
#
# It calibrates:
# 1. the realized movement of the sampled-plot center in km/decade;
# 2. the resulting species migration-estimation error;
# 3. correction performance across sample fractions and grid resolutions;
# 4. an empirical severity-0 null distribution within each state and fraction.
#
# This is still a semi-synthetic benchmark. It does not claim that real GBIF
# observers moved by the simulated amount.

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
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 300)

# ============================================================
# Configuration
# ============================================================
STATES = ["WV", "PA"]

SEVERITIES = [0.0, 0.10, 0.25, 0.50, 1.00]
SAMPLE_FRACTIONS = [0.30, 0.50, 0.70]
GRID_RESOLUTIONS = [0.25, 0.50, 1.00]

N_REPLICATES = 100
RANDOM_SEED = 20260710

DEFAULT_SPECIES_FRACTION = 0.50
MIN_OBSERVED_OCCURRENCES = 5

MIN_REALIZED_DRIFT_FOR_EVALUATION = 5.0
MAX_ACCEPTABLE_NULL_INFLATION = 1.25

MIN_RESPONSE_SPEARMAN = 0.80
MIN_EMPIRICAL_DETECTION_RATE = 0.80
MIN_EXCESS_ERROR_REDUCTION = 0.50
MIN_FRACTION_CORRECTED_BETTER = 0.70
MIN_FRACTION_PASSES_PER_STATE = 2

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
REPEATED_DIR = BASE_DIR / "derived" / "repeated_physical_plot_control"
PRESENCE_DIR = BASE_DIR / "derived" / "appalachian_four_state_replication"
BENCHMARK_DIR = BASE_DIR / "derived" / "semi_synthetic_observation_drift"
OUT_DIR = BASE_DIR / "derived" / "observation_drift_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_SPECIES_CSV = (
    BENCHMARK_DIR / "selected_benchmark_species.csv"
)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("STATES:", STATES)
print("SEVERITIES:", SEVERITIES)
print("SAMPLE_FRACTIONS:", SAMPLE_FRACTIONS)
print("GRID_RESOLUTIONS:", GRID_RESOLUTIONS)
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

    frame_coverage = (
        target_grid_counts[available].sum()
        / target_grid_counts.sum()
    )

    return plot_weights, float(frame_coverage)


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

    denominator = selected_weights @ selected_matrix
    numerator = (
        selected_weights * selected_values
    ) @ selected_matrix

    means = np.divide(
        numerator,
        denominator,
        out=np.full(
            presence_matrix.shape[1],
            np.nan,
            dtype=float,
        ),
        where=(
            (denominator > 0)
            & (raw_counts >= minimum_occurrences)
        ),
    )

    return means, raw_counts


def shift_km_decade(
    early_latitude,
    late_latitude,
    early_year,
    late_year,
):
    early_latitude = np.asarray(
        early_latitude,
        dtype=float,
    )
    late_latitude = np.asarray(
        late_latitude,
        dtype=float,
    )
    early_year = np.asarray(
        early_year,
        dtype=float,
    )
    late_year = np.asarray(
        late_year,
        dtype=float,
    )

    interval = late_year - early_year

    result = np.full(
        np.broadcast(
            early_latitude,
            late_latitude,
            early_year,
            late_year,
        ).shape,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(early_latitude)
        & np.isfinite(late_latitude)
        & np.isfinite(early_year)
        & np.isfinite(late_year)
        & (interval > 5)
    )

    result[valid] = (
        (late_latitude[valid] - early_latitude[valid])
        * 111.32
        * 10
        / interval[valid]
    )

    return result


def scalar_shift_km_decade(
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


def direction_error_array(
    estimates,
    truth,
    minimum_magnitude=1.0,
):
    estimates = np.asarray(estimates, dtype=float)
    truth = np.asarray(truth, dtype=float)

    return (
        np.isfinite(estimates)
        & np.isfinite(truth)
        & (np.abs(estimates) >= minimum_magnitude)
        & (np.abs(truth) >= minimum_magnitude)
        & (np.sign(estimates) != np.sign(truth))
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
        frame.corr(method="spearman").iloc[0, 1]
    )


def ordinary_slope(x, y):
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if (
        len(frame) < 3
        or frame["x"].nunique() < 2
    ):
        return np.nan

    x_values = frame["x"].to_numpy(dtype=float)
    y_values = frame["y"].to_numpy(dtype=float)

    centered_x = x_values - x_values.mean()
    denominator = np.dot(centered_x, centered_x)

    if denominator == 0:
        return np.nan

    return float(
        np.dot(
            centered_x,
            y_values - y_values.mean(),
        )
        / denominator
    )


def print_failure(status, error, next_step):
    append_readme(
        f"""

## Observation-drift calibration failure — {RUN_UTC}
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
    # 1. Validate and load the fixed species list
    # --------------------------------------------------------
    if not SELECTED_SPECIES_CSV.exists():
        raise FileNotFoundError(
            f"Missing Cell 07A species list: "
            f"{SELECTED_SPECIES_CSV}"
        )

    selected_species_df = pd.read_csv(
        SELECTED_SPECIES_CSV
    )

    selected_species_df["species_code"] = pd.to_numeric(
        selected_species_df["species_code"],
        errors="coerce",
    )

    selected_species_df = selected_species_df.dropna(
        subset=["species_code"]
    ).copy()

    selected_species_df["species_code"] = (
        selected_species_df["species_code"].astype(int)
    )

    selected_species_codes = selected_species_df[
        "species_code"
    ].tolist()

    if len(selected_species_codes) < 10:
        raise ValueError(
            "The benchmark species list is unexpectedly small."
        )

    print("\nFIXED SPECIES LIST")
    display(selected_species_df)

    # --------------------------------------------------------
    # 2. Simulate each state
    # --------------------------------------------------------
    scenario_rows = []
    species_result_parts = []
    frame_rows = []

    for state_index, state in enumerate(STATES):
        print("\n" + "=" * 80)
        print(f"CALIBRATING STATE: {state}")
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
                f"Missing presence table: {presence_path}"
            )

        pairs_df = pd.read_parquet(pair_path)
        presence_df = pd.read_parquet(presence_path)

        pair_columns = [
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
            for column in pair_columns
            if column not in pairs_df.columns
        ]

        if missing_pair_columns:
            raise KeyError(
                f"{state}: missing pair fields: "
                f"{missing_pair_columns}"
            )

        pairs_df = (
            pairs_df[pair_columns]
            .dropna(
                subset=[
                    "physical_plot_id",
                    "early_plot_key",
                    "late_plot_key",
                    "early_year",
                    "late_year",
                    "stable_latitude",
                    "stable_longitude",
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

        unique_presence_df = presence_df[
            ["plot_key", "species_code"]
        ].drop_duplicates()

        for plot_key, species_code in tqdm(
            unique_presence_df.itertuples(
                index=False,
                name=None,
            ),
            total=len(unique_presence_df),
            desc=f"{state}: building presence matrices",
            unit="presence",
        ):
            column_index = species_to_column.get(
                int(species_code)
            )

            if column_index is None:
                continue

            key = str(plot_key)

            early_row = early_key_to_row.get(key)
            if early_row is not None:
                early_matrix[
                    early_row,
                    column_index,
                ] = 1

            late_row = late_key_to_row.get(key)
            if late_row is not None:
                late_matrix[
                    late_row,
                    column_index,
                ] = 1

        stable_latitude = pairs_df[
            "stable_latitude"
        ].to_numpy(dtype=float)
        stable_longitude = pairs_df[
            "stable_longitude"
        ].to_numpy(dtype=float)
        early_year = pairs_df[
            "early_year"
        ].to_numpy(dtype=float)
        late_year = pairs_df[
            "late_year"
        ].to_numpy(dtype=float)

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

        true_shift = shift_km_decade(
            true_early_lat,
            true_late_lat,
            true_early_year,
            true_late_year,
        )

        grid_structures = {}

        for resolution in GRID_RESOLUTIONS:
            lat_index = np.floor(
                stable_latitude / resolution
            ).astype(int)
            lon_index = np.floor(
                stable_longitude / resolution
            ).astype(int)

            labels = pd.Series(
                [
                    f"{lat}_{lon}"
                    for lat, lon in zip(
                        lat_index,
                        lon_index,
                    )
                ]
            )

            unique_labels = sorted(
                labels.unique().tolist()
            )
            label_to_index = {
                label: index
                for index, label
                in enumerate(unique_labels)
            }

            grid_index = labels.map(
                label_to_index
            ).to_numpy(dtype=int)

            target_counts = np.bincount(
                grid_index,
                minlength=len(unique_labels),
            ).astype(float)

            grid_structures[resolution] = {
                "grid_index": grid_index,
                "target_counts": target_counts,
                "n_cells": len(unique_labels),
            }

        latitude_std = np.nanstd(stable_latitude)

        if not np.isfinite(latitude_std) or latitude_std == 0:
            raise ValueError(
                f"{state}: invalid latitude standard deviation."
            )

        latitude_z = (
            stable_latitude
            - np.nanmean(stable_latitude)
        ) / latitude_std

        frame_rows.append(
            {
                "state": state,
                "n_matched_plots": n_plots,
                "n_species": n_species,
                "minimum_true_early_occurrences": int(
                    np.nanmin(true_early_counts)
                ),
                "minimum_true_late_occurrences": int(
                    np.nanmin(true_late_counts)
                ),
                **{
                    f"n_cells_{resolution}": (
                        grid_structures[resolution][
                            "n_cells"
                        ]
                    )
                    for resolution in GRID_RESOLUTIONS
                },
            }
        )

        rng = np.random.default_rng(
            RANDOM_SEED
            + state_index * 1_000_000
        )

        scenario_iterator = [
            (
                sample_fraction,
                severity,
                replicate,
            )
            for sample_fraction in SAMPLE_FRACTIONS
            for severity in SEVERITIES
            for replicate in range(N_REPLICATES)
        ]

        for (
            sample_fraction,
            severity,
            replicate,
        ) in tqdm(
            scenario_iterator,
            desc=f"{state}: calibration simulations",
            unit="scenario",
        ):
            sample_size = int(
                round(sample_fraction * n_plots)
            )
            sample_size = max(20, sample_size)
            sample_size = min(
                sample_size,
                n_plots - 1,
            )

            early_weights = np.exp(
                -severity * latitude_z
            )
            late_weights = np.exp(
                severity * latitude_z
            )

            early_mask = weighted_without_replacement(
                rng,
                early_weights,
                sample_size,
            )
            late_mask = weighted_without_replacement(
                rng,
                late_weights,
                sample_size,
            )

            observer_center_shift = (
                scalar_shift_km_decade(
                    float(
                        np.mean(
                            stable_latitude[early_mask]
                        )
                    ),
                    float(
                        np.mean(
                            stable_latitude[late_mask]
                        )
                    ),
                    float(
                        np.mean(
                            early_year[early_mask]
                        )
                    ),
                    float(
                        np.mean(
                            late_year[late_mask]
                        )
                    ),
                )
            )

            raw_early_lat, raw_early_counts = matrix_center(
                early_matrix,
                early_mask,
                stable_latitude,
                MIN_OBSERVED_OCCURRENCES,
            )
            raw_late_lat, raw_late_counts = matrix_center(
                late_matrix,
                late_mask,
                stable_latitude,
                MIN_OBSERVED_OCCURRENCES,
            )
            raw_early_year, _ = matrix_center(
                early_matrix,
                early_mask,
                early_year,
                MIN_OBSERVED_OCCURRENCES,
            )
            raw_late_year, _ = matrix_center(
                late_matrix,
                late_mask,
                late_year,
                MIN_OBSERVED_OCCURRENCES,
            )

            raw_shift = shift_km_decade(
                raw_early_lat,
                raw_late_lat,
                raw_early_year,
                raw_late_year,
            )

            raw_error = raw_shift - true_shift
            raw_abs_error = np.abs(raw_error)
            raw_direction_error = (
                direction_error_array(
                    raw_shift,
                    true_shift,
                )
            )

            raw_valid = (
                np.isfinite(raw_shift)
                & np.isfinite(true_shift)
            )

            for resolution in GRID_RESOLUTIONS:
                structure = grid_structures[resolution]

                early_post_weights, early_coverage = (
                    build_poststratification_weights(
                        early_mask,
                        structure["grid_index"],
                        structure["target_counts"],
                    )
                )
                late_post_weights, late_coverage = (
                    build_poststratification_weights(
                        late_mask,
                        structure["grid_index"],
                        structure["target_counts"],
                    )
                )

                corrected_early_lat, _ = (
                    weighted_matrix_center(
                        early_matrix,
                        early_post_weights,
                        stable_latitude,
                        MIN_OBSERVED_OCCURRENCES,
                    )
                )
                corrected_late_lat, _ = (
                    weighted_matrix_center(
                        late_matrix,
                        late_post_weights,
                        stable_latitude,
                        MIN_OBSERVED_OCCURRENCES,
                    )
                )
                corrected_early_year, _ = (
                    weighted_matrix_center(
                        early_matrix,
                        early_post_weights,
                        early_year,
                        MIN_OBSERVED_OCCURRENCES,
                    )
                )
                corrected_late_year, _ = (
                    weighted_matrix_center(
                        late_matrix,
                        late_post_weights,
                        late_year,
                        MIN_OBSERVED_OCCURRENCES,
                    )
                )

                corrected_shift = shift_km_decade(
                    corrected_early_lat,
                    corrected_late_lat,
                    corrected_early_year,
                    corrected_late_year,
                )

                corrected_error = (
                    corrected_shift - true_shift
                )
                corrected_abs_error = np.abs(
                    corrected_error
                )

                corrected_direction_error = (
                    direction_error_array(
                        corrected_shift,
                        true_shift,
                    )
                )

                valid = (
                    raw_valid
                    & np.isfinite(corrected_shift)
                )

                if not valid.any():
                    continue

                scenario_rows.append(
                    {
                        "state": state,
                        "sample_fraction": (
                            sample_fraction
                        ),
                        "severity": severity,
                        "replicate": replicate,
                        "grid_resolution": resolution,
                        "sample_size": sample_size,
                        "realized_observer_center_shift_km_decade": (
                            observer_center_shift
                        ),
                        "absolute_observer_center_shift_km_decade": (
                            abs(observer_center_shift)
                        ),
                        "n_valid_species": int(
                            valid.sum()
                        ),
                        "raw_median_abs_error_km_decade": float(
                            np.median(
                                raw_abs_error[valid]
                            )
                        ),
                        "corrected_median_abs_error_km_decade": float(
                            np.median(
                                corrected_abs_error[valid]
                            )
                        ),
                        "fraction_corrected_better": float(
                            np.mean(
                                corrected_abs_error[valid]
                                < raw_abs_error[valid]
                            )
                        ),
                        "raw_direction_error_fraction": float(
                            np.mean(
                                raw_direction_error[valid]
                            )
                        ),
                        "corrected_direction_error_fraction": float(
                            np.mean(
                                corrected_direction_error[
                                    valid
                                ]
                            )
                        ),
                        "early_frame_coverage": (
                            early_coverage
                        ),
                        "late_frame_coverage": (
                            late_coverage
                        ),
                    }
                )

                if np.isclose(
                    sample_fraction,
                    DEFAULT_SPECIES_FRACTION,
                ):
                    species_part = pd.DataFrame(
                        {
                            "state": state,
                            "sample_fraction": (
                                sample_fraction
                            ),
                            "severity": severity,
                            "replicate": replicate,
                            "grid_resolution": (
                                resolution
                            ),
                            "species_code": (
                                selected_species_codes
                            ),
                            "true_shift_km_decade": (
                                true_shift
                            ),
                            "raw_shift_km_decade": (
                                raw_shift
                            ),
                            "corrected_shift_km_decade": (
                                corrected_shift
                            ),
                            "raw_abs_error_km_decade": (
                                raw_abs_error
                            ),
                            "corrected_abs_error_km_decade": (
                                corrected_abs_error
                            ),
                            "realized_observer_center_shift_km_decade": (
                                observer_center_shift
                            ),
                            "absolute_observer_center_shift_km_decade": (
                                abs(observer_center_shift)
                            ),
                            "raw_early_occurrences": (
                                raw_early_counts
                            ),
                            "raw_late_occurrences": (
                                raw_late_counts
                            ),
                        }
                    )

                    species_part = species_part.loc[
                        valid
                    ].copy()

                    species_part[
                        "correction_improved"
                    ] = (
                        species_part[
                            "corrected_abs_error_km_decade"
                        ]
                        < species_part[
                            "raw_abs_error_km_decade"
                        ]
                    )

                    species_result_parts.append(
                        species_part
                    )

        del (
            pairs_df,
            presence_df,
            early_matrix,
            late_matrix,
            grid_structures,
        )
        gc.collect()

    scenario_df = pd.DataFrame(scenario_rows)
    species_result_df = pd.concat(
        species_result_parts,
        ignore_index=True,
    )
    frame_df = pd.DataFrame(frame_rows)

    scenario_df.to_parquet(
        OUT_DIR / "calibration_scenario_results.parquet",
        index=False,
    )
    species_result_df.to_parquet(
        OUT_DIR / "calibration_species_results.parquet",
        index=False,
    )
    frame_df.to_csv(
        OUT_DIR / "calibration_frame_summary.csv",
        index=False,
    )

    print("\nCALIBRATION FRAME SUMMARY")
    display(frame_df)

    # --------------------------------------------------------
    # 3. Empirical severity-0 null calibration
    # --------------------------------------------------------
    null_df = (
        scenario_df[
            scenario_df["severity"] == 0.0
        ]
        .groupby(
            [
                "state",
                "sample_fraction",
                "grid_resolution",
            ],
            as_index=False,
        )
        .agg(
            raw_null_median=(
                "raw_median_abs_error_km_decade",
                "median",
            ),
            raw_null_p95=(
                "raw_median_abs_error_km_decade",
                lambda series: series.quantile(0.95),
            ),
            corrected_null_median=(
                "corrected_median_abs_error_km_decade",
                "median",
            ),
            corrected_null_p95=(
                "corrected_median_abs_error_km_decade",
                lambda series: series.quantile(0.95),
            ),
        )
    )

    scenario_df = scenario_df.merge(
        null_df,
        on=[
            "state",
            "sample_fraction",
            "grid_resolution",
        ],
        how="left",
        validate="many_to_one",
    )

    scenario_df["raw_excess_error"] = (
        scenario_df[
            "raw_median_abs_error_km_decade"
        ]
        - scenario_df["raw_null_median"]
    ).clip(lower=0)

    scenario_df["corrected_excess_error"] = (
        scenario_df[
            "corrected_median_abs_error_km_decade"
        ]
        - scenario_df[
            "corrected_null_median"
        ]
    ).clip(lower=0)

    scenario_df["excess_error_reduction_fraction"] = (
        1
        - scenario_df[
            "corrected_excess_error"
        ]
        / scenario_df["raw_excess_error"]
    )

    scenario_df.loc[
        scenario_df["raw_excess_error"] <= 0,
        "excess_error_reduction_fraction",
    ] = np.nan

    scenario_df["raw_exceeds_null95"] = (
        scenario_df[
            "raw_median_abs_error_km_decade"
        ]
        > scenario_df["raw_null_p95"]
    )

    scenario_df.to_parquet(
        OUT_DIR / "calibration_scenario_results_with_null.parquet",
        index=False,
    )
    null_df.to_csv(
        OUT_DIR / "empirical_null_summary.csv",
        index=False,
    )

    print("\nEMPIRICAL NULL SUMMARY")
    display(null_df)

    # --------------------------------------------------------
    # 4. Select correction grid transparently
    # --------------------------------------------------------
    evaluation_df = scenario_df[
        (scenario_df["severity"] > 0)
        & (
            scenario_df[
                "absolute_observer_center_shift_km_decade"
            ]
            >= MIN_REALIZED_DRIFT_FOR_EVALUATION
        )
    ].copy()

    if evaluation_df.empty:
        raise ValueError(
            "No scenarios reached the minimum realized drift."
        )

    grid_performance_rows = []

    for resolution, group in evaluation_df.groupby(
        "grid_resolution"
    ):
        null_group = null_df[
            null_df["grid_resolution"] == resolution
        ].copy()

        null_inflation = (
            null_group["corrected_null_median"]
            / null_group["raw_null_median"]
        ).replace([np.inf, -np.inf], np.nan)

        grid_performance_rows.append(
            {
                "grid_resolution": resolution,
                "n_evaluation_scenarios": len(group),
                "median_excess_error_reduction_fraction": float(
                    group[
                        "excess_error_reduction_fraction"
                    ].median()
                ),
                "fraction_corrected_better": float(
                    (
                        group[
                            "corrected_median_abs_error_km_decade"
                        ]
                        < group[
                            "raw_median_abs_error_km_decade"
                        ]
                    ).mean()
                ),
                "median_corrected_abs_error_km_decade": float(
                    group[
                        "corrected_median_abs_error_km_decade"
                    ].median()
                ),
                "median_frame_coverage": float(
                    pd.concat(
                        [
                            group["early_frame_coverage"],
                            group["late_frame_coverage"],
                        ]
                    ).median()
                ),
                "median_null_inflation_ratio": float(
                    null_inflation.median()
                ),
            }
        )

    grid_performance_df = pd.DataFrame(
        grid_performance_rows
    )

    grid_performance_df["null_inflation_acceptable"] = (
        grid_performance_df[
            "median_null_inflation_ratio"
        ]
        <= MAX_ACCEPTABLE_NULL_INFLATION
    )

    acceptable_grids = grid_performance_df[
        grid_performance_df[
            "null_inflation_acceptable"
        ]
    ].copy()

    if acceptable_grids.empty:
        candidate_grids = grid_performance_df.copy()
        grid_selection_warning = True
    else:
        candidate_grids = acceptable_grids
        grid_selection_warning = False

    best_grid_row = candidate_grids.sort_values(
        [
            "median_excess_error_reduction_fraction",
            "fraction_corrected_better",
        ],
        ascending=[False, False],
    ).iloc[0]

    best_grid = float(
        best_grid_row["grid_resolution"]
    )

    grid_performance_df.to_csv(
        OUT_DIR / "grid_performance_summary.csv",
        index=False,
    )

    print("\nGRID PERFORMANCE SUMMARY")
    display(grid_performance_df)

    # --------------------------------------------------------
    # 5. State × sample-fraction robustness
    # --------------------------------------------------------
    robustness_rows = []

    for state in STATES:
        for sample_fraction in SAMPLE_FRACTIONS:
            raw_unique = (
                scenario_df[
                    (scenario_df["state"] == state)
                    & (
                        scenario_df["sample_fraction"]
                        == sample_fraction
                    )
                ][
                    [
                        "severity",
                        "replicate",
                        "absolute_observer_center_shift_km_decade",
                        "raw_median_abs_error_km_decade",
                    ]
                ]
                .drop_duplicates(
                    ["severity", "replicate"]
                )
            )

            response_spearman = safe_spearman(
                raw_unique[
                    "absolute_observer_center_shift_km_decade"
                ],
                raw_unique[
                    "raw_median_abs_error_km_decade"
                ],
            )

            chosen_group = scenario_df[
                (scenario_df["state"] == state)
                & (
                    scenario_df["sample_fraction"]
                    == sample_fraction
                )
                & (
                    scenario_df["grid_resolution"]
                    == best_grid
                )
                & (scenario_df["severity"] > 0)
                & (
                    scenario_df[
                        "absolute_observer_center_shift_km_decade"
                    ]
                    >= MIN_REALIZED_DRIFT_FOR_EVALUATION
                )
            ].copy()

            if chosen_group.empty:
                detection_rate = np.nan
                excess_reduction = np.nan
                corrected_better = np.nan
                median_observer_drift = np.nan
            else:
                detection_rate = float(
                    chosen_group[
                        "raw_exceeds_null95"
                    ].mean()
                )
                excess_reduction = float(
                    chosen_group[
                        "excess_error_reduction_fraction"
                    ].median()
                )
                corrected_better = float(
                    (
                        chosen_group[
                            "corrected_median_abs_error_km_decade"
                        ]
                        < chosen_group[
                            "raw_median_abs_error_km_decade"
                        ]
                    ).mean()
                )
                median_observer_drift = float(
                    chosen_group[
                        "absolute_observer_center_shift_km_decade"
                    ].median()
                )

            response_pass = bool(
                np.isfinite(response_spearman)
                and response_spearman
                >= MIN_RESPONSE_SPEARMAN
            )
            detection_pass = bool(
                np.isfinite(detection_rate)
                and detection_rate
                >= MIN_EMPIRICAL_DETECTION_RATE
            )
            correction_pass = bool(
                np.isfinite(excess_reduction)
                and excess_reduction
                >= MIN_EXCESS_ERROR_REDUCTION
                and np.isfinite(corrected_better)
                and corrected_better
                >= MIN_FRACTION_CORRECTED_BETTER
            )

            robustness_rows.append(
                {
                    "state": state,
                    "sample_fraction": sample_fraction,
                    "best_grid_resolution": best_grid,
                    "response_spearman": response_spearman,
                    "median_evaluated_observer_drift_km_decade": (
                        median_observer_drift
                    ),
                    "empirical_detection_rate": detection_rate,
                    "median_excess_error_reduction_fraction": (
                        excess_reduction
                    ),
                    "fraction_corrected_better": corrected_better,
                    "response_pass": response_pass,
                    "detection_pass": detection_pass,
                    "correction_pass": correction_pass,
                    "fraction_pass": bool(
                        response_pass
                        and detection_pass
                        and correction_pass
                    ),
                }
            )

    robustness_df = pd.DataFrame(
        robustness_rows
    )

    robustness_df.to_csv(
        OUT_DIR / "state_fraction_robustness.csv",
        index=False,
    )

    print("\nSTATE × SAMPLE-FRACTION ROBUSTNESS")
    display(robustness_df)

    # --------------------------------------------------------
    # 6. Species sensitivity using default fraction and best grid
    # --------------------------------------------------------
    chosen_species_df = species_result_df[
        np.isclose(
            species_result_df["sample_fraction"],
            DEFAULT_SPECIES_FRACTION,
        )
        & np.isclose(
            species_result_df["grid_resolution"],
            best_grid,
        )
        & (species_result_df["severity"] > 0)
        & (
            species_result_df[
                "absolute_observer_center_shift_km_decade"
            ]
            >= MIN_REALIZED_DRIFT_FOR_EVALUATION
        )
    ].copy()

    chosen_species_df = chosen_species_df.merge(
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

    sensitivity_rows = []

    for (
        species_code,
        scientific_name,
        common_name,
    ), group in tqdm(
        chosen_species_df.groupby(
            [
                "species_code",
                "scientific_name",
                "common_name",
            ],
            dropna=False,
        ),
        desc="Building calibrated species sensitivity index",
        unit="species",
    ):
        raw_slope = ordinary_slope(
            group[
                "absolute_observer_center_shift_km_decade"
            ],
            group["raw_abs_error_km_decade"],
        )
        corrected_slope = ordinary_slope(
            group[
                "absolute_observer_center_shift_km_decade"
            ],
            group[
                "corrected_abs_error_km_decade"
            ],
        )

        correction_gain = (
            1 - corrected_slope / raw_slope
            if np.isfinite(raw_slope)
            and raw_slope > 0
            and np.isfinite(corrected_slope)
            else np.nan
        )

        sensitivity_rows.append(
            {
                "species_code": int(species_code),
                "scientific_name": scientific_name,
                "common_name": common_name,
                "n_states": int(
                    group["state"].nunique()
                ),
                "n_species_replicates": len(group),
                "raw_error_per_observer_drift_km": (
                    raw_slope
                ),
                "corrected_error_per_observer_drift_km": (
                    corrected_slope
                ),
                "calibrated_correction_gain_fraction": (
                    correction_gain
                ),
                "median_raw_abs_error_km_decade": float(
                    group[
                        "raw_abs_error_km_decade"
                    ].median()
                ),
                "median_corrected_abs_error_km_decade": float(
                    group[
                        "corrected_abs_error_km_decade"
                    ].median()
                ),
                "fraction_corrected_better": float(
                    group[
                        "correction_improved"
                    ].mean()
                ),
                "error_response_spearman": safe_spearman(
                    group[
                        "absolute_observer_center_shift_km_decade"
                    ],
                    group[
                        "raw_abs_error_km_decade"
                    ],
                ),
            }
        )

    sensitivity_df = pd.DataFrame(
        sensitivity_rows
    ).sort_values(
        "raw_error_per_observer_drift_km",
        ascending=False,
    ).reset_index(drop=True)

    sensitivity_df.to_parquet(
        OUT_DIR / "calibrated_species_sensitivity.parquet",
        index=False,
    )
    sensitivity_df.to_csv(
        OUT_DIR / "calibrated_species_sensitivity.csv",
        index=False,
    )

    print("\nCALIBRATED SPECIES SENSITIVITY")
    display(sensitivity_df)

    # --------------------------------------------------------
    # 7. Final decision
    # --------------------------------------------------------
    state_pass_rows = []

    for state in STATES:
        state_rows = robustness_df[
            robustness_df["state"] == state
        ]

        n_fraction_passes = int(
            state_rows["fraction_pass"].sum()
        )

        state_pass_rows.append(
            {
                "state": state,
                "n_fraction_passes": n_fraction_passes,
                "n_tested_fractions": len(state_rows),
                "state_pass": bool(
                    n_fraction_passes
                    >= MIN_FRACTION_PASSES_PER_STATE
                ),
            }
        )

    state_pass_df = pd.DataFrame(
        state_pass_rows
    )

    n_state_passes = int(
        state_pass_df["state_pass"].sum()
    )

    passed = bool(
        n_state_passes == len(STATES)
        and not grid_selection_warning
    )

    if passed:
        status = "GO_FOR_REAL_GBIF_PILOT"
        next_step = (
            "The causal benchmark is robust across states, sample "
            "fractions, and empirically calibrated nulls. Next obtain "
            "a small GBIF/iNaturalist sample and measure the real annual "
            "observation-system center drift for these species and regions."
        )
    elif n_state_passes == len(STATES):
        status = "GO_WITH_GRID_CAUTION"
        next_step = (
            "The drift-response replicated, but every grid increased "
            "null error beyond the preferred bound. Refine the spatial "
            "standardization before using real GBIF records."
        )
    elif n_state_passes == 1:
        status = "CALIBRATION_PARTIAL_REPLICATION"
        next_step = (
            "The calibrated response remains state-dependent. Inspect "
            "which sampling fraction or grid failed before starting GBIF."
        )
    else:
        status = "CALIBRATION_NOT_ROBUST"
        next_step = (
            "The benchmark is not robust enough for real-data transfer. "
            "Do not begin GBIF extraction yet."
        )

    state_pass_df.to_csv(
        OUT_DIR / "state_pass_summary.csv",
        index=False,
    )

    print("\nSTATE PASS SUMMARY")
    display(state_pass_df)

    # --------------------------------------------------------
    # 8. Visual previews
    # --------------------------------------------------------
    chosen_summary_df = (
        scenario_df[
            np.isclose(
                scenario_df["sample_fraction"],
                DEFAULT_SPECIES_FRACTION,
            )
            & np.isclose(
                scenario_df["grid_resolution"],
                best_grid,
            )
        ]
        .groupby(
            ["state", "severity"],
            as_index=False,
        )
        .agg(
            median_observer_shift=(
                "absolute_observer_center_shift_km_decade",
                "median",
            ),
            median_raw_error=(
                "raw_median_abs_error_km_decade",
                "median",
            ),
            median_corrected_error=(
                "corrected_median_abs_error_km_decade",
                "median",
            ),
        )
    )

    for state in STATES:
        plot_df = chosen_summary_df[
            chosen_summary_df["state"] == state
        ].sort_values("median_observer_shift")

        plt.figure(figsize=(9, 5))
        plt.plot(
            plot_df["median_observer_shift"],
            plot_df["median_raw_error"],
            marker="o",
            label="Raw",
        )
        plt.plot(
            plot_df["median_observer_shift"],
            plot_df["median_corrected_error"],
            marker="o",
            label=f"Post-stratified {best_grid}°",
        )
        plt.xlabel(
            "Realized observation-system center drift "
            "(km/decade)"
        )
        plt.ylabel(
            "Median absolute species migration error "
            "(km/decade)"
        )
        plt.title(
            f"{state}: calibrated observation-drift response"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    top_sensitivity_df = (
        sensitivity_df.head(15)
        .sort_values(
            "raw_error_per_observer_drift_km"
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
            "raw_error_per_observer_drift_km"
        ],
    )
    plt.xlabel(
        "Species error per 1 km/decade of "
        "observation-center drift"
    )
    plt.ylabel("Species")
    plt.title(
        "Calibrated observation-drift sensitivity"
    )
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 9. README
    # --------------------------------------------------------
    top_sensitive_species = (
        sensitivity_df[
            [
                "species_code",
                "scientific_name",
                "raw_error_per_observer_drift_km",
                "corrected_error_per_observer_drift_km",
                "calibrated_correction_gain_fraction",
                "error_response_spearman",
            ]
        ]
        .head(10)
        .round(4)
        .to_dict("records")
    )

    append_readme(
        f"""

## Observation-drift calibration — {RUN_UTC}

### Purpose
Replace the arbitrary severity-0 error threshold from Cell 07A with empirical
state- and sample-fraction-specific null distributions, and express drift in
interpretable km/decade of sampled-plot center movement.

### Configuration
- States: {STATES}
- Severities: {SEVERITIES}
- Sample fractions: {SAMPLE_FRACTIONS}
- Grid resolutions: {GRID_RESOLUTIONS}
- Replicates per state/fraction/severity: {N_REPLICATES}
- Minimum evaluated realized drift:
  {MIN_REALIZED_DRIFT_FOR_EVALUATION} km/decade
- Random seed: {RANDOM_SEED}
- GPU: not used

### Selected correction
- Best grid: {best_grid} degrees
- Grid-selection warning: {grid_selection_warning}
- Median excess-error reduction:
  {best_grid_row['median_excess_error_reduction_fraction']}
- Fraction of scenarios improved:
  {best_grid_row['fraction_corrected_better']}
- Median null inflation:
  {best_grid_row['median_null_inflation_ratio']}

### Decision
- State passes: {n_state_passes}/{len(STATES)}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
The relationship is calibrated against controlled sample-center movement in FIA
repeated plots. It does not estimate actual GBIF observer drift. The selected
grid is provisional and must be locked before a full real-data analysis.
"""
    )

    # --------------------------------------------------------
    # 10. Compact output
    # --------------------------------------------------------
    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATES: {STATES}")
    print(f"N_SELECTED_SPECIES: {len(selected_species_codes)}")
    print(f"BEST_GRID_RESOLUTION: {best_grid}")
    print(
        f"GRID_SELECTION_WARNING: "
        f"{grid_selection_warning}"
    )
    print(
        "GRID_PERFORMANCE: "
        + json.dumps(
            grid_performance_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "STATE_FRACTION_ROBUSTNESS: "
        + json.dumps(
            robustness_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "STATE_PASS_SUMMARY: "
        + json.dumps(
            state_pass_df.to_dict("records"),
            ensure_ascii=False,
        )
    )
    print(
        "TOP_CALIBRATED_SENSITIVE_SPECIES: "
        + json.dumps(
            top_sensitive_species,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_CRITERIA: in each state at least 2/3 sample "
        "fractions pass response Spearman >=0.80, empirical "
        "detection rate >=0.80, excess-error reduction >=0.50, "
        "and fraction corrected better >=0.70"
    )
    print(f"N_STATE_PASSES: {n_state_passes}")
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except Exception as exc:
    print_failure(
        "CELL07B_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        "Send this COPY block and the last displayed output. "
        "Do not rerun previous downloads.",
    )
