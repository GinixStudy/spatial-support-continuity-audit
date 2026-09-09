
# Cell 13A — Species-specific transportability and correction-failure audit
#
# Scientific pivot
# ----------------
# The locked 0.5-degree correction worked in the semi-synthetic benchmark but did
# not validate as a universal correction against real repeated-plot FIA data.
#
# This cell asks:
# Which pre-existing species/support characteristics predict whether correction
# improves or harms external agreement?
#
# This is an exploratory trend audit with a locked set of candidate features.
# It does not fit a flexible machine-learning model because only 15 species are
# currently available.
#
# Main outcome:
#   correction gain = raw absolute error - corrected absolute error
# Positive values mean improvement.
#
# Candidate mechanisms:
# - spatial-cell continuity between periods;
# - observer-label continuity;
# - record and observer growth imbalance;
# - species-share change within the expanded tree target group;
# - stable-cell support coverage;
# - latitudinal-support width change;
# - target-effort exposure change;
# - observer concentration change;
# - semi-synthetic species sensitivity.
#
# Success:
# At least one feature must show a large, permutation-supported and
# leave-one-species-out-stable association with correction gain, with the same
# direction for the median gain across 0.25°, 0.5° and 1.0° grids.

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 240)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 340)

# ============================================================
# Configuration
# ============================================================
EARLY_YEARS = list(range(2013, 2019))
LATE_YEARS = list(range(2020, 2026))

GRID_RESOLUTION = 0.50
STABLE_CELL_MIN_RECORDS = 10

GRID_RESOLUTIONS = [0.25, 0.50, 1.00]

N_PERMUTATIONS = 1500
N_BOOTSTRAP = 1000
RANDOM_SEED = 20260710

MIN_SPECIES = 12
MIN_ABS_RHO = 0.50
MAX_PERMUTATION_P = 0.10
MIN_BOOTSTRAP_SAME_SIGN = 0.90
MIN_LOO_SIGN_STABILITY = 0.80
MIN_SECONDARY_ABS_RHO = 0.35

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
VALIDATION_DIR = (
    BASE_DIR / "derived" / "pa_species_external_validation"
)
EXPANDED_DIR = (
    BASE_DIR
    / "derived"
    / "pa_expanded_tree_observer_decomposition"
)
CALIBRATION_DIR = (
    BASE_DIR / "derived" / "observation_drift_calibration"
)
OUT_DIR = (
    BASE_DIR
    / "derived"
    / "pa_correction_transportability_audit"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_PATH = (
    VALIDATION_DIR / "species_external_validation.parquet"
)
EXPANDED_INAT_PATH = (
    EXPANDED_DIR / "expanded_pa_tree_thinned_occurrences.parquet"
)
FIA_REFERENCE_PATH = (
    VALIDATION_DIR / "fia_same_window_species_reference.parquet"
)
SENSITIVITY_PATH = (
    CALIBRATION_DIR / "calibrated_species_sensitivity.parquet"
)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("OUT_DIR:", OUT_DIR)
print("N_PERMUTATIONS:", N_PERMUTATIONS)
print("N_BOOTSTRAP:", N_BOOTSTRAP)
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


def set_jaccard(first, second):
    union = first | second

    if not union:
        return np.nan

    return float(
        len(first & second) / len(union)
    )


def overlap_min_fraction(first, second):
    denominator = min(
        len(first),
        len(second),
    )

    if denominator == 0:
        return np.nan

    return float(
        len(first & second) / denominator
    )


def safe_spearman(x, y):
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if (
        len(frame) < 5
        or frame["x"].nunique() < 2
        or frame["y"].nunique() < 2
    ):
        return np.nan

    return float(
        spearmanr(
            frame["x"],
            frame["y"],
        ).statistic
    )


def benjamini_hochberg(p_values):
    values = np.asarray(
        p_values,
        dtype=float,
    )

    result = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        np.isfinite(values)
    )[0]

    if len(valid_indices) == 0:
        return result

    valid_values = values[
        valid_indices
    ]

    order = np.argsort(
        valid_values
    )
    ranked = valid_values[
        order
    ]

    adjusted = (
        ranked
        * len(ranked)
        / np.arange(
            1,
            len(ranked) + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1,
    )

    result[
        valid_indices[order]
    ] = adjusted

    return result


def correlation_inference(
    frame,
    feature,
    outcome,
    rng,
    progress,
):
    work = frame[
        [feature, outcome]
    ].dropna()

    observed_rho = safe_spearman(
        work[feature],
        work[outcome],
    )

    if not np.isfinite(observed_rho):
        progress.update(
            N_PERMUTATIONS
            + N_BOOTSTRAP
            + len(work)
        )

        return {
            "feature": feature,
            "outcome": outcome,
            "n_species": len(work),
            "spearman_rho": np.nan,
            "permutation_p": np.nan,
            "bootstrap_low": np.nan,
            "bootstrap_high": np.nan,
            "bootstrap_same_sign_fraction": np.nan,
            "loo_sign_stability": np.nan,
        }

    feature_values = work[
        feature
    ].to_numpy(dtype=float)
    outcome_values = work[
        outcome
    ].to_numpy(dtype=float)

    permutation_exceed = 0

    for _ in range(
        N_PERMUTATIONS
    ):
        permuted = rng.permutation(
            outcome_values
        )

        permuted_rho = safe_spearman(
            feature_values,
            permuted,
        )

        if (
            np.isfinite(permuted_rho)
            and abs(permuted_rho)
            >= abs(observed_rho)
        ):
            permutation_exceed += 1

        progress.update(1)

    permutation_p = float(
        (
            1 + permutation_exceed
        )
        / (
            N_PERMUTATIONS + 1
        )
    )

    bootstrap_values = []

    for _ in range(
        N_BOOTSTRAP
    ):
        indices = rng.integers(
            0,
            len(work),
            size=len(work),
        )

        rho = safe_spearman(
            feature_values[indices],
            outcome_values[indices],
        )

        if np.isfinite(rho):
            bootstrap_values.append(
                rho
            )

        progress.update(1)

    bootstrap_values = np.asarray(
        bootstrap_values,
        dtype=float,
    )

    if len(bootstrap_values):
        bootstrap_low = float(
            np.quantile(
                bootstrap_values,
                0.025,
            )
        )
        bootstrap_high = float(
            np.quantile(
                bootstrap_values,
                0.975,
            )
        )
        same_sign_fraction = float(
            np.mean(
                np.sign(
                    bootstrap_values
                )
                == np.sign(
                    observed_rho
                )
            )
        )
    else:
        bootstrap_low = np.nan
        bootstrap_high = np.nan
        same_sign_fraction = np.nan

    loo_signs = []

    for dropped_index in range(
        len(work)
    ):
        keep = np.ones(
            len(work),
            dtype=bool,
        )
        keep[
            dropped_index
        ] = False

        rho = safe_spearman(
            feature_values[keep],
            outcome_values[keep],
        )

        if np.isfinite(rho):
            loo_signs.append(
                np.sign(rho)
                == np.sign(
                    observed_rho
                )
            )

        progress.update(1)

    loo_stability = (
        float(
            np.mean(
                loo_signs
            )
        )
        if loo_signs
        else np.nan
    )

    return {
        "feature": feature,
        "outcome": outcome,
        "n_species": len(work),
        "spearman_rho": observed_rho,
        "permutation_p": permutation_p,
        "bootstrap_low": bootstrap_low,
        "bootstrap_high": bootstrap_high,
        "bootstrap_same_sign_fraction": (
            same_sign_fraction
        ),
        "loo_sign_stability": (
            loo_stability
        ),
    }


def resolution_file(
    resolution,
):
    tag = str(
        resolution
    ).replace(
        ".",
        "p",
    )

    return (
        VALIDATION_DIR
        / f"inat_species_estimands_grid_{tag}.parquet"
    )


def print_failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## PA transportability audit failure — {RUN_UTC}
- Status: **{status}**
- Error: `{error}`
- Next step: {next_step}
"""
    )

    print(
        "\n========== RUN SUMMARY =========="
    )
    print(
        f"CELL_STATUS: {status}"
    )
    print(
        f"ERROR: {error}"
    )
    print(
        f"OUT_DIR: {OUT_DIR}"
    )
    print(
        "PASSED: False"
    )
    print(
        f"NEXT_STEP: {next_step}"
    )
    print(
        "README_UPDATED: True"
    )
    print(
        "============================================"
    )


# ============================================================
# Main
# ============================================================
try:
    # --------------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------------
    required_paths = [
        VALIDATION_PATH,
        EXPANDED_INAT_PATH,
        FIA_REFERENCE_PATH,
        SENSITIVITY_PATH,
    ] + [
        resolution_file(
            resolution
        )
        for resolution in GRID_RESOLUTIONS
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing prior outputs: "
            + " | ".join(
                missing_paths
            )
        )

    evaluation_df = pd.read_parquet(
        VALIDATION_PATH
    )
    occurrence_df = pd.read_parquet(
        EXPANDED_INAT_PATH
    )
    fia_reference_df = pd.read_parquet(
        FIA_REFERENCE_PATH
    )
    sensitivity_df = pd.read_parquet(
        SENSITIVITY_PATH
    )

    evaluation_df[
        "fia_species_code"
    ] = pd.to_numeric(
        evaluation_df[
            "fia_species_code"
        ],
        errors="coerce",
    )

    evaluation_df = evaluation_df.dropna(
        subset=[
            "fia_species_code",
            "fia_shift_km_decade",
            "raw_shift_km_decade",
            "effort_standardized_shift_km_decade",
            "observer_balanced_shift_km_decade",
        ]
    ).copy()

    evaluation_df[
        "fia_species_code"
    ] = evaluation_df[
        "fia_species_code"
    ].astype(int)

    evaluation_codes = set(
        evaluation_df[
            "fia_species_code"
        ].tolist()
    )

    if (
        len(evaluation_codes)
        < MIN_SPECIES
    ):
        raise ValueError(
            f"Only {len(evaluation_codes)} "
            "evaluation species are available."
        )

    # Explicitly recompute outcomes.
    evaluation_df[
        "raw_abs_error"
    ] = (
        evaluation_df[
            "raw_shift_km_decade"
        ]
        - evaluation_df[
            "fia_shift_km_decade"
        ]
    ).abs()

    evaluation_df[
        "effort_abs_error"
    ] = (
        evaluation_df[
            "effort_standardized_shift_km_decade"
        ]
        - evaluation_df[
            "fia_shift_km_decade"
        ]
    ).abs()

    evaluation_df[
        "observer_abs_error"
    ] = (
        evaluation_df[
            "observer_balanced_shift_km_decade"
        ]
        - evaluation_df[
            "fia_shift_km_decade"
        ]
    ).abs()

    evaluation_df[
        "effort_gain_km_decade"
    ] = (
        evaluation_df[
            "raw_abs_error"
        ]
        - evaluation_df[
            "effort_abs_error"
        ]
    )

    evaluation_df[
        "observer_gain_km_decade"
    ] = (
        evaluation_df[
            "raw_abs_error"
        ]
        - evaluation_df[
            "observer_abs_error"
        ]
    )

    evaluation_df[
        "relative_effort_gain"
    ] = (
        evaluation_df[
            "effort_gain_km_decade"
        ]
        / evaluation_df[
            "raw_abs_error"
        ].replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # 2. Build grid-robust correction-gain outcomes
    # --------------------------------------------------------
    grid_gain_tables = []

    for resolution in tqdm(
        GRID_RESOLUTIONS,
        desc="Building grid-specific correction gains",
        unit="resolution",
    ):
        estimand_df = pd.read_parquet(
            resolution_file(
                resolution
            )
        )

        merged = estimand_df.merge(
            fia_reference_df,
            left_on="fia_species_code",
            right_on="species_code",
            how="inner",
        )

        merged = merged[
            merged[
                "fia_species_code"
            ].isin(
                evaluation_codes
            )
        ].copy()

        merged[
            "raw_grid_error"
        ] = (
            merged[
                "raw_shift_km_decade"
            ]
            - merged[
                "fia_shift_km_decade"
            ]
        ).abs()

        merged[
            "corrected_grid_error"
        ] = (
            merged[
                "effort_standardized_shift_km_decade"
            ]
            - merged[
                "fia_shift_km_decade"
            ]
        ).abs()

        tag = str(
            resolution
        ).replace(
            ".",
            "p",
        )

        grid_gain_tables.append(
            merged[
                [
                    "fia_species_code",
                ]
            ].assign(
                **{
                    f"gain_grid_{tag}": (
                        merged[
                            "raw_grid_error"
                        ]
                        - merged[
                            "corrected_grid_error"
                        ]
                    )
                }
            )
        )

    grid_gain_df = pd.DataFrame(
        {
            "fia_species_code": sorted(
                evaluation_codes
            )
        }
    )

    for table in grid_gain_tables:
        grid_gain_df = grid_gain_df.merge(
            table,
            on="fia_species_code",
            how="left",
        )

    gain_columns = [
        column
        for column in grid_gain_df.columns
        if column.startswith(
            "gain_grid_"
        )
    ]

    grid_gain_df[
        "grid_median_gain_km_decade"
    ] = grid_gain_df[
        gain_columns
    ].median(
        axis=1,
        skipna=True,
    )

    grid_gain_df[
        "grid_positive_gain_fraction"
    ] = (
        grid_gain_df[
            gain_columns
        ]
        .gt(0)
        .sum(axis=1)
        / grid_gain_df[
            gain_columns
        ]
        .notna()
        .sum(axis=1)
        .replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # 3. Prepare iNaturalist species support features
    # --------------------------------------------------------
    for column in [
        "year",
        "fia_species_code",
        "decimalLatitude",
        "decimalLongitude",
    ]:
        occurrence_df[
            column
        ] = pd.to_numeric(
            occurrence_df[
                column
            ],
            errors="coerce",
        )

    occurrence_df = occurrence_df[
        occurrence_df[
            "year"
        ].isin(
            EARLY_YEARS
            + LATE_YEARS
        )
        & occurrence_df[
            "fia_species_code"
        ].notna()
        & occurrence_df[
            "decimalLatitude"
        ].notna()
        & occurrence_df[
            "decimalLongitude"
        ].notna()
    ].copy()

    occurrence_df[
        "fia_species_code"
    ] = occurrence_df[
        "fia_species_code"
    ].astype(int)

    occurrence_df[
        "period"
    ] = np.where(
        occurrence_df[
            "year"
        ].isin(
            EARLY_YEARS
        ),
        "early",
        "late",
    )

    occurrence_df[
        "observer_label"
    ] = normalize_observer_label(
        occurrence_df[
            "recordedBy"
        ]
    )

    occurrence_df[
        "grid_lat_index"
    ] = np.floor(
        occurrence_df[
            "decimalLatitude"
        ]
        / GRID_RESOLUTION
    ).astype(
        "int32"
    )

    occurrence_df[
        "grid_lon_index"
    ] = np.floor(
        occurrence_df[
            "decimalLongitude"
        ]
        / GRID_RESOLUTION
    ).astype(
        "int32"
    )

    occurrence_df[
        "grid_id"
    ] = (
        occurrence_df[
            "grid_lat_index"
        ].astype(str)
        + "_"
        + occurrence_df[
            "grid_lon_index"
        ].astype(str)
    )

    target_effort_df = (
        occurrence_df.groupby(
            [
                "period",
                "grid_id",
            ],
            as_index=False,
        )
        .agg(
            target_effort_records=(
                "gbifID",
                "size",
            ),
        )
    )

    effort_wide = (
        target_effort_df.pivot(
            index="grid_id",
            columns="period",
            values="target_effort_records",
        )
        .fillna(0)
    )

    for period in [
        "early",
        "late",
    ]:
        if (
            period
            not in effort_wide.columns
        ):
            effort_wide[
                period
            ] = 0

    stable_cells = set(
        effort_wide[
            (
                effort_wide[
                    "early"
                ]
                >= STABLE_CELL_MIN_RECORDS
            )
            & (
                effort_wide[
                    "late"
                ]
                >= STABLE_CELL_MIN_RECORDS
            )
        ].index
    )

    occurrence_df = occurrence_df.merge(
        target_effort_df,
        on=[
            "period",
            "grid_id",
        ],
        how="left",
        validate="many_to_one",
    )

    total_target_records = (
        occurrence_df.groupby(
            "period"
        )[
            "gbifID"
        ]
        .size()
        .to_dict()
    )

    feature_rows = []

    for species_code in tqdm(
        sorted(
            evaluation_codes
        ),
        desc="Building species support features",
        unit="species",
    ):
        species_frame = occurrence_df[
            occurrence_df[
                "fia_species_code"
            ]
            == species_code
        ].copy()

        period_features = {}

        for period in [
            "early",
            "late",
        ]:
            period_frame = species_frame[
                species_frame[
                    "period"
                ]
                == period
            ].copy()

            observer_counts = (
                period_frame.dropna(
                    subset=[
                        "observer_label"
                    ]
                )
                .groupby(
                    "observer_label"
                )[
                    "gbifID"
                ]
                .size()
                .sort_values(
                    ascending=False
                )
            )

            n_records = len(
                period_frame
            )

            period_features[
                period
            ] = {
                "n_records": n_records,
                "n_observers": int(
                    period_frame[
                        "observer_label"
                    ].nunique(
                        dropna=True
                    )
                ),
                "n_cells": int(
                    period_frame[
                        "grid_id"
                    ].nunique()
                ),
                "observer_set": set(
                    period_frame[
                        "observer_label"
                    ].dropna()
                ),
                "cell_set": set(
                    period_frame[
                        "grid_id"
                    ].dropna()
                ),
                "latitude_iqr": float(
                    period_frame[
                        "decimalLatitude"
                    ].quantile(
                        0.75
                    )
                    - period_frame[
                        "decimalLatitude"
                    ].quantile(
                        0.25
                    )
                )
                if n_records
                else np.nan,
                "stable_cell_fraction": float(
                    period_frame[
                        "grid_id"
                    ].isin(
                        stable_cells
                    ).mean()
                )
                if n_records
                else np.nan,
                "species_share": float(
                    n_records
                    / total_target_records[
                        period
                    ]
                )
                if total_target_records.get(
                    period,
                    0,
                )
                else np.nan,
                "effort_exposure": float(
                    np.log1p(
                        period_frame[
                            "target_effort_records"
                        ]
                    ).mean()
                )
                if n_records
                else np.nan,
                "top1_observer_share": float(
                    observer_counts.head(
                        1
                    ).sum()
                    / observer_counts.sum()
                )
                if observer_counts.sum()
                else np.nan,
                "top10_observer_share": float(
                    observer_counts.head(
                        10
                    ).sum()
                    / observer_counts.sum()
                )
                if observer_counts.sum()
                else np.nan,
            }

        early = period_features[
            "early"
        ]
        late = period_features[
            "late"
        ]

        feature_rows.append(
            {
                "fia_species_code": (
                    species_code
                ),
                "observer_jaccard": (
                    set_jaccard(
                        early[
                            "observer_set"
                        ],
                        late[
                            "observer_set"
                        ],
                    )
                ),
                "observer_overlap_min_fraction": (
                    overlap_min_fraction(
                        early[
                            "observer_set"
                        ],
                        late[
                            "observer_set"
                        ],
                    )
                ),
                "cell_jaccard": (
                    set_jaccard(
                        early[
                            "cell_set"
                        ],
                        late[
                            "cell_set"
                        ],
                    )
                ),
                "cell_overlap_min_fraction": (
                    overlap_min_fraction(
                        early[
                            "cell_set"
                        ],
                        late[
                            "cell_set"
                        ],
                    )
                ),
                "abs_log_record_growth": abs(
                    np.log(
                        (
                            late[
                                "n_records"
                            ]
                            + 1
                        )
                        / (
                            early[
                                "n_records"
                            ]
                            + 1
                        )
                    )
                ),
                "abs_log_observer_growth": abs(
                    np.log(
                        (
                            late[
                                "n_observers"
                            ]
                            + 1
                        )
                        / (
                            early[
                                "n_observers"
                            ]
                            + 1
                        )
                    )
                ),
                "abs_species_share_change": abs(
                    late[
                        "species_share"
                    ]
                    - early[
                        "species_share"
                    ]
                ),
                "stable_frame_min_fraction": (
                    min(
                        early[
                            "stable_cell_fraction"
                        ],
                        late[
                            "stable_cell_fraction"
                        ],
                    )
                ),
                "abs_latitude_iqr_change": abs(
                    late[
                        "latitude_iqr"
                    ]
                    - early[
                        "latitude_iqr"
                    ]
                ),
                "abs_effort_exposure_change": abs(
                    late[
                        "effort_exposure"
                    ]
                    - early[
                        "effort_exposure"
                    ]
                ),
                "abs_top10_concentration_change": abs(
                    late[
                        "top10_observer_share"
                    ]
                    - early[
                        "top10_observer_share"
                    ]
                ),
                "min_period_records": min(
                    early[
                        "n_records"
                    ],
                    late[
                        "n_records"
                    ],
                ),
                "min_period_observers": min(
                    early[
                        "n_observers"
                    ],
                    late[
                        "n_observers"
                    ],
                ),
                "min_period_cells": min(
                    early[
                        "n_cells"
                    ],
                    late[
                        "n_cells"
                    ],
                ),
            }
        )

    feature_df = pd.DataFrame(
        feature_rows
    )

    sensitivity_subset = (
        sensitivity_df[
            [
                "species_code",
                "raw_error_per_observer_drift_km",
                "calibrated_correction_gain_fraction",
                "error_response_spearman",
            ]
        ]
        .rename(
            columns={
                "species_code": (
                    "fia_species_code"
                ),
                "raw_error_per_observer_drift_km": (
                    "synthetic_sensitivity"
                ),
                "calibrated_correction_gain_fraction": (
                    "synthetic_correction_gain"
                ),
                "error_response_spearman": (
                    "synthetic_response_spearman"
                ),
            }
        )
    )

    analysis_df = (
        evaluation_df.merge(
            grid_gain_df,
            on="fia_species_code",
            how="left",
        )
        .merge(
            feature_df,
            on="fia_species_code",
            how="left",
        )
        .merge(
            sensitivity_subset,
            on="fia_species_code",
            how="left",
        )
    )

    analysis_df.to_parquet(
        OUT_DIR
        / "species_transportability_features.parquet",
        index=False,
    )
    analysis_df.to_csv(
        OUT_DIR
        / "species_transportability_features.csv",
        index=False,
    )

    print(
        "\nSPECIES TRANSPORTABILITY FEATURES"
    )
    display(
        analysis_df[
            [
                "fia_species_code",
                "scientific_name",
                "effort_gain_km_decade",
                "observer_gain_km_decade",
                "grid_median_gain_km_decade",
                "grid_positive_gain_fraction",
                "observer_jaccard",
                "cell_jaccard",
                "stable_frame_min_fraction",
                "abs_log_record_growth",
                "abs_log_observer_growth",
                "abs_species_share_change",
                "abs_latitude_iqr_change",
                "synthetic_sensitivity",
            ]
        ]
    )

    # --------------------------------------------------------
    # 4. Locked candidate-feature inference
    # --------------------------------------------------------
    candidate_features = [
        "observer_jaccard",
        "observer_overlap_min_fraction",
        "cell_jaccard",
        "cell_overlap_min_fraction",
        "abs_log_record_growth",
        "abs_log_observer_growth",
        "abs_species_share_change",
        "stable_frame_min_fraction",
        "abs_latitude_iqr_change",
        "abs_effort_exposure_change",
        "abs_top10_concentration_change",
        "synthetic_sensitivity",
    ]

    outcomes = [
        "effort_gain_km_decade",
        "observer_gain_km_decade",
        "grid_median_gain_km_decade",
    ]

    total_iterations = (
        len(
            candidate_features
        )
        * len(
            outcomes
        )
        * (
            N_PERMUTATIONS
            + N_BOOTSTRAP
            + len(
                analysis_df
            )
        )
    )

    progress = tqdm(
        total=total_iterations,
        desc="Testing correction-failure mechanisms",
        unit="resample",
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    correlation_rows = []

    for feature in candidate_features:
        for outcome in outcomes:
            correlation_rows.append(
                correlation_inference(
                    analysis_df,
                    feature,
                    outcome,
                    rng,
                    progress,
                )
            )

    progress.close()

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    main_mask = (
        correlation_df[
            "outcome"
        ]
        == "effort_gain_km_decade"
    )

    correlation_df.loc[
        main_mask,
        "main_outcome_q_value",
    ] = benjamini_hochberg(
        correlation_df.loc[
            main_mask,
            "permutation_p",
        ]
    )

    correlation_df.to_csv(
        OUT_DIR
        / "feature_gain_correlation_inference.csv",
        index=False,
    )

    print(
        "\nFEATURE–GAIN CORRELATION INFERENCE"
    )
    display(
        correlation_df.sort_values(
            [
                "outcome",
                "spearman_rho",
            ],
            key=lambda series: (
                series.abs()
                if series.name
                == "spearman_rho"
                else series
            ),
            ascending=[
                True,
                False,
            ],
        )
    )

    # --------------------------------------------------------
    # 5. Robust candidate decision
    # --------------------------------------------------------
    main_results = correlation_df[
        correlation_df[
            "outcome"
        ]
        == "effort_gain_km_decade"
    ].copy()

    grid_results = correlation_df[
        correlation_df[
            "outcome"
        ]
        == "grid_median_gain_km_decade"
    ][
        [
            "feature",
            "spearman_rho",
            "permutation_p",
        ]
    ].rename(
        columns={
            "spearman_rho": (
                "grid_median_rho"
            ),
            "permutation_p": (
                "grid_median_permutation_p"
            ),
        }
    )

    candidate_decision_df = (
        main_results.merge(
            grid_results,
            on="feature",
            how="left",
        )
    )

    candidate_decision_df[
        "same_direction_across_grid_endpoint"
    ] = (
        np.sign(
            candidate_decision_df[
                "spearman_rho"
            ]
        )
        == np.sign(
            candidate_decision_df[
                "grid_median_rho"
            ]
        )
    )

    candidate_decision_df[
        "robust_candidate"
    ] = (
        candidate_decision_df[
            "spearman_rho"
        ].abs()
        >= MIN_ABS_RHO
    ) & (
        candidate_decision_df[
            "permutation_p"
        ]
        <= MAX_PERMUTATION_P
    ) & (
        candidate_decision_df[
            "bootstrap_same_sign_fraction"
        ]
        >= MIN_BOOTSTRAP_SAME_SIGN
    ) & (
        candidate_decision_df[
            "loo_sign_stability"
        ]
        >= MIN_LOO_SIGN_STABILITY
    ) & (
        candidate_decision_df[
            "same_direction_across_grid_endpoint"
        ]
    ) & (
        candidate_decision_df[
            "grid_median_rho"
        ].abs()
        >= MIN_SECONDARY_ABS_RHO
    )

    candidate_decision_df = (
        candidate_decision_df.sort_values(
            "spearman_rho",
            key=lambda series: series.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )

    candidate_decision_df.to_csv(
        OUT_DIR
        / "transportability_candidate_decision.csv",
        index=False,
    )

    print(
        "\nTRANSPORTABILITY CANDIDATE DECISION"
    )
    display(
        candidate_decision_df
    )

    robust_candidates = (
        candidate_decision_df[
            candidate_decision_df[
                "robust_candidate"
            ]
        ].copy()
    )

    n_robust_candidates = len(
        robust_candidates
    )

    effort_improved_fraction = float(
        (
            analysis_df[
                "effort_gain_km_decade"
            ]
            > 0
        ).mean()
    )

    observer_improved_fraction = float(
        (
            analysis_df[
                "observer_gain_km_decade"
            ]
            > 0
        ).mean()
    )

    grid_consistent_improvement_fraction = float(
        (
            analysis_df[
                "grid_positive_gain_fraction"
            ]
            >= 2 / 3
        ).mean()
    )

    if (
        n_robust_candidates
        >= 1
    ):
        status = (
            "GO_FOR_SENSITIVITY_INDEX_CONFIRMATION"
        )
        passed = True
        next_step = (
            "At least one predeclared support feature predicts "
            "whether correction helps or harms and is stable across "
            "grid endpoints. Confirm the candidate in a second region "
            "before constructing a final sensitivity index."
        )
    elif (
        observer_improved_fraction
        >= 0.60
    ):
        status = (
            "RUN_OBSERVER_BALANCED_WINDOW_CONFIRMATION"
        )
        passed = False
        next_step = (
            "No stable failure-condition predictor was identified, "
            "but observer balancing improves most species. Run "
            "prespecified time-window sensitivity before considering "
            "observer balancing as the main correction."
        )
    else:
        status = (
            "PA_FAILURE_CONDITIONS_NOT_IDENTIFIED"
        )
        passed = False
        next_step = (
            "Correction benefit remains heterogeneous and cannot be "
            "predicted reliably from the current 15-species PA sample. "
            "Do not build a sensitivity index yet; add a second high-"
            "coverage region using the same locked features."
        )

    # --------------------------------------------------------
    # 6. Visual previews
    # --------------------------------------------------------
    if (
        len(
            candidate_decision_df
        )
    ):
        top_feature = (
            candidate_decision_df.iloc[
                0
            ][
                "feature"
            ]
        )

        plot_df = analysis_df[
            [
                "scientific_name",
                top_feature,
                "effort_gain_km_decade",
            ]
        ].dropna()

        plt.figure(
            figsize=(8, 6)
        )
        plt.scatter(
            plot_df[
                top_feature
            ],
            plot_df[
                "effort_gain_km_decade"
            ],
        )
        plt.axhline(
            0,
            linestyle="--",
        )
        plt.xlabel(
            top_feature
        )
        plt.ylabel(
            "0.5° correction gain versus FIA "
            "(km/decade)"
        )
        plt.title(
            "Strongest exploratory correction-failure feature"
        )
        plt.tight_layout()
        plt.show()

    gain_plot_df = (
        analysis_df[
            [
                "scientific_name",
                "effort_gain_km_decade",
                "observer_gain_km_decade",
            ]
        ]
        .sort_values(
            "effort_gain_km_decade"
        )
        .reset_index(
            drop=True
        )
    )

    y = np.arange(
        len(
            gain_plot_df
        )
    )

    plt.figure(
        figsize=(
            11,
            max(
                6,
                0.55
                * len(
                    gain_plot_df
                ),
            ),
        )
    )
    plt.scatter(
        gain_plot_df[
            "effort_gain_km_decade"
        ],
        y,
        label="Target-effort correction",
    )
    plt.scatter(
        gain_plot_df[
            "observer_gain_km_decade"
        ],
        y,
        label="Observer-balanced",
    )
    plt.axvline(
        0,
        linestyle="--",
    )
    plt.yticks(
        y,
        gain_plot_df[
            "scientific_name"
        ],
    )
    plt.xlabel(
        "Error reduction versus raw estimate "
        "(km/decade; positive is better)"
    )
    plt.ylabel(
        "Species"
    )
    plt.title(
        "Species-specific correction benefit and harm"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 7. README
    # --------------------------------------------------------
    robust_compact = (
        robust_candidates[
            [
                "feature",
                "spearman_rho",
                "permutation_p",
                "main_outcome_q_value",
                "bootstrap_low",
                "bootstrap_high",
                "bootstrap_same_sign_fraction",
                "loo_sign_stability",
                "grid_median_rho",
            ]
        ]
        .round(6)
        .to_dict(
            "records"
        )
    )

    top_correlations = (
        candidate_decision_df[
            [
                "feature",
                "spearman_rho",
                "permutation_p",
                "main_outcome_q_value",
                "bootstrap_same_sign_fraction",
                "loo_sign_stability",
                "grid_median_rho",
                "robust_candidate",
            ]
        ]
        .head(8)
        .round(6)
        .to_dict(
            "records"
        )
    )

    append_readme(
        f"""

## PA correction transportability audit — {RUN_UTC}

### Reason for pivot
The universal 0.5-degree correction did not validate against same-window FIA:
median error fell by only 15.5%, bootstrap support was insufficient, and
direction/rank agreement worsened. Correction benefit was strongly
species-specific.

### Locked analysis
- Evaluation species: {len(analysis_df)}
- Main outcome:
  raw absolute error minus 0.5-degree corrected absolute error
- Secondary outcome:
  median correction gain across 0.25, 0.5 and 1.0 degree grids
- Candidate features declared before testing:
  observer continuity, spatial-cell continuity, record/observer growth,
  target-group share change, stable-frame coverage, latitude-width change,
  effort-exposure change, observer concentration change and semi-synthetic
  sensitivity
- Permutations per feature-outcome: {N_PERMUTATIONS}
- Bootstrap replicates per feature-outcome: {N_BOOTSTRAP}
- Random seed: {RANDOM_SEED}
- GPU: not used

### Results
- Effort correction improved fraction:
  {effort_improved_fraction}
- Observer-balanced improved fraction:
  {observer_improved_fraction}
- Improved in at least two of three grids:
  {grid_consistent_improvement_fraction}
- Robust candidate features:
  {n_robust_candidates}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
This is an exploratory mechanism audit with only 15 species. Even a robust
candidate is not a final sensitivity index until it is confirmed in another
region. No flexible predictive model was fitted.
"""
    )

    # --------------------------------------------------------
    # 8. Compact output
    # --------------------------------------------------------
    print(
        "\n========== RUN SUMMARY =========="
    )
    print(
        f"CELL_STATUS: {status}"
    )
    print(
        f"OUT_DIR: {OUT_DIR}"
    )
    print(
        f"N_EVALUATION_SPECIES: "
        f"{len(analysis_df)}"
    )
    print(
        f"EFFORT_CORRECTION_IMPROVED_FRACTION: "
        f"{effort_improved_fraction}"
    )
    print(
        f"OBSERVER_BALANCED_IMPROVED_FRACTION: "
        f"{observer_improved_fraction}"
    )
    print(
        f"GRID_CONSISTENT_IMPROVEMENT_FRACTION: "
        f"{grid_consistent_improvement_fraction}"
    )
    print(
        f"N_ROBUST_CANDIDATE_FEATURES: "
        f"{n_robust_candidates}"
    )
    print(
        "ROBUST_CANDIDATES: "
        + json.dumps(
            robust_compact,
            ensure_ascii=False,
        )
    )
    print(
        "TOP_FEATURE_RESULTS: "
        + json.dumps(
            top_correlations,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_CRITERIA: |rho| >=0.50, permutation p <=0.10, "
        "bootstrap same-sign >=0.90, leave-one-out sign stability "
        ">=0.80, and same-direction grid-median rho >=0.35"
    )
    print(
        f"PASSED: {passed}"
    )
    print(
        f"NEXT_STEP: {next_step}"
    )
    print(
        "README_UPDATED: True"
    )
    print(
        "============================================"
    )

except Exception as exc:
    print_failure(
        "CELL13A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed output. "
            "Do not download new data."
        ),
    )
