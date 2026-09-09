
# Cell 16C — Final locked North Carolina confirmation
#
# Final confirmatory question
# ---------------------------
# Do either of the two predeclared Pennsylvania transportability candidates
# independently predict correction gain in North Carolina?
#
# Locked candidates and directions
# --------------------------------
# 1. cell_jaccard: positive association with correction gain
# 2. abs_log_observer_growth: positive association with correction gain
#
# No additional features, thresholds, windows, species or regions are searched.
#
# Primary outcome
# ---------------
# 0.5-degree correction gain:
#
#   raw absolute error versus FIA
#   minus
#   effort-standardized absolute error versus FIA
#
# Positive values mean that correction improves agreement with FIA.
#
# Secondary robustness outcome
# ----------------------------
# Median correction gain across 0.25, 0.5 and 1.0-degree grids.
#
# Locked confirmation criteria per candidate
# ------------------------------------------
# - at least 10 species;
# - primary Spearman rho >= 0.50;
# - one-sided permutation p, Holm-adjusted across two candidates, <= 0.05;
# - bootstrap positive fraction >= 0.90;
# - leave-one-species-out positive fraction >= 0.80;
# - grid-median rho >= 0.35;
# - positive rho in at least two of three grids.
#
# Final stop rule
# ---------------
# If neither candidate confirms in North Carolina, permanently close the
# sensitivity-index route. Do not add regions, windows, features or tune cutoffs.

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
pd.set_option("display.width", 360)

# ============================================================
# Locked configuration
# ============================================================
STATE = "NC"

EARLY_YEARS = list(range(2013, 2019))
LATE_YEARS = list(range(2020, 2026))
EARLY_MID_YEAR = float(np.mean(EARLY_YEARS))
LATE_MID_YEAR = float(np.mean(LATE_YEARS))

GRID_RESOLUTIONS = [0.25, 0.50, 1.00]
PRIMARY_GRID_RESOLUTION = 0.50
MIN_TARGET_EFFORT_PER_CELL_PERIOD = 10

LOCKED_FEATURES = [
    "cell_jaccard",
    "abs_log_observer_growth",
]

PA_DISCOVERY_RHO = {
    "cell_jaccard": 0.717857,
    "abs_log_observer_growth": 0.653571,
}

VA_CONFIRMATION_RHO = {
    "cell_jaccard": 0.327273,
    "abs_log_observer_growth": 0.063636,
}

MIN_CONFIRMATION_SPECIES = 10
MIN_PRIMARY_RHO = 0.50
MAX_HOLM_P = 0.05
MIN_BOOTSTRAP_POSITIVE = 0.90
MIN_LOO_POSITIVE = 0.80
MIN_GRID_MEDIAN_RHO = 0.35
MIN_POSITIVE_GRID_CORRELATIONS = 2

N_PERMUTATIONS = 5000
N_BOOTSTRAP = 3000
RANDOM_SEED = 20260710

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
INPUT_DIR = (
    BASE_DIR
    / "derived"
    / "nc_final_confirmation_dataset"
)
OUT_DIR = (
    BASE_DIR
    / "derived"
    / "nc_final_candidate_confirmation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OCCURRENCE_PATH = (
    INPUT_DIR / "nc_thinned_occurrences.parquet"
)
FIA_REFERENCE_PATH = (
    INPUT_DIR / "nc_fia_same_window_reference.parquet"
)
COVERAGE_PATH = (
    INPUT_DIR / "nc_confirmation_species_coverage.csv"
)
SELECTED_TAXA_PATH = (
    INPUT_DIR / "nc_locked_selected_taxa.csv"
)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("STATE:", STATE)
print("LOCKED_FEATURES:", LOCKED_FEATURES)
print("PRIMARY_GRID_RESOLUTION:", PRIMARY_GRID_RESOLUTION)
print("N_PERMUTATIONS:", N_PERMUTATIONS)
print("N_BOOTSTRAP:", N_BOOTSTRAP)
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

    readme_path.write_text(
        existing + text,
        encoding="utf-8",
    )


def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
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

    return normalized.mask(
        normalized.isin(blocked)
    )


def set_jaccard(first, second):
    union = first | second

    if not union:
        return np.nan

    return float(
        len(first & second)
        / len(union)
    )


def shift_km_decade(
    early_latitude,
    late_latitude,
):
    if not (
        np.isfinite(early_latitude)
        and np.isfinite(late_latitude)
    ):
        return np.nan

    interval = (
        LATE_MID_YEAR
        - EARLY_MID_YEAR
    )

    if interval <= 0:
        return np.nan

    return float(
        (
            late_latitude
            - early_latitude
        )
        * 111.32
        * 10
        / interval
    )


def safe_spearman(x, y):
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
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


def holm_adjust(p_values):
    values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        np.isfinite(values)
    )[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_values = values[
        valid_indices
    ]

    order = np.argsort(
        valid_values
    )

    ordered_values = valid_values[
        order
    ]

    m = len(
        ordered_values
    )

    ordered_adjusted = np.empty(
        m,
        dtype=float,
    )

    running_maximum = 0.0

    for rank_index, p_value in enumerate(
        ordered_values
    ):
        multiplier = (
            m - rank_index
        )

        current = min(
            1.0,
            p_value * multiplier,
        )

        running_maximum = max(
            running_maximum,
            current,
        )

        ordered_adjusted[
            rank_index
        ] = running_maximum

    adjusted[
        valid_indices[order]
    ] = np.clip(
        ordered_adjusted,
        0,
        1,
    )

    return adjusted


def build_grid_estimands(
    occurrence_df,
    evaluation_codes,
    resolution,
):
    work = occurrence_df.copy()

    work["grid_lat_index"] = np.floor(
        work["decimalLatitude"]
        / resolution
    ).astype("int32")

    work["grid_lon_index"] = np.floor(
        work["decimalLongitude"]
        / resolution
    ).astype("int32")

    work["grid_id"] = (
        work["grid_lat_index"].astype(str)
        + "_"
        + work["grid_lon_index"].astype(str)
    )

    effort_frame = work[
        work["is_effort_frame_taxon"]
    ].copy()

    effort_df = (
        effort_frame.groupby(
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
        effort_df.pivot(
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
        if period not in effort_wide.columns:
            effort_wide[period] = 0

    stable_cells = set(
        effort_wide[
            (
                effort_wide["early"]
                >= MIN_TARGET_EFFORT_PER_CELL_PERIOD
            )
            & (
                effort_wide["late"]
                >= MIN_TARGET_EFFORT_PER_CELL_PERIOD
            )
        ].index
    )

    stable_effort_df = effort_df[
        effort_df["grid_id"].isin(
            stable_cells
        )
    ].copy()

    pooled_cell_latitude = (
        effort_frame[
            effort_frame["grid_id"].isin(
                stable_cells
            )
        ]
        .groupby("grid_id")[
            "decimalLatitude"
        ]
        .mean()
        .to_dict()
    )

    benchmark_frame = work[
        work["fia_species_code"].isin(
            evaluation_codes
        )
    ].copy()

    raw_period_df = (
        benchmark_frame.groupby(
            [
                "fia_species_code",
                "period",
            ],
            as_index=False,
        )
        .agg(
            raw_records=(
                "gbifID",
                "size",
            ),
            raw_latitude=(
                "decimalLatitude",
                "mean",
            ),
            raw_cells=(
                "grid_id",
                "nunique",
            ),
        )
    )

    raw_wide = raw_period_df.pivot(
        index="fia_species_code",
        columns="period",
        values=[
            "raw_records",
            "raw_latitude",
            "raw_cells",
        ],
    )

    raw_wide.columns = [
        f"{metric}_{period}"
        for metric, period
        in raw_wide.columns
    ]

    raw_wide = raw_wide.reset_index()

    corrected_rows = []

    for species_code in sorted(
        evaluation_codes
    ):
        species_counts = (
            benchmark_frame[
                benchmark_frame[
                    "fia_species_code"
                ]
                == species_code
            ]
            .groupby(
                [
                    "period",
                    "grid_id",
                ]
            )[
                "gbifID"
            ]
            .size()
            .rename(
                "species_records"
            )
            .reset_index()
        )

        merged = stable_effort_df.merge(
            species_counts,
            on=[
                "period",
                "grid_id",
            ],
            how="left",
        )

        merged[
            "species_records"
        ] = (
            merged[
                "species_records"
            ]
            .fillna(0)
            .astype(float)
        )

        merged[
            "relative_occurrence_rate"
        ] = (
            merged[
                "species_records"
            ]
            / merged[
                "target_effort_records"
            ]
        )

        merged[
            "cell_latitude"
        ] = merged[
            "grid_id"
        ].map(
            pooled_cell_latitude
        )

        for period in [
            "early",
            "late",
        ]:
            period_frame = merged[
                merged["period"]
                == period
            ].copy()

            weight_sum = float(
                period_frame[
                    "relative_occurrence_rate"
                ].sum()
            )

            corrected_latitude = (
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
                    weight_sum > 0
                    and period_frame[
                        "cell_latitude"
                    ].notna().all()
                )
                else np.nan
            )

            corrected_rows.append(
                {
                    "fia_species_code": (
                        species_code
                    ),
                    "period": period,
                    "corrected_latitude": (
                        corrected_latitude
                    ),
                    "stable_species_records": float(
                        period_frame[
                            "species_records"
                        ].sum()
                    ),
                    "stable_positive_cells": int(
                        (
                            period_frame[
                                "species_records"
                            ]
                            > 0
                        ).sum()
                    ),
                }
            )

    corrected_df = pd.DataFrame(
        corrected_rows
    )

    corrected_wide = corrected_df.pivot(
        index="fia_species_code",
        columns="period",
        values=[
            "corrected_latitude",
            "stable_species_records",
            "stable_positive_cells",
        ],
    )

    corrected_wide.columns = [
        f"{metric}_{period}"
        for metric, period
        in corrected_wide.columns
    ]

    corrected_wide = (
        corrected_wide.reset_index()
    )

    result = raw_wide.merge(
        corrected_wide,
        on="fia_species_code",
        how="outer",
    )

    result[
        "raw_shift_km_decade"
    ] = [
        shift_km_decade(
            row.get(
                "raw_latitude_early",
                np.nan,
            ),
            row.get(
                "raw_latitude_late",
                np.nan,
            ),
        )
        for _, row in result.iterrows()
    ]

    result[
        "corrected_shift_km_decade"
    ] = [
        shift_km_decade(
            row.get(
                "corrected_latitude_early",
                np.nan,
            ),
            row.get(
                "corrected_latitude_late",
                np.nan,
            ),
        )
        for _, row in result.iterrows()
    ]

    frame_summary = {
        "grid_resolution": resolution,
        "n_effort_cells": int(
            effort_frame[
                "grid_id"
            ].nunique()
        ),
        "n_stable_cells": int(
            len(stable_cells)
        ),
        "early_effort_records": int(
            (
                effort_frame[
                    "period"
                ]
                == "early"
            ).sum()
        ),
        "late_effort_records": int(
            (
                effort_frame[
                    "period"
                ]
                == "late"
            ).sum()
        ),
    }

    return result, frame_summary


def confirm_candidate(
    frame,
    feature,
    outcome,
    rng,
    progress,
):
    work = frame[
        [
            "fia_species_code",
            feature,
            outcome,
        ]
    ].dropna().copy()

    feature_values = work[
        feature
    ].to_numpy(
        dtype=float
    )

    outcome_values = work[
        outcome
    ].to_numpy(
        dtype=float
    )

    observed_rho = safe_spearman(
        feature_values,
        outcome_values,
    )

    permutation_exceed = 0

    for _ in range(
        N_PERMUTATIONS
    ):
        permuted_outcome = rng.permutation(
            outcome_values
        )

        permuted_rho = safe_spearman(
            feature_values,
            permuted_outcome,
        )

        if (
            np.isfinite(
                permuted_rho
            )
            and permuted_rho
            >= observed_rho
        ):
            permutation_exceed += 1

        progress.update(1)

    permutation_p = float(
        (
            1
            + permutation_exceed
        )
        / (
            N_PERMUTATIONS
            + 1
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

        bootstrap_rho = safe_spearman(
            feature_values[
                indices
            ],
            outcome_values[
                indices
            ],
        )

        if np.isfinite(
            bootstrap_rho
        ):
            bootstrap_values.append(
                bootstrap_rho
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
        bootstrap_positive_fraction = float(
            np.mean(
                bootstrap_values > 0
            )
        )
    else:
        bootstrap_low = np.nan
        bootstrap_high = np.nan
        bootstrap_positive_fraction = np.nan

    loo_positive = []

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

        loo_rho = safe_spearman(
            feature_values[
                keep
            ],
            outcome_values[
                keep
            ],
        )

        if np.isfinite(
            loo_rho
        ):
            loo_positive.append(
                loo_rho > 0
            )

        progress.update(1)

    loo_positive_fraction = (
        float(
            np.mean(
                loo_positive
            )
        )
        if loo_positive
        else np.nan
    )

    return {
        "feature": feature,
        "n_species": len(work),
        "pa_discovery_rho": (
            PA_DISCOVERY_RHO[
                feature
            ]
        ),
        "va_confirmation_rho": (
            VA_CONFIRMATION_RHO[
                feature
            ]
        ),
        "nc_primary_rho": observed_rho,
        "permutation_p_one_sided": (
            permutation_p
        ),
        "bootstrap_low": bootstrap_low,
        "bootstrap_high": bootstrap_high,
        "bootstrap_positive_fraction": (
            bootstrap_positive_fraction
        ),
        "loo_positive_fraction": (
            loo_positive_fraction
        ),
    }


def print_failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## NC final candidate confirmation failure — {RUN_UTC}
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
    # 1. Load final independent-confirmation data
    # --------------------------------------------------------
    required_paths = [
        OCCURRENCE_PATH,
        FIA_REFERENCE_PATH,
        COVERAGE_PATH,
        SELECTED_TAXA_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing Cell 16B outputs: "
            + " | ".join(
                missing_paths
            )
        )

    occurrence_df = pd.read_parquet(
        OCCURRENCE_PATH
    )

    fia_df = pd.read_parquet(
        FIA_REFERENCE_PATH
    )

    coverage_df = pd.read_csv(
        COVERAGE_PATH
    )

    selected_taxa_df = pd.read_csv(
        SELECTED_TAXA_PATH
    )

    coverage_df[
        "confirmation_coverage_pass"
    ] = to_bool(
        coverage_df[
            "confirmation_coverage_pass"
        ]
    )

    evaluation_codes = set(
        pd.to_numeric(
            coverage_df.loc[
                coverage_df[
                    "confirmation_coverage_pass"
                ],
                "fia_species_code",
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    if (
        len(evaluation_codes)
        < MIN_CONFIRMATION_SPECIES
    ):
        raise ValueError(
            f"Only {len(evaluation_codes)} "
            "North Carolina confirmation species remain."
        )

    selected_taxa_df[
        "is_effort_frame_taxon"
    ] = to_bool(
        selected_taxa_df[
            "is_effort_frame_taxon"
        ]
    )

    selected_taxa_df[
        "fia_species_code"
    ] = pd.to_numeric(
        selected_taxa_df[
            "fia_species_code"
        ],
        errors="coerce",
    )

    selected_taxa_df = selected_taxa_df.dropna(
        subset=[
            "fia_species_code",
        ]
    ).copy()

    selected_taxa_df[
        "fia_species_code"
    ] = selected_taxa_df[
        "fia_species_code"
    ].astype(int)

    effort_code_map = (
        selected_taxa_df.set_index(
            "fia_species_code"
        )[
            "is_effort_frame_taxon"
        ]
        .to_dict()
    )

    for column in [
        "fia_species_code",
        "year",
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
            "fia_species_code"
        ].notna()
        & occurrence_df[
            "year"
        ].isin(
            EARLY_YEARS
            + LATE_YEARS
        )
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
        "year"
    ] = occurrence_df[
        "year"
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

    if (
        "is_effort_frame_taxon"
        in occurrence_df.columns
    ):
        occurrence_df[
            "is_effort_frame_taxon"
        ] = to_bool(
            occurrence_df[
                "is_effort_frame_taxon"
            ]
        )
    else:
        occurrence_df[
            "is_effort_frame_taxon"
        ] = (
            occurrence_df[
                "fia_species_code"
            ]
            .map(
                effort_code_map
            )
            .fillna(False)
        )

    fia_df[
        "fia_species_code"
    ] = pd.to_numeric(
        fia_df[
            "fia_species_code"
        ],
        errors="coerce",
    )

    fia_df = fia_df.dropna(
        subset=[
            "fia_species_code",
            "fia_shift_km_decade",
        ]
    ).copy()

    fia_df[
        "fia_species_code"
    ] = fia_df[
        "fia_species_code"
    ].astype(int)

    fia_df = fia_df[
        fia_df[
            "fia_species_code"
        ].isin(
            evaluation_codes
        )
    ].copy()

    # --------------------------------------------------------
    # 2. Reproduce locked correction across three grids
    # --------------------------------------------------------
    grid_tables = {}
    grid_summary_rows = []

    for resolution in tqdm(
        GRID_RESOLUTIONS,
        desc="Computing NC correction gains",
        unit="resolution",
    ):
        grid_df, frame_summary = (
            build_grid_estimands(
                occurrence_df,
                evaluation_codes,
                resolution,
            )
        )

        merged_grid_df = grid_df.merge(
            fia_df[
                [
                    "fia_species_code",
                    "scientific_name",
                    "fia_shift_km_decade",
                ]
            ],
            on="fia_species_code",
            how="inner",
        )

        merged_grid_df[
            "raw_abs_error"
        ] = (
            merged_grid_df[
                "raw_shift_km_decade"
            ]
            - merged_grid_df[
                "fia_shift_km_decade"
            ]
        ).abs()

        merged_grid_df[
            "corrected_abs_error"
        ] = (
            merged_grid_df[
                "corrected_shift_km_decade"
            ]
            - merged_grid_df[
                "fia_shift_km_decade"
            ]
        ).abs()

        tag = str(
            resolution
        ).replace(
            ".",
            "p",
        )

        merged_grid_df[
            f"gain_grid_{tag}"
        ] = (
            merged_grid_df[
                "raw_abs_error"
            ]
            - merged_grid_df[
                "corrected_abs_error"
            ]
        )

        merged_grid_df.to_parquet(
            OUT_DIR
            / f"nc_grid_{tag}_species_results.parquet",
            index=False,
        )

        grid_tables[
            resolution
        ] = merged_grid_df

        frame_summary[
            "n_species_with_gain"
        ] = int(
            merged_grid_df[
                f"gain_grid_{tag}"
            ].notna().sum()
        )

        grid_summary_rows.append(
            frame_summary
        )

    grid_summary_df = pd.DataFrame(
        grid_summary_rows
    )

    grid_summary_df.to_csv(
        OUT_DIR
        / "nc_grid_frame_summary.csv",
        index=False,
    )

    print(
        "\nNORTH CAROLINA GRID FRAME SUMMARY"
    )
    display(
        grid_summary_df
    )

    primary_df = grid_tables[
        PRIMARY_GRID_RESOLUTION
    ].copy()

    gain_df = primary_df[
        [
            "fia_species_code",
            "scientific_name",
            "fia_shift_km_decade",
            "raw_shift_km_decade",
            "corrected_shift_km_decade",
            "raw_abs_error",
            "corrected_abs_error",
            "gain_grid_0p5",
        ]
    ].copy()

    for resolution in [
        0.25,
        1.00,
    ]:
        tag = str(
            resolution
        ).replace(
            ".",
            "p",
        )

        gain_df = gain_df.merge(
            grid_tables[
                resolution
            ][
                [
                    "fia_species_code",
                    f"gain_grid_{tag}",
                ]
            ],
            on="fia_species_code",
            how="left",
        )

    gain_columns = [
        "gain_grid_0p25",
        "gain_grid_0p5",
        "gain_grid_1p0",
    ]

    gain_df[
        "grid_median_gain_km_decade"
    ] = gain_df[
        gain_columns
    ].median(
        axis=1,
        skipna=True,
    )

    gain_df[
        "positive_grid_fraction"
    ] = (
        gain_df[
            gain_columns
        ]
        .gt(0)
        .sum(axis=1)
        / gain_df[
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
    # 3. Compute only the two locked features
    # --------------------------------------------------------
    feature_frame = occurrence_df[
        occurrence_df[
            "fia_species_code"
        ].isin(
            evaluation_codes
        )
    ].copy()

    feature_frame[
        "feature_grid_lat_index"
    ] = np.floor(
        feature_frame[
            "decimalLatitude"
        ]
        / PRIMARY_GRID_RESOLUTION
    ).astype("int32")

    feature_frame[
        "feature_grid_lon_index"
    ] = np.floor(
        feature_frame[
            "decimalLongitude"
        ]
        / PRIMARY_GRID_RESOLUTION
    ).astype("int32")

    feature_frame[
        "feature_grid_id"
    ] = (
        feature_frame[
            "feature_grid_lat_index"
        ].astype(str)
        + "_"
        + feature_frame[
            "feature_grid_lon_index"
        ].astype(str)
    )

    feature_rows = []

    for species_code in tqdm(
        sorted(
            evaluation_codes
        ),
        desc="Computing locked NC features",
        unit="species",
    ):
        species_df = feature_frame[
            feature_frame[
                "fia_species_code"
            ]
            == species_code
        ]

        early_df = species_df[
            species_df[
                "period"
            ]
            == "early"
        ]

        late_df = species_df[
            species_df[
                "period"
            ]
            == "late"
        ]

        early_cells = set(
            early_df[
                "feature_grid_id"
            ].dropna()
        )

        late_cells = set(
            late_df[
                "feature_grid_id"
            ].dropna()
        )

        early_observers = int(
            early_df[
                "observer_label"
            ].nunique(
                dropna=True
            )
        )

        late_observers = int(
            late_df[
                "observer_label"
            ].nunique(
                dropna=True
            )
        )

        feature_rows.append(
            {
                "fia_species_code": (
                    species_code
                ),
                "cell_jaccard": (
                    set_jaccard(
                        early_cells,
                        late_cells,
                    )
                ),
                "abs_log_observer_growth": abs(
                    np.log(
                        (
                            late_observers
                            + 1
                        )
                        / (
                            early_observers
                            + 1
                        )
                    )
                ),
                "early_cells": len(
                    early_cells
                ),
                "late_cells": len(
                    late_cells
                ),
                "early_observers": (
                    early_observers
                ),
                "late_observers": (
                    late_observers
                ),
            }
        )

    feature_df = pd.DataFrame(
        feature_rows
    )

    analysis_df = gain_df.merge(
        feature_df,
        on="fia_species_code",
        how="inner",
    )

    analysis_df.to_parquet(
        OUT_DIR
        / "nc_locked_candidate_species_table.parquet",
        index=False,
    )

    analysis_df.to_csv(
        OUT_DIR
        / "nc_locked_candidate_species_table.csv",
        index=False,
    )

    print(
        "\nNORTH CAROLINA LOCKED CANDIDATE SPECIES TABLE"
    )
    display(
        analysis_df[
            [
                "fia_species_code",
                "scientific_name",
                "cell_jaccard",
                "abs_log_observer_growth",
                "gain_grid_0p25",
                "gain_grid_0p5",
                "gain_grid_1p0",
                "grid_median_gain_km_decade",
                "raw_abs_error",
                "corrected_abs_error",
            ]
        ]
    )

    # --------------------------------------------------------
    # 4. Final one-sided confirmatory inference
    # --------------------------------------------------------
    total_iterations = (
        len(
            LOCKED_FEATURES
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
        desc="Final NC confirmation of locked candidates",
        unit="resample",
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    inference_rows = []

    for feature in LOCKED_FEATURES:
        row = confirm_candidate(
            analysis_df,
            feature,
            "gain_grid_0p5",
            rng,
            progress,
        )

        row[
            "grid_median_rho"
        ] = safe_spearman(
            analysis_df[
                feature
            ],
            analysis_df[
                "grid_median_gain_km_decade"
            ],
        )

        positive_grid_correlations = 0

        for grid_column in gain_columns:
            rho = safe_spearman(
                analysis_df[
                    feature
                ],
                analysis_df[
                    grid_column
                ],
            )

            row[
                f"rho_{grid_column}"
            ] = rho

            positive_grid_correlations += int(
                np.isfinite(rho)
                and rho > 0
            )

        row[
            "n_positive_grid_correlations"
        ] = (
            positive_grid_correlations
        )

        inference_rows.append(
            row
        )

    progress.close()

    inference_df = pd.DataFrame(
        inference_rows
    )

    inference_df[
        "holm_adjusted_p"
    ] = holm_adjust(
        inference_df[
            "permutation_p_one_sided"
        ]
    )

    inference_df[
        "confirmed"
    ] = (
        inference_df[
            "n_species"
        ]
        >= MIN_CONFIRMATION_SPECIES
    ) & (
        inference_df[
            "nc_primary_rho"
        ]
        >= MIN_PRIMARY_RHO
    ) & (
        inference_df[
            "holm_adjusted_p"
        ]
        <= MAX_HOLM_P
    ) & (
        inference_df[
            "bootstrap_positive_fraction"
        ]
        >= MIN_BOOTSTRAP_POSITIVE
    ) & (
        inference_df[
            "loo_positive_fraction"
        ]
        >= MIN_LOO_POSITIVE
    ) & (
        inference_df[
            "grid_median_rho"
        ]
        >= MIN_GRID_MEDIAN_RHO
    ) & (
        inference_df[
            "n_positive_grid_correlations"
        ]
        >= MIN_POSITIVE_GRID_CORRELATIONS
    )

    inference_df.to_csv(
        OUT_DIR
        / "nc_final_candidate_confirmation_results.csv",
        index=False,
    )

    print(
        "\nFINAL NORTH CAROLINA CONFIRMATION RESULTS"
    )
    display(
        inference_df
    )

    # --------------------------------------------------------
    # 5. Overall NC correction context
    # --------------------------------------------------------
    raw_median_error = float(
        analysis_df[
            "raw_abs_error"
        ].median()
    )

    corrected_median_error = float(
        analysis_df[
            "corrected_abs_error"
        ].median()
    )

    median_error_reduction = (
        1
        - corrected_median_error
        / raw_median_error
        if raw_median_error > 0
        else np.nan
    )

    corrected_improved_fraction = float(
        (
            analysis_df[
                "gain_grid_0p5"
            ]
            > 0
        ).mean()
    )

    overall_context_df = pd.DataFrame(
        [
            {
                "n_species": len(
                    analysis_df
                ),
                "raw_median_abs_error": (
                    raw_median_error
                ),
                "corrected_median_abs_error": (
                    corrected_median_error
                ),
                "median_error_reduction": (
                    median_error_reduction
                ),
                "corrected_improved_fraction": (
                    corrected_improved_fraction
                ),
            }
        ]
    )

    overall_context_df.to_csv(
        OUT_DIR
        / "nc_overall_correction_context.csv",
        index=False,
    )

    print(
        "\nNORTH CAROLINA OVERALL CORRECTION CONTEXT"
    )
    display(
        overall_context_df
    )

    # --------------------------------------------------------
    # 6. Final decision and stop rule
    # --------------------------------------------------------
    confirmed_features = inference_df.loc[
        inference_df[
            "confirmed"
        ],
        "feature",
    ].tolist()

    n_confirmed = len(
        confirmed_features
    )

    directional_features = inference_df[
        (
            inference_df[
                "nc_primary_rho"
            ]
            > 0
        )
        & (
            inference_df[
                "grid_median_rho"
            ]
            > 0
        )
    ][
        "feature"
    ].tolist()

    if n_confirmed == 2:
        status = (
            "BOTH_CANDIDATES_CONFIRMED_IN_FINAL_NC"
        )
        passed = True
        index_route_closed = False
        next_step = (
            "Retain both candidates. Run a locked PA–VA–NC "
            "cross-region synthesis and construct a two-feature "
            "correction-applicability score without searching "
            "additional predictors."
        )

    elif n_confirmed == 1:
        confirmed_feature = (
            confirmed_features[0]
        )

        if confirmed_feature == "cell_jaccard":
            status = (
                "CELL_JACCARD_CONFIRMED_IN_FINAL_NC"
            )
        else:
            status = (
                "OBSERVER_GROWTH_CONFIRMED_IN_FINAL_NC"
            )

        passed = True
        index_route_closed = False
        next_step = (
            f"Retain only {confirmed_feature}. Permanently discard "
            "the unconfirmed candidate. Run a locked three-region "
            "effect synthesis and threshold-free ranking validation."
        )

    else:
        status = (
            "SENSITIVITY_INDEX_ROUTE_CLOSED_AFTER_NC"
        )
        passed = False
        index_route_closed = True
        next_step = (
            "Neither locked candidate independently confirmed. "
            "Permanently close the sensitivity-index route. Do not "
            "search more regions, windows or features. Reframe the "
            "paper around causal semi-synthetic validity, real-world "
            "correction failure, species/region heterogeneity and "
            "non-transportability."
        )

    # --------------------------------------------------------
    # 7. Visual previews
    # --------------------------------------------------------
    for feature in LOCKED_FEATURES:
        plot_df = analysis_df[
            [
                "scientific_name",
                feature,
                "gain_grid_0p5",
            ]
        ].dropna()

        plt.figure(
            figsize=(8, 6)
        )
        plt.scatter(
            plot_df[
                feature
            ],
            plot_df[
                "gain_grid_0p5"
            ],
        )
        plt.axhline(
            0,
            linestyle="--",
        )
        plt.xlabel(
            feature
        )
        plt.ylabel(
            "0.5° correction gain versus FIA "
            "(km/decade)"
        )
        plt.title(
            f"Final NC confirmation: {feature}"
        )
        plt.tight_layout()
        plt.show()

    gain_plot_df = analysis_df[
        [
            "scientific_name",
            "gain_grid_0p25",
            "gain_grid_0p5",
            "gain_grid_1p0",
        ]
    ].sort_values(
        "gain_grid_0p5"
    ).reset_index(
        drop=True
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
            "gain_grid_0p25"
        ],
        y,
        label="0.25°",
    )
    plt.scatter(
        gain_plot_df[
            "gain_grid_0p5"
        ],
        y,
        label="0.5°",
    )
    plt.scatter(
        gain_plot_df[
            "gain_grid_1p0"
        ],
        y,
        label="1.0°",
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
        "Correction gain versus FIA "
        "(km/decade; positive is better)"
    )
    plt.ylabel(
        "Species"
    )
    plt.title(
        "North Carolina correction benefit across locked grids"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 8. README
    # --------------------------------------------------------
    compact_confirmation = (
        inference_df[
            [
                "feature",
                "n_species",
                "pa_discovery_rho",
                "va_confirmation_rho",
                "nc_primary_rho",
                "permutation_p_one_sided",
                "holm_adjusted_p",
                "bootstrap_low",
                "bootstrap_high",
                "bootstrap_positive_fraction",
                "loo_positive_fraction",
                "grid_median_rho",
                "rho_gain_grid_0p25",
                "rho_gain_grid_0p5",
                "rho_gain_grid_1p0",
                "n_positive_grid_correlations",
                "confirmed",
            ]
        ]
        .round(6)
        .to_dict(
            "records"
        )
    )

    append_readme(
        f"""

## Final North Carolina confirmation — {RUN_UTC}

### Locked hypotheses
1. Higher cell_jaccard predicts greater correction gain.
2. Greater abs_log_observer_growth predicts greater correction gain.

No additional features, thresholds, windows, species or regions were searched.

### Data
- Final independent region: North Carolina
- Confirmation species: {len(analysis_df)}
- Early window: {EARLY_YEARS}
- Late window: {LATE_YEARS}
- Primary grid: {PRIMARY_GRID_RESOLUTION} degrees
- Sensitivity grids: {GRID_RESOLUTIONS}
- GPU: not used

### Confirmatory inference
- One-sided permutations per candidate: {N_PERMUTATIONS}
- Species bootstrap replicates per candidate: {N_BOOTSTRAP}
- Multiple testing: Holm adjustment across two locked candidates
- Confirmed features: {confirmed_features}
- Directionally positive features: {directional_features}

### Overall North Carolina correction context
- Raw median absolute error:
  {raw_median_error} km/decade
- Corrected median absolute error:
  {corrected_median_error} km/decade
- Median error reduction:
  {median_error_reduction}
- Fraction of species improved:
  {corrected_improved_fraction}

### Final decision
- Status: **{status}**
- Passed: {passed}
- Sensitivity-index route closed:
  {index_route_closed}
- Next step: {next_step}

### Interpretation boundary
This is the final independent confirmatory test of two predeclared associations.
A confirmed feature is a transportable ranking/gating signal, not proof of
causality. If neither feature confirms, the stop rule prohibits any further
feature, region, window or threshold search for a sensitivity index.
"""
    )

    # --------------------------------------------------------
    # 9. Compact output
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
        f"STATE: {STATE}"
    )
    print(
        f"N_CONFIRMATION_SPECIES: "
        f"{len(analysis_df)}"
    )
    print(
        "GRID_FRAME_SUMMARY: "
        + json.dumps(
            grid_summary_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "OVERALL_CORRECTION_CONTEXT: "
        + json.dumps(
            overall_context_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "FINAL_CANDIDATE_CONFIRMATION: "
        + json.dumps(
            compact_confirmation,
            ensure_ascii=False,
        )
    )
    print(
        f"N_CONFIRMED_FEATURES: "
        f"{n_confirmed}"
    )
    print(
        "CONFIRMED_FEATURES: "
        + json.dumps(
            confirmed_features,
            ensure_ascii=False,
        )
    )
    print(
        "DIRECTIONAL_FEATURES: "
        + json.dumps(
            directional_features,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_CRITERIA_PER_FEATURE: n>=10, NC primary rho>=0.50, "
        "Holm-adjusted one-sided permutation p<=0.05, bootstrap "
        "positive fraction>=0.90, LOO positive fraction>=0.80, "
        "grid-median rho>=0.35, and positive rho in >=2/3 grids"
    )
    print(
        f"SENSITIVITY_INDEX_ROUTE_CLOSED: "
        f"{index_route_closed}"
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
        "CELL16C_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed output. "
            "Do not alter thresholds or search additional features."
        ),
    )
