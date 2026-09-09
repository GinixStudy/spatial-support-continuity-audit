
# Cell 22C — Orthogonal support-overlap phase diagram: redesigned trend test
#
# Why this redesign is necessary
# ------------------------------
# Cell 22B confirmed that the first phase diagram was structurally confounded:
# - target-frame overlap controlled corrected estimability;
# - low overlap enlarged raw error and therefore enlarged gain opportunity;
# - the nominal zero-drift condition still generated large realized drift;
# - manipulated effort-frame overlap was not the same construct as empirical
#   species occupied-cell continuity.
#
# This redesigned experiment separates three data-generating components:
#
# 1. Effort-frame support overlap
#    Controls whether the correction has a common spatial frame and is
#    estimable.
#
# 2. Species-specific observation-support overlap
#    Controls whether the species relative-occurrence surface is comparable
#    across periods.
#
# 3. Expected observation-footprint drift
#    Controlled by exponential tilting of late-period effort weights.
#    The zero-drift condition is exact at the expected data-generating level.
#
# The latent ecological distribution is stationary, so the true shift is zero.
#
# Primary outcomes
# ----------------
# - corrected estimability;
# - raw absolute error;
# - corrected absolute error;
# - correction gain;
# - corrected/raw error ratio.
#
# Trend-stage success requires:
# - exact zero-drift and zero species-centre mismatch;
# - effort overlap strongly predicts estimability;
# - species-support overlap predicts lower corrected error;
# - the two overlap axes show the expected monotonic roles within fixed strata.
#
# No cutoff, new empirical feature or final theory claim is produced here.

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr
from tqdm.auto import tqdm
from IPython.display import display, Image

# ============================================================
# Configuration
# ============================================================
STATES = ["PA", "VA", "NC"]

GRID_RESOLUTION = 0.50

EFFORT_SUPPORT_SIZE = 20
SPECIES_SUPPORT_SIZE = 12

MIN_AVAILABLE_CELL_EFFORT = 5
MIN_STABLE_CELL_EFFORT = 10
MIN_STABLE_CELLS = 3

SPECIES_PER_STATE = 5

EFFORT_OVERLAP_LEVELS = [
    0.20,
    0.50,
    0.80,
    1.00,
]

SPECIES_OVERLAP_LEVELS = [
    0.20,
    0.50,
    0.80,
    1.00,
]

# Expected absolute effort-centre drift in km/decade.
DRIFT_LEVELS_KM_DECADE = [
    0.0,
    20.0,
    40.0,
]

SAMPLE_SIZES = [
    500,
    1500,
]

WIDTH_MODES = {
    "narrow": 0.75,
    "broad": 1.25,
}

REPLICATES = 20

DETECTION_BASELINE = 0.01
DETECTION_AMPLITUDE = 0.34

MAX_SUPPORT_ATTEMPTS = 250
BISECTION_ITERATIONS = 100

RANDOM_SEED = 20260711

# Trend-stage criteria.
MAX_EXPECTED_ZERO_DRIFT_ERROR = 1e-6
MAX_SPECIES_CENTRE_MISMATCH = 1e-6

MIN_EFFORT_ESTIMABILITY_RHO = 0.50
MAX_EFFORT_ESTIMABILITY_P = 0.01

MAX_SPECIES_OVERLAP_ERROR_RHO = -0.30
MAX_SPECIES_OVERLAP_ERROR_P = 0.01

MIN_EFFORT_MONOTONIC_FRACTION = 0.65
MIN_SPECIES_MONOTONIC_FRACTION = 0.60

MIN_HIGH_LOW_ESTIMABILITY_DELTA = 0.12
MIN_LOW_HIGH_CORRECTED_ERROR_DELTA = 1.0

BASE_DIR = Path(
    "/kaggle/working/fia_temporal_observation_drift"
)

FORMAL_DIR = (
    BASE_DIR
    / "derived"
    / "registered_gbif_formal_parity"
)

OUT_DIR = (
    BASE_DIR
    / "derived"
    / "orthogonal_support_overlap_phase_diagram_trend"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

THINNED_PATHS = {
    state: FORMAL_DIR / f"{state}_formal_thinned.parquet"
    for state in STATES
}

SPECIES_INPUT_PATHS = {
    state: FORMAL_DIR / f"{state}_formal_species_inputs.parquet"
    for state in STATES
}

EFFORT_PLAN_PATHS = {
    "PA": (
        BASE_DIR
        / "derived"
        / "pa_expanded_tree_taxon_audit"
        / "selected_expanded_tree_taxa.csv"
    ),
    "VA": (
        BASE_DIR
        / "derived"
        / "va_confirmation_dataset"
        / "va_locked_selected_taxa.csv"
    ),
    "NC": (
        BASE_DIR
        / "derived"
        / "nc_final_confirmation_dataset"
        / "nc_locked_selected_taxa.csv"
    ),
}

GBIF_DOI = "10.15468/dl.sm6ygu"

RUN_UTC = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("OUT_DIR:", OUT_DIR)
print("STATES:", STATES)
print("REPLICATES:", REPLICATES)
print("GPU: not used")


# ============================================================
# General helpers
# ============================================================
def append_readme(text):
    path = BASE_DIR / "README.md"

    old = (
        path.read_text(
            encoding="utf-8"
        )
        if path.exists()
        else "# Temporal Observation Drift\n"
    )

    path.write_text(
        old + text,
        encoding="utf-8",
    )


def resolve_column(
    columns,
    candidates,
):
    lookup = {
        str(column).strip().lower(): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[
                candidate.lower()
            ]

    return None


def to_bool(series):
    if pd.api.types.is_bool_dtype(
        series
    ):
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


def load_effort_codes(
    state,
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing effort-frame plan: {path}"
        )

    frame = pd.read_csv(
        path
    )

    code_column = resolve_column(
        frame.columns,
        [
            "fia_species_code",
            "species_code",
        ],
    )

    if code_column is None:
        raise KeyError(
            f"{state}: species-code column missing."
        )

    frame[
        "fia_species_code"
    ] = pd.to_numeric(
        frame[
            code_column
        ],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "fia_species_code",
        ]
    ).copy()

    frame[
        "fia_species_code"
    ] = frame[
        "fia_species_code"
    ].astype(int)

    effort_column = resolve_column(
        frame.columns,
        [
            "is_effort_frame_taxon",
            "effort_frame_taxon",
        ],
    )

    if effort_column is None:
        return set(
            frame[
                "fia_species_code"
            ]
        )

    effort_mask = to_bool(
        frame[
            effort_column
        ]
    )

    return set(
        frame.loc[
            effort_mask,
            "fia_species_code",
        ]
    )


def safe_spearman(
    x,
    y,
):
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(
                pd.Series(x),
                errors="coerce",
            ),
            "y": pd.to_numeric(
                pd.Series(y),
                errors="coerce",
            ),
        }
    ).dropna()

    if (
        len(frame) < 5
        or frame["x"].nunique() < 2
        or frame["y"].nunique() < 2
    ):
        return (
            np.nan,
            np.nan,
        )

    result = spearmanr(
        frame["x"],
        frame["y"],
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def weighted_mean(
    values,
    weights,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.isfinite(
            values
        )
        & np.isfinite(
            weights
        )
        & (
            weights > 0
        )
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            values[
                valid
            ],
            weights=weights[
                valid
            ],
        )
    )


def set_jaccard(
    first,
    second,
):
    first = set(
        first
    )

    second = set(
        second
    )

    union = (
        first
        | second
    )

    if not union:
        return np.nan

    return float(
        len(
            first
            & second
        )
        / len(
            union
        )
    )


def exact_overlap_count(
    support_size,
    requested_jaccard,
):
    overlap = int(
        round(
            (
                2
                * support_size
                * requested_jaccard
            )
            / (
                1
                + requested_jaccard
            )
        )
    )

    return int(
        np.clip(
            overlap,
            0,
            support_size,
        )
    )


def weighted_choice_without_replacement(
    rng,
    values,
    size,
    weights,
):
    values = np.asarray(
        values
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if size == 0:
        return np.asarray(
            [],
            dtype=values.dtype,
        )

    if size > len(values):
        raise ValueError(
            "Requested sample exceeds available cells."
        )

    weights = np.where(
        np.isfinite(
            weights
        )
        & (
            weights > 0
        ),
        weights,
        0.0,
    )

    if weights.sum() <= 0:
        weights = np.ones(
            len(values),
            dtype=float,
        )

    weights = (
        weights
        / weights.sum()
    )

    return rng.choice(
        values,
        size=size,
        replace=False,
        p=weights,
    )


# ============================================================
# Exact exponential-tilt calibration
# ============================================================
def exponentially_tilted_weights(
    latitudes,
    base_weights,
    target_mean,
):
    latitudes = np.asarray(
        latitudes,
        dtype=float,
    )

    base_weights = np.asarray(
        base_weights,
        dtype=float,
    )

    valid = (
        np.isfinite(
            latitudes
        )
        & np.isfinite(
            base_weights
        )
        & (
            base_weights > 0
        )
    )

    latitudes = latitudes[
        valid
    ]

    base_weights = base_weights[
        valid
    ]

    if len(
        latitudes
    ) == 0:
        return None

    minimum = float(
        latitudes.min()
    )

    maximum = float(
        latitudes.max()
    )

    if (
        target_mean
        < minimum
        - 1e-12
        or target_mean
        > maximum
        + 1e-12
    ):
        return None

    centred_latitudes = (
        latitudes
        - latitudes.mean()
    )

    def compute(beta):
        exponent = np.clip(
            beta
            * centred_latitudes,
            -700,
            700,
        )

        weights = (
            base_weights
            * np.exp(
                exponent
            )
        )

        weights = (
            weights
            / weights.sum()
        )

        mean_value = float(
            np.sum(
                weights
                * latitudes
            )
        )

        return (
            weights,
            mean_value,
        )

    base_normalized = (
        base_weights
        / base_weights.sum()
    )

    base_mean = float(
        np.sum(
            base_normalized
            * latitudes
        )
    )

    if abs(
        base_mean
        - target_mean
    ) <= 1e-12:
        return (
            base_normalized,
            base_mean,
        )

    low = -200.0
    high = 200.0

    (
        _,
        low_mean,
    ) = compute(
        low
    )

    (
        _,
        high_mean,
    ) = compute(
        high
    )

    if (
        target_mean
        < low_mean
        - 1e-10
        or target_mean
        > high_mean
        + 1e-10
    ):
        return None

    for _ in range(
        BISECTION_ITERATIONS
    ):
        middle = (
            low
            + high
        ) / 2.0

        (
            weights,
            middle_mean,
        ) = compute(
            middle
        )

        if middle_mean < target_mean:
            low = middle
        else:
            high = middle

    final_beta = (
        low
        + high
    ) / 2.0

    (
        final_weights,
        final_mean,
    ) = compute(
        final_beta
    )

    return (
        final_weights,
        final_mean,
    )


# ============================================================
# Support-pair generators
# ============================================================
def generate_effort_support_pair(
    rng,
    cell_indices,
    latitudes,
    base_effort,
    requested_overlap,
    drift_km_decade,
):
    overlap_count = exact_overlap_count(
        EFFORT_SUPPORT_SIZE,
        requested_overlap,
    )

    actual_jaccard = (
        overlap_count
        / (
            2
            * EFFORT_SUPPORT_SIZE
            - overlap_count
        )
    )

    target_shift_degrees = (
        drift_km_decade
        * 7.0
        / (
            111.32
            * 10.0
        )
    )

    for _ in range(
        MAX_SUPPORT_ATTEMPTS
    ):
        early_support = (
            weighted_choice_without_replacement(
                rng,
                cell_indices,
                EFFORT_SUPPORT_SIZE,
                base_effort,
            )
        )

        early_weights = (
            base_effort[
                early_support
            ]
            / base_effort[
                early_support
            ].sum()
        )

        early_mean = weighted_mean(
            latitudes[
                early_support
            ],
            early_weights,
        )

        overlap_cells = (
            weighted_choice_without_replacement(
                rng,
                early_support,
                overlap_count,
                base_effort[
                    early_support
                ],
            )
        )

        outside_mask = ~np.isin(
            cell_indices,
            early_support,
        )

        outside_cells = cell_indices[
            outside_mask
        ]

        late_new_count = (
            EFFORT_SUPPORT_SIZE
            - overlap_count
        )

        late_new_cells = (
            weighted_choice_without_replacement(
                rng,
                outside_cells,
                late_new_count,
                base_effort[
                    outside_cells
                ],
            )
        )

        late_support = np.concatenate(
            [
                overlap_cells,
                late_new_cells,
            ]
        )

        possible_signs = (
            [0.0]
            if drift_km_decade == 0
            else rng.permutation(
                [
                    -1.0,
                    1.0,
                ]
            ).tolist()
        )

        for direction in possible_signs:
            target_late_mean = (
                early_mean
                + direction
                * target_shift_degrees
            )

            tilted = exponentially_tilted_weights(
                latitudes[
                    late_support
                ],
                base_effort[
                    late_support
                ],
                target_late_mean,
            )

            if tilted is None:
                continue

            (
                late_weights,
                calibrated_late_mean,
            ) = tilted

            expected_signed_drift = (
                (
                    calibrated_late_mean
                    - early_mean
                )
                * 111.32
                * 10.0
                / 7.0
            )

            expected_absolute_drift = abs(
                expected_signed_drift
            )

            return {
                "early_support": (
                    early_support
                ),
                "late_support": (
                    late_support
                ),
                "early_weights": (
                    early_weights
                ),
                "late_weights": (
                    late_weights
                ),
                "expected_early_mean": (
                    early_mean
                ),
                "expected_late_mean": (
                    calibrated_late_mean
                ),
                "expected_signed_drift": (
                    expected_signed_drift
                ),
                "expected_absolute_drift": (
                    expected_absolute_drift
                ),
                "requested_absolute_drift": (
                    drift_km_decade
                ),
                "requested_overlap": (
                    requested_overlap
                ),
                "actual_overlap": (
                    actual_jaccard
                ),
            }

    return None


def generate_species_support_pair(
    rng,
    cell_indices,
    latitudes,
    latent_kernel,
    requested_overlap,
):
    overlap_count = exact_overlap_count(
        SPECIES_SUPPORT_SIZE,
        requested_overlap,
    )

    actual_jaccard = (
        overlap_count
        / (
            2
            * SPECIES_SUPPORT_SIZE
            - overlap_count
        )
    )

    selection_weights = (
        latent_kernel
        + 1e-12
    )

    for _ in range(
        MAX_SUPPORT_ATTEMPTS
    ):
        early_support = (
            weighted_choice_without_replacement(
                rng,
                cell_indices,
                SPECIES_SUPPORT_SIZE,
                selection_weights,
            )
        )

        early_profile = (
            latent_kernel[
                early_support
            ]
            + 1e-12
        )

        early_profile = (
            early_profile
            / early_profile.sum()
        )

        early_mean = weighted_mean(
            latitudes[
                early_support
            ],
            early_profile,
        )

        overlap_cells = (
            weighted_choice_without_replacement(
                rng,
                early_support,
                overlap_count,
                latent_kernel[
                    early_support
                ]
                + 1e-12,
            )
        )

        outside_mask = ~np.isin(
            cell_indices,
            early_support,
        )

        outside_cells = cell_indices[
            outside_mask
        ]

        late_new_count = (
            SPECIES_SUPPORT_SIZE
            - overlap_count
        )

        late_new_cells = (
            weighted_choice_without_replacement(
                rng,
                outside_cells,
                late_new_count,
                latent_kernel[
                    outside_cells
                ]
                + 1e-12,
            )
        )

        late_support = np.concatenate(
            [
                overlap_cells,
                late_new_cells,
            ]
        )

        tilted = exponentially_tilted_weights(
            latitudes[
                late_support
            ],
            latent_kernel[
                late_support
            ]
            + 1e-12,
            early_mean,
        )

        if tilted is None:
            continue

        (
            late_profile,
            late_mean,
        ) = tilted

        return {
            "early_support": (
                early_support
            ),
            "late_support": (
                late_support
            ),
            "early_profile": (
                early_profile
            ),
            "late_profile": (
                late_profile
            ),
            "expected_early_mean": (
                early_mean
            ),
            "expected_late_mean": (
                late_mean
            ),
            "expected_centre_mismatch": abs(
                late_mean
                - early_mean
            ),
            "requested_overlap": (
                requested_overlap
            ),
            "actual_overlap": (
                actual_jaccard
            ),
        }

    return None


# ============================================================
# State and latent-species preparation
# ============================================================
def build_state_inputs(
    state,
    thinned_path,
    species_input_path,
    effort_plan_path,
):
    thinned = pd.read_parquet(
        thinned_path
    )

    species_inputs = pd.read_parquet(
        species_input_path
    )

    effort_codes = load_effort_codes(
        state,
        effort_plan_path,
    )

    for column in [
        "fia_species_code",
        "decimalLatitude",
        "decimalLongitude",
    ]:
        thinned[
            column
        ] = pd.to_numeric(
            thinned[
                column
            ],
            errors="coerce",
        )

    thinned = thinned.dropna(
        subset=[
            "fia_species_code",
            "period",
            "decimalLatitude",
            "decimalLongitude",
        ]
    ).copy()

    thinned[
        "fia_species_code"
    ] = thinned[
        "fia_species_code"
    ].astype(int)

    thinned[
        "grid_lat_index"
    ] = np.floor(
        thinned[
            "decimalLatitude"
        ]
        / GRID_RESOLUTION
    ).astype(
        "int32"
    )

    thinned[
        "grid_lon_index"
    ] = np.floor(
        thinned[
            "decimalLongitude"
        ]
        / GRID_RESOLUTION
    ).astype(
        "int32"
    )

    thinned[
        "grid_id"
    ] = (
        thinned[
            "grid_lat_index"
        ].astype(str)
        + "_"
        + thinned[
            "grid_lon_index"
        ].astype(str)
    )

    effort_frame = thinned[
        thinned[
            "fia_species_code"
        ].isin(
            effort_codes
        )
    ].copy()

    cell_frame = (
        effort_frame.groupby(
            "grid_id",
            as_index=False,
        )
        .agg(
            pooled_effort=(
                "gbifID",
                "size",
            ),
            cell_latitude=(
                "decimalLatitude",
                "mean",
            ),
        )
    )

    cell_frame = cell_frame[
        cell_frame[
            "pooled_effort"
        ]
        >= MIN_AVAILABLE_CELL_EFFORT
    ].copy()

    minimum_required_cells = max(
        2
        * EFFORT_SUPPORT_SIZE,
        2
        * SPECIES_SUPPORT_SIZE,
    )

    if len(
        cell_frame
    ) < minimum_required_cells:
        raise ValueError(
            f"{state}: only {len(cell_frame)} eligible cells; "
            f"need at least {minimum_required_cells}."
        )

    cell_frame = (
        cell_frame.sort_values(
            "grid_id"
        )
        .reset_index(
            drop=True
        )
    )

    cell_frame[
        "cell_index"
    ] = np.arange(
        len(
            cell_frame
        )
    )

    species_inputs[
        "fia_species_code"
    ] = pd.to_numeric(
        species_inputs[
            "fia_species_code"
        ],
        errors="coerce",
    )

    species_inputs = species_inputs.dropna(
        subset=[
            "fia_species_code",
        ]
    ).copy()

    species_inputs[
        "fia_species_code"
    ] = species_inputs[
        "fia_species_code"
    ].astype(int)

    evaluation_codes = set(
        species_inputs[
            "fia_species_code"
        ]
    )

    benchmark = thinned[
        thinned[
            "fia_species_code"
        ].isin(
            evaluation_codes
        )
    ].copy()

    support_counts = (
        benchmark.groupby(
            [
                "fia_species_code",
                "period",
            ],
            as_index=False,
        )
        .agg(
            n_records=(
                "gbifID",
                "size",
            ),
        )
    )

    support_wide = support_counts.pivot(
        index="fia_species_code",
        columns="period",
        values="n_records",
    ).fillna(
        0
    )

    for period in [
        "early",
        "late",
    ]:
        if period not in support_wide.columns:
            support_wide[
                period
            ] = 0

    support_wide[
        "min_period_records"
    ] = support_wide[
        [
            "early",
            "late",
        ]
    ].min(
        axis=1
    )

    selected_codes = (
        support_wide.sort_values(
            [
                "min_period_records",
                "early",
                "late",
            ],
            ascending=False,
        )
        .head(
            SPECIES_PER_STATE
        )
        .index.astype(int)
        .tolist()
    )

    cell_ids = set(
        cell_frame[
            "grid_id"
        ]
    )

    species_rows = []

    for species_code in selected_codes:
        species = benchmark[
            benchmark[
                "fia_species_code"
            ]
            == species_code
        ].copy()

        species_cell = (
            species.groupby(
                "grid_id",
                as_index=False,
            )
            .agg(
                species_records=(
                    "gbifID",
                    "size",
                ),
                species_latitude=(
                    "decimalLatitude",
                    "mean",
                ),
            )
        )

        species_cell = species_cell[
            species_cell[
                "grid_id"
            ].isin(
                cell_ids
            )
        ].copy()

        if len(
            species_cell
        ) < 3:
            continue

        centre = weighted_mean(
            species_cell[
                "species_latitude"
            ],
            species_cell[
                "species_records"
            ],
        )

        variance = weighted_mean(
            (
                species_cell[
                    "species_latitude"
                ]
                - centre
            )
            ** 2,
            species_cell[
                "species_records"
            ],
        )

        base_width = float(
            np.sqrt(
                max(
                    variance,
                    0.08 ** 2,
                )
            )
        )

        species_rows.append(
            {
                "state_code": state,
                "fia_species_code": int(
                    species_code
                ),
                "pooled_centre": (
                    centre
                ),
                "base_width_degrees": (
                    base_width
                ),
                "min_period_records": float(
                    support_wide.loc[
                        species_code,
                        "min_period_records",
                    ]
                ),
            }
        )

    species_frame = pd.DataFrame(
        species_rows
    )

    if len(
        species_frame
    ) < SPECIES_PER_STATE:
        raise ValueError(
            f"{state}: only {len(species_frame)} prepared species; "
            f"need {SPECIES_PER_STATE}."
        )

    return (
        cell_frame,
        species_frame,
    )


# ============================================================
# Single replicate
# ============================================================
def simulate_replicate(
    rng,
    cell_frame,
    species_row,
    effort_overlap,
    species_overlap,
    drift_km_decade,
    sample_size,
    width_mode,
    width_factor,
):
    cell_indices = cell_frame[
        "cell_index"
    ].to_numpy(
        dtype=int
    )

    latitudes = cell_frame[
        "cell_latitude"
    ].to_numpy(
        dtype=float
    )

    base_effort = cell_frame[
        "pooled_effort"
    ].to_numpy(
        dtype=float
    )

    effort_pair = generate_effort_support_pair(
        rng=rng,
        cell_indices=cell_indices,
        latitudes=latitudes,
        base_effort=base_effort,
        requested_overlap=effort_overlap,
        drift_km_decade=drift_km_decade,
    )

    if effort_pair is None:
        return {
            "design_generated": False,
        }

    species_centre = float(
        species_row[
            "pooled_centre"
        ]
    )

    species_width = max(
        float(
            species_row[
                "base_width_degrees"
            ]
        )
        * width_factor,
        0.08,
    )

    latent_kernel = np.exp(
        -0.5
        * (
            (
                latitudes
                - species_centre
            )
            / species_width
        )
        ** 2
    )

    latent_kernel = (
        latent_kernel
        / max(
            latent_kernel.max(),
            1e-12,
        )
    )

    species_pair = generate_species_support_pair(
        rng=rng,
        cell_indices=cell_indices,
        latitudes=latitudes,
        latent_kernel=latent_kernel,
        requested_overlap=species_overlap,
    )

    if species_pair is None:
        return {
            "design_generated": False,
        }

    early_effort_counts = rng.multinomial(
        sample_size,
        effort_pair[
            "early_weights"
        ],
    )

    late_effort_counts = rng.multinomial(
        sample_size,
        effort_pair[
            "late_weights"
        ],
    )

    early_effort_full = np.zeros(
        len(
            cell_indices
        ),
        dtype=int,
    )

    late_effort_full = np.zeros(
        len(
            cell_indices
        ),
        dtype=int,
    )

    early_effort_full[
        effort_pair[
            "early_support"
        ]
    ] = early_effort_counts

    late_effort_full[
        effort_pair[
            "late_support"
        ]
    ] = late_effort_counts

    early_detection = np.zeros(
        len(
            cell_indices
        ),
        dtype=float,
    )

    late_detection = np.zeros(
        len(
            cell_indices
        ),
        dtype=float,
    )

    early_profile = species_pair[
        "early_profile"
    ]

    late_profile = species_pair[
        "late_profile"
    ]

    early_detection[
        species_pair[
            "early_support"
        ]
    ] = (
        DETECTION_BASELINE
        + DETECTION_AMPLITUDE
        * (
            early_profile
            / max(
                early_profile.max(),
                1e-12,
            )
        )
    )

    late_detection[
        species_pair[
            "late_support"
        ]
    ] = (
        DETECTION_BASELINE
        + DETECTION_AMPLITUDE
        * (
            late_profile
            / max(
                late_profile.max(),
                1e-12,
            )
        )
    )

    early_species_counts = rng.binomial(
        early_effort_full,
        np.clip(
            early_detection,
            0.0,
            0.95,
        ),
    )

    late_species_counts = rng.binomial(
        late_effort_full,
        np.clip(
            late_detection,
            0.0,
            0.95,
        ),
    )

    early_species_total = int(
        early_species_counts.sum()
    )

    late_species_total = int(
        late_species_counts.sum()
    )

    raw_estimable = bool(
        early_species_total > 0
        and late_species_total > 0
    )

    raw_error = np.nan
    raw_shift = np.nan

    if raw_estimable:
        early_raw_centre = weighted_mean(
            latitudes,
            early_species_counts,
        )

        late_raw_centre = weighted_mean(
            latitudes,
            late_species_counts,
        )

        raw_shift = (
            late_raw_centre
            - early_raw_centre
        )

        raw_error = abs(
            raw_shift
            * 111.32
            * 10.0
            / 7.0
        )

    stable_mask = (
        early_effort_full
        >= MIN_STABLE_CELL_EFFORT
    ) & (
        late_effort_full
        >= MIN_STABLE_CELL_EFFORT
    )

    stable_cell_count = int(
        stable_mask.sum()
    )

    corrected_estimable = bool(
        raw_estimable
        and stable_cell_count
        >= MIN_STABLE_CELLS
    )

    corrected_error = np.nan
    corrected_shift = np.nan
    gain = np.nan
    error_ratio = np.nan

    if corrected_estimable:
        early_relative_rate = (
            early_species_counts[
                stable_mask
            ]
            / early_effort_full[
                stable_mask
            ]
        )

        late_relative_rate = (
            late_species_counts[
                stable_mask
            ]
            / late_effort_full[
                stable_mask
            ]
        )

        early_rate_sum = float(
            early_relative_rate.sum()
        )

        late_rate_sum = float(
            late_relative_rate.sum()
        )

        if (
            early_rate_sum <= 0
            or late_rate_sum <= 0
        ):
            corrected_estimable = False
        else:
            early_corrected_centre = weighted_mean(
                latitudes[
                    stable_mask
                ],
                early_relative_rate,
            )

            late_corrected_centre = weighted_mean(
                latitudes[
                    stable_mask
                ],
                late_relative_rate,
            )

            if (
                not np.isfinite(
                    early_corrected_centre
                )
                or not np.isfinite(
                    late_corrected_centre
                )
            ):
                corrected_estimable = False
            else:
                corrected_shift = (
                    late_corrected_centre
                    - early_corrected_centre
                )

                corrected_error = abs(
                    corrected_shift
                    * 111.32
                    * 10.0
                    / 7.0
                )

                gain = (
                    raw_error
                    - corrected_error
                )

                error_ratio = (
                    corrected_error
                    / raw_error
                    if raw_error > 0
                    else np.nan
                )

    occupied_early = set(
        cell_indices[
            early_species_counts
            > 0
        ]
    )

    occupied_late = set(
        cell_indices[
            late_species_counts
            > 0
        ]
    )

    realized_occupied_jaccard = (
        set_jaccard(
            occupied_early,
            occupied_late,
        )
    )

    realized_early_effort_centre = weighted_mean(
        latitudes,
        early_effort_full,
    )

    realized_late_effort_centre = weighted_mean(
        latitudes,
        late_effort_full,
    )

    realized_effort_drift = (
        (
            realized_late_effort_centre
            - realized_early_effort_centre
        )
        * 111.32
        * 10.0
        / 7.0
    )

    return {
        "design_generated": True,
        "state_code": (
            species_row[
                "state_code"
            ]
        ),
        "fia_species_code": int(
            species_row[
                "fia_species_code"
            ]
        ),
        "effort_overlap_requested": (
            effort_overlap
        ),
        "effort_overlap_actual": (
            effort_pair[
                "actual_overlap"
            ]
        ),
        "species_overlap_requested": (
            species_overlap
        ),
        "species_overlap_actual": (
            species_pair[
                "actual_overlap"
            ]
        ),
        "drift_requested_km_decade": (
            drift_km_decade
        ),
        "expected_effort_drift_km_decade": (
            effort_pair[
                "expected_absolute_drift"
            ]
        ),
        "expected_signed_effort_drift": (
            effort_pair[
                "expected_signed_drift"
            ]
        ),
        "realized_effort_drift_km_decade": abs(
            realized_effort_drift
        ),
        "species_expected_centre_mismatch_degrees": (
            species_pair[
                "expected_centre_mismatch"
            ]
        ),
        "sample_size": (
            sample_size
        ),
        "width_mode": (
            width_mode
        ),
        "width_factor": (
            width_factor
        ),
        "raw_estimable": (
            raw_estimable
        ),
        "corrected_estimable": (
            corrected_estimable
        ),
        "stable_cell_count": (
            stable_cell_count
        ),
        "early_species_records": (
            early_species_total
        ),
        "late_species_records": (
            late_species_total
        ),
        "raw_shift_degrees": (
            raw_shift
        ),
        "corrected_shift_degrees": (
            corrected_shift
        ),
        "raw_error_km_decade": (
            raw_error
        ),
        "corrected_error_km_decade": (
            corrected_error
        ),
        "gain_km_decade": (
            gain
        ),
        "corrected_raw_error_ratio": (
            error_ratio
        ),
        "correction_helped": bool(
            corrected_estimable
            and np.isfinite(
                gain
            )
            and gain > 0
        ),
        "realized_occupied_jaccard": (
            realized_occupied_jaccard
        ),
    }


# ============================================================
# Partial rank association
# ============================================================
def residualize(
    values,
    controls,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    controls = np.asarray(
        controls,
        dtype=float,
    )

    design = np.column_stack(
        [
            np.ones(
                len(
                    values
                )
            ),
            controls,
        ]
    )

    coefficients, _, _, _ = (
        np.linalg.lstsq(
            design,
            values,
            rcond=None,
        )
    )

    return (
        values
        - design
        @ coefficients
    )


def partial_species_overlap_gain_rho(
    scenario_df,
):
    required = [
        "species_overlap_actual",
        "median_gain",
        "median_raw_error",
        "corrected_estimable_fraction",
        "effort_overlap_actual",
        "drift_requested_km_decade",
        "sample_size",
        "width_mode",
        "state_code",
        "fia_species_code",
    ]

    work = scenario_df[
        required
    ].dropna().copy()

    numeric_controls = pd.DataFrame(
        {
            "raw_error_rank": rankdata(
                work[
                    "median_raw_error"
                ],
                method="average",
            ),
            "estimability_rank": rankdata(
                work[
                    "corrected_estimable_fraction"
                ],
                method="average",
            ),
            "effort_overlap_rank": rankdata(
                work[
                    "effort_overlap_actual"
                ],
                method="average",
            ),
            "drift_rank": rankdata(
                work[
                    "drift_requested_km_decade"
                ],
                method="average",
            ),
            "sample_rank": rankdata(
                work[
                    "sample_size"
                ],
                method="average",
            ),
        },
        index=work.index,
    )

    categorical_controls = pd.get_dummies(
        work[
            [
                "width_mode",
                "state_code",
                "fia_species_code",
            ]
        ].astype(str),
        drop_first=True,
        dtype=float,
    )

    controls = pd.concat(
        [
            numeric_controls,
            categorical_controls,
        ],
        axis=1,
    ).to_numpy(
        dtype=float
    )

    feature_rank = rankdata(
        work[
            "species_overlap_actual"
        ],
        method="average",
    )

    outcome_rank = rankdata(
        work[
            "median_gain"
        ],
        method="average",
    )

    feature_residual = residualize(
        feature_rank,
        controls,
    )

    outcome_residual = residualize(
        outcome_rank,
        controls,
    )

    return float(
        np.corrcoef(
            feature_residual,
            outcome_residual,
        )[0, 1]
    )


# ============================================================
# Failure helper
# ============================================================
def print_failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## Orthogonal phase-diagram trend failure — {RUN_UTC}
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
    required_paths = [
        *THINNED_PATHS.values(),
        *SPECIES_INPUT_PATHS.values(),
        *EFFORT_PLAN_PATHS.values(),
    ]

    missing_paths = [
        str(
            path
        )
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing required inputs: "
            + " | ".join(
                missing_paths
            )
        )

    state_inputs = {}
    selected_species_parts = []

    for state in tqdm(
        STATES,
        desc="Preparing orthogonal phase-diagram inputs",
        unit="state",
    ):
        (
            cell_frame,
            species_frame,
        ) = build_state_inputs(
            state=state,
            thinned_path=THINNED_PATHS[
                state
            ],
            species_input_path=SPECIES_INPUT_PATHS[
                state
            ],
            effort_plan_path=EFFORT_PLAN_PATHS[
                state
            ],
        )

        state_inputs[
            state
        ] = {
            "cell_frame": (
                cell_frame
            ),
            "species_frame": (
                species_frame
            ),
        }

        selected_species_parts.append(
            species_frame
        )

    selected_species_df = pd.concat(
        selected_species_parts,
        ignore_index=True,
    )

    selected_species_df.to_csv(
        OUT_DIR
        / "selected_orthogonal_phase_species.csv",
        index=False,
    )

    print(
        "\nSELECTED SPECIES"
    )
    display(
        selected_species_df
    )

    total_replicates = (
        len(
            STATES
        )
        * SPECIES_PER_STATE
        * len(
            EFFORT_OVERLAP_LEVELS
        )
        * len(
            SPECIES_OVERLAP_LEVELS
        )
        * len(
            DRIFT_LEVELS_KM_DECADE
        )
        * len(
            SAMPLE_SIZES
        )
        * len(
            WIDTH_MODES
        )
        * REPLICATES
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []

    progress = tqdm(
        total=total_replicates,
        desc="Simulating orthogonal support phase diagram",
        unit="replicate",
    )

    for state in STATES:
        cell_frame = state_inputs[
            state
        ][
            "cell_frame"
        ]

        species_frame = state_inputs[
            state
        ][
            "species_frame"
        ]

        for _, species_row in species_frame.iterrows():
            for effort_overlap in EFFORT_OVERLAP_LEVELS:
                for species_overlap in SPECIES_OVERLAP_LEVELS:
                    for drift_level in DRIFT_LEVELS_KM_DECADE:
                        for sample_size in SAMPLE_SIZES:
                            for (
                                width_mode,
                                width_factor,
                            ) in WIDTH_MODES.items():
                                for replicate in range(
                                    REPLICATES
                                ):
                                    result = simulate_replicate(
                                        rng=rng,
                                        cell_frame=cell_frame,
                                        species_row=species_row,
                                        effort_overlap=effort_overlap,
                                        species_overlap=species_overlap,
                                        drift_km_decade=drift_level,
                                        sample_size=sample_size,
                                        width_mode=width_mode,
                                        width_factor=width_factor,
                                    )

                                    result[
                                        "replicate"
                                    ] = replicate

                                    rows.append(
                                        result
                                    )

                                    progress.update(
                                        1
                                    )

    progress.close()

    replicate_df = pd.DataFrame(
        rows
    )

    replicate_df.to_parquet(
        OUT_DIR
        / "orthogonal_phase_replicates.parquet",
        index=False,
    )

    design_generation_fraction = float(
        replicate_df[
            "design_generated"
        ].mean()
    )

    valid_replicates = replicate_df[
        replicate_df[
            "design_generated"
        ].fillna(False)
    ].copy()

    if len(
        valid_replicates
    ) == 0:
        raise ValueError(
            "No valid redesigned simulation replicates were generated."
        )

    # --------------------------------------------------------
    # Calibration checks
    # --------------------------------------------------------
    zero_drift = valid_replicates[
        valid_replicates[
            "drift_requested_km_decade"
        ]
        == 0
    ]

    zero_expected_drift_max = float(
        zero_drift[
            "expected_effort_drift_km_decade"
        ].abs().max()
    )

    species_centre_mismatch_max = float(
        valid_replicates[
            "species_expected_centre_mismatch_degrees"
        ].max()
    )

    expected_drift_error = (
        valid_replicates[
            "expected_effort_drift_km_decade"
        ]
        - valid_replicates[
            "drift_requested_km_decade"
        ]
    ).abs()

    expected_drift_max_error = float(
        expected_drift_error.max()
    )

    # --------------------------------------------------------
    # Scenario summaries
    # --------------------------------------------------------
    scenario_columns = [
        "state_code",
        "fia_species_code",
        "effort_overlap_requested",
        "effort_overlap_actual",
        "species_overlap_requested",
        "species_overlap_actual",
        "drift_requested_km_decade",
        "sample_size",
        "width_mode",
    ]

    scenario_df = (
        valid_replicates.groupby(
            scenario_columns,
            as_index=False,
        )
        .agg(
            n_replicates=(
                "replicate",
                "size",
            ),
            raw_estimable_fraction=(
                "raw_estimable",
                "mean",
            ),
            corrected_estimable_fraction=(
                "corrected_estimable",
                "mean",
            ),
            correction_help_fraction=(
                "correction_helped",
                "mean",
            ),
            median_raw_error=(
                "raw_error_km_decade",
                "median",
            ),
            median_corrected_error=(
                "corrected_error_km_decade",
                "median",
            ),
            median_gain=(
                "gain_km_decade",
                "median",
            ),
            median_error_ratio=(
                "corrected_raw_error_ratio",
                "median",
            ),
            median_realized_effort_drift=(
                "realized_effort_drift_km_decade",
                "median",
            ),
            median_realized_occupied_jaccard=(
                "realized_occupied_jaccard",
                "median",
            ),
            median_stable_cells=(
                "stable_cell_count",
                "median",
            ),
            median_early_species_records=(
                "early_species_records",
                "median",
            ),
            median_late_species_records=(
                "late_species_records",
                "median",
            ),
        )
    )

    scenario_df.to_csv(
        OUT_DIR
        / "orthogonal_phase_scenario_summary.csv",
        index=False,
    )

    aggregate_df = (
        valid_replicates.groupby(
            [
                "effort_overlap_requested",
                "effort_overlap_actual",
                "species_overlap_requested",
                "species_overlap_actual",
                "drift_requested_km_decade",
                "sample_size",
                "width_mode",
            ],
            as_index=False,
        )
        .agg(
            n_replicates=(
                "replicate",
                "size",
            ),
            corrected_estimable_fraction=(
                "corrected_estimable",
                "mean",
            ),
            correction_help_fraction=(
                "correction_helped",
                "mean",
            ),
            median_raw_error=(
                "raw_error_km_decade",
                "median",
            ),
            median_corrected_error=(
                "corrected_error_km_decade",
                "median",
            ),
            median_gain=(
                "gain_km_decade",
                "median",
            ),
            median_error_ratio=(
                "corrected_raw_error_ratio",
                "median",
            ),
            median_realized_effort_drift=(
                "realized_effort_drift_km_decade",
                "median",
            ),
            median_realized_occupied_jaccard=(
                "realized_occupied_jaccard",
                "median",
            ),
        )
    )

    aggregate_df.to_csv(
        OUT_DIR
        / "orthogonal_phase_aggregate.csv",
        index=False,
    )

    print(
        "\nORTHOGONAL PHASE AGGREGATE"
    )
    display(
        aggregate_df
    )

    # --------------------------------------------------------
    # Axis-specific diagnostics
    # --------------------------------------------------------
    (
        effort_estimability_rho,
        effort_estimability_p,
    ) = safe_spearman(
        scenario_df[
            "effort_overlap_actual"
        ],
        scenario_df[
            "corrected_estimable_fraction"
        ],
    )

    error_scenarios = scenario_df.dropna(
        subset=[
            "median_corrected_error",
        ]
    ).copy()

    (
        species_error_rho,
        species_error_p,
    ) = safe_spearman(
        error_scenarios[
            "species_overlap_actual"
        ],
        error_scenarios[
            "median_corrected_error"
        ],
    )

    (
        species_gain_rho,
        species_gain_p,
    ) = safe_spearman(
        error_scenarios[
            "species_overlap_actual"
        ],
        error_scenarios[
            "median_gain"
        ],
    )

    partial_species_gain_rho = (
        partial_species_overlap_gain_rho(
            error_scenarios
        )
    )

    # --------------------------------------------------------
    # Within-stratum monotonicity
    # --------------------------------------------------------
    effort_monotonic_rows = []

    for keys, group in scenario_df.groupby(
        [
            "state_code",
            "fia_species_code",
            "species_overlap_actual",
            "drift_requested_km_decade",
            "sample_size",
            "width_mode",
        ]
    ):
        (
            rho,
            p_value,
        ) = safe_spearman(
            group[
                "effort_overlap_actual"
            ],
            group[
                "corrected_estimable_fraction"
            ],
        )

        effort_monotonic_rows.append(
            {
                "state_code": (
                    keys[0]
                ),
                "fia_species_code": (
                    keys[1]
                ),
                "species_overlap_actual": (
                    keys[2]
                ),
                "drift_requested_km_decade": (
                    keys[3]
                ),
                "sample_size": (
                    keys[4]
                ),
                "width_mode": (
                    keys[5]
                ),
                "rho": rho,
                "p_value": (
                    p_value
                ),
                "expected_direction": bool(
                    np.isfinite(
                        rho
                    )
                    and rho > 0
                ),
                "strong_expected_direction": bool(
                    np.isfinite(
                        rho
                    )
                    and rho >= 0.50
                ),
            }
        )

    effort_monotonic_df = pd.DataFrame(
        effort_monotonic_rows
    )

    species_monotonic_rows = []

    for keys, group in error_scenarios.groupby(
        [
            "state_code",
            "fia_species_code",
            "effort_overlap_actual",
            "drift_requested_km_decade",
            "sample_size",
            "width_mode",
        ]
    ):
        (
            rho,
            p_value,
        ) = safe_spearman(
            group[
                "species_overlap_actual"
            ],
            group[
                "median_corrected_error"
            ],
        )

        species_monotonic_rows.append(
            {
                "state_code": (
                    keys[0]
                ),
                "fia_species_code": (
                    keys[1]
                ),
                "effort_overlap_actual": (
                    keys[2]
                ),
                "drift_requested_km_decade": (
                    keys[3]
                ),
                "sample_size": (
                    keys[4]
                ),
                "width_mode": (
                    keys[5]
                ),
                "rho": rho,
                "p_value": (
                    p_value
                ),
                "expected_direction": bool(
                    np.isfinite(
                        rho
                    )
                    and rho < 0
                ),
                "strong_expected_direction": bool(
                    np.isfinite(
                        rho
                    )
                    and rho <= -0.50
                ),
            }
        )

    species_monotonic_df = pd.DataFrame(
        species_monotonic_rows
    )

    effort_monotonic_fraction = float(
        effort_monotonic_df[
            "strong_expected_direction"
        ].mean()
    )

    species_monotonic_fraction = float(
        species_monotonic_df[
            "strong_expected_direction"
        ].mean()
    )

    effort_monotonic_df.to_csv(
        OUT_DIR
        / "effort_overlap_estimability_monotonicity.csv",
        index=False,
    )

    species_monotonic_df.to_csv(
        OUT_DIR
        / "species_overlap_error_monotonicity.csv",
        index=False,
    )

    # --------------------------------------------------------
    # High-low contrasts
    # --------------------------------------------------------
    low_effort = valid_replicates[
        valid_replicates[
            "effort_overlap_requested"
        ]
        <= 0.20
    ]

    high_effort = valid_replicates[
        valid_replicates[
            "effort_overlap_requested"
        ]
        >= 0.80
    ]

    low_effort_estimability = float(
        low_effort[
            "corrected_estimable"
        ].mean()
    )

    high_effort_estimability = float(
        high_effort[
            "corrected_estimable"
        ].mean()
    )

    effort_estimability_delta = (
        high_effort_estimability
        - low_effort_estimability
    )

    low_species = valid_replicates[
        (
            valid_replicates[
                "species_overlap_requested"
            ]
            <= 0.20
        )
        & valid_replicates[
            "corrected_estimable"
        ].fillna(False)
    ]

    high_species = valid_replicates[
        (
            valid_replicates[
                "species_overlap_requested"
            ]
            >= 0.80
        )
        & valid_replicates[
            "corrected_estimable"
        ].fillna(False)
    ]

    low_species_corrected_error = float(
        low_species[
            "corrected_error_km_decade"
        ].median()
    )

    high_species_corrected_error = float(
        high_species[
            "corrected_error_km_decade"
        ].median()
    )

    low_minus_high_corrected_error = (
        low_species_corrected_error
        - high_species_corrected_error
    )

    calibration_pass = bool(
        zero_expected_drift_max
        <= MAX_EXPECTED_ZERO_DRIFT_ERROR
        and species_centre_mismatch_max
        <= MAX_SPECIES_CENTRE_MISMATCH
        and expected_drift_max_error
        <= MAX_EXPECTED_ZERO_DRIFT_ERROR
    )

    diagnostic_df = pd.DataFrame(
        [
            {
                "design_generation_fraction": (
                    design_generation_fraction
                ),
                "zero_expected_drift_max_error": (
                    zero_expected_drift_max
                ),
                "species_centre_mismatch_max_degrees": (
                    species_centre_mismatch_max
                ),
                "expected_drift_max_error": (
                    expected_drift_max_error
                ),
                "calibration_pass": (
                    calibration_pass
                ),
                "effort_overlap_estimability_rho": (
                    effort_estimability_rho
                ),
                "effort_overlap_estimability_p": (
                    effort_estimability_p
                ),
                "species_overlap_corrected_error_rho": (
                    species_error_rho
                ),
                "species_overlap_corrected_error_p": (
                    species_error_p
                ),
                "species_overlap_gain_rho": (
                    species_gain_rho
                ),
                "species_overlap_gain_p": (
                    species_gain_p
                ),
                "partial_species_overlap_gain_rho": (
                    partial_species_gain_rho
                ),
                "effort_monotonic_fraction": (
                    effort_monotonic_fraction
                ),
                "species_monotonic_fraction": (
                    species_monotonic_fraction
                ),
                "low_effort_estimability": (
                    low_effort_estimability
                ),
                "high_effort_estimability": (
                    high_effort_estimability
                ),
                "effort_estimability_delta": (
                    effort_estimability_delta
                ),
                "low_species_corrected_error": (
                    low_species_corrected_error
                ),
                "high_species_corrected_error": (
                    high_species_corrected_error
                ),
                "low_minus_high_corrected_error": (
                    low_minus_high_corrected_error
                ),
            }
        ]
    )

    diagnostic_df.to_csv(
        OUT_DIR
        / "orthogonal_phase_trend_diagnostic.csv",
        index=False,
    )

    print(
        "\nORTHOGONAL PHASE TREND DIAGNOSTIC"
    )
    display(
        diagnostic_df
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------
    passed = bool(
        calibration_pass
        and np.isfinite(
            effort_estimability_rho
        )
        and effort_estimability_rho
        >= MIN_EFFORT_ESTIMABILITY_RHO
        and effort_estimability_p
        <= MAX_EFFORT_ESTIMABILITY_P
        and np.isfinite(
            species_error_rho
        )
        and species_error_rho
        <= MAX_SPECIES_OVERLAP_ERROR_RHO
        and species_error_p
        <= MAX_SPECIES_OVERLAP_ERROR_P
        and effort_monotonic_fraction
        >= MIN_EFFORT_MONOTONIC_FRACTION
        and species_monotonic_fraction
        >= MIN_SPECIES_MONOTONIC_FRACTION
        and effort_estimability_delta
        >= MIN_HIGH_LOW_ESTIMABILITY_DELTA
        and low_minus_high_corrected_error
        >= MIN_LOW_HIGH_CORRECTED_ERROR_DELTA
    )

    if passed:
        status = (
            "GO_FOR_FULL_ORTHOGONAL_SUPPORT_PHASE_DIAGRAM"
        )

        next_step = (
            "Run a locked full confirmation with 200 replicates, finer "
            "effort/species overlap levels, species-region cluster uncertainty "
            "and negative controls. Keep estimability and corrected error as "
            "separate primary outcomes; do not optimize a cutoff."
        )

    else:
        status = (
            "ORTHOGONAL_SUPPORT_PHASE_TREND_INCOMPLETE"
        )

        next_step = (
            "Do not expand to full simulation. Inspect which axis failed: "
            "effort-overlap estimability, species-overlap corrected error, "
            "calibration or monotonicity."
        )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    figure_paths = []

    for drift_level in DRIFT_LEVELS_KM_DECADE:
        figure_data = aggregate_df[
            (
                aggregate_df[
                    "drift_requested_km_decade"
                ]
                == drift_level
            )
            & (
                aggregate_df[
                    "sample_size"
                ]
                == max(
                    SAMPLE_SIZES
                )
            )
            & (
                aggregate_df[
                    "width_mode"
                ]
                == "broad"
            )
        ].copy()

        estimability_pivot = figure_data.pivot(
            index="effort_overlap_actual",
            columns="species_overlap_actual",
            values="corrected_estimable_fraction",
        ).sort_index()

        plt.figure(
            figsize=(8, 5)
        )

        image = plt.imshow(
            estimability_pivot.to_numpy(
                dtype=float
            ),
            aspect="auto",
            origin="lower",
            vmin=0,
            vmax=1,
        )

        plt.colorbar(
            image,
            label=(
                "Corrected estimability probability"
            ),
        )

        plt.xticks(
            np.arange(
                len(
                    estimability_pivot.columns
                )
            ),
            [
                f"{value:.2f}"
                for value in estimability_pivot.columns
            ],
        )

        plt.yticks(
            np.arange(
                len(
                    estimability_pivot.index
                )
            ),
            [
                f"{value:.2f}"
                for value in estimability_pivot.index
            ],
        )

        plt.xlabel(
            "Species observation-support Jaccard"
        )

        plt.ylabel(
            "Effort-frame Jaccard"
        )

        plt.title(
            "Correction estimability "
            f"(drift={drift_level:.0f} km/decade)"
        )

        plt.tight_layout()

        estimability_figure = (
            OUT_DIR
            / (
                "estimability_phase_"
                f"drift_{int(drift_level)}.png"
            )
        )

        plt.savefig(
            estimability_figure,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

        display(
            Image(
                filename=str(
                    estimability_figure
                )
            )
        )

        figure_paths.append(
            estimability_figure
        )

        error_pivot = figure_data.pivot(
            index="effort_overlap_actual",
            columns="species_overlap_actual",
            values="median_corrected_error",
        ).sort_index()

        plt.figure(
            figsize=(8, 5)
        )

        image = plt.imshow(
            error_pivot.to_numpy(
                dtype=float
            ),
            aspect="auto",
            origin="lower",
        )

        plt.colorbar(
            image,
            label=(
                "Median corrected error (km/decade)"
            ),
        )

        plt.xticks(
            np.arange(
                len(
                    error_pivot.columns
                )
            ),
            [
                f"{value:.2f}"
                for value in error_pivot.columns
            ],
        )

        plt.yticks(
            np.arange(
                len(
                    error_pivot.index
                )
            ),
            [
                f"{value:.2f}"
                for value in error_pivot.index
            ],
        )

        plt.xlabel(
            "Species observation-support Jaccard"
        )

        plt.ylabel(
            "Effort-frame Jaccard"
        )

        plt.title(
            "Corrected-error phase surface "
            f"(drift={drift_level:.0f} km/decade)"
        )

        plt.tight_layout()

        error_figure = (
            OUT_DIR
            / (
                "corrected_error_phase_"
                f"drift_{int(drift_level)}.png"
            )
        )

        plt.savefig(
            error_figure,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

        display(
            Image(
                filename=str(
                    error_figure
                )
            )
        )

        figure_paths.append(
            error_figure
        )

    effort_line = (
        scenario_df.groupby(
            "effort_overlap_actual",
            as_index=False,
        )
        .agg(
            estimability=(
                "corrected_estimable_fraction",
                "mean",
            ),
        )
        .sort_values(
            "effort_overlap_actual"
        )
    )

    species_line = (
        error_scenarios.groupby(
            "species_overlap_actual",
            as_index=False,
        )
        .agg(
            corrected_error=(
                "median_corrected_error",
                "median",
            ),
            gain=(
                "median_gain",
                "median",
            ),
        )
        .sort_values(
            "species_overlap_actual"
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        effort_line[
            "effort_overlap_actual"
        ],
        effort_line[
            "estimability"
        ],
        marker="o",
    )

    plt.xlabel(
        "Effort-frame Jaccard"
    )

    plt.ylabel(
        "Corrected estimability probability"
    )

    plt.title(
        "Effort overlap controls whether correction is estimable"
    )

    plt.tight_layout()

    effort_figure = (
        OUT_DIR
        / "effort_overlap_estimability_trend.png"
    )

    plt.savefig(
        effort_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    display(
        Image(
            filename=str(
                effort_figure
            )
        )
    )

    figure_paths.append(
        effort_figure
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        species_line[
            "species_overlap_actual"
        ],
        species_line[
            "corrected_error"
        ],
        marker="o",
        label="Corrected error",
    )

    plt.plot(
        species_line[
            "species_overlap_actual"
        ],
        species_line[
            "gain"
        ],
        marker="o",
        label="Correction gain",
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "Species observation-support Jaccard"
    )

    plt.ylabel(
        "km/decade"
    )

    plt.title(
        "Species support overlap controls correction transportability"
    )

    plt.legend()
    plt.tight_layout()

    species_figure = (
        OUT_DIR
        / "species_overlap_error_gain_trend.png"
    )

    plt.savefig(
        species_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    display(
        Image(
            filename=str(
                species_figure
            )
        )
    )

    figure_paths.append(
        species_figure
    )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    append_readme(
        f"""

## Orthogonal support-overlap phase-diagram trend — {RUN_UTC}

### Redesign
Effort-frame overlap, species-specific observation-support overlap and expected
effort-centre drift were generated independently. The latent ecological
distribution was stationary.

### Formal spatial frames
- GBIF DOI: {GBIF_DOI}
- Regions: {STATES}
- Selected species-region units:
  {len(selected_species_df)}
- Grid:
  {GRID_RESOLUTION} degrees
- Replicates per scenario:
  {REPLICATES}
- Total requested replicates:
  {total_replicates}
- Design generation fraction:
  {design_generation_fraction}
- GPU: not used

### Manipulated factors
- Effort overlaps:
  {EFFORT_OVERLAP_LEVELS}
- Species-support overlaps:
  {SPECIES_OVERLAP_LEVELS}
- Expected drift:
  {DRIFT_LEVELS_KM_DECADE} km/decade
- Sample sizes:
  {SAMPLE_SIZES}
- Width modes:
  {list(WIDTH_MODES.keys())}

### Calibration
- Maximum zero expected drift error:
  {zero_expected_drift_max}
- Maximum expected drift calibration error:
  {expected_drift_max_error}
- Maximum species expected-centre mismatch:
  {species_centre_mismatch_max}
- Calibration passed:
  {calibration_pass}

### Axis-specific trend results
- Effort-overlap/estimability rho:
  {effort_estimability_rho}
- Effort-overlap/estimability p:
  {effort_estimability_p}
- Species-overlap/corrected-error rho:
  {species_error_rho}
- Species-overlap/corrected-error p:
  {species_error_p}
- Species-overlap/gain rho:
  {species_gain_rho}
- Partial species-overlap/gain rho:
  {partial_species_gain_rho}
- Effort monotonic strata fraction:
  {effort_monotonic_fraction}
- Species monotonic strata fraction:
  {species_monotonic_fraction}
- High-minus-low effort estimability:
  {effort_estimability_delta}
- Low-minus-high species corrected error:
  {low_minus_high_corrected_error}

### Decision
- Status: **{status}**
- Passed:
  {passed}
- Next step:
  {next_step}

### Boundary
This remains a trend-stage controlled experiment. Effort overlap is interpreted
as an estimability condition, while species support overlap is interpreted as a
transportability condition. No binary overlap cutoff is selected.
"""
    )

    # --------------------------------------------------------
    # Compact output
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
        f"GBIF_DOWNLOAD_DOI: {GBIF_DOI}"
    )
    print(
        f"N_SELECTED_SPECIES_REGION_UNITS: "
        f"{len(selected_species_df)}"
    )
    print(
        f"TOTAL_REQUESTED_REPLICATES: "
        f"{total_replicates:,}"
    )
    print(
        f"VALID_REPLICATES: "
        f"{len(valid_replicates):,}"
    )
    print(
        f"DESIGN_GENERATION_FRACTION: "
        f"{design_generation_fraction}"
    )
    print(
        f"ZERO_EXPECTED_DRIFT_MAX_ERROR: "
        f"{zero_expected_drift_max}"
    )
    print(
        f"EXPECTED_DRIFT_MAX_ERROR: "
        f"{expected_drift_max_error}"
    )
    print(
        "SPECIES_EXPECTED_CENTRE_MISMATCH_MAX: "
        f"{species_centre_mismatch_max}"
    )
    print(
        f"CALIBRATION_PASSED: "
        f"{calibration_pass}"
    )
    print(
        "EFFORT_OVERLAP_ESTIMABILITY_RHO: "
        f"{effort_estimability_rho}"
    )
    print(
        "EFFORT_OVERLAP_ESTIMABILITY_P: "
        f"{effort_estimability_p}"
    )
    print(
        "SPECIES_OVERLAP_CORRECTED_ERROR_RHO: "
        f"{species_error_rho}"
    )
    print(
        "SPECIES_OVERLAP_CORRECTED_ERROR_P: "
        f"{species_error_p}"
    )
    print(
        "SPECIES_OVERLAP_GAIN_RHO: "
        f"{species_gain_rho}"
    )
    print(
        "SPECIES_OVERLAP_GAIN_P: "
        f"{species_gain_p}"
    )
    print(
        "PARTIAL_SPECIES_OVERLAP_GAIN_RHO: "
        f"{partial_species_gain_rho}"
    )
    print(
        "EFFORT_MONOTONIC_STRATA_FRACTION: "
        f"{effort_monotonic_fraction}"
    )
    print(
        "SPECIES_MONOTONIC_STRATA_FRACTION: "
        f"{species_monotonic_fraction}"
    )
    print(
        f"LOW_EFFORT_ESTIMABILITY: "
        f"{low_effort_estimability}"
    )
    print(
        f"HIGH_EFFORT_ESTIMABILITY: "
        f"{high_effort_estimability}"
    )
    print(
        f"EFFORT_ESTIMABILITY_DELTA: "
        f"{effort_estimability_delta}"
    )
    print(
        f"LOW_SPECIES_CORRECTED_ERROR: "
        f"{low_species_corrected_error}"
    )
    print(
        f"HIGH_SPECIES_CORRECTED_ERROR: "
        f"{high_species_corrected_error}"
    )
    print(
        "LOW_MINUS_HIGH_SPECIES_CORRECTED_ERROR: "
        f"{low_minus_high_corrected_error}"
    )
    print(
        "FIGURE_PATHS: "
        + json.dumps(
            [
                str(
                    path
                )
                for path in figure_paths
            ],
            ensure_ascii=False,
        )
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
        "CELL22C_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed diagnostic table. "
            "Do not expand to the full phase diagram."
        ),
    )
