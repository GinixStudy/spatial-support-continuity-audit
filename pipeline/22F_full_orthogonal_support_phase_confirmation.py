
# Cell 22F — Full orthogonal support-overlap phase-diagram confirmation
#
# IMPORTANT
# ---------
# This cell reads and enforces the frozen plan created by Cell 22E.
# It does not change factor levels, species, thresholds or random seed.
#
# Full workload
# -------------
# - 576,000 core orthogonal-grid replicates
# - 24,000 targeted marginal-curve replicates
# - 5,000 within-scenario label permutations per primary axis
# - 5,000 species-region-cluster bootstrap replicates
#
# Frozen scientific questions
# ---------------------------
# 1. Does effort-frame overlap govern whether correction is estimable?
# 2. Does species-specific observation-support overlap govern corrected error?
#
# Primary outcomes
# ----------------
# Effort axis:
#   corrected_estimable
#
# Species axis:
#   corrected_error_km_decade
#
# Secondary outcomes
# ------------------
# raw error, gain, error ratio, correction-help probability and stable cells.
#
# No binary cell-Jaccard cutoff is selected.

from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import math
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr
from tqdm.auto import tqdm
from IPython.display import display, Image

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(
    "/kaggle/working/fia_temporal_observation_drift"
)

DERIVED_DIR = BASE_DIR / "derived"

PLAN_DIR = (
    DERIVED_DIR
    / "full_orthogonal_phase_diagram_plan"
)

PLAN_PATH = (
    PLAN_DIR
    / "full_orthogonal_phase_confirmation_plan.json"
)

TREND_DIR = (
    DERIVED_DIR
    / "orthogonal_support_overlap_phase_diagram_trend"
)

SELECTED_SPECIES_PATH = (
    TREND_DIR
    / "selected_orthogonal_phase_species.csv"
)

FORMAL_DIR = (
    DERIVED_DIR
    / "registered_gbif_formal_parity"
)

OUT_DIR = (
    DERIVED_DIR
    / "full_orthogonal_support_phase_confirmation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CORE_REPLICATE_PATH = (
    OUT_DIR
    / "full_core_phase_replicates.parquet"
)

MARGINAL_REPLICATE_PATH = (
    OUT_DIR
    / "full_marginal_phase_replicates.parquet"
)

THINNED_PATHS = {
    state: (
        FORMAL_DIR
        / f"{state}_formal_thinned.parquet"
    )
    for state in [
        "PA",
        "VA",
        "NC",
    ]
}

EFFORT_PLAN_PATHS = {
    "PA": (
        DERIVED_DIR
        / "pa_expanded_tree_taxon_audit"
        / "selected_expanded_tree_taxa.csv"
    ),
    "VA": (
        DERIVED_DIR
        / "va_confirmation_dataset"
        / "va_locked_selected_taxa.csv"
    ),
    "NC": (
        DERIVED_DIR
        / "nc_final_confirmation_dataset"
        / "nc_locked_selected_taxa.csv"
    ),
}

RUN_UTC = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("PLAN_PATH:", PLAN_PATH)
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")


# ============================================================
# Read and enforce frozen plan
# ============================================================
if not PLAN_PATH.exists():
    raise FileNotFoundError(
        f"Frozen plan not found: {PLAN_PATH}"
    )

plan = json.loads(
    PLAN_PATH.read_text(
        encoding="utf-8"
    )
)

if plan.get(
    "plan_status"
) != "FROZEN":
    raise RuntimeError(
        "The phase-diagram plan is not frozen."
    )

STATES = list(
    plan[
        "formal_data"
    ][
        "states"
    ]
)

SPECIES_PER_STATE = int(
    plan[
        "formal_data"
    ][
        "species_per_state"
    ]
)

N_SPECIES_REGION_UNITS = int(
    plan[
        "formal_data"
    ][
        "species_region_units"
    ]
)

RANDOM_SEED = int(
    plan[
        "random_seed"
    ]
)

CORE_EFFORT_OVERLAPS = [
    float(
        value
    )
    for value in plan[
        "core_design"
    ][
        "effort_overlap_levels"
    ]
]

CORE_SPECIES_OVERLAPS = [
    float(
        value
    )
    for value in plan[
        "core_design"
    ][
        "species_overlap_levels"
    ]
]

DRIFT_LEVELS_KM_DECADE = [
    float(
        value
    )
    for value in plan[
        "core_design"
    ][
        "drift_levels_km_decade"
    ]
]

SAMPLE_SIZES = [
    int(
        value
    )
    for value in plan[
        "core_design"
    ][
        "sample_sizes"
    ]
]

WIDTH_MODES = {
    str(
        key
    ): float(
        value
    )
    for key, value in plan[
        "core_design"
    ][
        "width_modes"
    ].items()
}

REPLICATES_PER_CORE_SCENARIO = int(
    plan[
        "core_design"
    ][
        "replicates_per_scenario"
    ]
)

MARGINAL_REFINEMENT_LEVELS = [
    float(
        value
    )
    for value in plan[
        "marginal_refinement"
    ][
        "levels"
    ]
]

MARGINAL_FIXED_SETTINGS = dict(
    plan[
        "marginal_refinement"
    ][
        "fixed_settings"
    ]
)

REPLICATES_PER_MARGINAL_SCENARIO = int(
    plan[
        "marginal_refinement"
    ][
        "replicates_per_scenario"
    ]
)

GRID_RESOLUTION = float(
    plan[
        "estimator_settings"
    ][
        "grid_resolution_degrees"
    ]
)

EFFORT_SUPPORT_SIZE = int(
    plan[
        "estimator_settings"
    ][
        "effort_support_size_cells"
    ]
)

SPECIES_SUPPORT_SIZE = int(
    plan[
        "estimator_settings"
    ][
        "species_support_size_cells"
    ]
)

MIN_STABLE_CELL_EFFORT = int(
    plan[
        "estimator_settings"
    ][
        "minimum_stable_cell_effort"
    ]
)

MIN_STABLE_CELLS = int(
    plan[
        "estimator_settings"
    ][
        "minimum_stable_cells"
    ]
)

N_CLUSTER_BOOTSTRAP = int(
    plan[
        "inference"
    ][
        "species_region_cluster_bootstrap"
    ]
)

N_LABEL_PERMUTATIONS = int(
    plan[
        "inference"
    ][
        "scenario_label_permutations"
    ]
)

SUCCESS = dict(
    plan[
        "success_criteria"
    ]
)

EXPECTED_CORE_TOTAL = int(
    plan[
        "task_scale"
    ][
        "core_total_replicates"
    ]
)

EXPECTED_MARGINAL_TOTAL = int(
    plan[
        "task_scale"
    ][
        "marginal_total_replicates"
    ]
)

EXPECTED_TOTAL = int(
    plan[
        "task_scale"
    ][
        "total_replicates"
    ]
)

GBIF_DOI = plan[
    "formal_data"
][
    "gbif_doi"
]

# Fixed settings inherited from the successful trend experiment.
MIN_AVAILABLE_CELL_EFFORT = 5
DETECTION_BASELINE = 0.01
DETECTION_AMPLITUDE = 0.34
MAX_SUPPORT_ATTEMPTS = 250
BISECTION_ITERATIONS = 100

PARQUET_BUFFER_ROWS = 10_000

print("PLAN_STATUS: FROZEN")
print("EXPECTED_CORE_TOTAL:", f"{EXPECTED_CORE_TOTAL:,}")
print("EXPECTED_MARGINAL_TOTAL:", f"{EXPECTED_MARGINAL_TOTAL:,}")
print("EXPECTED_TOTAL:", f"{EXPECTED_TOTAL:,}")
print("N_LABEL_PERMUTATIONS:", f"{N_LABEL_PERMUTATIONS:,}")
print("N_CLUSTER_BOOTSTRAP:", f"{N_CLUSTER_BOOTSTRAP:,}")


# ============================================================
# Helpers
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
        return series.fillna(
            False
        )

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

    if size > len(
        values
    ):
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
            len(
                values
            ),
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


def safe_spearman(
    x,
    y,
    minimum_rows=5,
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
        len(
            frame
        )
        < minimum_rows
        or frame[
            "x"
        ].nunique()
        < 2
        or frame[
            "y"
        ].nunique()
        < 2
    ):
        return (
            np.nan,
            np.nan,
        )

    result = spearmanr(
        frame[
            "x"
        ],
        frame[
            "y"
        ],
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


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


# ============================================================
# Buffered Parquet writer
# ============================================================
class BufferedParquetWriter:
    def __init__(
        self,
        path,
        buffer_rows=10_000,
    ):
        self.path = Path(
            path
        )

        if self.path.exists():
            self.path.unlink()

        self.buffer_rows = int(
            buffer_rows
        )

        self.buffer = []
        self.writer = None
        self.rows_written = 0

    def add(
        self,
        row,
    ):
        self.buffer.append(
            row
        )

        if len(
            self.buffer
        ) >= self.buffer_rows:
            self.flush()

    def flush(
        self,
    ):
        if not self.buffer:
            return

        frame = pd.DataFrame(
            self.buffer
        )

        table = pa.Table.from_pandas(
            frame,
            preserve_index=False,
        )

        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                table.schema,
                compression="snappy",
            )

        self.writer.write_table(
            table
        )

        self.rows_written += len(
            frame
        )

        self.buffer.clear()

    def close(
        self,
    ):
        self.flush()

        if self.writer is not None:
            self.writer.close()
            self.writer = None


# ============================================================
# Exponential-tilt calibration
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

    def compute(
        beta,
    ):
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
            _,
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

    return compute(
        final_beta
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

        outside_cells = cell_indices[
            ~np.isin(
                cell_indices,
                early_support,
            )
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
                "expected_absolute_drift": abs(
                    expected_signed_drift
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

        outside_cells = cell_indices[
            ~np.isin(
                cell_indices,
                early_support,
            )
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
# Formal state frame preparation
# ============================================================
def build_cell_frame(
    state,
    thinned_path,
    effort_plan_path,
):
    thinned = pd.read_parquet(
        thinned_path
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

    return cell_frame


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

        if (
            early_relative_rate.sum()
            <= 0
            or late_relative_rate.sum()
            <= 0
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

    realized_occupied_jaccard = set_jaccard(
        occupied_early,
        occupied_late,
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
        "state_code": str(
            species_row[
                "state_code"
            ]
        ),
        "fia_species_code": int(
            species_row[
                "fia_species_code"
            ]
        ),
        "species_region_cluster": (
            str(
                species_row[
                    "state_code"
                ]
            )
            + "_"
            + str(
                int(
                    species_row[
                        "fia_species_code"
                    ]
                )
            )
        ),
        "effort_overlap_requested": float(
            effort_overlap
        ),
        "effort_overlap_actual": float(
            effort_pair[
                "actual_overlap"
            ]
        ),
        "species_overlap_requested": float(
            species_overlap
        ),
        "species_overlap_actual": float(
            species_pair[
                "actual_overlap"
            ]
        ),
        "drift_requested_km_decade": float(
            drift_km_decade
        ),
        "expected_effort_drift_km_decade": float(
            effort_pair[
                "expected_absolute_drift"
            ]
        ),
        "expected_signed_effort_drift": float(
            effort_pair[
                "expected_signed_drift"
            ]
        ),
        "realized_effort_drift_km_decade": abs(
            float(
                realized_effort_drift
            )
        ),
        "species_expected_centre_mismatch_degrees": float(
            species_pair[
                "expected_centre_mismatch"
            ]
        ),
        "sample_size": int(
            sample_size
        ),
        "width_mode": str(
            width_mode
        ),
        "width_factor": float(
            width_factor
        ),
        "raw_estimable": bool(
            raw_estimable
        ),
        "corrected_estimable": bool(
            corrected_estimable
        ),
        "stable_cell_count": int(
            stable_cell_count
        ),
        "early_species_records": int(
            early_species_total
        ),
        "late_species_records": int(
            late_species_total
        ),
        "raw_shift_degrees": float(
            raw_shift
        )
        if np.isfinite(
            raw_shift
        )
        else np.nan,
        "corrected_shift_degrees": float(
            corrected_shift
        )
        if np.isfinite(
            corrected_shift
        )
        else np.nan,
        "raw_error_km_decade": float(
            raw_error
        )
        if np.isfinite(
            raw_error
        )
        else np.nan,
        "corrected_error_km_decade": float(
            corrected_error
        )
        if np.isfinite(
            corrected_error
        )
        else np.nan,
        "gain_km_decade": float(
            gain
        )
        if np.isfinite(
            gain
        )
        else np.nan,
        "corrected_raw_error_ratio": float(
            error_ratio
        )
        if np.isfinite(
            error_ratio
        )
        else np.nan,
        "correction_helped": bool(
            corrected_estimable
            and np.isfinite(
                gain
            )
            and gain > 0
        ),
        "realized_occupied_jaccard": float(
            realized_occupied_jaccard
        )
        if np.isfinite(
            realized_occupied_jaccard
        )
        else np.nan,
    }


# ============================================================
# Partial association
# ============================================================
def partial_species_overlap_gain_rho(
    frame,
    cluster_column=(
        "species_region_cluster"
    ),
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
        cluster_column,
    ]

    work = frame[
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
                cluster_column,
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
# Permutation helpers
# ============================================================
def build_group_indices(
    frame,
    group_columns,
):
    return [
        np.asarray(
            list(
                indices
            ),
            dtype=int,
        )
        for indices in frame.groupby(
            group_columns,
            sort=False,
        ).indices.values()
    ]


def rank_correlation_fast(
    x,
    y,
):
    valid = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    if (
        len(
            x
        )
        < 5
        or len(
            np.unique(
                x
            )
        )
        < 2
        or len(
            np.unique(
                y
            )
        )
        < 2
    ):
        return np.nan

    x_rank = rankdata(
        x,
        method="average",
    )

    y_rank = rankdata(
        y,
        method="average",
    )

    return float(
        np.corrcoef(
            x_rank,
            y_rank,
        )[0, 1]
    )


def permutation_test_axis(
    frame,
    x_column,
    y_column,
    group_columns,
    direction,
    n_permutations,
    rng,
    progress,
):
    work = frame[
        list(
            dict.fromkeys(
                [
                    x_column,
                    y_column,
                    *group_columns,
                ]
            )
        )
    ].dropna().reset_index(
        drop=True
    )

    x = work[
        x_column
    ].to_numpy(
        dtype=float
    )

    y = work[
        y_column
    ].to_numpy(
        dtype=float
    )

    observed = rank_correlation_fast(
        x,
        y,
    )

    group_indices = build_group_indices(
        work,
        group_columns,
    )

    null_values = np.empty(
        n_permutations,
        dtype=float,
    )

    exceed = 0

    for replicate in range(
        n_permutations
    ):
        permuted_x = x.copy()

        for indices in group_indices:
            permuted_x[
                indices
            ] = rng.permutation(
                permuted_x[
                    indices
                ]
            )

        value = rank_correlation_fast(
            permuted_x,
            y,
        )

        null_values[
            replicate
        ] = value

        if direction == "positive":
            exceed += int(
                np.isfinite(
                    value
                )
                and value
                >= observed
            )
        elif direction == "negative":
            exceed += int(
                np.isfinite(
                    value
                )
                and value
                <= observed
            )
        else:
            raise ValueError(
                f"Unknown direction: {direction}"
            )

        progress.update(
            1
        )

    p_value = float(
        (
            1
            + exceed
        )
        / (
            n_permutations
            + 1
        )
    )

    return {
        "observed_rho": float(
            observed
        ),
        "one_sided_p": (
            p_value
        ),
        "null_median": float(
            np.nanmedian(
                null_values
            )
        ),
        "null_low": float(
            np.nanquantile(
                null_values,
                0.025,
            )
        ),
        "null_high": float(
            np.nanquantile(
                null_values,
                0.975,
            )
        ),
        "null_values": (
            null_values
        ),
    }


# ============================================================
# Cluster bootstrap helpers
# ============================================================
def bootstrap_cluster_sample(
    frame,
    rng,
):
    clusters = np.array(
        sorted(
            frame[
                "species_region_cluster"
            ].unique()
        ),
        dtype=object,
    )

    sampled_clusters = rng.choice(
        clusters,
        size=len(
            clusters
        ),
        replace=True,
    )

    parts = []

    for bootstrap_id, cluster in enumerate(
        sampled_clusters
    ):
        part = frame[
            frame[
                "species_region_cluster"
            ]
            == cluster
        ].copy()

        part[
            "bootstrap_cluster_id"
        ] = str(
            bootstrap_id
        )

        parts.append(
            part
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def bootstrap_summary(
    values,
    direction,
):
    values = np.asarray(
        [
            value
            for value in values
            if np.isfinite(
                value
            )
        ],
        dtype=float,
    )

    if len(
        values
    ) == 0:
        return {
            "low": np.nan,
            "high": np.nan,
            "direction_fraction": np.nan,
        }

    if direction == "positive":
        direction_fraction = float(
            np.mean(
                values > 0
            )
        )
    elif direction == "negative":
        direction_fraction = float(
            np.mean(
                values < 0
            )
        )
    else:
        raise ValueError(
            f"Unknown direction: {direction}"
        )

    return {
        "low": float(
            np.quantile(
                values,
                0.025,
            )
        ),
        "high": float(
            np.quantile(
                values,
                0.975,
            )
        ),
        "direction_fraction": (
            direction_fraction
        ),
    }


# ============================================================
# Four-level monotonicity
# ============================================================
def monotonicity_table(
    frame,
    axis,
):
    if axis == "effort":
        group_columns = [
            "state_code",
            "fia_species_code",
            "species_overlap_requested",
            "drift_requested_km_decade",
            "sample_size",
            "width_mode",
        ]

        x_column = (
            "effort_overlap_requested"
        )

        y_column = (
            "corrected_estimable_fraction"
        )

        expected = "positive"

    elif axis == "species":
        group_columns = [
            "state_code",
            "fia_species_code",
            "effort_overlap_requested",
            "drift_requested_km_decade",
            "sample_size",
            "width_mode",
        ]

        x_column = (
            "species_overlap_requested"
        )

        y_column = (
            "median_corrected_error"
        )

        expected = "negative"

    else:
        raise ValueError(
            f"Unknown axis: {axis}"
        )

    rows = []

    groups = list(
        frame.groupby(
            group_columns,
            dropna=False,
        )
    )

    for keys, group in tqdm(
        groups,
        desc=f"{axis.title()}-axis monotonicity",
        unit="stratum",
        leave=False,
    ):
        (
            rho,
            p_value,
        ) = safe_spearman(
            group[
                x_column
            ],
            group[
                y_column
            ],
            minimum_rows=3,
        )

        ordered = group.sort_values(
            x_column
        )

        if len(
            ordered
        ) >= 2:
            low_value = float(
                ordered.iloc[
                    0
                ][
                    y_column
                ]
            )

            high_value = float(
                ordered.iloc[
                    -1
                ][
                    y_column
                ]
            )

            endpoint_delta = (
                high_value
                - low_value
                if expected
                == "positive"
                else low_value
                - high_value
            )
        else:
            endpoint_delta = np.nan

        if expected == "positive":
            expected_direction = bool(
                np.isfinite(
                    rho
                )
                and rho > 0
                and np.isfinite(
                    endpoint_delta
                )
                and endpoint_delta > 0
            )

            strong_direction = bool(
                np.isfinite(
                    rho
                )
                and rho >= 0.50
                and np.isfinite(
                    endpoint_delta
                )
                and endpoint_delta > 0
            )
        else:
            expected_direction = bool(
                np.isfinite(
                    rho
                )
                and rho < 0
                and np.isfinite(
                    endpoint_delta
                )
                and endpoint_delta > 0
            )

            strong_direction = bool(
                np.isfinite(
                    rho
                )
                and rho <= -0.50
                and np.isfinite(
                    endpoint_delta
                )
                and endpoint_delta > 0
            )

        row = {
            column: value
            for column, value in zip(
                group_columns,
                keys,
            )
        }

        row.update(
            {
                "rho": rho,
                "p_value": (
                    p_value
                ),
                "endpoint_delta": (
                    endpoint_delta
                ),
                "expected_direction": (
                    expected_direction
                ),
                "strong_expected_direction": (
                    strong_direction
                ),
            }
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
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

## Full orthogonal phase confirmation failure — {RUN_UTC}
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
        PLAN_PATH,
        SELECTED_SPECIES_PATH,
        *THINNED_PATHS.values(),
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
            "Missing frozen inputs: "
            + " | ".join(
                missing_paths
            )
        )

    selected_species_df = pd.read_csv(
        SELECTED_SPECIES_PATH
    )

    selected_species_df[
        "fia_species_code"
    ] = pd.to_numeric(
        selected_species_df[
            "fia_species_code"
        ],
        errors="coerce",
    )

    selected_species_df = (
        selected_species_df.dropna(
            subset=[
                "state_code",
                "fia_species_code",
                "pooled_centre",
                "base_width_degrees",
            ]
        )
        .copy()
    )

    selected_species_df[
        "fia_species_code"
    ] = selected_species_df[
        "fia_species_code"
    ].astype(int)

    if len(
        selected_species_df
    ) != N_SPECIES_REGION_UNITS:
        raise ValueError(
            "Selected species-region units do not match frozen plan: "
            f"{len(selected_species_df)} vs {N_SPECIES_REGION_UNITS}"
        )

    state_frames = {}

    for state in tqdm(
        STATES,
        desc="Preparing formal spatial frames",
        unit="state",
    ):
        state_frames[
            state
        ] = build_cell_frame(
            state=state,
            thinned_path=THINNED_PATHS[
                state
            ],
            effort_plan_path=EFFORT_PLAN_PATHS[
                state
            ],
        )

    # --------------------------------------------------------
    # Run core full confirmation
    # --------------------------------------------------------
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    core_writer = BufferedParquetWriter(
        CORE_REPLICATE_PATH,
        buffer_rows=(
            PARQUET_BUFFER_ROWS
        ),
    )

    core_progress = tqdm(
        total=EXPECTED_CORE_TOTAL,
        desc="Full core orthogonal simulation",
        unit="replicate",
    )

    core_rows_generated = 0

    for _, species_row in selected_species_df.iterrows():
        state = str(
            species_row[
                "state_code"
            ]
        )

        cell_frame = state_frames[
            state
        ]

        for effort_overlap in CORE_EFFORT_OVERLAPS:
            for species_overlap in CORE_SPECIES_OVERLAPS:
                for drift_level in DRIFT_LEVELS_KM_DECADE:
                    for sample_size in SAMPLE_SIZES:
                        for (
                            width_mode,
                            width_factor,
                        ) in WIDTH_MODES.items():
                            for replicate in range(
                                REPLICATES_PER_CORE_SCENARIO
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

                                result.update(
                                    {
                                        "design_component": (
                                            "core"
                                        ),
                                        "marginal_axis": (
                                            ""
                                        ),
                                        "replicate": int(
                                            replicate
                                        ),
                                    }
                                )

                                core_writer.add(
                                    result
                                )

                                core_rows_generated += 1

                                core_progress.update(
                                    1
                                )

    core_writer.close()
    core_progress.close()

    if (
        core_rows_generated
        != EXPECTED_CORE_TOTAL
    ):
        raise RuntimeError(
            "Core replicate count mismatch: "
            f"{core_rows_generated} vs {EXPECTED_CORE_TOTAL}"
        )

    # --------------------------------------------------------
    # Run targeted marginal refinements
    # --------------------------------------------------------
    marginal_writer = BufferedParquetWriter(
        MARGINAL_REPLICATE_PATH,
        buffer_rows=(
            PARQUET_BUFFER_ROWS
        ),
    )

    marginal_progress = tqdm(
        total=EXPECTED_MARGINAL_TOTAL,
        desc="Targeted marginal refinements",
        unit="replicate",
    )

    marginal_rows_generated = 0

    fixed_species_overlap = float(
        MARGINAL_FIXED_SETTINGS[
            "species_overlap_for_effort_curve"
        ]
    )

    fixed_effort_overlap = float(
        MARGINAL_FIXED_SETTINGS[
            "effort_overlap_for_species_curve"
        ]
    )

    fixed_drift = float(
        MARGINAL_FIXED_SETTINGS[
            "drift_km_decade"
        ]
    )

    fixed_sample_size = int(
        MARGINAL_FIXED_SETTINGS[
            "sample_size"
        ]
    )

    fixed_width_mode = str(
        MARGINAL_FIXED_SETTINGS[
            "width_mode"
        ]
    )

    fixed_width_factor = float(
        WIDTH_MODES[
            fixed_width_mode
        ]
    )

    for _, species_row in selected_species_df.iterrows():
        state = str(
            species_row[
                "state_code"
            ]
        )

        cell_frame = state_frames[
            state
        ]

        # Effort-overlap marginal curve.
        for effort_overlap in MARGINAL_REFINEMENT_LEVELS:
            for replicate in range(
                REPLICATES_PER_MARGINAL_SCENARIO
            ):
                result = simulate_replicate(
                    rng=rng,
                    cell_frame=cell_frame,
                    species_row=species_row,
                    effort_overlap=effort_overlap,
                    species_overlap=(
                        fixed_species_overlap
                    ),
                    drift_km_decade=(
                        fixed_drift
                    ),
                    sample_size=(
                        fixed_sample_size
                    ),
                    width_mode=(
                        fixed_width_mode
                    ),
                    width_factor=(
                        fixed_width_factor
                    ),
                )

                result.update(
                    {
                        "design_component": (
                            "marginal"
                        ),
                        "marginal_axis": (
                            "effort"
                        ),
                        "replicate": int(
                            replicate
                        ),
                    }
                )

                marginal_writer.add(
                    result
                )

                marginal_rows_generated += 1

                marginal_progress.update(
                    1
                )

        # Species-overlap marginal curve.
        for species_overlap in MARGINAL_REFINEMENT_LEVELS:
            for replicate in range(
                REPLICATES_PER_MARGINAL_SCENARIO
            ):
                result = simulate_replicate(
                    rng=rng,
                    cell_frame=cell_frame,
                    species_row=species_row,
                    effort_overlap=(
                        fixed_effort_overlap
                    ),
                    species_overlap=(
                        species_overlap
                    ),
                    drift_km_decade=(
                        fixed_drift
                    ),
                    sample_size=(
                        fixed_sample_size
                    ),
                    width_mode=(
                        fixed_width_mode
                    ),
                    width_factor=(
                        fixed_width_factor
                    ),
                )

                result.update(
                    {
                        "design_component": (
                            "marginal"
                        ),
                        "marginal_axis": (
                            "species"
                        ),
                        "replicate": int(
                            replicate
                        ),
                    }
                )

                marginal_writer.add(
                    result
                )

                marginal_rows_generated += 1

                marginal_progress.update(
                    1
                )

    marginal_writer.close()
    marginal_progress.close()

    if (
        marginal_rows_generated
        != EXPECTED_MARGINAL_TOTAL
    ):
        raise RuntimeError(
            "Marginal replicate count mismatch: "
            f"{marginal_rows_generated} vs {EXPECTED_MARGINAL_TOTAL}"
        )

    # --------------------------------------------------------
    # Load simulation outputs
    # --------------------------------------------------------
    core_df = pd.read_parquet(
        CORE_REPLICATE_PATH
    )

    marginal_df = pd.read_parquet(
        MARGINAL_REPLICATE_PATH
    )

    if (
        len(
            core_df
        )
        != EXPECTED_CORE_TOTAL
        or len(
            marginal_df
        )
        != EXPECTED_MARGINAL_TOTAL
    ):
        raise RuntimeError(
            "Saved Parquet row counts do not match the frozen plan."
        )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------
    core_valid = core_df[
        core_df[
            "design_generated"
        ].fillna(
            False
        )
    ].copy()

    marginal_valid = marginal_df[
        marginal_df[
            "design_generated"
        ].fillna(
            False
        )
    ].copy()

    design_generation_fraction = float(
        (
            len(
                core_valid
            )
            + len(
                marginal_valid
            )
        )
        / EXPECTED_TOTAL
    )

    zero_drift = core_valid[
        core_valid[
            "drift_requested_km_decade"
        ]
        == 0
    ]

    zero_drift_max_error = float(
        zero_drift[
            "expected_effort_drift_km_decade"
        ].abs().max()
    )

    expected_drift_max_error = float(
        (
            core_valid[
                "expected_effort_drift_km_decade"
            ]
            - core_valid[
                "drift_requested_km_decade"
            ]
        ).abs().max()
    )

    species_centre_mismatch_max = float(
        core_valid[
            "species_expected_centre_mismatch_degrees"
        ].max()
    )

    calibration_pass = bool(
        design_generation_fraction
        >= float(
            SUCCESS[
                "design_generation_fraction_min"
            ]
        )
        and zero_drift_max_error
        <= float(
            SUCCESS[
                "calibration_max_error"
            ]
        )
        and expected_drift_max_error
        <= float(
            SUCCESS[
                "calibration_max_error"
            ]
        )
        and species_centre_mismatch_max
        <= float(
            SUCCESS[
                "calibration_max_error"
            ]
        )
    )

    # --------------------------------------------------------
    # Scenario summaries
    # --------------------------------------------------------
    core_scenario_columns = [
        "state_code",
        "fia_species_code",
        "species_region_cluster",
        "effort_overlap_requested",
        "effort_overlap_actual",
        "species_overlap_requested",
        "species_overlap_actual",
        "drift_requested_km_decade",
        "sample_size",
        "width_mode",
    ]

    core_scenario_df = (
        core_valid.groupby(
            core_scenario_columns,
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
            median_stable_cells=(
                "stable_cell_count",
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

    core_scenario_df.to_parquet(
        OUT_DIR
        / "full_core_scenario_summary.parquet",
        index=False,
    )

    core_scenario_df.to_csv(
        OUT_DIR
        / "full_core_scenario_summary.csv",
        index=False,
    )

    marginal_scenario_columns = [
        "state_code",
        "fia_species_code",
        "species_region_cluster",
        "marginal_axis",
        "effort_overlap_requested",
        "effort_overlap_actual",
        "species_overlap_requested",
        "species_overlap_actual",
        "drift_requested_km_decade",
        "sample_size",
        "width_mode",
    ]

    marginal_scenario_df = (
        marginal_valid.groupby(
            marginal_scenario_columns,
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
        )
    )

    marginal_scenario_df.to_csv(
        OUT_DIR
        / "full_marginal_scenario_summary.csv",
        index=False,
    )

    print(
        "\nCORE SCENARIO SUMMARY"
    )

    display(
        core_scenario_df
    )

    print(
        "\nMARGINAL SCENARIO SUMMARY"
    )

    display(
        marginal_scenario_df
    )

    # --------------------------------------------------------
    # Global observed associations
    # --------------------------------------------------------
    (
        effort_global_rho,
        _,
    ) = safe_spearman(
        core_scenario_df[
            "effort_overlap_actual"
        ],
        core_scenario_df[
            "corrected_estimable_fraction"
        ],
    )

    error_scenarios = core_scenario_df.dropna(
        subset=[
            "median_corrected_error",
        ]
    ).copy()

    (
        species_global_rho,
        _,
    ) = safe_spearman(
        error_scenarios[
            "species_overlap_actual"
        ],
        error_scenarios[
            "median_corrected_error"
        ],
    )

    partial_gain_rho = (
        partial_species_overlap_gain_rho(
            error_scenarios,
            cluster_column=(
                "species_region_cluster"
            ),
        )
    )

    # Cross-axis associations for specificity.
    (
        species_overlap_estimability_rho,
        _,
    ) = safe_spearman(
        core_scenario_df[
            "species_overlap_actual"
        ],
        core_scenario_df[
            "corrected_estimable_fraction"
        ],
    )

    (
        effort_overlap_corrected_error_rho,
        _,
    ) = safe_spearman(
        error_scenarios[
            "effort_overlap_actual"
        ],
        error_scenarios[
            "median_corrected_error"
        ],
    )

    effort_axis_specificity = bool(
        abs(
            effort_global_rho
        )
        > abs(
            species_overlap_estimability_rho
        )
    )

    species_axis_specificity = bool(
        abs(
            species_global_rho
        )
        > abs(
            effort_overlap_corrected_error_rho
        )
    )

    axis_specificity_pass = bool(
        effort_axis_specificity
        and species_axis_specificity
    )

    # --------------------------------------------------------
    # Scenario-label permutations
    # --------------------------------------------------------
    rng_permutation = np.random.default_rng(
        RANDOM_SEED
        + 1
    )

    permutation_progress = tqdm(
        total=(
            2
            * N_LABEL_PERMUTATIONS
        ),
        desc="Scenario-label permutations",
        unit="permutation",
    )

    effort_permutation = permutation_test_axis(
        frame=core_scenario_df,
        x_column=(
            "effort_overlap_actual"
        ),
        y_column=(
            "corrected_estimable_fraction"
        ),
        group_columns=[
            "state_code",
            "fia_species_code",
            "species_overlap_requested",
            "drift_requested_km_decade",
            "sample_size",
            "width_mode",
        ],
        direction="positive",
        n_permutations=(
            N_LABEL_PERMUTATIONS
        ),
        rng=rng_permutation,
        progress=(
            permutation_progress
        ),
    )

    species_permutation = permutation_test_axis(
        frame=error_scenarios,
        x_column=(
            "species_overlap_actual"
        ),
        y_column=(
            "median_corrected_error"
        ),
        group_columns=[
            "state_code",
            "fia_species_code",
            "effort_overlap_requested",
            "drift_requested_km_decade",
            "sample_size",
            "width_mode",
        ],
        direction="negative",
        n_permutations=(
            N_LABEL_PERMUTATIONS
        ),
        rng=rng_permutation,
        progress=(
            permutation_progress
        ),
    )

    permutation_progress.close()

    permutation_summary_df = pd.DataFrame(
        [
            {
                "axis": "effort",
                "observed_rho": (
                    effort_permutation[
                        "observed_rho"
                    ]
                ),
                "one_sided_p": (
                    effort_permutation[
                        "one_sided_p"
                    ]
                ),
                "null_median": (
                    effort_permutation[
                        "null_median"
                    ]
                ),
                "null_low": (
                    effort_permutation[
                        "null_low"
                    ]
                ),
                "null_high": (
                    effort_permutation[
                        "null_high"
                    ]
                ),
            },
            {
                "axis": "species",
                "observed_rho": (
                    species_permutation[
                        "observed_rho"
                    ]
                ),
                "one_sided_p": (
                    species_permutation[
                        "one_sided_p"
                    ]
                ),
                "null_median": (
                    species_permutation[
                        "null_median"
                    ]
                ),
                "null_low": (
                    species_permutation[
                        "null_low"
                    ]
                ),
                "null_high": (
                    species_permutation[
                        "null_high"
                    ]
                ),
            },
        ]
    )

    permutation_summary_df.to_csv(
        OUT_DIR
        / "scenario_label_permutation_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "effort_null_rho": (
                effort_permutation[
                    "null_values"
                ]
            ),
            "species_null_rho": (
                species_permutation[
                    "null_values"
                ]
            ),
        }
    ).to_parquet(
        OUT_DIR
        / "scenario_label_permutation_nulls.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # Species-region-cluster bootstrap
    # --------------------------------------------------------
    rng_bootstrap = np.random.default_rng(
        RANDOM_SEED
        + 2
    )

    effort_bootstrap_values = []
    species_bootstrap_values = []
    partial_gain_bootstrap_values = []

    for _ in tqdm(
        range(
            N_CLUSTER_BOOTSTRAP
        ),
        desc="Species-region cluster bootstrap",
        unit="replicate",
    ):
        sample = bootstrap_cluster_sample(
            core_scenario_df,
            rng_bootstrap,
        )

        (
            effort_rho,
            _,
        ) = safe_spearman(
            sample[
                "effort_overlap_actual"
            ],
            sample[
                "corrected_estimable_fraction"
            ],
        )

        sample_error = sample.dropna(
            subset=[
                "median_corrected_error",
            ]
        )

        (
            species_rho,
            _,
        ) = safe_spearman(
            sample_error[
                "species_overlap_actual"
            ],
            sample_error[
                "median_corrected_error"
            ],
        )

        partial_rho = (
            partial_species_overlap_gain_rho(
                sample_error,
                cluster_column=(
                    "bootstrap_cluster_id"
                ),
            )
        )

        if np.isfinite(
            effort_rho
        ):
            effort_bootstrap_values.append(
                effort_rho
            )

        if np.isfinite(
            species_rho
        ):
            species_bootstrap_values.append(
                species_rho
            )

        if np.isfinite(
            partial_rho
        ):
            partial_gain_bootstrap_values.append(
                partial_rho
            )

    effort_bootstrap_summary = bootstrap_summary(
        effort_bootstrap_values,
        direction="positive",
    )

    species_bootstrap_summary = bootstrap_summary(
        species_bootstrap_values,
        direction="negative",
    )

    partial_bootstrap_summary = bootstrap_summary(
        partial_gain_bootstrap_values,
        direction="positive",
    )

    bootstrap_summary_df = pd.DataFrame(
        [
            {
                "metric": (
                    "effort_overlap_estimability_rho"
                ),
                **effort_bootstrap_summary,
            },
            {
                "metric": (
                    "species_overlap_corrected_error_rho"
                ),
                **species_bootstrap_summary,
            },
            {
                "metric": (
                    "partial_species_overlap_gain_rho"
                ),
                **partial_bootstrap_summary,
            },
        ]
    )

    bootstrap_summary_df.to_csv(
        OUT_DIR
        / "species_region_cluster_bootstrap_summary.csv",
        index=False,
    )

    bootstrap_length = min(
        len(
            effort_bootstrap_values
        ),
        len(
            species_bootstrap_values
        ),
        len(
            partial_gain_bootstrap_values
        ),
    )

    pd.DataFrame(
        {
            "effort_rho": np.asarray(
                effort_bootstrap_values[
                    :bootstrap_length
                ],
                dtype=float,
            ),
            "species_error_rho": np.asarray(
                species_bootstrap_values[
                    :bootstrap_length
                ],
                dtype=float,
            ),
            "partial_gain_rho": np.asarray(
                partial_gain_bootstrap_values[
                    :bootstrap_length
                ],
                dtype=float,
            ),
        }
    ).to_parquet(
        OUT_DIR
        / "species_region_cluster_bootstrap_values.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # Monotonicity
    # --------------------------------------------------------
    effort_monotonic_df = monotonicity_table(
        core_scenario_df,
        axis="effort",
    )

    species_monotonic_df = monotonicity_table(
        core_scenario_df,
        axis="species",
    )

    effort_valid_monotonic = (
        effort_monotonic_df[
            effort_monotonic_df[
                "rho"
            ].notna()
        ]
    )

    species_valid_monotonic = (
        species_monotonic_df[
            species_monotonic_df[
                "rho"
            ].notna()
        ]
    )

    effort_monotonic_fraction = float(
        effort_valid_monotonic[
            "strong_expected_direction"
        ].mean()
    )

    species_monotonic_fraction = float(
        species_valid_monotonic[
            "strong_expected_direction"
        ].mean()
    )

    effort_monotonic_df.to_csv(
        OUT_DIR
        / "full_effort_axis_monotonicity.csv",
        index=False,
    )

    species_monotonic_df.to_csv(
        OUT_DIR
        / "full_species_axis_monotonicity.csv",
        index=False,
    )

    # --------------------------------------------------------
    # High-low contrasts
    # --------------------------------------------------------
    low_effort = core_valid[
        core_valid[
            "effort_overlap_requested"
        ]
        <= 0.20
    ]

    high_effort = core_valid[
        core_valid[
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

    low_species = core_valid[
        (
            core_valid[
                "species_overlap_requested"
            ]
            <= 0.20
        )
        & core_valid[
            "corrected_estimable"
        ].fillna(
            False
        )
    ]

    high_species = core_valid[
        (
            core_valid[
                "species_overlap_requested"
            ]
            >= 0.80
        )
        & core_valid[
            "corrected_estimable"
        ].fillna(
            False
        )
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

    # --------------------------------------------------------
    # Marginal curves
    # --------------------------------------------------------
    effort_marginal = marginal_scenario_df[
        marginal_scenario_df[
            "marginal_axis"
        ]
        == "effort"
    ].copy()

    species_marginal = marginal_scenario_df[
        marginal_scenario_df[
            "marginal_axis"
        ]
        == "species"
    ].copy()

    effort_marginal_curve = (
        effort_marginal.groupby(
            "effort_overlap_requested",
            as_index=False,
        )
        .agg(
            corrected_estimable_fraction=(
                "corrected_estimable_fraction",
                "mean",
            ),
            correction_help_fraction=(
                "correction_help_fraction",
                "mean",
            ),
        )
        .sort_values(
            "effort_overlap_requested"
        )
    )

    species_marginal_curve = (
        species_marginal.groupby(
            "species_overlap_requested",
            as_index=False,
        )
        .agg(
            median_corrected_error=(
                "median_corrected_error",
                "median",
            ),
            median_gain=(
                "median_gain",
                "median",
            ),
        )
        .sort_values(
            "species_overlap_requested"
        )
    )

    (
        marginal_effort_rho,
        marginal_effort_p,
    ) = safe_spearman(
        effort_marginal_curve[
            "effort_overlap_requested"
        ],
        effort_marginal_curve[
            "corrected_estimable_fraction"
        ],
        minimum_rows=3,
    )

    (
        marginal_species_rho,
        marginal_species_p,
    ) = safe_spearman(
        species_marginal_curve[
            "species_overlap_requested"
        ],
        species_marginal_curve[
            "median_corrected_error"
        ],
        minimum_rows=3,
    )

    marginal_effort_direction_pass = bool(
        np.isfinite(
            marginal_effort_rho
        )
        and marginal_effort_rho > 0
        and float(
            effort_marginal_curve.iloc[
                -1
            ][
                "corrected_estimable_fraction"
            ]
        )
        > float(
            effort_marginal_curve.iloc[
                0
            ][
                "corrected_estimable_fraction"
            ]
        )
    )

    marginal_species_direction_pass = bool(
        np.isfinite(
            marginal_species_rho
        )
        and marginal_species_rho < 0
        and float(
            species_marginal_curve.iloc[
                -1
            ][
                "median_corrected_error"
            ]
        )
        < float(
            species_marginal_curve.iloc[
                0
            ][
                "median_corrected_error"
            ]
        )
    )

    effort_marginal_curve.to_csv(
        OUT_DIR
        / "marginal_effort_overlap_curve.csv",
        index=False,
    )

    species_marginal_curve.to_csv(
        OUT_DIR
        / "marginal_species_overlap_curve.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Negative controls
    # --------------------------------------------------------
    permutation_null_centred = bool(
        abs(
            effort_permutation[
                "null_median"
            ]
        )
        <= 0.05
        and abs(
            species_permutation[
                "null_median"
            ]
        )
        <= 0.05
    )

    permutation_significance_pass = bool(
        effort_permutation[
            "one_sided_p"
        ]
        <= 0.01
        and species_permutation[
            "one_sided_p"
        ]
        <= 0.01
    )

    negative_controls_pass = bool(
        calibration_pass
        and permutation_null_centred
        and permutation_significance_pass
    )

    negative_control_df = pd.DataFrame(
        [
            {
                "control": (
                    "zero_drift_calibration"
                ),
                "value": (
                    zero_drift_max_error
                ),
                "passed": bool(
                    zero_drift_max_error
                    <= float(
                        SUCCESS[
                            "calibration_max_error"
                        ]
                    )
                ),
            },
            {
                "control": (
                    "stationary_species_centre"
                ),
                "value": (
                    species_centre_mismatch_max
                ),
                "passed": bool(
                    species_centre_mismatch_max
                    <= float(
                        SUCCESS[
                            "calibration_max_error"
                        ]
                    )
                ),
            },
            {
                "control": (
                    "effort_label_permutation_null_median"
                ),
                "value": (
                    effort_permutation[
                        "null_median"
                    ]
                ),
                "passed": bool(
                    abs(
                        effort_permutation[
                            "null_median"
                        ]
                    )
                    <= 0.05
                ),
            },
            {
                "control": (
                    "species_label_permutation_null_median"
                ),
                "value": (
                    species_permutation[
                        "null_median"
                    ]
                ),
                "passed": bool(
                    abs(
                        species_permutation[
                            "null_median"
                        ]
                    )
                    <= 0.05
                ),
            },
            {
                "control": (
                    "axis_specificity"
                ),
                "value": (
                    float(
                        axis_specificity_pass
                    )
                ),
                "passed": (
                    axis_specificity_pass
                ),
            },
        ]
    )

    negative_control_df.to_csv(
        OUT_DIR
        / "negative_control_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Frozen success decision
    # --------------------------------------------------------
    effort_axis_pass = bool(
        effort_global_rho
        >= float(
            SUCCESS[
                "effort_overlap_estimability_rho_min"
            ]
        )
        and effort_permutation[
            "one_sided_p"
        ]
        <= float(
            SUCCESS[
                "effort_overlap_estimability_p_max"
            ]
        )
        and effort_monotonic_fraction
        >= float(
            SUCCESS[
                "effort_monotonic_strata_fraction_min"
            ]
        )
        and effort_bootstrap_summary[
            "direction_fraction"
        ]
        >= float(
            SUCCESS[
                "effort_cluster_bootstrap_positive_fraction_min"
            ]
        )
        and effort_estimability_delta
        >= float(
            SUCCESS[
                "high_minus_low_effort_estimability_min"
            ]
        )
    )

    species_axis_pass = bool(
        species_global_rho
        <= float(
            SUCCESS[
                "species_overlap_corrected_error_rho_max"
            ]
        )
        and species_permutation[
            "one_sided_p"
        ]
        <= float(
            SUCCESS[
                "species_overlap_corrected_error_p_max"
            ]
        )
        and species_monotonic_fraction
        >= float(
            SUCCESS[
                "species_monotonic_strata_fraction_min"
            ]
        )
        and species_bootstrap_summary[
            "direction_fraction"
        ]
        >= float(
            SUCCESS[
                "species_cluster_bootstrap_negative_fraction_min"
            ]
        )
        and low_minus_high_corrected_error
        >= float(
            SUCCESS[
                "low_minus_high_species_corrected_error_min"
            ]
        )
    )

    partial_gain_pass = bool(
        partial_gain_rho
        >= float(
            SUCCESS[
                "partial_species_overlap_gain_rho_min"
            ]
        )
        and partial_bootstrap_summary[
            "direction_fraction"
        ]
        >= float(
            SUCCESS[
                "partial_species_overlap_gain_bootstrap_positive_fraction_min"
            ]
        )
    )

    marginal_curves_pass = bool(
        marginal_effort_direction_pass
        and marginal_species_direction_pass
    )

    passed = bool(
        calibration_pass
        and effort_axis_pass
        and species_axis_pass
        and partial_gain_pass
        and marginal_curves_pass
        and axis_specificity_pass
        and negative_controls_pass
    )

    if passed:
        status = (
            "ORTHOGONAL_SUPPORT_PHASE_DIAGRAM_CONFIRMED"
        )

        next_step = (
            "Freeze the mechanism experiments. Update the manuscript package "
            "with the two-axis support principle, full phase surfaces, "
            "negative controls and formal statistical results."
        )
    else:
        status = (
            "ORTHOGONAL_SUPPORT_PHASE_CONFIRMATION_PARTIAL"
        )

        next_step = (
            "Do not change the frozen plan or rerun selectively. Report which "
            "predeclared criterion failed and retain only the supported axis."
        )

    # --------------------------------------------------------
    # Final result tables
    # --------------------------------------------------------
    result_summary_df = pd.DataFrame(
        [
            {
                "design_generation_fraction": (
                    design_generation_fraction
                ),
                "zero_drift_max_error": (
                    zero_drift_max_error
                ),
                "expected_drift_max_error": (
                    expected_drift_max_error
                ),
                "species_centre_mismatch_max": (
                    species_centre_mismatch_max
                ),
                "calibration_pass": (
                    calibration_pass
                ),
                "effort_global_rho": (
                    effort_global_rho
                ),
                "effort_permutation_p": (
                    effort_permutation[
                        "one_sided_p"
                    ]
                ),
                "effort_bootstrap_low": (
                    effort_bootstrap_summary[
                        "low"
                    ]
                ),
                "effort_bootstrap_high": (
                    effort_bootstrap_summary[
                        "high"
                    ]
                ),
                "effort_bootstrap_positive_fraction": (
                    effort_bootstrap_summary[
                        "direction_fraction"
                    ]
                ),
                "effort_monotonic_fraction": (
                    effort_monotonic_fraction
                ),
                "effort_estimability_delta": (
                    effort_estimability_delta
                ),
                "species_global_rho": (
                    species_global_rho
                ),
                "species_permutation_p": (
                    species_permutation[
                        "one_sided_p"
                    ]
                ),
                "species_bootstrap_low": (
                    species_bootstrap_summary[
                        "low"
                    ]
                ),
                "species_bootstrap_high": (
                    species_bootstrap_summary[
                        "high"
                    ]
                ),
                "species_bootstrap_negative_fraction": (
                    species_bootstrap_summary[
                        "direction_fraction"
                    ]
                ),
                "species_monotonic_fraction": (
                    species_monotonic_fraction
                ),
                "low_minus_high_corrected_error": (
                    low_minus_high_corrected_error
                ),
                "partial_gain_rho": (
                    partial_gain_rho
                ),
                "partial_gain_bootstrap_low": (
                    partial_bootstrap_summary[
                        "low"
                    ]
                ),
                "partial_gain_bootstrap_high": (
                    partial_bootstrap_summary[
                        "high"
                    ]
                ),
                "partial_gain_bootstrap_positive_fraction": (
                    partial_bootstrap_summary[
                        "direction_fraction"
                    ]
                ),
                "species_overlap_estimability_rho": (
                    species_overlap_estimability_rho
                ),
                "effort_overlap_corrected_error_rho": (
                    effort_overlap_corrected_error_rho
                ),
                "axis_specificity_pass": (
                    axis_specificity_pass
                ),
                "marginal_effort_rho": (
                    marginal_effort_rho
                ),
                "marginal_species_rho": (
                    marginal_species_rho
                ),
                "marginal_curves_pass": (
                    marginal_curves_pass
                ),
                "negative_controls_pass": (
                    negative_controls_pass
                ),
                "effort_axis_pass": (
                    effort_axis_pass
                ),
                "species_axis_pass": (
                    species_axis_pass
                ),
                "partial_gain_pass": (
                    partial_gain_pass
                ),
                "passed": (
                    passed
                ),
            }
        ]
    )

    result_summary_df.to_csv(
        OUT_DIR
        / "full_phase_confirmation_summary.csv",
        index=False,
    )

    print(
        "\nFULL PHASE CONFIRMATION SUMMARY"
    )

    display(
        result_summary_df
    )

    print(
        "\nPERMUTATION SUMMARY"
    )

    display(
        permutation_summary_df
    )

    print(
        "\nCLUSTER BOOTSTRAP SUMMARY"
    )

    display(
        bootstrap_summary_df
    )

    print(
        "\nNEGATIVE CONTROL SUMMARY"
    )

    display(
        negative_control_df
    )

    print(
        "\nMARGINAL EFFORT CURVE"
    )

    display(
        effort_marginal_curve
    )

    print(
        "\nMARGINAL SPECIES CURVE"
    )

    display(
        species_marginal_curve
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    figure_paths = []

    aggregate_core = (
        core_valid.groupby(
            [
                "effort_overlap_actual",
                "species_overlap_actual",
                "drift_requested_km_decade",
                "sample_size",
                "width_mode",
            ],
            as_index=False,
        )
        .agg(
            corrected_estimable_fraction=(
                "corrected_estimable",
                "mean",
            ),
            median_corrected_error=(
                "corrected_error_km_decade",
                "median",
            ),
            median_gain=(
                "gain_km_decade",
                "median",
            ),
        )
    )

    for drift_level in DRIFT_LEVELS_KM_DECADE:
        figure_data = aggregate_core[
            (
                aggregate_core[
                    "drift_requested_km_decade"
                ]
                == drift_level
            )
            & (
                aggregate_core[
                    "sample_size"
                ]
                == max(
                    SAMPLE_SIZES
                )
            )
            & (
                aggregate_core[
                    "width_mode"
                ]
                == "broad"
            )
        ].copy()

        estimability_pivot = figure_data.pivot(
            index=(
                "effort_overlap_actual"
            ),
            columns=(
                "species_overlap_actual"
            ),
            values=(
                "corrected_estimable_fraction"
            ),
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

        path = (
            OUT_DIR
            / (
                "full_estimability_phase_"
                f"drift_{int(drift_level)}.png"
            )
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

        display(
            Image(
                filename=str(
                    path
                )
            )
        )

        figure_paths.append(
            path
        )

        error_pivot = figure_data.pivot(
            index=(
                "effort_overlap_actual"
            ),
            columns=(
                "species_overlap_actual"
            ),
            values=(
                "median_corrected_error"
            ),
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

        path = (
            OUT_DIR
            / (
                "full_corrected_error_phase_"
                f"drift_{int(drift_level)}.png"
            )
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

        display(
            Image(
                filename=str(
                    path
                )
            )
        )

        figure_paths.append(
            path
        )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        effort_marginal_curve[
            "effort_overlap_requested"
        ],
        effort_marginal_curve[
            "corrected_estimable_fraction"
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
        "Frozen marginal effort-overlap confirmation"
    )

    plt.tight_layout()

    marginal_effort_figure = (
        OUT_DIR
        / "full_marginal_effort_curve.png"
    )

    plt.savefig(
        marginal_effort_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    display(
        Image(
            filename=str(
                marginal_effort_figure
            )
        )
    )

    figure_paths.append(
        marginal_effort_figure
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        species_marginal_curve[
            "species_overlap_requested"
        ],
        species_marginal_curve[
            "median_corrected_error"
        ],
        marker="o",
        label=(
            "Corrected error"
        ),
    )

    plt.plot(
        species_marginal_curve[
            "species_overlap_requested"
        ],
        species_marginal_curve[
            "median_gain"
        ],
        marker="o",
        label=(
            "Correction gain"
        ),
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
        "Frozen marginal species-overlap confirmation"
    )

    plt.legend()
    plt.tight_layout()

    marginal_species_figure = (
        OUT_DIR
        / "full_marginal_species_curve.png"
    )

    plt.savefig(
        marginal_species_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    display(
        Image(
            filename=str(
                marginal_species_figure
            )
        )
    )

    figure_paths.append(
        marginal_species_figure
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        effort_permutation[
            "null_values"
        ],
        bins=40,
        alpha=0.7,
        label=(
            "Effort-axis null"
        ),
    )

    plt.hist(
        species_permutation[
            "null_values"
        ],
        bins=40,
        alpha=0.7,
        label=(
            "Species-axis null"
        ),
    )

    plt.axvline(
        effort_global_rho,
        linestyle="--",
        label=(
            "Observed effort rho"
        ),
    )

    plt.axvline(
        species_global_rho,
        linestyle=":",
        label=(
            "Observed species rho"
        ),
    )

    plt.xlabel(
        "Permuted Spearman rho"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Scenario-label negative controls"
    )

    plt.legend()
    plt.tight_layout()

    permutation_figure = (
        OUT_DIR
        / "full_permutation_negative_controls.png"
    )

    plt.savefig(
        permutation_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    display(
        Image(
            filename=str(
                permutation_figure
            )
        )
    )

    figure_paths.append(
        permutation_figure
    )

    # --------------------------------------------------------
    # Manuscript-ready result
    # --------------------------------------------------------
    manuscript_text = f"""
FULL ORTHOGONAL SUPPORT-OVERLAP CONFIRMATION

Formal spatial frames:
GBIF Occurrence Download DOI {GBIF_DOI}

Design:
- species-region units: {N_SPECIES_REGION_UNITS}
- core replicates: {EXPECTED_CORE_TOTAL:,}
- marginal replicates: {EXPECTED_MARGINAL_TOTAL:,}
- species-region cluster bootstraps: {N_CLUSTER_BOOTSTRAP:,}
- scenario-label permutations per axis: {N_LABEL_PERMUTATIONS:,}

Calibration:
- design-generation fraction:
  {design_generation_fraction:.6f}
- maximum expected zero-drift error:
  {zero_drift_max_error:.8g}
- maximum expected drift calibration error:
  {expected_drift_max_error:.8g}
- maximum stationary-species centre mismatch:
  {species_centre_mismatch_max:.8g}

Effort-frame overlap and estimability:
- global rho:
  {effort_global_rho:.4f}
- one-sided permutation P:
  {effort_permutation['one_sided_p']:.6f}
- cluster-bootstrap 95% interval:
  [{effort_bootstrap_summary['low']:.4f},
   {effort_bootstrap_summary['high']:.4f}]
- bootstrap positive fraction:
  {effort_bootstrap_summary['direction_fraction']:.4f}
- strong monotonic strata fraction:
  {effort_monotonic_fraction:.4f}
- high-minus-low estimability:
  {effort_estimability_delta:.4f}

Species support overlap and corrected error:
- global rho:
  {species_global_rho:.4f}
- one-sided permutation P:
  {species_permutation['one_sided_p']:.6f}
- cluster-bootstrap 95% interval:
  [{species_bootstrap_summary['low']:.4f},
   {species_bootstrap_summary['high']:.4f}]
- bootstrap negative fraction:
  {species_bootstrap_summary['direction_fraction']:.4f}
- strong monotonic strata fraction:
  {species_monotonic_fraction:.4f}
- low-minus-high corrected error:
  {low_minus_high_corrected_error:.4f} km/decade

Adjusted correction gain:
- partial rank rho:
  {partial_gain_rho:.4f}
- cluster-bootstrap 95% interval:
  [{partial_bootstrap_summary['low']:.4f},
   {partial_bootstrap_summary['high']:.4f}]
- bootstrap positive fraction:
  {partial_bootstrap_summary['direction_fraction']:.4f}

Interpretation:
Effort-frame overlap and species-specific support overlap have distinct roles.
The former governs whether correction can be estimated on a common frame,
whereas the latter governs the error of the corrected species shift. This
supports a two-axis support principle rather than a universal correction or a
single optimized cutoff.

Decision:
{status}
"""

    manuscript_path = (
        OUT_DIR
        / "manuscript_ready_full_phase_result.txt"
    )

    manuscript_path.write_text(
        manuscript_text.strip()
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    append_readme(
        f"""

## Full orthogonal support-overlap confirmation — {RUN_UTC}

### Frozen plan
- Plan file:
  {PLAN_PATH}
- Plan status:
  FROZEN
- Core factor levels changed:
  No
- Binary cutoff allowed:
  No

### Workload
- Core replicates:
  {EXPECTED_CORE_TOTAL:,}
- Marginal replicates:
  {EXPECTED_MARGINAL_TOTAL:,}
- Total:
  {EXPECTED_TOTAL:,}
- Scenario-label permutations per axis:
  {N_LABEL_PERMUTATIONS:,}
- Species-region cluster bootstraps:
  {N_CLUSTER_BOOTSTRAP:,}
- GPU:
  not used

### Results
- Calibration passed:
  {calibration_pass}
- Effort axis passed:
  {effort_axis_pass}
- Species axis passed:
  {species_axis_pass}
- Partial gain passed:
  {partial_gain_pass}
- Marginal curves passed:
  {marginal_curves_pass}
- Axis specificity passed:
  {axis_specificity_pass}
- Negative controls passed:
  {negative_controls_pass}
- Status:
  **{status}**
- Passed:
  {passed}
- Next step:
  {next_step}

### Interpretation boundary
The two support axes are distinct. Effort overlap is an estimability condition;
species observation-support overlap is a corrected-error and transportability
condition. Gain remains secondary. No universal correction or binary overlap
threshold is claimed.
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
        f"PLAN_JSON: {PLAN_PATH}"
    )
    print(
        f"GBIF_DOWNLOAD_DOI: {GBIF_DOI}"
    )
    print(
        f"CORE_REPLICATES: "
        f"{len(core_df):,}"
    )
    print(
        f"MARGINAL_REPLICATES: "
        f"{len(marginal_df):,}"
    )
    print(
        f"TOTAL_REPLICATES: "
        f"{len(core_df) + len(marginal_df):,}"
    )
    print(
        f"DESIGN_GENERATION_FRACTION: "
        f"{design_generation_fraction}"
    )
    print(
        f"ZERO_DRIFT_MAX_ERROR: "
        f"{zero_drift_max_error}"
    )
    print(
        f"EXPECTED_DRIFT_MAX_ERROR: "
        f"{expected_drift_max_error}"
    )
    print(
        "SPECIES_CENTRE_MISMATCH_MAX: "
        f"{species_centre_mismatch_max}"
    )
    print(
        f"CALIBRATION_PASSED: "
        f"{calibration_pass}"
    )
    print(
        f"EFFORT_GLOBAL_RHO: "
        f"{effort_global_rho}"
    )
    print(
        f"EFFORT_PERMUTATION_P: "
        f"{effort_permutation['one_sided_p']}"
    )
    print(
        f"EFFORT_BOOTSTRAP_LOW: "
        f"{effort_bootstrap_summary['low']}"
    )
    print(
        f"EFFORT_BOOTSTRAP_HIGH: "
        f"{effort_bootstrap_summary['high']}"
    )
    print(
        "EFFORT_BOOTSTRAP_POSITIVE_FRACTION: "
        f"{effort_bootstrap_summary['direction_fraction']}"
    )
    print(
        f"EFFORT_MONOTONIC_FRACTION: "
        f"{effort_monotonic_fraction}"
    )
    print(
        f"EFFORT_ESTIMABILITY_DELTA: "
        f"{effort_estimability_delta}"
    )
    print(
        f"SPECIES_GLOBAL_RHO: "
        f"{species_global_rho}"
    )
    print(
        f"SPECIES_PERMUTATION_P: "
        f"{species_permutation['one_sided_p']}"
    )
    print(
        f"SPECIES_BOOTSTRAP_LOW: "
        f"{species_bootstrap_summary['low']}"
    )
    print(
        f"SPECIES_BOOTSTRAP_HIGH: "
        f"{species_bootstrap_summary['high']}"
    )
    print(
        "SPECIES_BOOTSTRAP_NEGATIVE_FRACTION: "
        f"{species_bootstrap_summary['direction_fraction']}"
    )
    print(
        f"SPECIES_MONOTONIC_FRACTION: "
        f"{species_monotonic_fraction}"
    )
    print(
        "LOW_MINUS_HIGH_CORRECTED_ERROR: "
        f"{low_minus_high_corrected_error}"
    )
    print(
        f"PARTIAL_GAIN_RHO: "
        f"{partial_gain_rho}"
    )
    print(
        f"PARTIAL_GAIN_BOOTSTRAP_LOW: "
        f"{partial_bootstrap_summary['low']}"
    )
    print(
        f"PARTIAL_GAIN_BOOTSTRAP_HIGH: "
        f"{partial_bootstrap_summary['high']}"
    )
    print(
        "PARTIAL_GAIN_BOOTSTRAP_POSITIVE_FRACTION: "
        f"{partial_bootstrap_summary['direction_fraction']}"
    )
    print(
        f"SPECIES_OVERLAP_ESTIMABILITY_RHO: "
        f"{species_overlap_estimability_rho}"
    )
    print(
        "EFFORT_OVERLAP_CORRECTED_ERROR_RHO: "
        f"{effort_overlap_corrected_error_rho}"
    )
    print(
        f"AXIS_SPECIFICITY_PASSED: "
        f"{axis_specificity_pass}"
    )
    print(
        f"MARGINAL_EFFORT_RHO: "
        f"{marginal_effort_rho}"
    )
    print(
        f"MARGINAL_SPECIES_RHO: "
        f"{marginal_species_rho}"
    )
    print(
        f"MARGINAL_CURVES_PASSED: "
        f"{marginal_curves_pass}"
    )
    print(
        f"NEGATIVE_CONTROLS_PASSED: "
        f"{negative_controls_pass}"
    )
    print(
        f"EFFORT_AXIS_PASSED: "
        f"{effort_axis_pass}"
    )
    print(
        f"SPECIES_AXIS_PASSED: "
        f"{species_axis_pass}"
    )
    print(
        f"PARTIAL_GAIN_PASSED: "
        f"{partial_gain_pass}"
    )
    print(
        "PERMUTATION_SUMMARY: "
        + json.dumps(
            permutation_summary_df.round(
                8
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "BOOTSTRAP_SUMMARY: "
        + json.dumps(
            bootstrap_summary_df.round(
                8
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "NEGATIVE_CONTROL_SUMMARY: "
        + json.dumps(
            negative_control_df.round(
                8
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "MARGINAL_EFFORT_CURVE: "
        + json.dumps(
            effort_marginal_curve.round(
                8
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "MARGINAL_SPECIES_CURVE: "
        + json.dumps(
            species_marginal_curve.round(
                8
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
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
        f"MANUSCRIPT_RESULT_PATH: "
        f"{manuscript_path}"
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
        "CELL22F_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed summary. "
            "Do not modify or selectively rerun the frozen plan."
        ),
    )
