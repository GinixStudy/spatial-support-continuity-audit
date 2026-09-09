
# Cell 21A — Cross-method correction benchmark: trend validation
#
# Q1 upgrade question
# -------------------
# Is spatial-support continuity (cell_jaccard) a general applicability
# principle across different correction strategies, rather than a diagnostic
# specific to the locked target-effort correction?
#
# Existing formal data only:
# - PA, VA and NC registered GBIF records
# - same-window repeated FIA reference
#
# Compared correction strategies
# ------------------------------
# 1. stable_frame_raw
#    Raw record-weighted centre restricted to cells with sufficient
#    target-group effort in both periods.
#
# 2. equal_cell
#    Equal weight for each species-occupied 0.5-degree cell.
#
# 3. observer_balanced
#    Equal weight for each observer's species-period centre.
#
# 4. target_effort
#    Locked target-group effort-standardized correction.
#
# Baseline
# --------
# Raw record-weighted shift.
#
# Trend-only inference
# --------------------
# - 2,000 state-stratified permutations
# - 1,000 FIA-species-cluster bootstrap replicates
#
# No method is selected or tuned in this cell.
# No new feature or cell-Jaccard cutoff is searched.

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from tqdm.auto import tqdm
from IPython.display import display, Image

# ============================================================
# Locked configuration
# ============================================================
STATES = ["PA", "VA", "NC"]

EARLY_YEARS = list(range(2013, 2019))
LATE_YEARS = list(range(2020, 2026))
EARLY_MID_YEAR = float(np.mean(EARLY_YEARS))
LATE_MID_YEAR = float(np.mean(LATE_YEARS))

GRID_RESOLUTION = 0.50
MIN_TARGET_EFFORT_PER_CELL_PERIOD = 10

METHODS = [
    "stable_frame_raw",
    "equal_cell",
    "observer_balanced",
    "target_effort",
]

N_PERMUTATIONS = 2_000
N_CLUSTER_BOOTSTRAP = 1_000
RANDOM_SEED = 20260711

MIN_POSITIVE_METHODS = 3
MIN_CONSENSUS_RHO = 0.35
MAX_CONSENSUS_PERMUTATION_P = 0.05
MIN_CONSENSUS_BOOTSTRAP_POSITIVE = 0.80
MIN_METHODS_WITH_AUC_065 = 2

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
DERIVED_DIR = BASE_DIR / "derived"

FORMAL_DIR = (
    DERIVED_DIR
    / "registered_gbif_formal_parity"
)

OUT_DIR = (
    DERIVED_DIR
    / "cross_method_correction_benchmark_trend"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES_INPUT_PATHS = {
    state: FORMAL_DIR / f"{state}_formal_species_inputs.parquet"
    for state in STATES
}

THINNED_PATHS = {
    state: FORMAL_DIR / f"{state}_formal_thinned.parquet"
    for state in STATES
}

FIA_REFERENCE_PATHS = {
    "PA": (
        DERIVED_DIR
        / "pa_species_external_validation"
        / "fia_same_window_species_reference.parquet"
    ),
    "VA": (
        DERIVED_DIR
        / "va_confirmation_dataset"
        / "va_fia_same_window_reference.parquet"
    ),
    "NC": (
        DERIVED_DIR
        / "nc_final_confirmation_dataset"
        / "nc_fia_same_window_reference.parquet"
    ),
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

GBIF_DOI = "10.15468/dl.sm6ygu"
DOWNLOAD_KEY = "0032735-260623161305970"

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("OUT_DIR:", OUT_DIR)
print("METHODS:", METHODS)
print("N_PERMUTATIONS:", N_PERMUTATIONS)
print("N_CLUSTER_BOOTSTRAP:", N_CLUSTER_BOOTSTRAP)
print("GPU: not used")


# ============================================================
# Helpers
# ============================================================
def append_readme(text):
    path = BASE_DIR / "README.md"

    old = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else "# Temporal Observation Drift\n"
    )

    path.write_text(
        old + text,
        encoding="utf-8",
    )


def resolve_column(columns, candidates):
    lookup = {
        str(column).strip().lower(): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


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


def shift_km_decade(
    early_latitude,
    late_latitude,
):
    if (
        not np.isfinite(early_latitude)
        or not np.isfinite(late_latitude)
    ):
        return np.nan

    return float(
        (
            late_latitude
            - early_latitude
        )
        * 111.32
        * 10
        / (
            LATE_MID_YEAR
            - EARLY_MID_YEAR
        )
    )


def safe_correlation(x, y):
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
        return np.nan

    return float(
        np.corrcoef(
            frame["x"],
            frame["y"],
        )[0, 1]
    )


def safe_spearman(x, y):
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
        return np.nan

    return safe_correlation(
        rankdata(
            frame["x"],
            method="average",
        ),
        rankdata(
            frame["y"],
            method="average",
        ),
    )


def binary_auc(scores, labels):
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(
                pd.Series(scores),
                errors="coerce",
            ),
            "label": pd.Series(labels).astype("boolean"),
        }
    ).dropna()

    positive = frame.loc[
        frame["label"],
        "score",
    ].to_numpy(dtype=float)

    negative = frame.loc[
        ~frame["label"],
        "score",
    ].to_numpy(dtype=float)

    if (
        len(positive) == 0
        or len(negative) == 0
    ):
        return np.nan, 0

    comparisons = (
        positive[:, None]
        - negative[None, :]
    )

    auc = float(
        (
            np.sum(
                comparisons > 0
            )
            + 0.5
            * np.sum(
                comparisons == 0
            )
        )
        / comparisons.size
    )

    return auc, int(
        comparisons.size
    )


def within_state_rank(
    frame,
    column,
):
    return (
        frame.groupby(
            "state_code"
        )[column]
        .rank(
            method="average",
            pct=True,
        )
    )


def stratified_rank_rho(
    frame,
    gain_column="gain",
):
    work = frame[
        [
            "state_code",
            "cell_jaccard",
            gain_column,
        ]
    ].dropna().copy()

    work[
        "feature_rank"
    ] = within_state_rank(
        work,
        "cell_jaccard",
    )

    work[
        "gain_rank"
    ] = within_state_rank(
        work,
        gain_column,
    )

    return safe_correlation(
        work[
            "feature_rank"
        ],
        work[
            "gain_rank"
        ],
    )


def stratified_auc(
    frame,
    gain_column="gain",
):
    weighted_sum = 0.0
    total_pairs = 0

    for _, state_frame in frame.groupby(
        "state_code"
    ):
        auc, pairs = binary_auc(
            state_frame[
                "cell_jaccard"
            ],
            state_frame[
                gain_column
            ]
            > 0,
        )

        if (
            np.isfinite(auc)
            and pairs > 0
        ):
            weighted_sum += (
                auc * pairs
            )
            total_pairs += pairs

    return (
        float(
            weighted_sum
            / total_pairs
        )
        if total_pairs > 0
        else np.nan
    )


def load_effort_codes(
    state,
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing effort plan: {path}"
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
            f"{state}: FIA species-code column missing."
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
        # PA plan already contains only effort-frame taxa.
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


def load_fia_reference(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FIA reference: {path}"
        )

    frame = pd.read_parquet(
        path
    )

    if (
        "fia_species_code"
        not in frame.columns
        and "species_code"
        in frame.columns
    ):
        frame = frame.rename(
            columns={
                "species_code": (
                    "fia_species_code"
                )
            }
        )

    required = {
        "fia_species_code",
        "fia_shift_km_decade",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise KeyError(
            f"FIA reference missing: {sorted(missing)}"
        )

    frame[
        "fia_species_code"
    ] = pd.to_numeric(
        frame[
            "fia_species_code"
        ],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "fia_species_code",
            "fia_shift_km_decade",
        ]
    ).copy()

    frame[
        "fia_species_code"
    ] = frame[
        "fia_species_code"
    ].astype(int)

    return frame[
        [
            "fia_species_code",
            "fia_shift_km_decade",
        ]
    ].drop_duplicates(
        "fia_species_code"
    )


def two_period_shift(
    frame,
    latitude_column,
):
    indexed = frame.set_index(
        "period"
    )

    if not {
        "early",
        "late",
    }.issubset(
        indexed.index
    ):
        return np.nan

    return shift_km_decade(
        float(
            indexed.loc[
                "early",
                latitude_column,
            ]
        ),
        float(
            indexed.loc[
                "late",
                latitude_column,
            ]
        ),
    )


def build_state_method_table(
    state,
    thinned,
    species_inputs,
    fia_reference,
    effort_codes,
):
    work = thinned.copy()

    for column in [
        "fia_species_code",
        "decimalLatitude",
        "decimalLongitude",
    ]:
        work[
            column
        ] = pd.to_numeric(
            work[
                column
            ],
            errors="coerce",
        )

    work = work.dropna(
        subset=[
            "fia_species_code",
            "period",
            "decimalLatitude",
            "decimalLongitude",
        ]
    ).copy()

    work[
        "fia_species_code"
    ] = work[
        "fia_species_code"
    ].astype(int)

    if (
        "observer_label"
        not in work.columns
    ):
        work[
            "observer_label"
        ] = normalize_observer_label(
            work[
                "recordedBy"
            ]
        )
    else:
        work[
            "observer_label"
        ] = normalize_observer_label(
            work[
                "observer_label"
            ]
        )

    work[
        "grid_lat_index"
    ] = np.floor(
        work[
            "decimalLatitude"
        ]
        / GRID_RESOLUTION
    ).astype(
        "int32"
    )

    work[
        "grid_lon_index"
    ] = np.floor(
        work[
            "decimalLongitude"
        ]
        / GRID_RESOLUTION
    ).astype(
        "int32"
    )

    work[
        "grid_id"
    ] = (
        work[
            "grid_lat_index"
        ].astype(str)
        + "_"
        + work[
            "grid_lon_index"
        ].astype(str)
    )

    effort_frame = work[
        work[
            "fia_species_code"
        ].isin(
            effort_codes
        )
    ].copy()

    effort_cell = (
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
        effort_cell.pivot(
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
            effort_wide[
                period
            ] = 0

    stable_cells = set(
        effort_wide[
            (
                effort_wide[
                    "early"
                ]
                >= MIN_TARGET_EFFORT_PER_CELL_PERIOD
            )
            & (
                effort_wide[
                    "late"
                ]
                >= MIN_TARGET_EFFORT_PER_CELL_PERIOD
            )
        ].index
    )

    species_inputs = species_inputs.copy()

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
            "cell_jaccard",
            "raw_shift_grid_0p5",
            "corrected_shift_grid_0p5",
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

    benchmark = work[
        work[
            "fia_species_code"
        ].isin(
            evaluation_codes
        )
    ].copy()

    base = (
        species_inputs[
            [
                "fia_species_code",
                "cell_jaccard",
                "raw_shift_grid_0p5",
                "corrected_shift_grid_0p5",
            ]
        ]
        .rename(
            columns={
                "raw_shift_grid_0p5": (
                    "raw_shift"
                ),
                "corrected_shift_grid_0p5": (
                    "target_effort_shift"
                ),
            }
        )
        .merge(
            fia_reference,
            on="fia_species_code",
            how="inner",
        )
    )

    rows = []

    for species_code in tqdm(
        sorted(
            evaluation_codes
        ),
        desc=f"{state} cross-method species",
        unit="species",
        leave=False,
    ):
        species = benchmark[
            benchmark[
                "fia_species_code"
            ]
            == species_code
        ].copy()

        base_row = base[
            base[
                "fia_species_code"
            ]
            == species_code
        ]

        if base_row.empty:
            continue

        base_row = base_row.iloc[
            0
        ]

        # Method 1: raw records restricted to the stable effort frame.
        stable_period = (
            species[
                species[
                    "grid_id"
                ].isin(
                    stable_cells
                )
            ]
            .groupby(
                "period",
                as_index=False,
            )
            .agg(
                latitude=(
                    "decimalLatitude",
                    "mean",
                ),
            )
        )

        stable_frame_shift = (
            two_period_shift(
                stable_period,
                "latitude",
            )
        )

        # Method 2: equal weight for each occupied cell.
        species_cell = (
            species.groupby(
                [
                    "period",
                    "grid_id",
                ],
                as_index=False,
            )
            .agg(
                cell_latitude=(
                    "decimalLatitude",
                    "mean",
                ),
            )
        )

        equal_cell_period = (
            species_cell.groupby(
                "period",
                as_index=False,
            )
            .agg(
                latitude=(
                    "cell_latitude",
                    "mean",
                ),
            )
        )

        equal_cell_shift = (
            two_period_shift(
                equal_cell_period,
                "latitude",
            )
        )

        # Method 3: equal weight for each observer.
        observer_period = (
            species.dropna(
                subset=[
                    "observer_label",
                ]
            )
            .groupby(
                [
                    "period",
                    "observer_label",
                ],
                as_index=False,
            )
            .agg(
                observer_latitude=(
                    "decimalLatitude",
                    "mean",
                ),
            )
            .groupby(
                "period",
                as_index=False,
            )
            .agg(
                latitude=(
                    "observer_latitude",
                    "mean",
                ),
            )
        )

        observer_balanced_shift = (
            two_period_shift(
                observer_period,
                "latitude",
            )
        )

        method_shifts = {
            "stable_frame_raw": (
                stable_frame_shift
            ),
            "equal_cell": (
                equal_cell_shift
            ),
            "observer_balanced": (
                observer_balanced_shift
            ),
            "target_effort": float(
                base_row[
                    "target_effort_shift"
                ]
            ),
        }

        raw_shift = float(
            base_row[
                "raw_shift"
            ]
        )

        fia_shift = float(
            base_row[
                "fia_shift_km_decade"
            ]
        )

        raw_error = abs(
            raw_shift
            - fia_shift
        )

        for method, method_shift in method_shifts.items():
            method_error = (
                abs(
                    method_shift
                    - fia_shift
                )
                if np.isfinite(
                    method_shift
                )
                else np.nan
            )

            rows.append(
                {
                    "state_code": state,
                    "fia_species_code": (
                        species_code
                    ),
                    "cell_jaccard": float(
                        base_row[
                            "cell_jaccard"
                        ]
                    ),
                    "method": method,
                    "raw_shift": raw_shift,
                    "method_shift": (
                        method_shift
                    ),
                    "fia_shift": fia_shift,
                    "raw_error": raw_error,
                    "method_error": (
                        method_error
                    ),
                    "gain": (
                        raw_error
                        - method_error
                        if np.isfinite(
                            method_error
                        )
                        else np.nan
                    ),
                    "improved": (
                        bool(
                            raw_error
                            - method_error
                            > 0
                        )
                        if np.isfinite(
                            method_error
                        )
                        else False
                    ),
                    "n_stable_effort_cells": (
                        len(
                            stable_cells
                        )
                    ),
                }
            )

    state_summary = {
        "state_code": state,
        "raw_records": len(
            work
        ),
        "effort_taxa": len(
            effort_codes
        ),
        "evaluation_species": len(
            evaluation_codes
        ),
        "stable_effort_cells": len(
            stable_cells
        ),
    }

    return (
        pd.DataFrame(
            rows
        ),
        state_summary,
    )


def method_inference(
    frame,
    rng,
    progress,
):
    observed_rho = (
        stratified_rank_rho(
            frame,
            "gain",
        )
    )

    observed_auc = (
        stratified_auc(
            frame,
            "gain",
        )
    )

    permutation_rho_exceed = 0
    permutation_auc_exceed = 0

    for _ in range(
        N_PERMUTATIONS
    ):
        permuted = frame.copy()

        for _, indices in (
            frame.groupby(
                "state_code"
            ).groups.items()
        ):
            index_array = np.asarray(
                list(indices)
            )

            permuted.loc[
                index_array,
                "gain",
            ] = rng.permutation(
                frame.loc[
                    index_array,
                    "gain",
                ].to_numpy()
            )

        rho = stratified_rank_rho(
            permuted,
            "gain",
        )

        auc = stratified_auc(
            permuted,
            "gain",
        )

        permutation_rho_exceed += int(
            np.isfinite(
                rho
            )
            and rho
            >= observed_rho
        )

        permutation_auc_exceed += int(
            np.isfinite(
                auc
            )
            and auc
            >= observed_auc
        )

        progress.update(1)

    species_codes = np.array(
        sorted(
            frame[
                "fia_species_code"
            ].unique()
        ),
        dtype=int,
    )

    bootstrap_rho = []
    bootstrap_auc = []

    for _ in range(
        N_CLUSTER_BOOTSTRAP
    ):
        sampled_codes = rng.choice(
            species_codes,
            size=len(
                species_codes
            ),
            replace=True,
        )

        parts = []

        for bootstrap_id, species_code in enumerate(
            sampled_codes
        ):
            part = frame[
                frame[
                    "fia_species_code"
                ]
                == species_code
            ].copy()

            part[
                "bootstrap_cluster_id"
            ] = bootstrap_id

            parts.append(
                part
            )

        sample = pd.concat(
            parts,
            ignore_index=True,
        )

        rho = stratified_rank_rho(
            sample,
            "gain",
        )

        auc = stratified_auc(
            sample,
            "gain",
        )

        if np.isfinite(rho):
            bootstrap_rho.append(
                rho
            )

        if np.isfinite(auc):
            bootstrap_auc.append(
                auc
            )

        progress.update(1)

    bootstrap_rho = np.asarray(
        bootstrap_rho,
        dtype=float,
    )

    bootstrap_auc = np.asarray(
        bootstrap_auc,
        dtype=float,
    )

    return {
        "stratified_rho": (
            observed_rho
        ),
        "rho_permutation_p": (
            (
                1
                + permutation_rho_exceed
            )
            / (
                N_PERMUTATIONS
                + 1
            )
        ),
        "rho_bootstrap_low": float(
            np.quantile(
                bootstrap_rho,
                0.025,
            )
        ),
        "rho_bootstrap_high": float(
            np.quantile(
                bootstrap_rho,
                0.975,
            )
        ),
        "rho_bootstrap_positive_fraction": float(
            np.mean(
                bootstrap_rho
                > 0
            )
        ),
        "stratified_auc": (
            observed_auc
        ),
        "auc_permutation_p": (
            (
                1
                + permutation_auc_exceed
            )
            / (
                N_PERMUTATIONS
                + 1
            )
        ),
        "auc_bootstrap_low": float(
            np.quantile(
                bootstrap_auc,
                0.025,
            )
        ),
        "auc_bootstrap_high": float(
            np.quantile(
                bootstrap_auc,
                0.975,
            )
        ),
        "auc_bootstrap_above_half_fraction": float(
            np.mean(
                bootstrap_auc
                > 0.5
            )
        ),
    }


def print_failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## Cross-method correction benchmark failure — {RUN_UTC}
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
        *SPECIES_INPUT_PATHS.values(),
        *THINNED_PATHS.values(),
        *FIA_REFERENCE_PATHS.values(),
        *EFFORT_PLAN_PATHS.values(),
    ]

    missing_paths = [
        str(path)
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

    method_parts = []
    state_summary_rows = []

    for state in tqdm(
        STATES,
        desc="Building cross-method benchmark",
        unit="state",
    ):
        species_inputs = pd.read_parquet(
            SPECIES_INPUT_PATHS[
                state
            ]
        )

        thinned = pd.read_parquet(
            THINNED_PATHS[
                state
            ]
        )

        fia_reference = load_fia_reference(
            FIA_REFERENCE_PATHS[
                state
            ]
        )

        effort_codes = load_effort_codes(
            state,
            EFFORT_PLAN_PATHS[
                state
            ],
        )

        (
            state_methods,
            state_summary,
        ) = build_state_method_table(
            state=state,
            thinned=thinned,
            species_inputs=species_inputs,
            fia_reference=fia_reference,
            effort_codes=effort_codes,
        )

        method_parts.append(
            state_methods
        )

        state_summary_rows.append(
            state_summary
        )

    method_df = pd.concat(
        method_parts,
        ignore_index=True,
    )

    state_summary_df = pd.DataFrame(
        state_summary_rows
    )

    method_df.to_parquet(
        OUT_DIR
        / "species_region_method_results.parquet",
        index=False,
    )

    method_df.to_csv(
        OUT_DIR
        / "species_region_method_results.csv",
        index=False,
    )

    state_summary_df.to_csv(
        OUT_DIR
        / "state_processing_summary.csv",
        index=False,
    )

    print(
        "\nSTATE PROCESSING SUMMARY"
    )
    display(
        state_summary_df
    )

    print(
        "\nSPECIES–REGION–METHOD RESULTS"
    )
    display(
        method_df.sort_values(
            [
                "state_code",
                "method",
                "cell_jaccard",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
    )

    # --------------------------------------------------------
    # State-method performance and associations
    # --------------------------------------------------------
    state_method_rows = []

    for (
        state_code,
        method,
    ), group in method_df.groupby(
        [
            "state_code",
            "method",
        ]
    ):
        valid = group.dropna(
            subset=[
                "gain",
                "method_error",
                "cell_jaccard",
            ]
        )

        auc, pairs = binary_auc(
            valid[
                "cell_jaccard"
            ],
            valid[
                "gain"
            ]
            > 0,
        )

        state_method_rows.append(
            {
                "state_code": (
                    state_code
                ),
                "method": method,
                "n_species": len(
                    valid
                ),
                "median_raw_error": float(
                    valid[
                        "raw_error"
                    ].median()
                ),
                "median_method_error": float(
                    valid[
                        "method_error"
                    ].median()
                ),
                "median_gain": float(
                    valid[
                        "gain"
                    ].median()
                ),
                "improved_fraction": float(
                    (
                        valid[
                            "gain"
                        ]
                        > 0
                    ).mean()
                ),
                "spearman_cell_jaccard_gain": (
                    safe_spearman(
                        valid[
                            "cell_jaccard"
                        ],
                        valid[
                            "gain"
                        ],
                    )
                ),
                "auc_improved_vs_harmed": (
                    auc
                ),
                "auc_pairs": pairs,
            }
        )

    state_method_df = pd.DataFrame(
        state_method_rows
    )

    state_method_df.to_csv(
        OUT_DIR
        / "state_method_summary.csv",
        index=False,
    )

    print(
        "\nSTATE–METHOD SUMMARY"
    )
    display(
        state_method_df
    )

    # --------------------------------------------------------
    # Method-level trend inference
    # --------------------------------------------------------
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    total_iterations = (
        (
            len(
                METHODS
            )
            + 1
        )
        * (
            N_PERMUTATIONS
            + N_CLUSTER_BOOTSTRAP
        )
    )

    progress = tqdm(
        total=total_iterations,
        desc="Cross-method trend inference",
        unit="resample",
    )

    method_inference_rows = []

    for method in METHODS:
        method_frame = method_df[
            method_df[
                "method"
            ]
            == method
        ].dropna(
            subset=[
                "gain",
                "cell_jaccard",
            ]
        ).copy()

        inference = method_inference(
            method_frame,
            rng,
            progress,
        )

        method_inference_rows.append(
            {
                "method": method,
                "n_species_region_rows": len(
                    method_frame
                ),
                **inference,
            }
        )

    # --------------------------------------------------------
    # Consensus gain across methods
    # --------------------------------------------------------
    consensus_wide = (
        method_df.pivot_table(
            index=[
                "state_code",
                "fia_species_code",
                "cell_jaccard",
            ],
            columns="method",
            values="gain",
            aggfunc="first",
        )
        .reset_index()
    )

    consensus_wide[
        "median_gain_across_methods"
    ] = consensus_wide[
        METHODS
    ].median(
        axis=1,
        skipna=True,
    )

    consensus_wide[
        "positive_method_fraction"
    ] = (
        consensus_wide[
            METHODS
        ]
        .gt(0)
        .sum(
            axis=1
        )
        / consensus_wide[
            METHODS
        ]
        .notna()
        .sum(
            axis=1
        )
        .replace(
            0,
            np.nan,
        )
    )

    consensus_for_inference = (
        consensus_wide[
            [
                "state_code",
                "fia_species_code",
                "cell_jaccard",
                "median_gain_across_methods",
            ]
        ]
        .rename(
            columns={
                "median_gain_across_methods": (
                    "gain"
                )
            }
        )
        .dropna()
    )

    consensus_inference = method_inference(
        consensus_for_inference,
        rng,
        progress,
    )

    progress.close()

    method_inference_rows.append(
        {
            "method": (
                "cross_method_median"
            ),
            "n_species_region_rows": len(
                consensus_for_inference
            ),
            **consensus_inference,
        }
    )

    method_inference_df = pd.DataFrame(
        method_inference_rows
    )

    method_inference_df.to_csv(
        OUT_DIR
        / "method_trend_inference.csv",
        index=False,
    )

    consensus_wide.to_csv(
        OUT_DIR
        / "cross_method_species_consensus.csv",
        index=False,
    )

    print(
        "\nMETHOD TREND INFERENCE"
    )
    display(
        method_inference_df
    )

    print(
        "\nCROSS-METHOD SPECIES CONSENSUS"
    )
    display(
        consensus_wide
    )

    # --------------------------------------------------------
    # Trend decision
    # --------------------------------------------------------
    method_only = method_inference_df[
        method_inference_df[
            "method"
        ].isin(
            METHODS
        )
    ]

    consensus_row = method_inference_df[
        method_inference_df[
            "method"
        ]
        == "cross_method_median"
    ].iloc[0]

    n_positive_methods = int(
        (
            method_only[
                "stratified_rho"
            ]
            > 0
        ).sum()
    )

    n_methods_auc_065 = int(
        (
            method_only[
                "stratified_auc"
            ]
            >= 0.65
        ).sum()
    )

    trend_pass = bool(
        n_positive_methods
        >= MIN_POSITIVE_METHODS
        and consensus_row[
            "stratified_rho"
        ]
        >= MIN_CONSENSUS_RHO
        and consensus_row[
            "rho_permutation_p"
        ]
        <= MAX_CONSENSUS_PERMUTATION_P
        and consensus_row[
            "rho_bootstrap_positive_fraction"
        ]
        >= MIN_CONSENSUS_BOOTSTRAP_POSITIVE
        and n_methods_auc_065
        >= MIN_METHODS_WITH_AUC_065
    )

    if trend_pass:
        status = (
            "GO_FOR_FULL_CROSS_METHOD_CONFIRMATION"
        )

        passed = True

        next_step = (
            "Run a locked full cross-method confirmation with "
            "5,000 permutations, 5,000 FIA-species-cluster bootstraps, "
            "cross-grid analysis and negative controls. Do not choose "
            "a best method or optimize a cutoff."
        )

    else:
        status = (
            "CROSS_METHOD_GENERALIZATION_WEAK"
        )

        passed = False

        next_step = (
            "Do not claim a general support-overlap principle across "
            "correction methods. Retain cell Jaccard as specific to "
            "target-effort correction and proceed to the support-overlap "
            "phase-diagram experiment."
        )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_summary = (
        state_method_df.pivot(
            index="state_code",
            columns="method",
            values="median_method_error",
        )
    )

    x = np.arange(
        len(
            plot_summary.index
        )
    )

    width = (
        0.8
        / len(
            METHODS
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    for method_index, method in enumerate(
        METHODS
    ):
        plt.bar(
            x
            + (
                method_index
                - (
                    len(
                        METHODS
                    )
                    - 1
                )
                / 2
            )
            * width,
            plot_summary[
                method
            ],
            width=width,
            label=method,
        )

    plt.xticks(
        x,
        plot_summary.index,
    )

    plt.ylabel(
        "Median absolute error versus FIA (km/decade)"
    )

    plt.xlabel(
        "Region"
    )

    plt.title(
        "Real-data performance across correction strategies"
    )

    plt.legend()
    plt.tight_layout()

    performance_figure = (
        OUT_DIR
        / "cross_method_state_error.png"
    )

    plt.savefig(
        performance_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    display(
        Image(
            filename=str(
                performance_figure
            )
        )
    )

    plot_inference = method_inference_df.copy()

    x = np.arange(
        len(
            plot_inference
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - 0.2,
        plot_inference[
            "stratified_rho"
        ],
        width=0.4,
        label="Stratified rho",
    )

    plt.bar(
        x + 0.2,
        plot_inference[
            "stratified_auc"
        ],
        width=0.4,
        label="Stratified AUROC",
    )

    plt.axhline(
        0.5,
        linestyle="--",
    )

    plt.xticks(
        x,
        plot_inference[
            "method"
        ],
        rotation=20,
    )

    plt.ylabel(
        "Cross-region trend metric"
    )

    plt.xlabel(
        "Correction strategy"
    )

    plt.title(
        "Does spatial continuity generalize across methods?"
    )

    plt.legend()
    plt.tight_layout()

    inference_figure = (
        OUT_DIR
        / "cell_jaccard_cross_method_metrics.png"
    )

    plt.savefig(
        inference_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    display(
        Image(
            filename=str(
                inference_figure
            )
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    for state_code, state_frame in (
        consensus_wide.groupby(
            "state_code"
        )
    ):
        plt.scatter(
            state_frame[
                "cell_jaccard"
            ],
            state_frame[
                "median_gain_across_methods"
            ],
            label=state_code,
        )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "Cell Jaccard"
    )

    plt.ylabel(
        "Median correction gain across methods "
        "(km/decade)"
    )

    plt.title(
        "Cross-method applicability consensus"
    )

    plt.legend()
    plt.tight_layout()

    consensus_figure = (
        OUT_DIR
        / "cell_jaccard_cross_method_consensus.png"
    )

    plt.savefig(
        consensus_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    display(
        Image(
            filename=str(
                consensus_figure
            )
        )
    )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    append_readme(
        f"""

## Cross-method correction benchmark trend — {RUN_UTC}

### Q1 upgrade question
Test whether cell Jaccard predicts correction benefit across multiple correction
strategies rather than only target-group effort standardization.

### Formal data
- GBIF DOI: {GBIF_DOI}
- Download key: {DOWNLOAD_KEY}
- Regions: {STATES}
- Grid: {GRID_RESOLUTION} degrees
- Stable effort threshold:
  {MIN_TARGET_EFFORT_PER_CELL_PERIOD} records per cell-period
- GPU: not used

### Compared methods
- stable_frame_raw
- equal_cell
- observer_balanced
- target_effort

### Trend inference
- State-stratified permutations per method:
  {N_PERMUTATIONS}
- FIA-species-cluster bootstraps per method:
  {N_CLUSTER_BOOTSTRAP}
- Positive method associations:
  {n_positive_methods}/{len(METHODS)}
- Methods with AUROC >= 0.65:
  {n_methods_auc_065}/{len(METHODS)}
- Consensus rho:
  {consensus_row['stratified_rho']}
- Consensus permutation p:
  {consensus_row['rho_permutation_p']}
- Consensus bootstrap positive fraction:
  {consensus_row['rho_bootstrap_positive_fraction']}

### Decision
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Boundary
This is a trend-stage cross-method benchmark. No best method, optimized weight,
new applicability feature or cell-Jaccard cutoff was selected.
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
        "METHODS: "
        + json.dumps(
            METHODS
        )
    )
    print(
        "STATE_PROCESSING: "
        + json.dumps(
            state_summary_df.to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "STATE_METHOD_SUMMARY: "
        + json.dumps(
            state_method_df.round(
                6
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "METHOD_INFERENCE: "
        + json.dumps(
            method_inference_df.round(
                6
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        f"N_POSITIVE_METHODS: "
        f"{n_positive_methods}"
    )
    print(
        f"N_METHODS_AUC_GE_0P65: "
        f"{n_methods_auc_065}"
    )
    print(
        f"CONSENSUS_RHO: "
        f"{consensus_row['stratified_rho']}"
    )
    print(
        f"CONSENSUS_PERMUTATION_P: "
        f"{consensus_row['rho_permutation_p']}"
    )
    print(
        "CONSENSUS_BOOTSTRAP_POSITIVE_FRACTION: "
        f"{consensus_row['rho_bootstrap_positive_fraction']}"
    )
    print(
        f"PERFORMANCE_FIGURE: "
        f"{performance_figure}"
    )
    print(
        f"INFERENCE_FIGURE: "
        f"{inference_figure}"
    )
    print(
        f"CONSENSUS_FIGURE: "
        f"{consensus_figure}"
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
        "CELL21A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed table. "
            "Do not add correction methods or tune cell Jaccard."
        ),
    )
