
# Cell 18B — FIA-species-clustered robustness for the cell-Jaccard gate
#
# Reviewer concern
# ----------------
# The 43 rows are species-region observations, but the same FIA species may
# occur in multiple states. Rows are therefore not fully independent.
#
# This cell treats FIA species code as the dependence cluster and performs:
# 1. Cluster-robust OLS on state-stratified ranks.
# 2. Species-cluster bootstrap, resampling entire species across all states.
# 3. Leave-one-species-cluster-out validation.
# 4. Primary, partial-adjusted and cross-grid analyses.
#
# No new feature, cutoff, species or model weight is selected.
#
# Locked gate:
#   cell_jaccard
#
# Locked outcomes:
#   Primary: gain_grid_0p5
#   Cross-grid: grid_median_gain_km_decade
#
# Adjustment variables:
#   log_min_period_records
#   min_period_cells
#   raw_error_grid_0p5

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import rankdata, t as student_t
from tqdm.auto import tqdm
from IPython.display import display

# ============================================================
# Configuration
# ============================================================
PRIMARY_FEATURE = "cell_jaccard"
PRIMARY_OUTCOME = "gain_grid_0p5"
CROSS_GRID_OUTCOME = "grid_median_gain_km_decade"

CONTROL_COLUMNS = [
    "log_min_period_records",
    "min_period_cells",
    "raw_error_grid_0p5",
]

N_CLUSTER_BOOTSTRAP = 5_000
RANDOM_SEED = 20260711

PRIMARY_P_THRESHOLD = 0.05
PARTIAL_P_THRESHOLD = 0.10
CROSS_GRID_P_THRESHOLD = 0.05

MIN_PRIMARY_BOOTSTRAP_POSITIVE = 0.90
MIN_PARTIAL_BOOTSTRAP_POSITIVE = 0.80
MIN_CROSS_GRID_BOOTSTRAP_POSITIVE = 0.90
MIN_AUC_BOOTSTRAP_ABOVE_HALF = 0.90

MIN_PRIMARY_LOO_POSITIVE = 0.90
MIN_PARTIAL_LOO_POSITIVE = 0.80
MIN_CROSS_GRID_LOO_POSITIVE = 0.90

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
INPUT_DIR = (
    BASE_DIR
    / "derived"
    / "cell_jaccard_threshold_free_validation"
)
INPUT_PATH = (
    INPUT_DIR
    / "cell_jaccard_cross_region_species_table.parquet"
)

OUT_DIR = (
    BASE_DIR
    / "derived"
    / "cell_jaccard_species_cluster_robustness"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

GBIF_DOI = "10.15468/dl.sm6ygu"
DOWNLOAD_KEY = "0032735-260623161305970"

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("INPUT_PATH:", INPUT_PATH)
print("OUT_DIR:", OUT_DIR)
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
            "label": pd.Series(
                labels
            ).astype("boolean"),
        }
    ).dropna()

    positive = frame[
        frame["label"]
    ]["score"].to_numpy(
        dtype=float
    )

    negative = frame[
        ~frame["label"]
    ]["score"].to_numpy(
        dtype=float
    )

    if (
        len(positive) == 0
        or len(negative) == 0
    ):
        return np.nan

    comparisons = (
        positive[:, None]
        - negative[None, :]
    )

    return float(
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


def stratified_auc(frame, outcome_column):
    weighted_sum = 0.0
    total_pairs = 0

    for _, state_frame in frame.groupby(
        "state_code"
    ):
        labels = (
            state_frame[
                outcome_column
            ]
            > 0
        )

        positive_count = int(
            labels.sum()
        )
        negative_count = int(
            (~labels).sum()
        )

        n_pairs = (
            positive_count
            * negative_count
        )

        auc = binary_auc(
            state_frame[
                PRIMARY_FEATURE
            ],
            labels,
        )

        if (
            np.isfinite(auc)
            and n_pairs > 0
        ):
            weighted_sum += (
                auc
                * n_pairs
            )
            total_pairs += n_pairs

    return (
        float(
            weighted_sum
            / total_pairs
        )
        if total_pairs
        else np.nan
    )


def within_state_rank(
    frame,
    column,
):
    return (
        frame.groupby(
            "state_code"
        )[
            column
        ]
        .rank(
            method="average",
            pct=True,
        )
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
                len(values)
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


def prepare_rank_frame(
    frame,
    outcome_column,
):
    required_columns = list(
        dict.fromkeys(
            [
                "state_code",
                "fia_species_code",
                PRIMARY_FEATURE,
                outcome_column,
                *CONTROL_COLUMNS,
            ]
        )
    )

    work = frame[
        required_columns
    ].dropna().copy()

    work[
        "feature_rank"
    ] = within_state_rank(
        work,
        PRIMARY_FEATURE,
    )

    work[
        "outcome_rank"
    ] = within_state_rank(
        work,
        outcome_column,
    )

    for control in CONTROL_COLUMNS:
        work[
            f"{control}_rank"
        ] = within_state_rank(
            work,
            control,
        )

    return work


def stratified_rank_rho(
    frame,
    outcome_column,
):
    work = prepare_rank_frame(
        frame,
        outcome_column,
    )

    return safe_correlation(
        work[
            "feature_rank"
        ],
        work[
            "outcome_rank"
        ],
    )


def partial_rank_rho(
    frame,
    outcome_column,
):
    work = prepare_rank_frame(
        frame,
        outcome_column,
    )

    control_matrix = np.column_stack(
        [
            work[
                f"{control}_rank"
            ].to_numpy(
                dtype=float
            )
            for control in CONTROL_COLUMNS
        ]
    )

    feature_residual = residualize(
        work[
            "feature_rank"
        ].to_numpy(
            dtype=float
        ),
        control_matrix,
    )

    outcome_residual = residualize(
        work[
            "outcome_rank"
        ].to_numpy(
            dtype=float
        ),
        control_matrix,
    )

    return safe_correlation(
        feature_residual,
        outcome_residual,
    )


def cluster_robust_model(
    frame,
    outcome_column,
    adjusted,
):
    work = prepare_rank_frame(
        frame,
        outcome_column,
    )

    state_dummies = pd.get_dummies(
        work[
            "state_code"
        ],
        prefix="state",
        drop_first=True,
        dtype=float,
    )

    predictor_parts = [
        work[
            [
                "feature_rank",
            ]
        ].astype(float),
    ]

    if adjusted:
        predictor_parts.append(
            work[
                [
                    f"{control}_rank"
                    for control
                    in CONTROL_COLUMNS
                ]
            ].astype(float)
        )

    predictor_parts.append(
        state_dummies
    )

    design = pd.concat(
        predictor_parts,
        axis=1,
    )

    design = sm.add_constant(
        design,
        has_constant="add",
    )

    outcome = work[
        "outcome_rank"
    ].astype(float)

    groups = work[
        "fia_species_code"
    ].astype(int)

    model = sm.OLS(
        outcome,
        design,
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True,
        },
    )

    coefficient = float(
        result.params[
            "feature_rank"
        ]
    )

    standard_error = float(
        result.bse[
            "feature_rank"
        ]
    )

    n_clusters = int(
        groups.nunique()
    )

    degrees_freedom = max(
        n_clusters - 1,
        1,
    )

    t_statistic = (
        coefficient
        / standard_error
        if standard_error > 0
        else np.nan
    )

    one_sided_p = (
        float(
            student_t.sf(
                t_statistic,
                df=degrees_freedom,
            )
        )
        if np.isfinite(
            t_statistic
        )
        else np.nan
    )

    critical_value = float(
        student_t.ppf(
            0.975,
            df=degrees_freedom,
        )
    )

    confidence_low = (
        coefficient
        - critical_value
        * standard_error
    )

    confidence_high = (
        coefficient
        + critical_value
        * standard_error
    )

    return {
        "outcome": outcome_column,
        "adjusted": adjusted,
        "n_rows": len(
            work
        ),
        "n_species_clusters": (
            n_clusters
        ),
        "coefficient": (
            coefficient
        ),
        "cluster_robust_se": (
            standard_error
        ),
        "t_statistic": (
            t_statistic
        ),
        "degrees_freedom": (
            degrees_freedom
        ),
        "one_sided_p": (
            one_sided_p
        ),
        "confidence_low": (
            confidence_low
        ),
        "confidence_high": (
            confidence_high
        ),
        "r_squared": float(
            result.rsquared
        ),
    }


def cluster_bootstrap_sample(
    frame,
    species_codes,
    rng,
):
    sampled_codes = rng.choice(
        species_codes,
        size=len(
            species_codes
        ),
        replace=True,
    )

    parts = []

    for bootstrap_cluster_id, species_code in enumerate(
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
        ] = (
            bootstrap_cluster_id
        )

        parts.append(
            part
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def bootstrap_summary(values):
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

    if len(values) == 0:
        return {
            "low": np.nan,
            "high": np.nan,
            "positive_fraction": np.nan,
        }

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
        "positive_fraction": float(
            np.mean(
                values > 0
            )
        ),
    }


def failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## FIA-species-clustered robustness failure — {RUN_UTC}
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
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Cell 18A output: {INPUT_PATH}"
        )

    analysis_df = pd.read_parquet(
        INPUT_PATH
    )

    required_columns = {
        "state_code",
        "fia_species_code",
        PRIMARY_FEATURE,
        PRIMARY_OUTCOME,
        CROSS_GRID_OUTCOME,
        *CONTROL_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(
            analysis_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Input table is missing columns: "
            + " | ".join(
                sorted(
                    missing_columns
                )
            )
        )

    for column in [
        "fia_species_code",
        PRIMARY_FEATURE,
        PRIMARY_OUTCOME,
        CROSS_GRID_OUTCOME,
        *CONTROL_COLUMNS,
    ]:
        analysis_df[
            column
        ] = pd.to_numeric(
            analysis_df[
                column
            ],
            errors="coerce",
        )

    analysis_df = analysis_df.dropna(
        subset=[
            "state_code",
            "fia_species_code",
            PRIMARY_FEATURE,
            PRIMARY_OUTCOME,
            CROSS_GRID_OUTCOME,
            *CONTROL_COLUMNS,
        ]
    ).copy()

    analysis_df[
        "fia_species_code"
    ] = analysis_df[
        "fia_species_code"
    ].astype(int)

    species_codes = np.array(
        sorted(
            analysis_df[
                "fia_species_code"
            ].unique()
        ),
        dtype=int,
    )

    cluster_sizes_df = (
        analysis_df.groupby(
            "fia_species_code",
            as_index=False,
        )
        .agg(
            n_regions=(
                "state_code",
                "nunique",
            ),
            regions=(
                "state_code",
                lambda values: ",".join(
                    sorted(
                        set(
                            values
                        )
                    )
                ),
            ),
        )
        .sort_values(
            [
                "n_regions",
                "fia_species_code",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    overlap_summary_df = pd.DataFrame(
        [
            {
                "n_species_region_rows": len(
                    analysis_df
                ),
                "n_unique_species_clusters": len(
                    species_codes
                ),
                "species_in_1_region": int(
                    (
                        cluster_sizes_df[
                            "n_regions"
                        ]
                        == 1
                    ).sum()
                ),
                "species_in_2_regions": int(
                    (
                        cluster_sizes_df[
                            "n_regions"
                        ]
                        == 2
                    ).sum()
                ),
                "species_in_3_regions": int(
                    (
                        cluster_sizes_df[
                            "n_regions"
                        ]
                        == 3
                    ).sum()
                ),
            }
        ]
    )

    cluster_sizes_df.to_csv(
        OUT_DIR
        / "species_cluster_overlap.csv",
        index=False,
    )

    print(
        "\nSPECIES CLUSTER OVERLAP"
    )
    display(
        overlap_summary_df
    )
    display(
        cluster_sizes_df
    )

    # --------------------------------------------------------
    # Cluster-robust models
    # --------------------------------------------------------
    model_rows = [
        cluster_robust_model(
            analysis_df,
            PRIMARY_OUTCOME,
            adjusted=False,
        ),
        cluster_robust_model(
            analysis_df,
            PRIMARY_OUTCOME,
            adjusted=True,
        ),
        cluster_robust_model(
            analysis_df,
            CROSS_GRID_OUTCOME,
            adjusted=False,
        ),
        cluster_robust_model(
            analysis_df,
            CROSS_GRID_OUTCOME,
            adjusted=True,
        ),
    ]

    model_df = pd.DataFrame(
        model_rows
    )

    model_df.to_csv(
        OUT_DIR
        / "cluster_robust_models.csv",
        index=False,
    )

    print(
        "\nCLUSTER-ROBUST MODELS"
    )
    display(
        model_df
    )

    # --------------------------------------------------------
    # Species-cluster bootstrap
    # --------------------------------------------------------
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_rows = []

    for replicate in tqdm(
        range(
            N_CLUSTER_BOOTSTRAP
        ),
        desc="FIA-species cluster bootstrap",
        unit="replicate",
    ):
        sample = cluster_bootstrap_sample(
            analysis_df,
            species_codes,
            rng,
        )

        bootstrap_rows.append(
            {
                "replicate": replicate,
                "primary_rank_rho": (
                    stratified_rank_rho(
                        sample,
                        PRIMARY_OUTCOME,
                    )
                ),
                "partial_rank_rho": (
                    partial_rank_rho(
                        sample,
                        PRIMARY_OUTCOME,
                    )
                ),
                "cross_grid_rank_rho": (
                    stratified_rank_rho(
                        sample,
                        CROSS_GRID_OUTCOME,
                    )
                ),
                "stratified_auc": (
                    stratified_auc(
                        sample,
                        PRIMARY_OUTCOME,
                    )
                ),
            }
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    bootstrap_df.to_parquet(
        OUT_DIR
        / "species_cluster_bootstrap.parquet",
        index=False,
    )

    primary_bootstrap = bootstrap_summary(
        bootstrap_df[
            "primary_rank_rho"
        ]
    )

    partial_bootstrap = bootstrap_summary(
        bootstrap_df[
            "partial_rank_rho"
        ]
    )

    cross_grid_bootstrap = bootstrap_summary(
        bootstrap_df[
            "cross_grid_rank_rho"
        ]
    )

    auc_values = pd.to_numeric(
        bootstrap_df[
            "stratified_auc"
        ],
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    auc_bootstrap_summary = {
        "low": (
            float(
                np.quantile(
                    auc_values,
                    0.025,
                )
            )
            if len(
                auc_values
            )
            else np.nan
        ),
        "high": (
            float(
                np.quantile(
                    auc_values,
                    0.975,
                )
            )
            if len(
                auc_values
            )
            else np.nan
        ),
        "above_half_fraction": (
            float(
                np.mean(
                    auc_values > 0.5
                )
            )
            if len(
                auc_values
            )
            else np.nan
        ),
    }

    bootstrap_summary_df = pd.DataFrame(
        [
            {
                "metric": "primary_rank_rho",
                **primary_bootstrap,
            },
            {
                "metric": "partial_rank_rho",
                **partial_bootstrap,
            },
            {
                "metric": "cross_grid_rank_rho",
                **cross_grid_bootstrap,
            },
            {
                "metric": "stratified_auc",
                "low": auc_bootstrap_summary[
                    "low"
                ],
                "high": auc_bootstrap_summary[
                    "high"
                ],
                "positive_fraction": (
                    auc_bootstrap_summary[
                        "above_half_fraction"
                    ]
                ),
            },
        ]
    )

    bootstrap_summary_df.to_csv(
        OUT_DIR
        / "species_cluster_bootstrap_summary.csv",
        index=False,
    )

    print(
        "\nSPECIES-CLUSTER BOOTSTRAP SUMMARY"
    )
    display(
        bootstrap_summary_df
    )

    # --------------------------------------------------------
    # Leave-one-species-cluster-out
    # --------------------------------------------------------
    loo_rows = []

    for species_code in tqdm(
        species_codes,
        desc="Leave-one-FIA-species-out",
        unit="species",
    ):
        reduced = analysis_df[
            analysis_df[
                "fia_species_code"
            ]
            != species_code
        ].copy()

        loo_rows.append(
            {
                "dropped_species_code": int(
                    species_code
                ),
                "dropped_regions": ",".join(
                    sorted(
                        analysis_df.loc[
                            analysis_df[
                                "fia_species_code"
                            ]
                            == species_code,
                            "state_code",
                        ].unique()
                    )
                ),
                "remaining_rows": len(
                    reduced
                ),
                "remaining_clusters": int(
                    reduced[
                        "fia_species_code"
                    ].nunique()
                ),
                "primary_rank_rho": (
                    stratified_rank_rho(
                        reduced,
                        PRIMARY_OUTCOME,
                    )
                ),
                "partial_rank_rho": (
                    partial_rank_rho(
                        reduced,
                        PRIMARY_OUTCOME,
                    )
                ),
                "cross_grid_rank_rho": (
                    stratified_rank_rho(
                        reduced,
                        CROSS_GRID_OUTCOME,
                    )
                ),
                "stratified_auc": (
                    stratified_auc(
                        reduced,
                        PRIMARY_OUTCOME,
                    )
                ),
            }
        )

    loo_df = pd.DataFrame(
        loo_rows
    )

    loo_df.to_csv(
        OUT_DIR
        / "leave_one_species_cluster_out.csv",
        index=False,
    )

    loo_summary_df = pd.DataFrame(
        [
            {
                "metric": "primary_rank_rho",
                "minimum": float(
                    loo_df[
                        "primary_rank_rho"
                    ].min()
                ),
                "maximum": float(
                    loo_df[
                        "primary_rank_rho"
                    ].max()
                ),
                "positive_fraction": float(
                    (
                        loo_df[
                            "primary_rank_rho"
                        ]
                        > 0
                    ).mean()
                ),
            },
            {
                "metric": "partial_rank_rho",
                "minimum": float(
                    loo_df[
                        "partial_rank_rho"
                    ].min()
                ),
                "maximum": float(
                    loo_df[
                        "partial_rank_rho"
                    ].max()
                ),
                "positive_fraction": float(
                    (
                        loo_df[
                            "partial_rank_rho"
                        ]
                        > 0
                    ).mean()
                ),
            },
            {
                "metric": "cross_grid_rank_rho",
                "minimum": float(
                    loo_df[
                        "cross_grid_rank_rho"
                    ].min()
                ),
                "maximum": float(
                    loo_df[
                        "cross_grid_rank_rho"
                    ].max()
                ),
                "positive_fraction": float(
                    (
                        loo_df[
                            "cross_grid_rank_rho"
                        ]
                        > 0
                    ).mean()
                ),
            },
            {
                "metric": "stratified_auc",
                "minimum": float(
                    loo_df[
                        "stratified_auc"
                    ].min()
                ),
                "maximum": float(
                    loo_df[
                        "stratified_auc"
                    ].max()
                ),
                "positive_fraction": float(
                    (
                        loo_df[
                            "stratified_auc"
                        ]
                        > 0.5
                    ).mean()
                ),
            },
        ]
    )

    loo_summary_df.to_csv(
        OUT_DIR
        / "leave_one_species_cluster_summary.csv",
        index=False,
    )

    print(
        "\nLEAVE-ONE-SPECIES-CLUSTER SUMMARY"
    )
    display(
        loo_summary_df
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------
    primary_model = model_df[
        (
            model_df[
                "outcome"
            ]
            == PRIMARY_OUTCOME
        )
        & (
            ~model_df[
                "adjusted"
            ]
        )
    ].iloc[0]

    partial_model = model_df[
        (
            model_df[
                "outcome"
            ]
            == PRIMARY_OUTCOME
        )
        & (
            model_df[
                "adjusted"
            ]
        )
    ].iloc[0]

    cross_grid_model = model_df[
        (
            model_df[
                "outcome"
            ]
            == CROSS_GRID_OUTCOME
        )
        & (
            ~model_df[
                "adjusted"
            ]
        )
    ].iloc[0]

    primary_model_pass = bool(
        primary_model[
            "coefficient"
        ]
        > 0
        and primary_model[
            "one_sided_p"
        ]
        <= PRIMARY_P_THRESHOLD
    )

    partial_model_pass = bool(
        partial_model[
            "coefficient"
        ]
        > 0
        and partial_model[
            "one_sided_p"
        ]
        <= PARTIAL_P_THRESHOLD
    )

    cross_grid_model_pass = bool(
        cross_grid_model[
            "coefficient"
        ]
        > 0
        and cross_grid_model[
            "one_sided_p"
        ]
        <= CROSS_GRID_P_THRESHOLD
    )

    bootstrap_pass = bool(
        primary_bootstrap[
            "positive_fraction"
        ]
        >= MIN_PRIMARY_BOOTSTRAP_POSITIVE
        and partial_bootstrap[
            "positive_fraction"
        ]
        >= MIN_PARTIAL_BOOTSTRAP_POSITIVE
        and cross_grid_bootstrap[
            "positive_fraction"
        ]
        >= MIN_CROSS_GRID_BOOTSTRAP_POSITIVE
        and auc_bootstrap_summary[
            "above_half_fraction"
        ]
        >= MIN_AUC_BOOTSTRAP_ABOVE_HALF
    )

    loo_pass = bool(
        loo_summary_df.loc[
            loo_summary_df[
                "metric"
            ]
            == "primary_rank_rho",
            "positive_fraction",
        ].iloc[0]
        >= MIN_PRIMARY_LOO_POSITIVE
        and loo_summary_df.loc[
            loo_summary_df[
                "metric"
            ]
            == "partial_rank_rho",
            "positive_fraction",
        ].iloc[0]
        >= MIN_PARTIAL_LOO_POSITIVE
        and loo_summary_df.loc[
            loo_summary_df[
                "metric"
            ]
            == "cross_grid_rank_rho",
            "positive_fraction",
        ].iloc[0]
        >= MIN_CROSS_GRID_LOO_POSITIVE
    )

    passed = bool(
        primary_model_pass
        and partial_model_pass
        and cross_grid_model_pass
        and bootstrap_pass
        and loo_pass
    )

    if passed:
        status = (
            "CELL_JACCARD_SPECIES_CLUSTER_ROBUSTNESS_PASSED"
        )
        next_step = (
            "Freeze the computational experiments. Generate final "
            "manuscript tables, figures, methods, limitations and "
            "supplementary reporting checklist."
        )
    elif (
        primary_model_pass
        and cross_grid_model_pass
        and bootstrap_pass
        and loo_pass
    ):
        status = (
            "CELL_JACCARD_CLUSTER_ROBUST_PARTIAL_ADJUSTMENT"
        )
        next_step = (
            "Freeze experiments. Present cell Jaccard as a replicated "
            "ranking diagnostic, but state that adjusted evidence is weaker."
        )
    else:
        status = (
            "CELL_JACCARD_CLUSTER_ROBUSTNESS_INCOMPLETE"
        )
        next_step = (
            "Do not define an applicability score. Retain cell Jaccard "
            "as a replicated association and emphasize correction "
            "non-transportability."
        )

    # --------------------------------------------------------
    # Manuscript-ready result
    # --------------------------------------------------------
    manuscript_text = f"""
FIA-SPECIES-CLUSTERED ROBUSTNESS RESULT

The cross-region analysis contained {len(analysis_df)} species-region rows from
{len(species_codes)} unique FIA species clusters. Dependence caused by the same
species appearing in multiple states was addressed using species-clustered
standard errors, species-cluster bootstrap resampling and leave-one-species-out
analysis.

Primary cluster-robust rank model:
- coefficient = {primary_model['coefficient']:.4f}
- cluster-robust SE = {primary_model['cluster_robust_se']:.4f}
- one-sided P = {primary_model['one_sided_p']:.6f}
- 95% CI =
  [{primary_model['confidence_low']:.4f},
   {primary_model['confidence_high']:.4f}]

Adjusted cluster-robust rank model:
- coefficient = {partial_model['coefficient']:.4f}
- cluster-robust SE = {partial_model['cluster_robust_se']:.4f}
- one-sided P = {partial_model['one_sided_p']:.6f}
- 95% CI =
  [{partial_model['confidence_low']:.4f},
   {partial_model['confidence_high']:.4f}]

Species-cluster bootstrap:
- primary rank 95% interval =
  [{primary_bootstrap['low']:.4f},
   {primary_bootstrap['high']:.4f}]
- primary positive fraction =
  {primary_bootstrap['positive_fraction']:.4f}
- adjusted positive fraction =
  {partial_bootstrap['positive_fraction']:.4f}
- cross-grid positive fraction =
  {cross_grid_bootstrap['positive_fraction']:.4f}
- AUROC above 0.5 fraction =
  {auc_bootstrap_summary['above_half_fraction']:.4f}

Leave-one-species-cluster-out:
- primary positive fraction =
  {loo_summary_df.loc[
      loo_summary_df['metric'] == 'primary_rank_rho',
      'positive_fraction'
  ].iloc[0]:.4f}
- adjusted positive fraction =
  {loo_summary_df.loc[
      loo_summary_df['metric'] == 'partial_rank_rho',
      'positive_fraction'
  ].iloc[0]:.4f}
- cross-grid positive fraction =
  {loo_summary_df.loc[
      loo_summary_df['metric'] == 'cross_grid_rank_rho',
      'positive_fraction'
  ].iloc[0]:.4f}

Interpretation:
The cell-Jaccard applicability ranking was evaluated after accounting for
cross-state dependence of repeated species. The signal remains an association-
based ranking diagnostic rather than a causal mechanism or optimized cutoff.

Decision:
{status}
"""

    manuscript_path = (
        OUT_DIR
        / "manuscript_ready_species_cluster_result.txt"
    )

    manuscript_path.write_text(
        manuscript_text.strip()
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        bootstrap_df[
            "primary_rank_rho"
        ].dropna(),
        bins=35,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "State-stratified rank association"
    )

    plt.ylabel(
        "Species-cluster bootstrap frequency"
    )

    plt.title(
        "Cluster-bootstrap stability of the cell-Jaccard gate"
    )

    plt.tight_layout()

    bootstrap_figure_path = (
        OUT_DIR
        / "species_cluster_bootstrap_primary_rho.png"
    )

    plt.savefig(
        bootstrap_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plot_loo = loo_df.sort_values(
        "primary_rank_rho"
    ).reset_index(
        drop=True
    )

    y = np.arange(
        len(
            plot_loo
        )
    )

    plt.figure(
        figsize=(
            9,
            max(
                6,
                0.35
                * len(
                    plot_loo
                ),
            ),
        )
    )

    plt.scatter(
        plot_loo[
            "primary_rank_rho"
        ],
        y,
        label="Primary",
    )

    plt.scatter(
        plot_loo[
            "partial_rank_rho"
        ],
        y,
        label="Adjusted",
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.yticks(
        y,
        plot_loo[
            "dropped_species_code"
        ].astype(str),
    )

    plt.xlabel(
        "Association after dropping one FIA species"
    )

    plt.ylabel(
        "Dropped FIA species code"
    )

    plt.title(
        "Leave-one-species-cluster-out robustness"
    )

    plt.legend()
    plt.tight_layout()

    loo_figure_path = (
        OUT_DIR
        / "leave_one_species_cluster_robustness.png"
    )

    plt.savefig(
        loo_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    append_readme(
        f"""

## FIA-species-clustered cell-Jaccard robustness — {RUN_UTC}

### Formal data
- GBIF DOI: {GBIF_DOI}
- Download key: {DOWNLOAD_KEY}
- Species-region rows: {len(analysis_df)}
- Unique FIA species clusters: {len(species_codes)}
- Cluster bootstrap replicates: {N_CLUSTER_BOOTSTRAP}
- GPU: not used

### Dependence structure
- Species in one region:
  {overlap_summary_df['species_in_1_region'].iloc[0]}
- Species in two regions:
  {overlap_summary_df['species_in_2_regions'].iloc[0]}
- Species in three regions:
  {overlap_summary_df['species_in_3_regions'].iloc[0]}

### Cluster-robust model decisions
- Primary model passed:
  {primary_model_pass}
- Adjusted model passed:
  {partial_model_pass}
- Cross-grid model passed:
  {cross_grid_model_pass}
- Species-cluster bootstrap passed:
  {bootstrap_pass}
- Leave-one-species-cluster-out passed:
  {loo_pass}

### Decision
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
The same species may appear in multiple states. All inferential robustness in
this cell clusters or resamples at FIA species code. Cell Jaccard remains a
threshold-free applicability ranking diagnostic, not a causal mechanism.
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
        f"N_SPECIES_REGION_ROWS: "
        f"{len(analysis_df)}"
    )
    print(
        f"N_UNIQUE_FIA_SPECIES_CLUSTERS: "
        f"{len(species_codes)}"
    )
    print(
        "CLUSTER_OVERLAP: "
        + json.dumps(
            overlap_summary_df.to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "CLUSTER_ROBUST_MODELS: "
        + json.dumps(
            model_df.round(
                6
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "CLUSTER_BOOTSTRAP_SUMMARY: "
        + json.dumps(
            bootstrap_summary_df.round(
                6
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "LEAVE_ONE_SPECIES_SUMMARY: "
        + json.dumps(
            loo_summary_df.round(
                6
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        f"PRIMARY_MODEL_PASSED: "
        f"{primary_model_pass}"
    )
    print(
        f"PARTIAL_MODEL_PASSED: "
        f"{partial_model_pass}"
    )
    print(
        f"CROSS_GRID_MODEL_PASSED: "
        f"{cross_grid_model_pass}"
    )
    print(
        f"CLUSTER_BOOTSTRAP_PASSED: "
        f"{bootstrap_pass}"
    )
    print(
        f"LEAVE_ONE_SPECIES_PASSED: "
        f"{loo_pass}"
    )
    print(
        f"MANUSCRIPT_RESULT_PATH: "
        f"{manuscript_path}"
    )
    print(
        f"BOOTSTRAP_FIGURE_PATH: "
        f"{bootstrap_figure_path}"
    )
    print(
        f"LOO_FIGURE_PATH: "
        f"{loo_figure_path}"
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
    failure(
        "CELL18B_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Save this diagnostic summary and the last displayed table. "
            "Do not add new features or define a cutoff."
        ),
    )
