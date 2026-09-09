
# Cell 18A — Threshold-free validation of the replicated cell-Jaccard gate
#
# Final replicated gate
# ---------------------
# cell_jaccard
#
# Scientific question
# -------------------
# Without selecting a cutoff, does higher spatial-support continuity rank
# species for which target-effort correction is more likely to improve
# agreement with FIA?
#
# This cell also addresses a major reviewer concern:
# cell_jaccard and correction gain are derived from the same occurrence data.
# We therefore test whether the association remains after controlling for:
# - minimum early/late record support;
# - minimum early/late occupied-cell support;
# - raw 0.5-degree error versus FIA.
#
# No feature search, threshold optimization or model fitting is performed.
#
# Primary threshold-free metrics
# ------------------------------
# 1. State-stratified Spearman rank association with 0.5° correction gain.
# 2. State-stratified AUROC for ranking improved versus harmed species.
# 3. Pairwise ranking concordance.
# 4. Partial rank association after support/error adjustment.
# 5. Cross-grid validation using median gain across 0.25°, 0.5° and 1.0°.
#
# Locked interpretation
# ---------------------
# cell_jaccard is an applicability ranking signal, not a causal mechanism and
# not a binary cutoff.

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from tqdm.auto import tqdm
from IPython.display import display

# ============================================================
# Configuration
# ============================================================
STATES = ["PA", "VA", "NC"]

PRIMARY_FEATURE = "cell_jaccard"
PRIMARY_GAIN = "gain_grid_0p5"
CROSS_GRID_GAIN = "grid_median_gain_km_decade"

N_PERMUTATIONS = 5_000
N_BOOTSTRAP = 3_000
RANDOM_SEED = 20260711

MIN_STRATIFIED_AUC = 0.65
MAX_PRIMARY_PERMUTATION_P = 0.05
MIN_BOOTSTRAP_POSITIVE = 0.90
MIN_POSITIVE_STATES = 2
MAX_PARTIAL_PERMUTATION_P = 0.10

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
INPUT_DIR = (
    BASE_DIR
    / "derived"
    / "registered_gbif_formal_parity"
)
OUT_DIR = (
    BASE_DIR
    / "derived"
    / "cell_jaccard_threshold_free_validation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES_PATHS = {
    state: INPUT_DIR / f"{state}_formal_species_inputs.parquet"
    for state in STATES
}

THINNED_PATHS = {
    state: INPUT_DIR / f"{state}_formal_thinned.parquet"
    for state in STATES
}

GBIF_DOI = "10.15468/dl.sm6ygu"
DOWNLOAD_KEY = "0032735-260623161305970"

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("PRIMARY_FEATURE:", PRIMARY_FEATURE)
print("PRIMARY_GAIN:", PRIMARY_GAIN)
print("N_PERMUTATIONS:", N_PERMUTATIONS)
print("N_BOOTSTRAP:", N_BOOTSTRAP)
print("OUT_DIR:", OUT_DIR)
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


def pairwise_concordance(feature, outcome):
    frame = pd.DataFrame(
        {
            "feature": pd.to_numeric(
                pd.Series(feature),
                errors="coerce",
            ),
            "outcome": pd.to_numeric(
                pd.Series(outcome),
                errors="coerce",
            ),
        }
    ).dropna()

    feature_values = frame[
        "feature"
    ].to_numpy(
        dtype=float
    )

    outcome_values = frame[
        "outcome"
    ].to_numpy(
        dtype=float
    )

    concordant = 0.0
    comparable = 0

    for first in range(
        len(frame)
    ):
        for second in range(
            first + 1,
            len(frame),
        ):
            feature_difference = (
                feature_values[first]
                - feature_values[second]
            )

            outcome_difference = (
                outcome_values[first]
                - outcome_values[second]
            )

            if (
                feature_difference == 0
                or outcome_difference == 0
            ):
                continue

            comparable += 1

            if (
                feature_difference
                * outcome_difference
                > 0
            ):
                concordant += 1.0

    return (
        float(
            concordant
            / comparable
        )
        if comparable
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


def stratified_rank_rho(
    frame,
    outcome_column,
):
    work = frame[
        [
            "state_code",
            PRIMARY_FEATURE,
            outcome_column,
        ]
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

    return safe_correlation(
        work[
            "feature_rank"
        ],
        work[
            "outcome_rank"
        ],
    )


def stratified_auc(
    frame,
    gain_column,
):
    weighted_auc_sum = 0.0
    total_pairs = 0

    for _, state_frame in frame.groupby(
        "state_code"
    ):
        auc, n_pairs = binary_auc(
            state_frame[
                PRIMARY_FEATURE
            ],
            state_frame[
                gain_column
            ]
            > 0,
        )

        if (
            np.isfinite(auc)
            and n_pairs > 0
        ):
            weighted_auc_sum += (
                auc
                * n_pairs
            )
            total_pairs += n_pairs

    return (
        float(
            weighted_auc_sum
            / total_pairs
        )
        if total_pairs
        else np.nan
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


def partial_rank_rho(
    frame,
    outcome_column,
):
    required = [
        "state_code",
        PRIMARY_FEATURE,
        outcome_column,
        "log_min_period_records",
        "min_period_cells",
        "raw_error_grid_0p5",
    ]

    work = frame[
        required
    ].dropna().copy()

    ranked_columns = {}

    for column in [
        PRIMARY_FEATURE,
        outcome_column,
        "log_min_period_records",
        "min_period_cells",
        "raw_error_grid_0p5",
    ]:
        ranked_columns[
            column
        ] = within_state_rank(
            work,
            column,
        ).to_numpy(
            dtype=float
        )

    feature_residual = residualize(
        ranked_columns[
            PRIMARY_FEATURE
        ],
        np.column_stack(
            [
                ranked_columns[
                    "log_min_period_records"
                ],
                ranked_columns[
                    "min_period_cells"
                ],
                ranked_columns[
                    "raw_error_grid_0p5"
                ],
            ]
        ),
    )

    outcome_residual = residualize(
        ranked_columns[
            outcome_column
        ],
        np.column_stack(
            [
                ranked_columns[
                    "log_min_period_records"
                ],
                ranked_columns[
                    "min_period_cells"
                ],
                ranked_columns[
                    "raw_error_grid_0p5"
                ],
            ]
        ),
    )

    return safe_correlation(
        feature_residual,
        outcome_residual,
    )


def permute_within_states(
    frame,
    column,
    rng,
):
    permuted = frame.copy()

    for state_code, indices in (
        frame.groupby(
            "state_code"
        ).groups.items()
    ):
        index_array = np.asarray(
            list(indices)
        )

        permuted.loc[
            index_array,
            column,
        ] = rng.permutation(
            frame.loc[
                index_array,
                column,
            ].to_numpy()
        )

    return permuted


def bootstrap_within_states(
    frame,
    rng,
):
    parts = []

    for state_code, state_frame in frame.groupby(
        "state_code"
    ):
        sampled_indices = rng.integers(
            0,
            len(
                state_frame
            ),
            size=len(
                state_frame
            ),
        )

        sample = (
            state_frame.iloc[
                sampled_indices
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        sample[
            "state_code"
        ] = state_code

        parts.append(
            sample
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def inference_for_metric(
    frame,
    outcome_column,
    rng,
    progress,
):
    observed_rank_rho = (
        stratified_rank_rho(
            frame,
            outcome_column,
        )
    )

    observed_auc = (
        stratified_auc(
            frame,
            outcome_column,
        )
    )

    observed_partial_rho = (
        partial_rank_rho(
            frame,
            outcome_column,
        )
    )

    permutation_rank_exceed = 0
    permutation_auc_exceed = 0
    permutation_partial_exceed = 0

    for _ in range(
        N_PERMUTATIONS
    ):
        permuted = permute_within_states(
            frame,
            outcome_column,
            rng,
        )

        rank_rho = stratified_rank_rho(
            permuted,
            outcome_column,
        )

        auc = stratified_auc(
            permuted,
            outcome_column,
        )

        partial_rho = partial_rank_rho(
            permuted,
            outcome_column,
        )

        permutation_rank_exceed += int(
            np.isfinite(
                rank_rho
            )
            and rank_rho
            >= observed_rank_rho
        )

        permutation_auc_exceed += int(
            np.isfinite(
                auc
            )
            and auc
            >= observed_auc
        )

        permutation_partial_exceed += int(
            np.isfinite(
                partial_rho
            )
            and partial_rho
            >= observed_partial_rho
        )

        progress.update(1)

    bootstrap_rank = []
    bootstrap_auc = []
    bootstrap_partial = []

    for _ in range(
        N_BOOTSTRAP
    ):
        sample = bootstrap_within_states(
            frame,
            rng,
        )

        rank_rho = stratified_rank_rho(
            sample,
            outcome_column,
        )

        auc = stratified_auc(
            sample,
            outcome_column,
        )

        partial_rho = partial_rank_rho(
            sample,
            outcome_column,
        )

        if np.isfinite(
            rank_rho
        ):
            bootstrap_rank.append(
                rank_rho
            )

        if np.isfinite(
            auc
        ):
            bootstrap_auc.append(
                auc
            )

        if np.isfinite(
            partial_rho
        ):
            bootstrap_partial.append(
                partial_rho
            )

        progress.update(1)

    bootstrap_rank = np.asarray(
        bootstrap_rank,
        dtype=float,
    )

    bootstrap_auc = np.asarray(
        bootstrap_auc,
        dtype=float,
    )

    bootstrap_partial = np.asarray(
        bootstrap_partial,
        dtype=float,
    )

    return {
        "outcome": outcome_column,
        "n_species": len(
            frame
        ),
        "stratified_rank_rho": (
            observed_rank_rho
        ),
        "rank_permutation_p": (
            (
                1
                + permutation_rank_exceed
            )
            / (
                N_PERMUTATIONS
                + 1
            )
        ),
        "rank_bootstrap_low": float(
            np.quantile(
                bootstrap_rank,
                0.025,
            )
        ),
        "rank_bootstrap_high": float(
            np.quantile(
                bootstrap_rank,
                0.975,
            )
        ),
        "rank_bootstrap_positive_fraction": float(
            np.mean(
                bootstrap_rank > 0
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
                bootstrap_auc > 0.5
            )
        ),
        "partial_rank_rho": (
            observed_partial_rho
        ),
        "partial_permutation_p": (
            (
                1
                + permutation_partial_exceed
            )
            / (
                N_PERMUTATIONS
                + 1
            )
        ),
        "partial_bootstrap_low": float(
            np.quantile(
                bootstrap_partial,
                0.025,
            )
        ),
        "partial_bootstrap_high": float(
            np.quantile(
                bootstrap_partial,
                0.975,
            )
        ),
        "partial_bootstrap_positive_fraction": float(
            np.mean(
                bootstrap_partial > 0
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

## Cell-Jaccard ranking validation failure — {RUN_UTC}
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
        *SPECIES_PATHS.values(),
        *THINNED_PATHS.values(),
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing formal input files: "
            + " | ".join(
                missing_paths
            )
        )

    combined_parts = []

    for state_code in tqdm(
        STATES,
        desc="Building cross-region ranking table",
        unit="state",
    ):
        species_df = pd.read_parquet(
            SPECIES_PATHS[
                state_code
            ]
        )

        thinned_df = pd.read_parquet(
            THINNED_PATHS[
                state_code
            ]
        )

        species_df[
            "fia_species_code"
        ] = pd.to_numeric(
            species_df[
                "fia_species_code"
            ],
            errors="coerce",
        )

        species_df = species_df.dropna(
            subset=[
                "fia_species_code",
                PRIMARY_FEATURE,
                PRIMARY_GAIN,
                CROSS_GRID_GAIN,
                "raw_error_grid_0p5",
            ]
        ).copy()

        species_df[
            "fia_species_code"
        ] = species_df[
            "fia_species_code"
        ].astype(int)

        thinned_df[
            "fia_species_code"
        ] = pd.to_numeric(
            thinned_df[
                "fia_species_code"
            ],
            errors="coerce",
        )

        thinned_df = thinned_df.dropna(
            subset=[
                "fia_species_code",
                "period",
            ]
        ).copy()

        thinned_df[
            "fia_species_code"
        ] = thinned_df[
            "fia_species_code"
        ].astype(int)

        support_df = (
            thinned_df.groupby(
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
                n_observers=(
                    "observer_label",
                    "nunique",
                ),
            )
        )

        support_wide = support_df.pivot(
            index="fia_species_code",
            columns="period",
            values=[
                "n_records",
                "n_observers",
            ],
        )

        support_wide.columns = [
            f"{metric}_{period}"
            for metric, period
            in support_wide.columns
        ]

        support_wide = (
            support_wide.reset_index()
        )

        state_df = species_df.merge(
            support_wide,
            on="fia_species_code",
            how="left",
        )

        state_df[
            "state_code"
        ] = state_code

        state_df[
            "min_period_records"
        ] = state_df[
            [
                "n_records_early",
                "n_records_late",
            ]
        ].min(
            axis=1
        )

        state_df[
            "log_min_period_records"
        ] = np.log1p(
            state_df[
                "min_period_records"
            ]
        )

        state_df[
            "min_period_cells"
        ] = state_df[
            [
                "early_cells",
                "late_cells",
            ]
        ].min(
            axis=1
        )

        state_df[
            "improved_0p5"
        ] = (
            state_df[
                PRIMARY_GAIN
            ]
            > 0
        )

        combined_parts.append(
            state_df
        )

    analysis_df = pd.concat(
        combined_parts,
        ignore_index=True,
    )

    analysis_df[
        "cell_jaccard_percentile"
    ] = within_state_rank(
        analysis_df,
        PRIMARY_FEATURE,
    )

    analysis_df[
        "gain_percentile"
    ] = within_state_rank(
        analysis_df,
        PRIMARY_GAIN,
    )

    analysis_df.to_parquet(
        OUT_DIR
        / "cell_jaccard_cross_region_species_table.parquet",
        index=False,
    )

    analysis_df.to_csv(
        OUT_DIR
        / "cell_jaccard_cross_region_species_table.csv",
        index=False,
    )

    print(
        "\nCROSS-REGION SPECIES RANKING TABLE"
    )
    display(
        analysis_df[
            [
                "state_code",
                "fia_species_code",
                PRIMARY_FEATURE,
                PRIMARY_GAIN,
                CROSS_GRID_GAIN,
                "improved_0p5",
                "min_period_records",
                "min_period_cells",
                "raw_error_grid_0p5",
                "cell_jaccard_percentile",
            ]
        ].sort_values(
            [
                "state_code",
                PRIMARY_FEATURE,
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    # --------------------------------------------------------
    # State-level threshold-free metrics
    # --------------------------------------------------------
    state_rows = []

    for state_code, state_df in analysis_df.groupby(
        "state_code"
    ):
        auc, n_auc_pairs = binary_auc(
            state_df[
                PRIMARY_FEATURE
            ],
            state_df[
                "improved_0p5"
            ],
        )

        state_rows.append(
            {
                "state_code": state_code,
                "n_species": len(
                    state_df
                ),
                "n_improved": int(
                    state_df[
                        "improved_0p5"
                    ].sum()
                ),
                "improved_fraction": float(
                    state_df[
                        "improved_0p5"
                    ].mean()
                ),
                "spearman_gain": safe_spearman(
                    state_df[
                        PRIMARY_FEATURE
                    ],
                    state_df[
                        PRIMARY_GAIN
                    ],
                ),
                "spearman_grid_median_gain": safe_spearman(
                    state_df[
                        PRIMARY_FEATURE
                    ],
                    state_df[
                        CROSS_GRID_GAIN
                    ],
                ),
                "auc_improved_vs_harmed": auc,
                "auc_positive_negative_pairs": (
                    n_auc_pairs
                ),
                "pairwise_concordance": (
                    pairwise_concordance(
                        state_df[
                            PRIMARY_FEATURE
                        ],
                        state_df[
                            PRIMARY_GAIN
                        ],
                    )
                ),
                "spearman_raw_error": (
                    safe_spearman(
                        state_df[
                            PRIMARY_FEATURE
                        ],
                        state_df[
                            "raw_error_grid_0p5"
                        ],
                    )
                ),
                "spearman_corrected_error": (
                    safe_spearman(
                        state_df[
                            PRIMARY_FEATURE
                        ],
                        state_df[
                            "corrected_error_grid_0p5"
                        ],
                    )
                ),
            }
        )

    state_metrics_df = pd.DataFrame(
        state_rows
    )

    state_metrics_df.to_csv(
        OUT_DIR
        / "state_threshold_free_ranking_metrics.csv",
        index=False,
    )

    print(
        "\nSTATE THRESHOLD-FREE RANKING METRICS"
    )
    display(
        state_metrics_df
    )

    # --------------------------------------------------------
    # Stratified inference
    # --------------------------------------------------------
    progress = tqdm(
        total=(
            2
            * (
                N_PERMUTATIONS
                + N_BOOTSTRAP
            )
        ),
        desc="Stratified ranking validation",
        unit="resample",
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    primary_inference = inference_for_metric(
        analysis_df,
        PRIMARY_GAIN,
        rng,
        progress,
    )

    cross_grid_inference = inference_for_metric(
        analysis_df,
        CROSS_GRID_GAIN,
        rng,
        progress,
    )

    progress.close()

    inference_df = pd.DataFrame(
        [
            primary_inference,
            cross_grid_inference,
        ]
    )

    inference_df.to_csv(
        OUT_DIR
        / "stratified_threshold_free_inference.csv",
        index=False,
    )

    print(
        "\nSTRATIFIED THRESHOLD-FREE INFERENCE"
    )
    display(
        inference_df
    )

    # --------------------------------------------------------
    # Leave-one-region-out descriptive stability
    # --------------------------------------------------------
    loo_rows = []

    for held_out_state in STATES:
        training_states = [
            state
            for state in STATES
            if state != held_out_state
        ]

        held_out = analysis_df[
            analysis_df[
                "state_code"
            ]
            == held_out_state
        ]

        training = analysis_df[
            analysis_df[
                "state_code"
            ].isin(
                training_states
            )
        ]

        held_out_auc, _ = binary_auc(
            held_out[
                PRIMARY_FEATURE
            ],
            held_out[
                "improved_0p5"
            ],
        )

        loo_rows.append(
            {
                "held_out_state": held_out_state,
                "training_states": ",".join(
                    training_states
                ),
                "training_direction_positive": bool(
                    stratified_rank_rho(
                        training,
                        PRIMARY_GAIN,
                    )
                    > 0
                ),
                "held_out_spearman": safe_spearman(
                    held_out[
                        PRIMARY_FEATURE
                    ],
                    held_out[
                        PRIMARY_GAIN
                    ],
                ),
                "held_out_auc": held_out_auc,
                "held_out_direction_positive": bool(
                    safe_spearman(
                        held_out[
                            PRIMARY_FEATURE
                        ],
                        held_out[
                            PRIMARY_GAIN
                        ],
                    )
                    > 0
                ),
            }
        )

    loo_df = pd.DataFrame(
        loo_rows
    )

    loo_df.to_csv(
        OUT_DIR
        / "leave_one_region_out_ranking.csv",
        index=False,
    )

    print(
        "\nLEAVE-ONE-REGION-OUT RANKING"
    )
    display(
        loo_df
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------
    primary_row = inference_df[
        inference_df[
            "outcome"
        ]
        == PRIMARY_GAIN
    ].iloc[0]

    cross_grid_row = inference_df[
        inference_df[
            "outcome"
        ]
        == CROSS_GRID_GAIN
    ].iloc[0]

    n_positive_state_spearman = int(
        (
            state_metrics_df[
                "spearman_gain"
            ]
            > 0
        ).sum()
    )

    n_positive_state_auc = int(
        (
            state_metrics_df[
                "auc_improved_vs_harmed"
            ]
            > 0.5
        ).sum()
    )

    ranking_pass = bool(
        primary_row[
            "stratified_auc"
        ]
        >= MIN_STRATIFIED_AUC
        and primary_row[
            "auc_permutation_p"
        ]
        <= MAX_PRIMARY_PERMUTATION_P
        and primary_row[
            "rank_permutation_p"
        ]
        <= MAX_PRIMARY_PERMUTATION_P
        and primary_row[
            "rank_bootstrap_positive_fraction"
        ]
        >= MIN_BOOTSTRAP_POSITIVE
        and primary_row[
            "auc_bootstrap_above_half_fraction"
        ]
        >= MIN_BOOTSTRAP_POSITIVE
        and n_positive_state_spearman
        >= MIN_POSITIVE_STATES
        and n_positive_state_auc
        >= MIN_POSITIVE_STATES
    )

    partial_support_pass = bool(
        primary_row[
            "partial_rank_rho"
        ]
        > 0
        and primary_row[
            "partial_permutation_p"
        ]
        <= MAX_PARTIAL_PERMUTATION_P
        and primary_row[
            "partial_bootstrap_positive_fraction"
        ]
        >= 0.80
    )

    cross_grid_pass = bool(
        cross_grid_row[
            "stratified_rank_rho"
        ]
        > 0
        and cross_grid_row[
            "rank_permutation_p"
        ]
        <= MAX_PRIMARY_PERMUTATION_P
        and cross_grid_row[
            "rank_bootstrap_positive_fraction"
        ]
        >= MIN_BOOTSTRAP_POSITIVE
    )

    loo_positive_states = int(
        loo_df[
            "held_out_direction_positive"
        ].sum()
    )

    loo_pass = bool(
        loo_positive_states
        >= MIN_POSITIVE_STATES
    )

    passed = bool(
        ranking_pass
        and cross_grid_pass
        and loo_pass
        and partial_support_pass
    )

    if passed:
        status = (
            "CELL_JACCARD_GATE_READY_FOR_MANUSCRIPT"
        )
        next_step = (
            "Freeze cell_jaccard as a threshold-free applicability "
            "ranking signal. Generate final manuscript tables, figures, "
            "limitations and reporting checklist without defining a cutoff."
        )
    elif (
        ranking_pass
        and cross_grid_pass
        and loo_pass
    ):
        status = (
            "CELL_JACCARD_RANKING_VALIDATED_PARTIAL_ADJUSTMENT"
        )
        next_step = (
            "The threshold-free ranking is validated, but adjusted "
            "association is weaker. Present cell_jaccard as a practical "
            "ranking diagnostic, not an independent mechanism."
        )
    else:
        status = (
            "CELL_JACCARD_REPLICATED_BUT_RANKING_NOT_ROBUST"
        )
        next_step = (
            "Do not promote cell_jaccard to a final applicability score. "
            "Report it only as a replicated association and retain the "
            "paper's main emphasis on correction non-transportability."
        )

    # --------------------------------------------------------
    # Manuscript-ready text
    # --------------------------------------------------------
    manuscript_text = f"""
FINAL THRESHOLD-FREE APPLICABILITY RESULT

Formal GBIF data source:
GBIF.org (10 July 2026) GBIF Occurrence Download.
DOI: {GBIF_DOI}

Across {len(analysis_df)} species-region observations from Pennsylvania,
Virginia and North Carolina, spatial-support continuity (cell Jaccard) was
evaluated as a threshold-free indicator of correction applicability.

Primary state-stratified results:
- Spearman rank association:
  rho = {primary_row['stratified_rank_rho']:.4f}
  one-sided permutation P = {primary_row['rank_permutation_p']:.6f}
  95% bootstrap interval =
  [{primary_row['rank_bootstrap_low']:.4f},
   {primary_row['rank_bootstrap_high']:.4f}]
- Stratified AUROC for ranking improved versus harmed species:
  AUC = {primary_row['stratified_auc']:.4f}
  one-sided permutation P = {primary_row['auc_permutation_p']:.6f}
  95% bootstrap interval =
  [{primary_row['auc_bootstrap_low']:.4f},
   {primary_row['auc_bootstrap_high']:.4f}]
- Partial rank association after adjustment for minimum record support,
  minimum occupied-cell support and raw FIA error:
  rho = {primary_row['partial_rank_rho']:.4f}
  one-sided permutation P = {primary_row['partial_permutation_p']:.6f}
  95% bootstrap interval =
  [{primary_row['partial_bootstrap_low']:.4f},
   {primary_row['partial_bootstrap_high']:.4f}]

Cross-grid rank association:
- rho = {cross_grid_row['stratified_rank_rho']:.4f}
- one-sided permutation P =
  {cross_grid_row['rank_permutation_p']:.6f}

Interpretation:
Higher spatial-support continuity ranked species for which target-effort
correction was more likely to reduce error relative to repeated FIA plots.
This result supports cell Jaccard as an applicability ranking diagnostic, not
as a causal mechanism and not as a universal binary cutoff.

Decision status:
{status}
"""

    manuscript_path = (
        OUT_DIR
        / "manuscript_ready_cell_jaccard_result.txt"
    )

    manuscript_path.write_text(
        manuscript_text.strip()
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Visual previews
    # --------------------------------------------------------
    plt.figure(
        figsize=(8, 6)
    )

    for state_code, state_df in analysis_df.groupby(
        "state_code"
    ):
        plt.scatter(
            state_df[
                PRIMARY_FEATURE
            ],
            state_df[
                PRIMARY_GAIN
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
        "0.5° correction gain versus FIA "
        "(km/decade)"
    )

    plt.title(
        "Threshold-free correction-applicability ranking"
    )

    plt.legend()
    plt.tight_layout()

    figure_path = (
        OUT_DIR
        / "cell_jaccard_gain_by_region.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    x = np.arange(
        len(
            state_metrics_df
        )
    )

    width = 0.35

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        x - width / 2,
        state_metrics_df[
            "spearman_gain"
        ],
        width=width,
        label="Spearman rho",
    )

    plt.bar(
        x + width / 2,
        state_metrics_df[
            "auc_improved_vs_harmed"
        ],
        width=width,
        label="AUROC",
    )

    plt.axhline(
        0.5,
        linestyle="--",
    )

    plt.xticks(
        x,
        state_metrics_df[
            "state_code"
        ],
    )

    plt.ylabel(
        "Threshold-free ranking metric"
    )

    plt.xlabel(
        "Region"
    )

    plt.title(
        "Regional stability of the cell-Jaccard gate"
    )

    plt.legend()
    plt.tight_layout()

    metric_figure_path = (
        OUT_DIR
        / "cell_jaccard_state_ranking_metrics.png"
    )

    plt.savefig(
        metric_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    append_readme(
        f"""

## Threshold-free cell-Jaccard applicability validation — {RUN_UTC}

### Formal data
- GBIF DOI: {GBIF_DOI}
- Download key: {DOWNLOAD_KEY}
- Species-region observations: {len(analysis_df)}
- Regions: {STATES}
- GPU: not used

### Locked gate
- Feature: cell_jaccard
- Use: threshold-free applicability ranking
- Binary cutoff selected: No
- Additional features searched: No

### Primary ranking result
- Stratified Spearman rho:
  {primary_row['stratified_rank_rho']}
- Rank permutation p:
  {primary_row['rank_permutation_p']}
- Stratified AUROC:
  {primary_row['stratified_auc']}
- AUROC permutation p:
  {primary_row['auc_permutation_p']}
- Partial rank rho:
  {primary_row['partial_rank_rho']}
- Partial permutation p:
  {primary_row['partial_permutation_p']}

### Robustness
- Cross-grid rank rho:
  {cross_grid_row['stratified_rank_rho']}
- Cross-grid permutation p:
  {cross_grid_row['rank_permutation_p']}
- Regions with positive Spearman:
  {n_positive_state_spearman}/3
- Regions with AUROC > 0.5:
  {n_positive_state_auc}/3
- Leave-one-region-out positive:
  {loo_positive_states}/3

### Decision
- Ranking passed: {ranking_pass}
- Partial support passed: {partial_support_pass}
- Cross-grid passed: {cross_grid_pass}
- Leave-one-region-out passed: {loo_pass}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
cell_jaccard is a replicated ranking diagnostic for correction applicability.
It is not treated as a causal driver, universal correction rule or optimized
binary threshold.
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
        f"N_SPECIES_REGION_OBSERVATIONS: "
        f"{len(analysis_df)}"
    )
    print(
        "STATE_RANKING_METRICS: "
        + json.dumps(
            state_metrics_df.round(
                6
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "STRATIFIED_INFERENCE: "
        + json.dumps(
            inference_df.round(
                6
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "LEAVE_ONE_REGION_OUT: "
        + json.dumps(
            loo_df.round(
                6
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        f"RANKING_PASSED: {ranking_pass}"
    )
    print(
        f"PARTIAL_SUPPORT_PASSED: "
        f"{partial_support_pass}"
    )
    print(
        f"CROSS_GRID_PASSED: "
        f"{cross_grid_pass}"
    )
    print(
        f"LOO_PASSED: {loo_pass}"
    )
    print(
        f"MANUSCRIPT_RESULT_PATH: "
        f"{manuscript_path}"
    )
    print(
        f"FIGURE_PATH: {figure_path}"
    )
    print(
        f"METRIC_FIGURE_PATH: "
        f"{metric_figure_path}"
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
        "CELL18A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Save this diagnostic summary and the last displayed table. "
            "Do not define a cell-Jaccard cutoff."
        ),
    )
