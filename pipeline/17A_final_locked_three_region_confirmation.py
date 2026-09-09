
# Cell 17A — Final locked three-region confirmatory inference
#
# Regions
# -------
# PA: discovery region
# VA: first independent confirmation region
# NC: second independent confirmation region
#
# Formal data source
# ------------------
# GBIF registered occurrence download:
# DOI 10.15468/dl.sm6ygu
#
# Locked candidate features
# -------------------------
# 1. cell_jaccard
# 2. abs_log_observer_growth
#
# Locked primary outcome
# ----------------------
# 0.5-degree correction gain:
# raw absolute error versus FIA
# minus corrected absolute error versus FIA.
#
# Positive gain means the correction improved agreement with FIA.
#
# Confirmatory rule
# -----------------
# PA is not reused as confirmation.
# VA and NC are evaluated independently using the thresholds locked before
# Virginia confirmation.
#
# A candidate is considered regionally confirmed when:
# - n >= 10 species;
# - primary rho >= 0.50;
# - one-sided permutation p, Holm-adjusted across two features, <= 0.05;
# - bootstrap positive fraction >= 0.90;
# - leave-one-species-out positive fraction >= 0.80;
# - grid-median rho >= 0.35;
# - positive rho in at least 2 of 3 grids.
#
# Final replication
# -----------------
# A feature becomes a replicated applicability gate only if:
# - it was discovered in PA;
# - it independently passes the locked criteria in NC or VA;
# - the VA+NC fixed-effect meta-association remains positive.
#
# No new features, thresholds or species are selected in this cell.

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm, chi2
from tqdm.auto import tqdm
from IPython.display import display

# ============================================================
# Locked configuration
# ============================================================
STATES = ["PA", "VA", "NC"]
CONFIRMATION_STATES = ["VA", "NC"]

STATE_ROLE = {
    "PA": "discovery",
    "VA": "confirmation_1",
    "NC": "confirmation_2",
}

LOCKED_FEATURES = [
    "cell_jaccard",
    "abs_log_observer_growth",
]

FEATURE_LABELS = {
    "cell_jaccard": "Cell Jaccard",
    "abs_log_observer_growth": "Absolute log observer growth",
}

PRIMARY_OUTCOME = "gain_grid_0p5"
GRID_MEDIAN_OUTCOME = "grid_median_gain_km_decade"

GRID_OUTCOMES = [
    "gain_grid_0p25",
    "gain_grid_0p5",
    "gain_grid_1p0",
]

MIN_CONFIRMATION_SPECIES = 10
MIN_PRIMARY_RHO = 0.50
MAX_HOLM_P = 0.05
MIN_BOOTSTRAP_POSITIVE = 0.90
MIN_LOO_POSITIVE = 0.80
MIN_GRID_MEDIAN_RHO = 0.35
MIN_POSITIVE_GRID_CORRELATIONS = 2

N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 5_000
RANDOM_SEED = 20260711

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
INPUT_DIR = (
    BASE_DIR
    / "derived"
    / "registered_gbif_formal_parity"
)
OUT_DIR = (
    BASE_DIR
    / "derived"
    / "final_three_region_confirmation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATHS = {
    state: INPUT_DIR / f"{state}_formal_species_inputs.parquet"
    for state in STATES
}

GBIF_DOI = "10.15468/dl.sm6ygu"
DOWNLOAD_KEY = "0032735-260623161305970"

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("INPUT_DIR:", INPUT_DIR)
print("OUT_DIR:", OUT_DIR)
print("LOCKED_FEATURES:", LOCKED_FEATURES)
print("N_PERMUTATIONS:", N_PERMUTATIONS)
print("N_BOOTSTRAP:", N_BOOTSTRAP)
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

    ordered = valid_values[
        order
    ]

    m = len(ordered)
    adjusted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for index, p_value in enumerate(
        ordered
    ):
        candidate = min(
            1.0,
            p_value * (
                m - index
            ),
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[
            index
        ] = running_max

    result[
        valid_indices[
            order
        ]
    ] = adjusted

    return result


def state_feature_inference(
    state,
    frame,
    feature,
    rng,
    progress,
):
    work = frame[
        [
            "fia_species_code",
            feature,
            PRIMARY_OUTCOME,
            GRID_MEDIAN_OUTCOME,
            *GRID_OUTCOMES,
        ]
    ].dropna(
        subset=[
            feature,
            PRIMARY_OUTCOME,
        ]
    ).copy()

    feature_values = work[
        feature
    ].to_numpy(
        dtype=float
    )

    outcome_values = work[
        PRIMARY_OUTCOME
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
        permuted = rng.permutation(
            outcome_values
        )

        rho = safe_spearman(
            feature_values,
            permuted,
        )

        if (
            np.isfinite(rho)
            and rho >= observed_rho
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
            feature_values[
                indices
            ],
            outcome_values[
                indices
            ],
        )

        if np.isfinite(
            rho
        ):
            bootstrap_values.append(
                rho
            )

        progress.update(1)

    bootstrap_values = np.asarray(
        bootstrap_values,
        dtype=float,
    )

    if len(
        bootstrap_values
    ):
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
        bootstrap_positive = float(
            np.mean(
                bootstrap_values > 0
            )
        )
    else:
        bootstrap_low = np.nan
        bootstrap_high = np.nan
        bootstrap_positive = np.nan

    loo_positive_values = []

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
            feature_values[
                keep
            ],
            outcome_values[
                keep
            ],
        )

        if np.isfinite(
            rho
        ):
            loo_positive_values.append(
                rho > 0
            )

        progress.update(1)

    loo_positive = (
        float(
            np.mean(
                loo_positive_values
            )
        )
        if loo_positive_values
        else np.nan
    )

    grid_median_rho = safe_spearman(
        work[
            feature
        ],
        work[
            GRID_MEDIAN_OUTCOME
        ],
    )

    grid_rhos = {}

    for grid_outcome in GRID_OUTCOMES:
        grid_rhos[
            grid_outcome
        ] = safe_spearman(
            work[
                feature
            ],
            work[
                grid_outcome
            ],
        )

    positive_grid_count = int(
        sum(
            np.isfinite(
                rho
            )
            and rho > 0
            for rho in grid_rhos.values()
        )
    )

    return {
        "state_code": state,
        "state_role": STATE_ROLE[
            state
        ],
        "feature": feature,
        "n_species": len(work),
        "primary_rho": observed_rho,
        "permutation_p_one_sided": (
            permutation_p
        ),
        "bootstrap_low": (
            bootstrap_low
        ),
        "bootstrap_high": (
            bootstrap_high
        ),
        "bootstrap_positive_fraction": (
            bootstrap_positive
        ),
        "loo_positive_fraction": (
            loo_positive
        ),
        "grid_median_rho": (
            grid_median_rho
        ),
        "rho_grid_0p25": grid_rhos[
            "gain_grid_0p25"
        ],
        "rho_grid_0p5": grid_rhos[
            "gain_grid_0p5"
        ],
        "rho_grid_1p0": grid_rhos[
            "gain_grid_1p0"
        ],
        "n_positive_grid_correlations": (
            positive_grid_count
        ),
    }


def fixed_effect_meta(
    frame,
    feature,
    states,
):
    subset = frame[
        (
            frame["feature"]
            == feature
        )
        & frame[
            "state_code"
        ].isin(
            states
        )
    ].dropna(
        subset=[
            "primary_rho",
            "n_species",
        ]
    ).copy()

    subset = subset[
        subset[
            "n_species"
        ] > 3
    ].copy()

    if len(
        subset
    ) == 0:
        return {
            "feature": feature,
            "states": ",".join(
                states
            ),
            "n_regions": 0,
            "total_species": 0,
            "meta_rho": np.nan,
            "meta_z": np.nan,
            "one_sided_p": np.nan,
            "two_sided_p": np.nan,
            "q_statistic": np.nan,
            "q_p": np.nan,
            "i2_percent": np.nan,
        }

    clipped_rho = np.clip(
        subset[
            "primary_rho"
        ].to_numpy(
            dtype=float
        ),
        -0.999999,
        0.999999,
    )

    fisher_z = np.arctanh(
        clipped_rho
    )

    weights = (
        subset[
            "n_species"
        ].to_numpy(
            dtype=float
        )
        - 3.0
    )

    pooled_z = float(
        np.sum(
            weights
            * fisher_z
        )
        / np.sum(
            weights
        )
    )

    standard_error = float(
        1.0
        / np.sqrt(
            np.sum(
                weights
            )
        )
    )

    z_statistic = float(
        pooled_z
        / standard_error
    )

    one_sided_p = float(
        norm.sf(
            z_statistic
        )
    )

    two_sided_p = float(
        2
        * norm.sf(
            abs(
                z_statistic
            )
        )
    )

    meta_rho = float(
        np.tanh(
            pooled_z
        )
    )

    q_statistic = float(
        np.sum(
            weights
            * (
                fisher_z
                - pooled_z
            )
            ** 2
        )
    )

    degrees_freedom = max(
        len(
            subset
        )
        - 1,
        0,
    )

    q_p = (
        float(
            chi2.sf(
                q_statistic,
                degrees_freedom,
            )
        )
        if degrees_freedom > 0
        else np.nan
    )

    i2 = (
        float(
            max(
                0.0,
                (
                    q_statistic
                    - degrees_freedom
                )
                / q_statistic
                * 100
            )
        )
        if (
            degrees_freedom > 0
            and q_statistic > 0
        )
        else 0.0
    )

    return {
        "feature": feature,
        "states": ",".join(
            states
        ),
        "n_regions": len(
            subset
        ),
        "total_species": int(
            subset[
                "n_species"
            ].sum()
        ),
        "meta_rho": meta_rho,
        "meta_z": z_statistic,
        "one_sided_p": (
            one_sided_p
        ),
        "two_sided_p": (
            two_sided_p
        ),
        "q_statistic": (
            q_statistic
        ),
        "q_p": q_p,
        "i2_percent": i2,
    }


def failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## Final three-region inference failure — {RUN_UTC}
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
    missing_paths = [
        str(path)
        for path in INPUT_PATHS.values()
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing formal species-input tables: "
            + " | ".join(
                missing_paths
            )
        )

    state_frames = {}

    for state in STATES:
        frame = pd.read_parquet(
            INPUT_PATHS[
                state
            ]
        )

        required_columns = {
            "fia_species_code",
            *LOCKED_FEATURES,
            PRIMARY_OUTCOME,
            GRID_MEDIAN_OUTCOME,
            *GRID_OUTCOMES,
            "raw_error_grid_0p5",
            "corrected_error_grid_0p5",
        }

        missing_columns = (
            required_columns
            - set(
                frame.columns
            )
        )

        if missing_columns:
            raise KeyError(
                f"{state} formal table is missing "
                f"{sorted(missing_columns)}"
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
            ]
        ).copy()

        frame[
            "fia_species_code"
        ] = frame[
            "fia_species_code"
        ].astype(int)

        state_frames[
            state
        ] = frame

    # --------------------------------------------------------
    # State-level correction context
    # --------------------------------------------------------
    context_rows = []

    for state, frame in state_frames.items():
        raw_error = pd.to_numeric(
            frame[
                "raw_error_grid_0p5"
            ],
            errors="coerce",
        )

        corrected_error = pd.to_numeric(
            frame[
                "corrected_error_grid_0p5"
            ],
            errors="coerce",
        )

        gain = pd.to_numeric(
            frame[
                PRIMARY_OUTCOME
            ],
            errors="coerce",
        )

        valid = (
            raw_error.notna()
            & corrected_error.notna()
        )

        raw_valid = raw_error[
            valid
        ]
        corrected_valid = corrected_error[
            valid
        ]
        gain_valid = gain[
            valid
        ]

        raw_median = float(
            raw_valid.median()
        )

        corrected_median = float(
            corrected_valid.median()
        )

        context_rows.append(
            {
                "state_code": state,
                "state_role": (
                    STATE_ROLE[
                        state
                    ]
                ),
                "n_species": len(
                    raw_valid
                ),
                "raw_median_abs_error": (
                    raw_median
                ),
                "corrected_median_abs_error": (
                    corrected_median
                ),
                "median_error_reduction": (
                    1
                    - corrected_median
                    / raw_median
                    if raw_median > 0
                    else np.nan
                ),
                "species_improved_fraction": float(
                    (
                        gain_valid > 0
                    ).mean()
                ),
                "median_gain_km_decade": float(
                    gain_valid.median()
                ),
            }
        )

    context_df = pd.DataFrame(
        context_rows
    )

    context_df.to_csv(
        OUT_DIR
        / "state_correction_context.csv",
        index=False,
    )

    print(
        "\nSTATE CORRECTION CONTEXT"
    )
    display(
        context_df
    )

    # --------------------------------------------------------
    # Locked state-feature inference
    # --------------------------------------------------------
    total_iterations = sum(
        len(
            state_frames[
                state
            ]
        )
        + N_PERMUTATIONS
        + N_BOOTSTRAP
        for state in STATES
        for _ in LOCKED_FEATURES
    )

    progress = tqdm(
        total=total_iterations,
        desc="Final locked three-region inference",
        unit="resample",
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    inference_rows = []

    for state in STATES:
        for feature in LOCKED_FEATURES:
            inference_rows.append(
                state_feature_inference(
                    state=state,
                    frame=state_frames[
                        state
                    ],
                    feature=feature,
                    rng=rng,
                    progress=progress,
                )
            )

    progress.close()

    inference_df = pd.DataFrame(
        inference_rows
    )

    # Holm correction is performed separately within each region
    # across the two locked candidates.
    inference_df[
        "holm_adjusted_p"
    ] = np.nan

    for state in STATES:
        mask = (
            inference_df[
                "state_code"
            ]
            == state
        )

        inference_df.loc[
            mask,
            "holm_adjusted_p",
        ] = holm_adjust(
            inference_df.loc[
                mask,
                "permutation_p_one_sided",
            ]
        )

    inference_df[
        "regional_confirmed"
    ] = (
        inference_df[
            "state_code"
        ].isin(
            CONFIRMATION_STATES
        )
        & (
            inference_df[
                "n_species"
            ]
            >= MIN_CONFIRMATION_SPECIES
        )
        & (
            inference_df[
                "primary_rho"
            ]
            >= MIN_PRIMARY_RHO
        )
        & (
            inference_df[
                "holm_adjusted_p"
            ]
            <= MAX_HOLM_P
        )
        & (
            inference_df[
                "bootstrap_positive_fraction"
            ]
            >= MIN_BOOTSTRAP_POSITIVE
        )
        & (
            inference_df[
                "loo_positive_fraction"
            ]
            >= MIN_LOO_POSITIVE
        )
        & (
            inference_df[
                "grid_median_rho"
            ]
            >= MIN_GRID_MEDIAN_RHO
        )
        & (
            inference_df[
                "n_positive_grid_correlations"
            ]
            >= MIN_POSITIVE_GRID_CORRELATIONS
        )
    )

    inference_df[
        "directionally_positive"
    ] = (
        inference_df[
            "primary_rho"
        ]
        > 0
    ) & (
        inference_df[
            "grid_median_rho"
        ]
        > 0
    )

    inference_df.to_csv(
        OUT_DIR
        / "state_feature_confirmation.csv",
        index=False,
    )

    print(
        "\nSTATE-FEATURE CONFIRMATION"
    )
    display(
        inference_df
    )

    # --------------------------------------------------------
    # Confirmation-region and all-region synthesis
    # --------------------------------------------------------
    meta_rows = []

    for feature in LOCKED_FEATURES:
        meta_rows.append(
            fixed_effect_meta(
                frame=inference_df,
                feature=feature,
                states=(
                    CONFIRMATION_STATES
                ),
            )
        )

        all_region_result = (
            fixed_effect_meta(
                frame=inference_df,
                feature=feature,
                states=STATES,
            )
        )

        all_region_result[
            "states"
        ] = "PA,VA,NC"

        meta_rows.append(
            all_region_result
        )

    meta_df = pd.DataFrame(
        meta_rows
    )

    meta_df.to_csv(
        OUT_DIR
        / "cross_region_meta_analysis.csv",
        index=False,
    )

    print(
        "\nCROSS-REGION META-ANALYSIS"
    )
    display(
        meta_df
    )

    # --------------------------------------------------------
    # Feature-level final decision
    # --------------------------------------------------------
    decision_rows = []

    for feature in LOCKED_FEATURES:
        feature_states = inference_df[
            inference_df[
                "feature"
            ]
            == feature
        ].copy()

        pa_row = feature_states[
            feature_states[
                "state_code"
            ]
            == "PA"
        ].iloc[0]

        va_row = feature_states[
            feature_states[
                "state_code"
            ]
            == "VA"
        ].iloc[0]

        nc_row = feature_states[
            feature_states[
                "state_code"
            ]
            == "NC"
        ].iloc[0]

        confirmation_meta = meta_df[
            (
                meta_df[
                    "feature"
                ]
                == feature
            )
            & (
                meta_df[
                    "states"
                ]
                == "VA,NC"
            )
        ].iloc[0]

        any_independent_confirmation = bool(
            va_row[
                "regional_confirmed"
            ]
            or nc_row[
                "regional_confirmed"
            ]
        )

        both_confirmation_directions = bool(
            va_row[
                "directionally_positive"
            ]
            and nc_row[
                "directionally_positive"
            ]
        )

        confirmation_meta_positive = bool(
            np.isfinite(
                confirmation_meta[
                    "meta_rho"
                ]
            )
            and confirmation_meta[
                "meta_rho"
            ]
            > 0
            and confirmation_meta[
                "one_sided_p"
            ]
            <= 0.05
        )

        replicated_gate = bool(
            pa_row[
                "directionally_positive"
            ]
            and any_independent_confirmation
            and confirmation_meta[
                "meta_rho"
            ]
            > 0
        )

        pooled_support_only = bool(
            (
                not replicated_gate
            )
            and both_confirmation_directions
            and confirmation_meta_positive
        )

        decision_rows.append(
            {
                "feature": feature,
                "pa_discovery_rho": (
                    pa_row[
                        "primary_rho"
                    ]
                ),
                "va_rho": (
                    va_row[
                        "primary_rho"
                    ]
                ),
                "va_confirmed": bool(
                    va_row[
                        "regional_confirmed"
                    ]
                ),
                "nc_rho": (
                    nc_row[
                        "primary_rho"
                    ]
                ),
                "nc_confirmed": bool(
                    nc_row[
                        "regional_confirmed"
                    ]
                ),
                "confirmation_meta_rho": (
                    confirmation_meta[
                        "meta_rho"
                    ]
                ),
                "confirmation_meta_p": (
                    confirmation_meta[
                        "one_sided_p"
                    ]
                ),
                "confirmation_i2_percent": (
                    confirmation_meta[
                        "i2_percent"
                    ]
                ),
                "both_confirmation_directions_positive": (
                    both_confirmation_directions
                ),
                "replicated_gate": (
                    replicated_gate
                ),
                "pooled_support_only": (
                    pooled_support_only
                ),
            }
        )

    decision_df = pd.DataFrame(
        decision_rows
    )

    decision_df.to_csv(
        OUT_DIR
        / "final_feature_decision.csv",
        index=False,
    )

    print(
        "\nFINAL FEATURE DECISION"
    )
    display(
        decision_df
    )

    replicated_features = (
        decision_df.loc[
            decision_df[
                "replicated_gate"
            ],
            "feature",
        ].tolist()
    )

    pooled_only_features = (
        decision_df.loc[
            decision_df[
                "pooled_support_only"
            ],
            "feature",
        ].tolist()
    )

    n_replicated = len(
        replicated_features
    )

    if n_replicated == 2:
        status = (
            "TWO_TRANSPORTABILITY_GATES_REPLICATED"
        )
        passed = True
        next_step = (
            "Construct a two-feature applicability ranking using only "
            "the replicated directions, then validate it with leave-one-"
            "region-out ranking analysis. Do not optimize weights."
        )
    elif n_replicated == 1:
        status = (
            "ONE_TRANSPORTABILITY_GATE_REPLICATED"
        )
        passed = True
        next_step = (
            f"Retain only {replicated_features[0]} as the replicated "
            "correction-applicability gate. Run threshold-free ranking "
            "validation and manuscript synthesis; discard the other feature "
            "from the final gate."
        )
    elif len(
        pooled_only_features
    ) >= 1:
        status = (
            "POOLED_DIRECTIONAL_SUPPORT_WITHOUT_REGIONAL_CONFIRMATION"
        )
        passed = False
        next_step = (
            "Do not construct a final applicability index. Report the "
            "candidate associations as directional cross-region evidence "
            "that did not meet locked regional confirmation."
        )
    else:
        status = (
            "NO_TRANSPORTABILITY_GATE_REPLICATED"
        )
        passed = False
        next_step = (
            "Do not construct an applicability index. Finalize the paper "
            "as a real-world non-transportability and correction-harm study."
        )

    # --------------------------------------------------------
    # Visual previews
    # --------------------------------------------------------
    for feature in LOCKED_FEATURES:
        plot_df = inference_df[
            inference_df[
                "feature"
            ]
            == feature
        ].copy()

        x = np.arange(
            len(
                plot_df
            )
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.errorbar(
            x,
            plot_df[
                "primary_rho"
            ],
            yerr=[
                plot_df[
                    "primary_rho"
                ]
                - plot_df[
                    "bootstrap_low"
                ],
                plot_df[
                    "bootstrap_high"
                ]
                - plot_df[
                    "primary_rho"
                ],
            ],
            fmt="o",
            capsize=4,
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xticks(
            x,
            plot_df[
                "state_code"
            ],
        )

        plt.ylabel(
            "Spearman rho with 0.5° correction gain"
        )

        plt.xlabel(
            "Region"
        )

        plt.title(
            f"Cross-region association: "
            f"{FEATURE_LABELS[feature]}"
        )

        plt.tight_layout()
        plt.show()

    plot_context = context_df.copy()
    x = np.arange(
        len(
            plot_context
        )
    )
    width = 0.35

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        x - width / 2,
        plot_context[
            "raw_median_abs_error"
        ],
        width=width,
        label="Raw",
    )

    plt.bar(
        x + width / 2,
        plot_context[
            "corrected_median_abs_error"
        ],
        width=width,
        label="Corrected",
    )

    plt.xticks(
        x,
        plot_context[
            "state_code"
        ],
    )

    plt.ylabel(
        "Median absolute error versus FIA "
        "(km/decade)"
    )

    plt.xlabel(
        "Region"
    )

    plt.title(
        "Real-data correction performance by region"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    compact_state_results = (
        inference_df[
            [
                "state_code",
                "state_role",
                "feature",
                "n_species",
                "primary_rho",
                "permutation_p_one_sided",
                "holm_adjusted_p",
                "bootstrap_low",
                "bootstrap_high",
                "bootstrap_positive_fraction",
                "loo_positive_fraction",
                "grid_median_rho",
                "rho_grid_0p25",
                "rho_grid_0p5",
                "rho_grid_1p0",
                "n_positive_grid_correlations",
                "regional_confirmed",
            ]
        ]
        .round(6)
        .to_dict(
            "records"
        )
    )

    compact_meta = (
        meta_df.round(6).to_dict(
            "records"
        )
    )

    compact_decision = (
        decision_df.round(6).to_dict(
            "records"
        )
    )

    append_readme(
        f"""

## Final locked PA–VA–NC confirmation — {RUN_UTC}

### Formal data
- GBIF download DOI: {GBIF_DOI}
- Download key: {DOWNLOAD_KEY}
- Formal registered data parity with temporary API inputs:
  exact after NC effort-frame repair
- GPU: not used

### Locked design
- PA: discovery region
- VA: first independent confirmation region
- NC: second independent confirmation region
- Features:
  {LOCKED_FEATURES}
- Primary outcome:
  {PRIMARY_OUTCOME}
- Permutations per state-feature:
  {N_PERMUTATIONS}
- Bootstrap replicates per state-feature:
  {N_BOOTSTRAP}
- Random seed:
  {RANDOM_SEED}

### Regional confirmation criteria
- n >= {MIN_CONFIRMATION_SPECIES}
- primary rho >= {MIN_PRIMARY_RHO}
- Holm-adjusted one-sided p <= {MAX_HOLM_P}
- bootstrap positive fraction >= {MIN_BOOTSTRAP_POSITIVE}
- LOO positive fraction >= {MIN_LOO_POSITIVE}
- grid-median rho >= {MIN_GRID_MEDIAN_RHO}
- positive correlations in >=
  {MIN_POSITIVE_GRID_CORRELATIONS}/3 grids

### Result
- Replicated features:
  {replicated_features}
- Pooled-support-only features:
  {pooled_only_features}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
PA is a discovery region and is not counted as independent confirmation.
A replicated gate is an association-based applicability signal, not a causal
mechanism. The universal 0.5-degree correction is not considered validated.
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
        f"GBIF_DOWNLOAD_KEY: {DOWNLOAD_KEY}"
    )
    print(
        "STATE_CORRECTION_CONTEXT: "
        + json.dumps(
            context_df.round(
                6
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "STATE_FEATURE_RESULTS: "
        + json.dumps(
            compact_state_results,
            ensure_ascii=False,
        )
    )
    print(
        "CROSS_REGION_META: "
        + json.dumps(
            compact_meta,
            ensure_ascii=False,
        )
    )
    print(
        "FINAL_FEATURE_DECISION: "
        + json.dumps(
            compact_decision,
            ensure_ascii=False,
        )
    )
    print(
        f"N_REPLICATED_FEATURES: "
        f"{n_replicated}"
    )
    print(
        "REPLICATED_FEATURES: "
        + json.dumps(
            replicated_features,
            ensure_ascii=False,
        )
    )
    print(
        "POOLED_SUPPORT_ONLY_FEATURES: "
        + json.dumps(
            pooled_only_features,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_RULE: PA discovery plus locked independent regional "
        "confirmation in VA or NC, with positive VA+NC meta-association"
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
        "CELL17A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed table. "
            "Do not change thresholds or search new features."
        ),
    )
