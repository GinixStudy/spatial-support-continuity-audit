
# Cell 17A — Locked PA–VA–NC synthesis of the replicated cell-continuity signal
#
# Final retained feature
# ----------------------
# cell_jaccard only.
#
# Permanently discarded feature
# -----------------------------
# abs_log_observer_growth.
#
# Scientific question
# -------------------
# Across Pennsylvania, Virginia and North Carolina, does greater continuity
# of a species' occupied 0.5-degree cells rank species for which the locked
# target-effort correction is more likely to improve agreement with FIA?
#
# This cell performs:
# 1. Region-specific Spearman effects.
# 2. Approximate Fisher-z fixed/random-effects synthesis.
# 3. Independent-region-only synthesis using VA and NC.
# 4. Region-stratified, threshold-free pooled rank validation.
# 5. Within-region pairwise concordance.
# 6. Species-cluster bootstrap.
# 7. Leave-one-region-out validation.
# 8. Partial rank audit controlling raw error and minimum cell support.
#
# No new feature, region, window, correction or cutoff is searched.
# No hard cell_jaccard threshold is constructed in this cell.

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 240)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 360)

# ============================================================
# Locked configuration
# ============================================================
FEATURE = "cell_jaccard"

N_REGION_PERMUTATIONS = 5000
N_STRATIFIED_PERMUTATIONS = 10000
N_CLUSTER_BOOTSTRAP = 5000
N_PARTIAL_PERMUTATIONS = 10000
RANDOM_SEED = 20260710

MIN_RANDOM_EFFECT_RHO = 0.35
MIN_POOLED_RANK_RHO = 0.40
MAX_POOLED_PERMUTATION_P = 0.01
MIN_PAIRWISE_CONCORDANCE = 0.60
MAX_CONCORDANCE_PERMUTATION_P = 0.01
MIN_BOOTSTRAP_POSITIVE_FRACTION = 0.95
MIN_GRID_MEDIAN_RHO = 0.40
MAX_GRID_MEDIAN_PERMUTATION_P = 0.01

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")

PA_PATH = (
    BASE_DIR
    / "derived"
    / "pa_correction_transportability_audit"
    / "species_transportability_features.parquet"
)

VA_PATH = (
    BASE_DIR
    / "derived"
    / "va_candidate_confirmation"
    / "va_locked_candidate_species_table.parquet"
)

NC_PATH = (
    BASE_DIR
    / "derived"
    / "nc_final_candidate_confirmation"
    / "nc_locked_candidate_species_table.parquet"
)

OUT_DIR = (
    BASE_DIR
    / "derived"
    / "three_region_cell_jaccard_synthesis"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("FEATURE:", FEATURE)
print("REGIONS: PA, VA, NC")
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")


# ============================================================
# General helpers
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


def first_existing_column(frame, candidates, required=True):
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise KeyError(
            "None of the required columns were found: "
            + " | ".join(candidates)
        )

    return None


def safe_spearman(x, y):
    work = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if (
        len(work) < 5
        or work["x"].nunique() < 2
        or work["y"].nunique() < 2
    ):
        return np.nan

    return float(
        spearmanr(
            work["x"],
            work["y"],
        ).statistic
    )


def percentile_rank(series):
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .rank(
            method="average",
            pct=True,
        )
    )


def add_within_region_ranks(
    frame,
    outcome_column,
):
    ranked_parts = []

    for region, group in frame.groupby(
        "region",
        sort=False,
    ):
        part = group.copy()

        part["feature_rank"] = percentile_rank(
            part["cell_jaccard"]
        )
        part["outcome_rank"] = percentile_rank(
            part[outcome_column]
        )
        part["raw_error_rank"] = percentile_rank(
            part["raw_abs_error"]
        )
        part["min_cells_rank"] = percentile_rank(
            part["min_period_cells"]
        )

        ranked_parts.append(part)

    return pd.concat(
        ranked_parts,
        ignore_index=True,
    )


def pairwise_concordance(
    frame,
    feature_column,
    outcome_column,
):
    concordant = 0
    discordant = 0
    tied = 0

    for _, group in frame.groupby(
        "region",
        sort=False,
    ):
        x = pd.to_numeric(
            group[feature_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        y = pd.to_numeric(
            group[outcome_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        x = x[valid]
        y = y[valid]

        for first in range(len(x)):
            for second in range(
                first + 1,
                len(x),
            ):
                product = (
                    x[first] - x[second]
                ) * (
                    y[first] - y[second]
                )

                if product > 0:
                    concordant += 1
                elif product < 0:
                    discordant += 1
                else:
                    tied += 1

    informative = (
        concordant + discordant
    )

    score = (
        concordant / informative
        if informative > 0
        else np.nan
    )

    return {
        "pairwise_concordance": float(score),
        "concordant_pairs": int(concordant),
        "discordant_pairs": int(discordant),
        "tied_pairs": int(tied),
        "informative_pairs": int(informative),
    }


def fisher_meta_analysis(
    region_effect_df,
    included_regions,
    label,
):
    work = region_effect_df[
        region_effect_df["region"].isin(
            included_regions
        )
    ].dropna(
        subset=[
            "rho_primary_gain",
            "n_species",
        ]
    ).copy()

    work = work[
        work["n_species"] > 3
    ].copy()

    if len(work) == 0:
        raise ValueError(
            f"No valid regions for meta-analysis: {label}"
        )

    rho = np.clip(
        work["rho_primary_gain"].to_numpy(
            dtype=float
        ),
        -0.999999,
        0.999999,
    )

    z_values = np.arctanh(rho)
    fixed_weights = (
        work["n_species"].to_numpy(
            dtype=float
        )
        - 3.0
    )

    fixed_z = float(
        np.sum(
            fixed_weights * z_values
        )
        / np.sum(
            fixed_weights
        )
    )

    q_statistic = float(
        np.sum(
            fixed_weights
            * (
                z_values - fixed_z
            ) ** 2
        )
    )

    degrees_freedom = max(
        len(work) - 1,
        0,
    )

    c_value = float(
        np.sum(fixed_weights)
        - np.sum(
            fixed_weights ** 2
        )
        / np.sum(fixed_weights)
    )

    tau_squared = (
        max(
            0.0,
            (
                q_statistic
                - degrees_freedom
            )
            / c_value,
        )
        if (
            len(work) > 1
            and c_value > 0
        )
        else 0.0
    )

    sampling_variance = (
        1.0 / fixed_weights
    )

    random_weights = (
        1.0
        / (
            sampling_variance
            + tau_squared
        )
    )

    random_z = float(
        np.sum(
            random_weights
            * z_values
        )
        / np.sum(
            random_weights
        )
    )

    random_se = float(
        math.sqrt(
            1.0
            / np.sum(
                random_weights
            )
        )
    )

    lower_z = (
        random_z
        - 1.96 * random_se
    )
    upper_z = (
        random_z
        + 1.96 * random_se
    )

    one_sided_p = float(
        norm.sf(
            random_z
            / random_se
        )
    )

    i_squared = (
        max(
            0.0,
            (
                q_statistic
                - degrees_freedom
            )
            / q_statistic
            * 100.0,
        )
        if q_statistic > 0
        else 0.0
    )

    return {
        "analysis": label,
        "regions": ",".join(
            included_regions
        ),
        "n_regions": len(work),
        "n_species_region_rows": int(
            work["n_species"].sum()
        ),
        "fixed_effect_rho": float(
            np.tanh(fixed_z)
        ),
        "random_effect_rho": float(
            np.tanh(random_z)
        ),
        "random_effect_ci_low": float(
            np.tanh(lower_z)
        ),
        "random_effect_ci_high": float(
            np.tanh(upper_z)
        ),
        "random_effect_one_sided_p": (
            one_sided_p
        ),
        "q_statistic": q_statistic,
        "q_df": degrees_freedom,
        "tau_squared_fisher_z": tau_squared,
        "i_squared_percent": i_squared,
    }


def residualize(
    values,
    covariate_matrix,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(
            covariate_matrix
        ).all(axis=1)
    )

    residuals = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    if valid.sum() < 4:
        return residuals

    design = np.column_stack(
        [
            np.ones(
                valid.sum()
            ),
            covariate_matrix[
                valid
            ],
        ]
    )

    beta = np.linalg.lstsq(
        design,
        values[valid],
        rcond=None,
    )[0]

    residuals[
        valid
    ] = (
        values[valid]
        - design @ beta
    )

    return residuals


def build_partial_residual_frame(
    ranked_df,
):
    parts = []

    for region, group in ranked_df.groupby(
        "region",
        sort=False,
    ):
        part = group.copy()

        covariates = part[
            [
                "raw_error_rank",
                "min_cells_rank",
            ]
        ].to_numpy(
            dtype=float
        )

        part["feature_residual"] = residualize(
            part[
                "feature_rank"
            ].to_numpy(
                dtype=float
            ),
            covariates,
        )

        part["outcome_residual"] = residualize(
            part[
                "outcome_rank"
            ].to_numpy(
                dtype=float
            ),
            covariates,
        )

        parts.append(part)

    return pd.concat(
        parts,
        ignore_index=True,
    )


def print_failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## Three-region cell-jaccard synthesis failure — {RUN_UTC}
- Status: **{status}**
- Error: `{error}`
- Next step: {next_step}
"""
    )

    print(
        "\n========== RUN SUMMARY =========="
    )
    print(f"CELL_STATUS: {status}")
    print(f"ERROR: {error}")
    print(f"OUT_DIR: {OUT_DIR}")
    print("PASSED: False")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")


# ============================================================
# Region harmonization
# ============================================================
def harmonize_pa(frame):
    primary_gain_column = first_existing_column(
        frame,
        [
            "effort_gain_km_decade",
            "gain_grid_0p5",
        ],
    )

    grid_median_column = first_existing_column(
        frame,
        [
            "grid_median_gain_km_decade",
        ],
    )

    corrected_error_column = first_existing_column(
        frame,
        [
            "effort_abs_error",
            "corrected_abs_error",
        ],
        required=False,
    )

    min_cells_column = first_existing_column(
        frame,
        [
            "min_period_cells",
        ],
        required=False,
    )

    output = pd.DataFrame(
        {
            "region": "PA",
            "fia_species_code": pd.to_numeric(
                frame["fia_species_code"],
                errors="coerce",
            ),
            "scientific_name": frame[
                "scientific_name"
            ],
            "cell_jaccard": pd.to_numeric(
                frame["cell_jaccard"],
                errors="coerce",
            ),
            "primary_gain": pd.to_numeric(
                frame[
                    primary_gain_column
                ],
                errors="coerce",
            ),
            "grid_median_gain": pd.to_numeric(
                frame[
                    grid_median_column
                ],
                errors="coerce",
            ),
            "raw_abs_error": pd.to_numeric(
                frame["raw_abs_error"],
                errors="coerce",
            ),
        }
    )

    if corrected_error_column is not None:
        output[
            "corrected_abs_error"
        ] = pd.to_numeric(
            frame[
                corrected_error_column
            ],
            errors="coerce",
        )
    else:
        output[
            "corrected_abs_error"
        ] = (
            output["raw_abs_error"]
            - output["primary_gain"]
        )

    if min_cells_column is not None:
        output[
            "min_period_cells"
        ] = pd.to_numeric(
            frame[
                min_cells_column
            ],
            errors="coerce",
        )
    else:
        output[
            "min_period_cells"
        ] = np.nan

    for resolution_tag in [
        "0p25",
        "0p5",
        "1p0",
    ]:
        column = (
            f"gain_grid_{resolution_tag}"
        )

        output[column] = (
            pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            if column in frame.columns
            else np.nan
        )

    return output


def harmonize_confirmation_region(
    frame,
    region,
):
    required_columns = [
        "fia_species_code",
        "scientific_name",
        "cell_jaccard",
        "gain_grid_0p25",
        "gain_grid_0p5",
        "gain_grid_1p0",
        "grid_median_gain_km_decade",
        "raw_abs_error",
        "corrected_abs_error",
        "early_cells",
        "late_cells",
    ]

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"{region} table is missing: "
            + " | ".join(missing)
        )

    return pd.DataFrame(
        {
            "region": region,
            "fia_species_code": pd.to_numeric(
                frame["fia_species_code"],
                errors="coerce",
            ),
            "scientific_name": frame[
                "scientific_name"
            ],
            "cell_jaccard": pd.to_numeric(
                frame["cell_jaccard"],
                errors="coerce",
            ),
            "primary_gain": pd.to_numeric(
                frame["gain_grid_0p5"],
                errors="coerce",
            ),
            "grid_median_gain": pd.to_numeric(
                frame[
                    "grid_median_gain_km_decade"
                ],
                errors="coerce",
            ),
            "raw_abs_error": pd.to_numeric(
                frame["raw_abs_error"],
                errors="coerce",
            ),
            "corrected_abs_error": pd.to_numeric(
                frame[
                    "corrected_abs_error"
                ],
                errors="coerce",
            ),
            "min_period_cells": np.minimum(
                pd.to_numeric(
                    frame["early_cells"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    frame["late_cells"],
                    errors="coerce",
                ),
            ),
            "gain_grid_0p25": pd.to_numeric(
                frame["gain_grid_0p25"],
                errors="coerce",
            ),
            "gain_grid_0p5": pd.to_numeric(
                frame["gain_grid_0p5"],
                errors="coerce",
            ),
            "gain_grid_1p0": pd.to_numeric(
                frame["gain_grid_1p0"],
                errors="coerce",
            ),
        }
    )


# ============================================================
# Main
# ============================================================
try:
    # --------------------------------------------------------
    # 1. Load and harmonize the three locked region datasets
    # --------------------------------------------------------
    required_paths = [
        PA_PATH,
        VA_PATH,
        NC_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing prior region outputs: "
            + " | ".join(
                missing_paths
            )
        )

    pa_raw = pd.read_parquet(
        PA_PATH
    )
    va_raw = pd.read_parquet(
        VA_PATH
    )
    nc_raw = pd.read_parquet(
        NC_PATH
    )

    combined_df = pd.concat(
        [
            harmonize_pa(
                pa_raw
            ),
            harmonize_confirmation_region(
                va_raw,
                "VA",
            ),
            harmonize_confirmation_region(
                nc_raw,
                "NC",
            ),
        ],
        ignore_index=True,
    )

    combined_df = combined_df.dropna(
        subset=[
            "region",
            "fia_species_code",
            "cell_jaccard",
            "primary_gain",
            "grid_median_gain",
            "raw_abs_error",
            "min_period_cells",
        ]
    ).copy()

    combined_df[
        "fia_species_code"
    ] = combined_df[
        "fia_species_code"
    ].astype(int)

    combined_df = combined_df.drop_duplicates(
        [
            "region",
            "fia_species_code",
        ]
    ).reset_index(
        drop=True
    )

    expected_regions = {
        "PA",
        "VA",
        "NC",
    }

    if set(
        combined_df["region"]
    ) != expected_regions:
        raise ValueError(
            "The harmonized table does not contain exactly PA, VA and NC."
        )

    combined_df.to_parquet(
        OUT_DIR
        / "three_region_harmonized_species_table.parquet",
        index=False,
    )

    combined_df.to_csv(
        OUT_DIR
        / "three_region_harmonized_species_table.csv",
        index=False,
    )

    print(
        "\nTHREE-REGION HARMONIZED SPECIES TABLE"
    )
    display(
        combined_df[
            [
                "region",
                "fia_species_code",
                "scientific_name",
                "cell_jaccard",
                "primary_gain",
                "grid_median_gain",
                "raw_abs_error",
                "corrected_abs_error",
                "min_period_cells",
            ]
        ].sort_values(
            [
                "region",
                "cell_jaccard",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    # --------------------------------------------------------
    # 2. Region-specific effects and one-sided permutations
    # --------------------------------------------------------
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    region_rows = []

    for region, group in combined_df.groupby(
        "region",
        sort=True,
    ):
        primary_rho = safe_spearman(
            group["cell_jaccard"],
            group["primary_gain"],
        )

        grid_median_rho = safe_spearman(
            group["cell_jaccard"],
            group["grid_median_gain"],
        )

        exceed = 0

        feature_values = group[
            "cell_jaccard"
        ].to_numpy(
            dtype=float
        )

        gain_values = group[
            "primary_gain"
        ].to_numpy(
            dtype=float
        )

        for _ in tqdm(
            range(
                N_REGION_PERMUTATIONS
            ),
            desc=(
                f"Region permutation {region}"
            ),
            unit="replicate",
        ):
            permuted_rho = safe_spearman(
                feature_values,
                rng.permutation(
                    gain_values
                ),
            )

            exceed += int(
                np.isfinite(
                    permuted_rho
                )
                and permuted_rho
                >= primary_rho
            )

        permutation_p = float(
            (
                1 + exceed
            )
            / (
                N_REGION_PERMUTATIONS
                + 1
            )
        )

        pairwise = pairwise_concordance(
            group,
            "cell_jaccard",
            "primary_gain",
        )

        n_species = len(
            group
        )

        clipped_rho = float(
            np.clip(
                primary_rho,
                -0.999999,
                0.999999,
            )
        )

        fisher_z = float(
            np.arctanh(
                clipped_rho
            )
        )

        fisher_se = float(
            1.0
            / math.sqrt(
                n_species - 3
            )
        )

        region_rows.append(
            {
                "region": region,
                "n_species": n_species,
                "rho_primary_gain": primary_rho,
                "rho_grid_median_gain": grid_median_rho,
                "one_sided_permutation_p": permutation_p,
                "rho_ci_low_approx": float(
                    np.tanh(
                        fisher_z
                        - 1.96
                        * fisher_se
                    )
                ),
                "rho_ci_high_approx": float(
                    np.tanh(
                        fisher_z
                        + 1.96
                        * fisher_se
                    )
                ),
                "median_primary_gain": float(
                    group[
                        "primary_gain"
                    ].median()
                ),
                "fraction_species_improved": float(
                    (
                        group[
                            "primary_gain"
                        ]
                        > 0
                    ).mean()
                ),
                **pairwise,
            }
        )

    region_effect_df = pd.DataFrame(
        region_rows
    )

    region_effect_df.to_csv(
        OUT_DIR
        / "region_specific_cell_jaccard_effects.csv",
        index=False,
    )

    print(
        "\nREGION-SPECIFIC EFFECTS"
    )
    display(
        region_effect_df
    )

    # --------------------------------------------------------
    # 3. Fixed/random effects synthesis
    # --------------------------------------------------------
    meta_df = pd.DataFrame(
        [
            fisher_meta_analysis(
                region_effect_df,
                [
                    "PA",
                    "VA",
                    "NC",
                ],
                "all_three_regions",
            ),
            fisher_meta_analysis(
                region_effect_df,
                [
                    "VA",
                    "NC",
                ],
                "independent_regions_only",
            ),
        ]
    )

    meta_df.to_csv(
        OUT_DIR
        / "cell_jaccard_meta_analysis.csv",
        index=False,
    )

    print(
        "\nFISHER-Z EFFECT SYNTHESIS"
    )
    display(
        meta_df
    )

    # --------------------------------------------------------
    # 4. Region-stratified threshold-free rank validation
    # --------------------------------------------------------
    ranked_primary_df = add_within_region_ranks(
        combined_df,
        "primary_gain",
    )

    observed_pooled_rho = safe_spearman(
        ranked_primary_df[
            "feature_rank"
        ],
        ranked_primary_df[
            "outcome_rank"
        ],
    )

    observed_pairwise = pairwise_concordance(
        ranked_primary_df,
        "feature_rank",
        "outcome_rank",
    )

    pooled_exceed = 0
    concordance_exceed = 0

    region_indices = {
        region: group.index.to_numpy()
        for region, group
        in ranked_primary_df.groupby(
            "region"
        )
    }

    original_outcome_rank = (
        ranked_primary_df[
            "outcome_rank"
        ].to_numpy(
            dtype=float
        )
    )

    for _ in tqdm(
        range(
            N_STRATIFIED_PERMUTATIONS
        ),
        desc="Region-stratified rank permutations",
        unit="replicate",
    ):
        permuted_rank = (
            original_outcome_rank.copy()
        )

        for indices in region_indices.values():
            permuted_rank[
                indices
            ] = rng.permutation(
                permuted_rank[
                    indices
                ]
            )

        permuted_df = (
            ranked_primary_df[
                [
                    "region",
                    "feature_rank",
                ]
            ].copy()
        )

        permuted_df[
            "permuted_outcome_rank"
        ] = permuted_rank

        permuted_rho = safe_spearman(
            permuted_df[
                "feature_rank"
            ],
            permuted_df[
                "permuted_outcome_rank"
            ],
        )

        permuted_concordance = (
            pairwise_concordance(
                permuted_df,
                "feature_rank",
                "permuted_outcome_rank",
            )[
                "pairwise_concordance"
            ]
        )

        pooled_exceed += int(
            np.isfinite(
                permuted_rho
            )
            and permuted_rho
            >= observed_pooled_rho
        )

        concordance_exceed += int(
            np.isfinite(
                permuted_concordance
            )
            and permuted_concordance
            >= observed_pairwise[
                "pairwise_concordance"
            ]
        )

    pooled_permutation_p = float(
        (
            1 + pooled_exceed
        )
        / (
            N_STRATIFIED_PERMUTATIONS
            + 1
        )
    )

    concordance_permutation_p = float(
        (
            1
            + concordance_exceed
        )
        / (
            N_STRATIFIED_PERMUTATIONS
            + 1
        )
    )

    # --------------------------------------------------------
    # 5. Species-cluster bootstrap
    # --------------------------------------------------------
    unique_species = combined_df[
        "fia_species_code"
    ].drop_duplicates().to_numpy()

    bootstrap_rho = []
    bootstrap_concordance = []

    for bootstrap_index in tqdm(
        range(
            N_CLUSTER_BOOTSTRAP
        ),
        desc="Species-cluster bootstrap",
        unit="replicate",
    ):
        sampled_species = rng.choice(
            unique_species,
            size=len(
                unique_species
            ),
            replace=True,
        )

        pieces = []

        for draw_index, species_code in enumerate(
            sampled_species
        ):
            species_rows = combined_df[
                combined_df[
                    "fia_species_code"
                ]
                == species_code
            ].copy()

            species_rows[
                "bootstrap_cluster"
            ] = draw_index

            pieces.append(
                species_rows
            )

        sampled_df = pd.concat(
            pieces,
            ignore_index=True,
        )

        ranked_sample = (
            add_within_region_ranks(
                sampled_df,
                "primary_gain",
            )
        )

        rho_value = safe_spearman(
            ranked_sample[
                "feature_rank"
            ],
            ranked_sample[
                "outcome_rank"
            ],
        )

        concordance_value = (
            pairwise_concordance(
                ranked_sample,
                "feature_rank",
                "outcome_rank",
            )[
                "pairwise_concordance"
            ]
        )

        if np.isfinite(
            rho_value
        ):
            bootstrap_rho.append(
                rho_value
            )

        if np.isfinite(
            concordance_value
        ):
            bootstrap_concordance.append(
                concordance_value
            )

    bootstrap_rho = np.asarray(
        bootstrap_rho,
        dtype=float,
    )

    bootstrap_concordance = np.asarray(
        bootstrap_concordance,
        dtype=float,
    )

    bootstrap_summary = {
        "pooled_rho_bootstrap_low": float(
            np.quantile(
                bootstrap_rho,
                0.025,
            )
        ),
        "pooled_rho_bootstrap_high": float(
            np.quantile(
                bootstrap_rho,
                0.975,
            )
        ),
        "pooled_rho_bootstrap_positive_fraction": float(
            np.mean(
                bootstrap_rho > 0
            )
        ),
        "concordance_bootstrap_low": float(
            np.quantile(
                bootstrap_concordance,
                0.025,
            )
        ),
        "concordance_bootstrap_high": float(
            np.quantile(
                bootstrap_concordance,
                0.975,
            )
        ),
        "concordance_bootstrap_above_half_fraction": float(
            np.mean(
                bootstrap_concordance
                > 0.5
            )
        ),
    }

    # --------------------------------------------------------
    # 6. Grid-median endpoint validation
    # --------------------------------------------------------
    ranked_grid_df = add_within_region_ranks(
        combined_df,
        "grid_median_gain",
    )

    observed_grid_rho = safe_spearman(
        ranked_grid_df[
            "feature_rank"
        ],
        ranked_grid_df[
            "outcome_rank"
        ],
    )

    grid_exceed = 0

    grid_region_indices = {
        region: group.index.to_numpy()
        for region, group
        in ranked_grid_df.groupby(
            "region"
        )
    }

    original_grid_rank = ranked_grid_df[
        "outcome_rank"
    ].to_numpy(
        dtype=float
    )

    for _ in tqdm(
        range(
            N_STRATIFIED_PERMUTATIONS
        ),
        desc="Grid-median rank permutations",
        unit="replicate",
    ):
        permuted_grid_rank = (
            original_grid_rank.copy()
        )

        for indices in grid_region_indices.values():
            permuted_grid_rank[
                indices
            ] = rng.permutation(
                permuted_grid_rank[
                    indices
                ]
            )

        permuted_rho = safe_spearman(
            ranked_grid_df[
                "feature_rank"
            ],
            permuted_grid_rank,
        )

        grid_exceed += int(
            np.isfinite(
                permuted_rho
            )
            and permuted_rho
            >= observed_grid_rho
        )

    grid_permutation_p = float(
        (
            1 + grid_exceed
        )
        / (
            N_STRATIFIED_PERMUTATIONS
            + 1
        )
    )

    # --------------------------------------------------------
    # 7. Leave-one-region-out validation
    # --------------------------------------------------------
    loo_rows = []

    for omitted_region in [
        "PA",
        "VA",
        "NC",
    ]:
        retained_df = combined_df[
            combined_df[
                "region"
            ]
            != omitted_region
        ].copy()

        retained_ranked = (
            add_within_region_ranks(
                retained_df,
                "primary_gain",
            )
        )

        loo_rows.append(
            {
                "omitted_region": omitted_region,
                "retained_regions": ",".join(
                    sorted(
                        retained_df[
                            "region"
                        ].unique()
                    )
                ),
                "n_species_region_rows": len(
                    retained_ranked
                ),
                "pooled_rank_rho": (
                    safe_spearman(
                        retained_ranked[
                            "feature_rank"
                        ],
                        retained_ranked[
                            "outcome_rank"
                        ],
                    )
                ),
                "pairwise_concordance": (
                    pairwise_concordance(
                        retained_ranked,
                        "feature_rank",
                        "outcome_rank",
                    )[
                        "pairwise_concordance"
                    ]
                ),
            }
        )

    loo_df = pd.DataFrame(
        loo_rows
    )

    loo_df.to_csv(
        OUT_DIR
        / "leave_one_region_out_validation.csv",
        index=False,
    )

    print(
        "\nLEAVE-ONE-REGION-OUT VALIDATION"
    )
    display(
        loo_df
    )

    # --------------------------------------------------------
    # 8. Partial rank audit
    #    Controls:
    #    - raw absolute error (opportunity to improve)
    #    - minimum early/late occupied-cell count (support size)
    # --------------------------------------------------------
    partial_primary_df = (
        build_partial_residual_frame(
            ranked_primary_df
        )
    )

    observed_partial_rho = safe_spearman(
        partial_primary_df[
            "feature_residual"
        ],
        partial_primary_df[
            "outcome_residual"
        ],
    )

    partial_exceed = 0

    partial_region_indices = {
        region: group.index.to_numpy()
        for region, group
        in partial_primary_df.groupby(
            "region"
        )
    }

    original_outcome_residual = (
        partial_primary_df[
            "outcome_residual"
        ].to_numpy(
            dtype=float
        )
    )

    for _ in tqdm(
        range(
            N_PARTIAL_PERMUTATIONS
        ),
        desc="Partial-rank residual permutations",
        unit="replicate",
    ):
        permuted_residual = (
            original_outcome_residual.copy()
        )

        for indices in partial_region_indices.values():
            valid_indices = indices[
                np.isfinite(
                    permuted_residual[
                        indices
                    ]
                )
            ]

            permuted_residual[
                valid_indices
            ] = rng.permutation(
                permuted_residual[
                    valid_indices
                ]
            )

        permuted_rho = safe_spearman(
            partial_primary_df[
                "feature_residual"
            ],
            permuted_residual,
        )

        partial_exceed += int(
            np.isfinite(
                permuted_rho
            )
            and permuted_rho
            >= observed_partial_rho
        )

    partial_permutation_p = float(
        (
            1 + partial_exceed
        )
        / (
            N_PARTIAL_PERMUTATIONS
            + 1
        )
    )

    # --------------------------------------------------------
    # 9. Main threshold-free synthesis table
    # --------------------------------------------------------
    synthesis_df = pd.DataFrame(
        [
            {
                "analysis": (
                    "region_stratified_primary_gain"
                ),
                "n_regions": 3,
                "n_species_region_rows": len(
                    ranked_primary_df
                ),
                "pooled_rank_rho": observed_pooled_rho,
                "one_sided_permutation_p": (
                    pooled_permutation_p
                ),
                "pairwise_concordance": (
                    observed_pairwise[
                        "pairwise_concordance"
                    ]
                ),
                "concordance_permutation_p": (
                    concordance_permutation_p
                ),
                **bootstrap_summary,
            },
            {
                "analysis": (
                    "region_stratified_grid_median_gain"
                ),
                "n_regions": 3,
                "n_species_region_rows": len(
                    ranked_grid_df
                ),
                "pooled_rank_rho": observed_grid_rho,
                "one_sided_permutation_p": (
                    grid_permutation_p
                ),
                "pairwise_concordance": np.nan,
                "concordance_permutation_p": np.nan,
                "pooled_rho_bootstrap_low": np.nan,
                "pooled_rho_bootstrap_high": np.nan,
                "pooled_rho_bootstrap_positive_fraction": np.nan,
                "concordance_bootstrap_low": np.nan,
                "concordance_bootstrap_high": np.nan,
                "concordance_bootstrap_above_half_fraction": np.nan,
            },
            {
                "analysis": (
                    "partial_rank_primary_gain"
                ),
                "n_regions": 3,
                "n_species_region_rows": len(
                    partial_primary_df
                ),
                "pooled_rank_rho": observed_partial_rho,
                "one_sided_permutation_p": (
                    partial_permutation_p
                ),
                "pairwise_concordance": np.nan,
                "concordance_permutation_p": np.nan,
                "pooled_rho_bootstrap_low": np.nan,
                "pooled_rho_bootstrap_high": np.nan,
                "pooled_rho_bootstrap_positive_fraction": np.nan,
                "concordance_bootstrap_low": np.nan,
                "concordance_bootstrap_high": np.nan,
                "concordance_bootstrap_above_half_fraction": np.nan,
            },
        ]
    )

    synthesis_df.to_csv(
        OUT_DIR
        / "threshold_free_ranking_synthesis.csv",
        index=False,
    )

    print(
        "\nTHRESHOLD-FREE RANKING SYNTHESIS"
    )
    display(
        synthesis_df
    )

    # --------------------------------------------------------
    # 10. Locked decision
    # --------------------------------------------------------
    all_region_rhos_positive = bool(
        (
            region_effect_df[
                "rho_primary_gain"
            ]
            > 0
        ).all()
    )

    all_region_grid_rhos_positive = bool(
        (
            region_effect_df[
                "rho_grid_median_gain"
            ]
            > 0
        ).all()
    )

    independent_meta_row = meta_df[
        meta_df[
            "analysis"
        ]
        == "independent_regions_only"
    ].iloc[0]

    all_region_meta_row = meta_df[
        meta_df[
            "analysis"
        ]
        == "all_three_regions"
    ].iloc[0]

    meta_pass = bool(
        independent_meta_row[
            "random_effect_rho"
        ]
        >= MIN_RANDOM_EFFECT_RHO
        and independent_meta_row[
            "random_effect_ci_low"
        ]
        > 0
        and all_region_meta_row[
            "random_effect_ci_low"
        ]
        > 0
    )

    pooled_rank_pass = bool(
        observed_pooled_rho
        >= MIN_POOLED_RANK_RHO
        and pooled_permutation_p
        <= MAX_POOLED_PERMUTATION_P
        and bootstrap_summary[
            "pooled_rho_bootstrap_positive_fraction"
        ]
        >= MIN_BOOTSTRAP_POSITIVE_FRACTION
    )

    concordance_pass = bool(
        observed_pairwise[
            "pairwise_concordance"
        ]
        >= MIN_PAIRWISE_CONCORDANCE
        and concordance_permutation_p
        <= MAX_CONCORDANCE_PERMUTATION_P
    )

    grid_endpoint_pass = bool(
        observed_grid_rho
        >= MIN_GRID_MEDIAN_RHO
        and grid_permutation_p
        <= MAX_GRID_MEDIAN_PERMUTATION_P
        and all_region_grid_rhos_positive
    )

    loo_pass = bool(
        (
            loo_df[
                "pooled_rank_rho"
            ]
            > 0
        ).all()
        and (
            loo_df[
                "pairwise_concordance"
            ]
            > 0.5
        ).all()
    )

    ranking_signal_passed = bool(
        all_region_rhos_positive
        and meta_pass
        and pooled_rank_pass
        and concordance_pass
        and grid_endpoint_pass
        and loo_pass
    )

    partial_rank_passed = bool(
        np.isfinite(
            observed_partial_rho
        )
        and observed_partial_rho > 0
        and partial_permutation_p <= 0.05
    )

    if (
        ranking_signal_passed
        and partial_rank_passed
    ):
        status = (
            "CELL_JACCARD_TRANSPORTABLE_APPLICABILITY_GATE"
        )
        passed = True
        interpretation = (
            "Cell continuity is a replicated, threshold-free ranking "
            "signal and retains evidence beyond raw-error opportunity "
            "and minimum spatial support."
        )
        next_step = (
            "Freeze cell_jaccard as the only applicability feature. "
            "Do not choose a hard cutoff yet. Run registered-GBIF-"
            "download confirmation and manuscript-ready robustness "
            "figures using continuous ranks."
        )

    elif ranking_signal_passed:
        status = (
            "CELL_JACCARD_TRANSPORTABLE_SUPPORT_SIGNAL"
        )
        passed = True
        interpretation = (
            "Cell continuity is a replicated ranking signal, but its "
            "incremental association after controlling support and raw "
            "error is not definitive. Present it as a support-quality "
            "gate rather than a causal mechanism."
        )
        next_step = (
            "Freeze cell_jaccard as a continuous support-quality signal. "
            "Do not claim an independent causal mechanism or choose a "
            "hard threshold. Proceed to registered-download confirmation."
        )

    else:
        status = (
            "CELL_JACCARD_SYNTHESIS_NOT_STABLE"
        )
        passed = False
        interpretation = (
            "The NC confirmation does not produce a stable three-region "
            "threshold-free ranking signal under the locked synthesis."
        )
        next_step = (
            "Do not construct an applicability gate. Retain the NC result "
            "as one successful confirmation and frame the paper around "
            "heterogeneous, incompletely transportable correction behavior."
        )

    decision_df = pd.DataFrame(
        [
            {
                "feature": FEATURE,
                "all_region_rhos_positive": (
                    all_region_rhos_positive
                ),
                "all_region_grid_rhos_positive": (
                    all_region_grid_rhos_positive
                ),
                "meta_pass": meta_pass,
                "pooled_rank_pass": (
                    pooled_rank_pass
                ),
                "concordance_pass": (
                    concordance_pass
                ),
                "grid_endpoint_pass": (
                    grid_endpoint_pass
                ),
                "leave_one_region_out_pass": (
                    loo_pass
                ),
                "partial_rank_rho": (
                    observed_partial_rho
                ),
                "partial_rank_permutation_p": (
                    partial_permutation_p
                ),
                "partial_rank_pass": (
                    partial_rank_passed
                ),
                "ranking_signal_passed": (
                    ranking_signal_passed
                ),
                "status": status,
                "passed": passed,
                "interpretation": interpretation,
                "next_step": next_step,
            }
        ]
    )

    decision_df.to_csv(
        OUT_DIR
        / "locked_three_region_decision.csv",
        index=False,
    )

    print(
        "\nLOCKED THREE-REGION DECISION"
    )
    display(
        decision_df
    )

    # --------------------------------------------------------
    # 11. Visual previews
    # --------------------------------------------------------
    forest_df = region_effect_df.sort_values(
        "rho_primary_gain"
    ).reset_index(
        drop=True
    )

    y = np.arange(
        len(
            forest_df
        )
    )

    lower_error = (
        forest_df[
            "rho_primary_gain"
        ]
        - forest_df[
            "rho_ci_low_approx"
        ]
    )

    upper_error = (
        forest_df[
            "rho_ci_high_approx"
        ]
        - forest_df[
            "rho_primary_gain"
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )
    plt.errorbar(
        forest_df[
            "rho_primary_gain"
        ],
        y,
        xerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        capsize=4,
    )
    plt.axvline(
        0,
        linestyle="--",
    )
    plt.yticks(
        y,
        forest_df[
            "region"
        ],
    )
    plt.xlabel(
        "Spearman rho: cell_jaccard vs correction gain"
    )
    plt.ylabel(
        "Region"
    )
    plt.title(
        "Region-specific cell-continuity effects"
    )
    plt.tight_layout()
    plt.show()

    scatter_df = ranked_primary_df.copy()

    plt.figure(
        figsize=(8, 6)
    )

    for region, group in scatter_df.groupby(
        "region"
    ):
        plt.scatter(
            group[
                "feature_rank"
            ],
            group[
                "outcome_rank"
            ],
            label=region,
        )

    plt.xlabel(
        "Within-region percentile rank of cell_jaccard"
    )
    plt.ylabel(
        "Within-region percentile rank of correction gain"
    )
    plt.title(
        "Threshold-free three-region ranking validation"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 12. README
    # --------------------------------------------------------
    region_compact = (
        region_effect_df.round(6).to_dict(
            "records"
        )
    )

    meta_compact = (
        meta_df.round(6).to_dict(
            "records"
        )
    )

    synthesis_compact = (
        synthesis_df.round(6).to_dict(
            "records"
        )
    )

    loo_compact = (
        loo_df.round(6).to_dict(
            "records"
        )
    )

    append_readme(
        f"""

## Locked PA–VA–NC cell-jaccard synthesis — {RUN_UTC}

### Final feature policy
- Retained feature: cell_jaccard
- Permanently discarded feature: abs_log_observer_growth
- New features searched: No
- Hard threshold selected: No

### Data
- PA species: {int((combined_df['region'] == 'PA').sum())}
- VA species: {int((combined_df['region'] == 'VA').sum())}
- NC species: {int((combined_df['region'] == 'NC').sum())}
- Species-region rows: {len(combined_df)}
- Unique species: {combined_df['fia_species_code'].nunique()}
- GPU: not used

### Analyses
- Region-specific Spearman correlations
- Fisher-z fixed/random-effects synthesis
- Independent-region-only synthesis: VA + NC
- Region-stratified rank permutation
- Within-region pairwise concordance
- Species-cluster bootstrap
- Leave-one-region-out validation
- Partial rank audit controlling raw absolute error and minimum cell support

### Locked result
- All region primary rhos positive: {all_region_rhos_positive}
- Independent random-effect rho:
  {independent_meta_row['random_effect_rho']}
- Independent random-effect 95% CI:
  [{independent_meta_row['random_effect_ci_low']},
   {independent_meta_row['random_effect_ci_high']}]
- Pooled within-region rank rho:
  {observed_pooled_rho}
- Pooled rank permutation p:
  {pooled_permutation_p}
- Pairwise concordance:
  {observed_pairwise['pairwise_concordance']}
- Concordance permutation p:
  {concordance_permutation_p}
- Grid-median pooled rho:
  {observed_grid_rho}
- Grid-median permutation p:
  {grid_permutation_p}
- Partial rank rho:
  {observed_partial_rho}
- Partial rank permutation p:
  {partial_permutation_p}
- Status: **{status}**
- Passed: {passed}
- Interpretation: {interpretation}
- Next step: {next_step}

### Interpretation boundary
cell_jaccard is an applicability-ranking or support-quality signal. It is not
proof that spatial continuity causally determines correction performance.
The external reference is repeated-plot FIA occurrence, not perfect ecological
truth. No hard deployment threshold was estimated.
"""
    )

    # --------------------------------------------------------
    # 13. Compact output
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
        "FINAL_RETAINED_FEATURE: cell_jaccard"
    )
    print(
        "PERMANENTLY_DISCARDED_FEATURE: "
        "abs_log_observer_growth"
    )
    print(
        f"N_SPECIES_REGION_ROWS: "
        f"{len(combined_df)}"
    )
    print(
        f"N_UNIQUE_SPECIES: "
        f"{combined_df['fia_species_code'].nunique()}"
    )
    print(
        "REGION_EFFECTS: "
        + json.dumps(
            region_compact,
            ensure_ascii=False,
        )
    )
    print(
        "META_ANALYSIS: "
        + json.dumps(
            meta_compact,
            ensure_ascii=False,
        )
    )
    print(
        "THRESHOLD_FREE_SYNTHESIS: "
        + json.dumps(
            synthesis_compact,
            ensure_ascii=False,
        )
    )
    print(
        "LEAVE_ONE_REGION_OUT: "
        + json.dumps(
            loo_compact,
            ensure_ascii=False,
        )
    )
    print(
        f"ALL_REGION_RHOS_POSITIVE: "
        f"{all_region_rhos_positive}"
    )
    print(
        f"RANKING_SIGNAL_PASSED: "
        f"{ranking_signal_passed}"
    )
    print(
        f"PARTIAL_RANK_PASSED: "
        f"{partial_rank_passed}"
    )
    print(
        "HARD_THRESHOLD_SELECTED: False"
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
        "CELL17A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed output. "
            "Do not search additional features or choose a cutoff."
        ),
    )
