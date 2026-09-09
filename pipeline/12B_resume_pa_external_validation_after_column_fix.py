
# Cell 12B — Resume PA external validation after duplicate-column bug
#
# Root cause in Cell 12A
# ----------------------
# For method == "raw", the selected column list contained `raw_abs_error` twice:
#
#   [estimate_column, "fia_shift_km_decade", "raw_abs_error", error_column]
#
# Because error_column was also "raw_abs_error", pandas returned a DataFrame when
# indexing method_eval["raw_abs_error"]. Converting its median to float caused:
#
#   TypeError: cannot convert the series to <class 'float'>
#
# This repair cell:
# 1. Reuses Cell 12A's completed species-level and grid outputs.
# 2. Rebuilds each method table with unique internal column names.
# 3. Completes bootstrap, Wilcoxon, grid sensitivity, figures and decision.
# 4. Does NOT reread PA_TREE.csv or rescan FIA data.

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 220)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 320)

# ============================================================
# Configuration — unchanged from Cell 12A
# ============================================================
PRIMARY_GRID_RESOLUTION = 0.50
GRID_RESOLUTIONS = [0.25, 0.50, 1.00]

MIN_INAT_RECORDS_PER_SPECIES_PERIOD = 30
MIN_INAT_POSITIVE_CELLS_PER_SPECIES_PERIOD = 3
MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD = 15

MIN_EVALUATION_SPECIES = 10
TARGET_MEDIAN_ERROR_REDUCTION = 0.20
TARGET_BOOTSTRAP_IMPROVEMENT_PROBABILITY = 0.90

N_SPECIES_BOOTSTRAP = 1000
RANDOM_SEED = 20260710

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
OUT_DIR = (
    BASE_DIR
    / "derived"
    / "pa_species_external_validation"
)

PRIMARY_COMPARISON_PATH = (
    OUT_DIR / "species_external_validation.parquet"
)
FIA_REFERENCE_PATH = (
    OUT_DIR / "fia_same_window_species_reference.parquet"
)
FRAME_SUMMARY_PATH = (
    OUT_DIR / "grid_frame_summary.csv"
)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("OUT_DIR:", OUT_DIR)
print("N_SPECIES_BOOTSTRAP:", N_SPECIES_BOOTSTRAP)
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
        spearmanr(
            frame["x"],
            frame["y"],
        ).statistic
    )


def direction_agreement(
    estimate,
    reference,
    minimum_reference=1.0,
):
    estimate = np.asarray(
        estimate,
        dtype=float,
    )
    reference = np.asarray(
        reference,
        dtype=float,
    )

    valid = (
        np.isfinite(estimate)
        & np.isfinite(reference)
        & (
            np.abs(reference)
            >= minimum_reference
        )
    )

    if not valid.any():
        return np.nan

    return float(
        np.mean(
            np.sign(estimate[valid])
            == np.sign(reference[valid])
        )
    )


def bootstrap_method_comparison(
    raw_errors,
    method_errors,
    rng,
    label,
):
    raw_errors = np.asarray(
        raw_errors,
        dtype=float,
    )
    method_errors = np.asarray(
        method_errors,
        dtype=float,
    )

    valid = (
        np.isfinite(raw_errors)
        & np.isfinite(method_errors)
    )

    raw_errors = raw_errors[valid]
    method_errors = method_errors[valid]

    if len(raw_errors) == 0:
        return {
            "bootstrap_reduction_low": np.nan,
            "bootstrap_reduction_high": np.nan,
            "bootstrap_probability_lower_error": np.nan,
        }

    reduction_values = np.empty(
        N_SPECIES_BOOTSTRAP,
        dtype=float,
    )
    lower_count = 0

    for replicate in tqdm(
        range(N_SPECIES_BOOTSTRAP),
        desc=f"Bootstrap {label}",
        unit="replicate",
        leave=False,
    ):
        indices = rng.integers(
            0,
            len(raw_errors),
            size=len(raw_errors),
        )

        raw_median = float(
            np.median(
                raw_errors[indices]
            )
        )
        method_median = float(
            np.median(
                method_errors[indices]
            )
        )

        reduction_values[replicate] = (
            1 - method_median / raw_median
            if raw_median > 0
            else np.nan
        )

        lower_count += int(
            method_median < raw_median
        )

    return {
        "bootstrap_reduction_low": float(
            np.nanquantile(
                reduction_values,
                0.025,
            )
        ),
        "bootstrap_reduction_high": float(
            np.nanquantile(
                reduction_values,
                0.975,
            )
        ),
        "bootstrap_probability_lower_error": float(
            lower_count
            / N_SPECIES_BOOTSTRAP
        ),
    }


def resolution_file(resolution):
    tag = str(resolution).replace(
        ".",
        "p",
    )

    return (
        OUT_DIR
        / f"inat_species_estimands_grid_{tag}.parquet"
    )


def print_failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## Cell 12B repair failure — {RUN_UTC}
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
    # 1. Validate cached outputs from Cell 12A
    # --------------------------------------------------------
    required_paths = [
        PRIMARY_COMPARISON_PATH,
        FIA_REFERENCE_PATH,
        FRAME_SUMMARY_PATH,
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
            "Cell 12A did not save all resumable outputs: "
            + " | ".join(missing_paths)
        )

    evaluation_df = pd.read_parquet(
        PRIMARY_COMPARISON_PATH
    )
    fia_reference_df = pd.read_parquet(
        FIA_REFERENCE_PATH
    )
    frame_summary_df = pd.read_csv(
        FRAME_SUMMARY_PATH
    )

    print("\nCACHED GRID FRAME SUMMARY")
    display(frame_summary_df)

    print("\nCACHED PRIMARY SPECIES COMPARISON")
    display(
        evaluation_df[
            [
                column
                for column in [
                    "fia_species_code",
                    "scientific_name",
                    "raw_shift_km_decade",
                    "equal_cell_shift_km_decade",
                    "effort_standardized_shift_km_decade",
                    "observer_balanced_shift_km_decade",
                    "fia_shift_km_decade",
                    "raw_abs_error",
                    "equal_cell_abs_error",
                    "effort_standardized_abs_error",
                    "observer_balanced_abs_error",
                ]
                if column in evaluation_df.columns
            ]
        ]
    )

    required_evaluation_columns = [
        "fia_species_code",
        "scientific_name",
        "raw_shift_km_decade",
        "equal_cell_shift_km_decade",
        "effort_standardized_shift_km_decade",
        "observer_balanced_shift_km_decade",
        "fia_shift_km_decade",
    ]

    missing_columns = [
        column
        for column in required_evaluation_columns
        if column not in evaluation_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Cached comparison table is missing columns: "
            + " | ".join(missing_columns)
        )

    # Recompute all errors explicitly to avoid relying on duplicated names.
    method_columns = {
        "raw": "raw_shift_km_decade",
        "equal_cell": "equal_cell_shift_km_decade",
        "effort_standardized": (
            "effort_standardized_shift_km_decade"
        ),
        "observer_balanced": (
            "observer_balanced_shift_km_decade"
        ),
    }

    for method, estimate_column in method_columns.items():
        evaluation_df[
            f"{method}_abs_error"
        ] = (
            pd.to_numeric(
                evaluation_df[estimate_column],
                errors="coerce",
            )
            - pd.to_numeric(
                evaluation_df[
                    "fia_shift_km_decade"
                ],
                errors="coerce",
            )
        ).abs()

    evaluation_df.to_parquet(
        PRIMARY_COMPARISON_PATH,
        index=False,
    )
    evaluation_df.to_csv(
        OUT_DIR
        / "species_external_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Fixed method-level external validation summary
    # --------------------------------------------------------
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    method_rows = []

    for method, estimate_column in method_columns.items():
        method_eval = pd.DataFrame(
            {
                "estimate": pd.to_numeric(
                    evaluation_df[
                        estimate_column
                    ],
                    errors="coerce",
                ),
                "fia_reference": pd.to_numeric(
                    evaluation_df[
                        "fia_shift_km_decade"
                    ],
                    errors="coerce",
                ),
                "raw_abs_error": pd.to_numeric(
                    evaluation_df[
                        "raw_abs_error"
                    ],
                    errors="coerce",
                ),
                "method_abs_error": pd.to_numeric(
                    evaluation_df[
                        f"{method}_abs_error"
                    ],
                    errors="coerce",
                ),
            }
        ).dropna()

        if method_eval.empty:
            continue

        raw_median_abs_error = float(
            method_eval[
                "raw_abs_error"
            ].median()
        )
        method_median_abs_error = float(
            method_eval[
                "method_abs_error"
            ].median()
        )
        method_mean_abs_error = float(
            method_eval[
                "method_abs_error"
            ].mean()
        )

        median_error_reduction = (
            1
            - method_median_abs_error
            / raw_median_abs_error
            if raw_median_abs_error > 0
            else np.nan
        )

        if method == "raw":
            bootstrap = {
                "bootstrap_reduction_low": 0.0,
                "bootstrap_reduction_high": 0.0,
                "bootstrap_probability_lower_error": 0.0,
            }
            wilcoxon_p = np.nan

        else:
            bootstrap = bootstrap_method_comparison(
                method_eval[
                    "raw_abs_error"
                ],
                method_eval[
                    "method_abs_error"
                ],
                rng,
                method,
            )

            paired_difference = (
                method_eval[
                    "raw_abs_error"
                ]
                - method_eval[
                    "method_abs_error"
                ]
            )

            if paired_difference.abs().sum() > 0:
                try:
                    wilcoxon_p = float(
                        wilcoxon(
                            method_eval[
                                "raw_abs_error"
                            ],
                            method_eval[
                                "method_abs_error"
                            ],
                            alternative="greater",
                            zero_method="wilcox",
                        ).pvalue
                    )
                except Exception:
                    wilcoxon_p = np.nan
            else:
                wilcoxon_p = np.nan

        method_rows.append(
            {
                "method": method,
                "n_species": len(
                    method_eval
                ),
                "median_abs_error_km_decade": (
                    method_median_abs_error
                ),
                "mean_abs_error_km_decade": (
                    method_mean_abs_error
                ),
                "median_error_reduction_vs_raw": (
                    median_error_reduction
                ),
                "direction_agreement": (
                    direction_agreement(
                        method_eval[
                            "estimate"
                        ],
                        method_eval[
                            "fia_reference"
                        ],
                    )
                ),
                "spearman_with_fia": (
                    safe_spearman(
                        method_eval[
                            "estimate"
                        ],
                        method_eval[
                            "fia_reference"
                        ],
                    )
                ),
                "wilcoxon_p_lower_error_vs_raw": (
                    wilcoxon_p
                ),
                **bootstrap,
            }
        )

    method_summary_df = pd.DataFrame(
        method_rows
    )

    method_summary_df.to_csv(
        OUT_DIR
        / "method_external_validation_summary.csv",
        index=False,
    )

    print("\nFIXED METHOD EXTERNAL VALIDATION SUMMARY")
    display(method_summary_df)

    # --------------------------------------------------------
    # 3. Rebuild grid-resolution sensitivity from cached files
    # --------------------------------------------------------
    grid_rows = []

    for resolution in tqdm(
        GRID_RESOLUTIONS,
        desc="Rebuilding grid sensitivity",
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

        required_grid_columns = [
            "raw_n_records_early",
            "raw_n_records_late",
            "effort_standardized_species_records_early",
            "effort_standardized_species_records_late",
            "effort_standardized_positive_cells_early",
            "effort_standardized_positive_cells_late",
            "raw_shift_km_decade",
            "effort_standardized_shift_km_decade",
            "fia_n_occupied_plots_early",
            "fia_n_occupied_plots_late",
            "fia_shift_km_decade",
        ]

        missing_grid_columns = [
            column
            for column in required_grid_columns
            if column not in merged.columns
        ]

        if missing_grid_columns:
            raise KeyError(
                f"Grid {resolution} is missing columns: "
                + " | ".join(
                    missing_grid_columns
                )
            )

        valid = (
            merged[
                "raw_n_records_early"
            ].ge(
                MIN_INAT_RECORDS_PER_SPECIES_PERIOD
            )
            & merged[
                "raw_n_records_late"
            ].ge(
                MIN_INAT_RECORDS_PER_SPECIES_PERIOD
            )
            & merged[
                "effort_standardized_species_records_early"
            ].ge(
                MIN_INAT_RECORDS_PER_SPECIES_PERIOD
            )
            & merged[
                "effort_standardized_species_records_late"
            ].ge(
                MIN_INAT_RECORDS_PER_SPECIES_PERIOD
            )
            & merged[
                "effort_standardized_positive_cells_early"
            ].ge(
                MIN_INAT_POSITIVE_CELLS_PER_SPECIES_PERIOD
            )
            & merged[
                "effort_standardized_positive_cells_late"
            ].ge(
                MIN_INAT_POSITIVE_CELLS_PER_SPECIES_PERIOD
            )
            & merged[
                "fia_n_occupied_plots_early"
            ].ge(
                MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD
            )
            & merged[
                "fia_n_occupied_plots_late"
            ].ge(
                MIN_FIA_OCCUPIED_PLOTS_PER_SPECIES_PERIOD
            )
        )

        valid_df = merged.loc[
            valid
        ].dropna(
            subset=[
                "raw_shift_km_decade",
                "effort_standardized_shift_km_decade",
                "fia_shift_km_decade",
            ]
        )

        if valid_df.empty:
            continue

        raw_error = (
            valid_df[
                "raw_shift_km_decade"
            ]
            - valid_df[
                "fia_shift_km_decade"
            ]
        ).abs()

        corrected_error = (
            valid_df[
                "effort_standardized_shift_km_decade"
            ]
            - valid_df[
                "fia_shift_km_decade"
            ]
        ).abs()

        raw_median = float(
            raw_error.median()
        )
        corrected_median = float(
            corrected_error.median()
        )

        grid_rows.append(
            {
                "grid_resolution": resolution,
                "n_species": len(
                    valid_df
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
                "direction_agreement": (
                    direction_agreement(
                        valid_df[
                            "effort_standardized_shift_km_decade"
                        ],
                        valid_df[
                            "fia_shift_km_decade"
                        ],
                    )
                ),
                "spearman_with_fia": (
                    safe_spearman(
                        valid_df[
                            "effort_standardized_shift_km_decade"
                        ],
                        valid_df[
                            "fia_shift_km_decade"
                        ],
                    )
                ),
            }
        )

    grid_sensitivity_df = pd.DataFrame(
        grid_rows
    )

    grid_sensitivity_df.to_csv(
        OUT_DIR
        / "grid_resolution_external_validation.csv",
        index=False,
    )

    print("\nGRID-RESOLUTION EXTERNAL VALIDATION")
    display(grid_sensitivity_df)

    # --------------------------------------------------------
    # 4. Decision
    # --------------------------------------------------------
    n_evaluation_species = len(
        evaluation_df
    )

    primary_rows = method_summary_df[
        method_summary_df["method"]
        == "effort_standardized"
    ]
    raw_rows = method_summary_df[
        method_summary_df["method"]
        == "raw"
    ]

    if (
        primary_rows.empty
        or raw_rows.empty
    ):
        primary_row = None
        raw_row = None
    else:
        primary_row = primary_rows.iloc[0]
        raw_row = raw_rows.iloc[0]

    coverage_pass = bool(
        n_evaluation_species
        >= MIN_EVALUATION_SPECIES
    )

    if primary_row is not None:
        error_reduction_pass = bool(
            primary_row[
                "median_error_reduction_vs_raw"
            ]
            >= TARGET_MEDIAN_ERROR_REDUCTION
        )

        bootstrap_pass = bool(
            primary_row[
                "bootstrap_probability_lower_error"
            ]
            >= TARGET_BOOTSTRAP_IMPROVEMENT_PROBABILITY
        )

        direction_not_worse = bool(
            pd.isna(
                raw_row[
                    "direction_agreement"
                ]
            )
            or pd.isna(
                primary_row[
                    "direction_agreement"
                ]
            )
            or primary_row[
                "direction_agreement"
            ]
            >= raw_row[
                "direction_agreement"
            ]
        )

        rank_not_worse = bool(
            pd.isna(
                raw_row[
                    "spearman_with_fia"
                ]
            )
            or pd.isna(
                primary_row[
                    "spearman_with_fia"
                ]
            )
            or primary_row[
                "spearman_with_fia"
            ]
            >= raw_row[
                "spearman_with_fia"
            ]
        )

        agreement_pass = bool(
            direction_not_worse
            or rank_not_worse
        )
    else:
        error_reduction_pass = False
        bootstrap_pass = False
        agreement_pass = False

    passed = bool(
        coverage_pass
        and error_reduction_pass
        and bootstrap_pass
        and agreement_pass
    )

    if passed:
        status = (
            "GO_FOR_FULL_PA_CORRECTION_CONFIRMATION"
        )
        next_step = (
            "The locked 0.5-degree target-group effort correction "
            "improves agreement with same-window repeated FIA plots. "
            "Next run annual-window sensitivity, observer-cluster "
            "bootstrap and registered GBIF download confirmation."
        )
    elif (
        coverage_pass
        and error_reduction_pass
    ):
        status = (
            "REAL_CORRECTION_PARTIAL_EXTERNAL_VALIDATION"
        )
        next_step = (
            "The correction reduces point-estimate error, but the "
            "across-species evidence is not fully stable. Inspect "
            "species drivers and window sensitivity before expansion."
        )
    elif coverage_pass:
        status = (
            "REAL_CORRECTION_NOT_VALIDATED_AGAINST_FIA"
        )
        next_step = (
            "The semi-synthetic correction does not improve real PA "
            "agreement with FIA. Do not scale the correction; analyze "
            "failure conditions and consider a sensitivity-index paper."
        )
    else:
        status = (
            "EXTERNAL_VALIDATION_COVERAGE_INSUFFICIENT"
        )
        next_step = (
            "Too few species support a fair comparison. Do not "
            "interpret correction performance."
        )

    # --------------------------------------------------------
    # 5. Visual previews
    # --------------------------------------------------------
    if not evaluation_df.empty:
        plt.figure(figsize=(8, 7))
        plt.scatter(
            evaluation_df[
                "fia_shift_km_decade"
            ],
            evaluation_df[
                "raw_shift_km_decade"
            ],
            label="Raw iNaturalist",
        )
        plt.scatter(
            evaluation_df[
                "fia_shift_km_decade"
            ],
            evaluation_df[
                "effort_standardized_shift_km_decade"
            ],
            label="Effort-standardized iNaturalist",
        )

        combined_values = pd.concat(
            [
                evaluation_df[
                    "fia_shift_km_decade"
                ],
                evaluation_df[
                    "raw_shift_km_decade"
                ],
                evaluation_df[
                    "effort_standardized_shift_km_decade"
                ],
            ]
        ).dropna()

        if len(combined_values):
            lower = float(
                combined_values.min()
            )
            upper = float(
                combined_values.max()
            )
            plt.plot(
                [lower, upper],
                [lower, upper],
                linestyle="--",
                label="1:1",
            )

        plt.xlabel(
            "FIA repeated-plot shift (km/decade)"
        )
        plt.ylabel(
            "iNaturalist shift (km/decade)"
        )
        plt.title(
            "PA species shifts: iNaturalist versus FIA"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

        error_plot_df = (
            evaluation_df[
                [
                    "scientific_name",
                    "raw_abs_error",
                    "effort_standardized_abs_error",
                ]
            ]
            .sort_values(
                "raw_abs_error",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        y = np.arange(
            len(error_plot_df)
        )

        plt.figure(
            figsize=(
                11,
                max(
                    6,
                    0.55
                    * len(error_plot_df),
                ),
            )
        )
        plt.scatter(
            error_plot_df[
                "raw_abs_error"
            ],
            y,
            label="Raw error",
        )
        plt.scatter(
            error_plot_df[
                "effort_standardized_abs_error"
            ],
            y,
            label="Corrected error",
        )

        for index, row in error_plot_df.iterrows():
            plt.plot(
                [
                    row[
                        "raw_abs_error"
                    ],
                    row[
                        "effort_standardized_abs_error"
                    ],
                ],
                [index, index],
            )

        plt.yticks(
            y,
            error_plot_df[
                "scientific_name"
            ],
        )
        plt.xlabel(
            "Absolute error versus FIA (km/decade)"
        )
        plt.ylabel("Species")
        plt.title(
            "Effect of the locked 0.5-degree correction"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 6. README
    # --------------------------------------------------------
    compact_methods = (
        method_summary_df.round(6).to_dict(
            "records"
        )
    )
    compact_grids = (
        grid_sensitivity_df.round(6).to_dict(
            "records"
        )
    )

    top_species_comparison = (
        evaluation_df[
            [
                "fia_species_code",
                "scientific_name",
                "raw_shift_km_decade",
                "effort_standardized_shift_km_decade",
                "fia_shift_km_decade",
                "raw_abs_error",
                "effort_standardized_abs_error",
            ]
        ]
        .sort_values(
            "raw_abs_error",
            ascending=False,
        )
        .head(15)
        .round(4)
        .to_dict("records")
    )

    append_readme(
        f"""

## Cell 12B external-validation repair — {RUN_UTC}

### Code correction
Cell 12A duplicated the `raw_abs_error` column when building the raw-method
summary table. Pandas therefore returned a DataFrame rather than a Series, and
`float(median)` raised a TypeError. Cell 12B rebuilt every method table with
unique internal columns and resumed from cached outputs without rescanning FIA.

### External validation
- Evaluation species: {n_evaluation_species}
- Primary grid: {PRIMARY_GRID_RESOLUTION} degrees
- Coverage passed: {coverage_pass}
- Median-error reduction passed: {error_reduction_pass}
- Bootstrap improvement passed: {bootstrap_pass}
- Direction/rank agreement not worsened: {agreement_pass}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}
"""
    )

    # --------------------------------------------------------
    # 7. Compact output
    # --------------------------------------------------------
    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(
        "PATCH_APPLIED: Removed duplicated raw_abs_error "
        "selection and resumed from cached Cell 12A outputs"
    )
    print(f"OUT_DIR: {OUT_DIR}")
    print(
        f"N_EVALUATION_SPECIES: "
        f"{n_evaluation_species}"
    )
    print(
        "METHOD_VALIDATION: "
        + json.dumps(
            compact_methods,
            ensure_ascii=False,
        )
    )
    print(
        "GRID_SENSITIVITY: "
        + json.dumps(
            compact_grids,
            ensure_ascii=False,
        )
    )
    print(
        "TOP_SPECIES_COMPARISON: "
        + json.dumps(
            top_species_comparison,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_CRITERIA: >=10 species; primary 0.5-degree "
        "effort correction reduces median absolute error by >=20%; "
        "bootstrap probability of lower median error >=0.90; "
        "direction agreement or rank correlation not worsened"
    )
    print(f"COVERAGE_PASSED: {coverage_pass}")
    print(
        f"ERROR_REDUCTION_PASSED: "
        f"{error_reduction_pass}"
    )
    print(
        f"BOOTSTRAP_IMPROVEMENT_PASSED: "
        f"{bootstrap_pass}"
    )
    print(
        f"AGREEMENT_PASSED: "
        f"{agreement_pass}"
    )
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except Exception as exc:
    print_failure(
        "CELL12B_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed output. "
            "Do not rerun the large FIA scan."
        ),
    )
