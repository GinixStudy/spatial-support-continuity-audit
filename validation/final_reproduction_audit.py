
# Final full-reproduction audit and environment capture
#
# Run after:
# - the locked manuscript pipeline has generated its final package
# - the exact Pennsylvania matched-plot reconstruction has completed
#
# Purpose
# -------
# 1. Verify all manuscript-critical locked results from saved outputs.
# 2. Verify PA/VA/NC same-window matched physical-plot counts.
# 3. Verify the 600,000-replicate mechanism confirmation.
# 4. Capture the exact Kaggle environment that reproduced the full run.
# 5. Copy small reproducibility materials into the final Q1 mechanism package.
#
# This cell performs no new scientific inference.
# Expected runtime: 1–4 minutes, CPU only.

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
try:
    from IPython.display import display
except ImportError:
    def display(value):
        """Plain-text fallback when the audit runs outside IPython."""
        if hasattr(value, "to_string"):
            print(value.to_string(index=False))
        else:
            print(value)


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(
    os.environ.get(
        "REPRO_BASE_DIR",
        "/kaggle/working/fia_temporal_observation_drift",
    )
)

DERIVED_DIR = BASE_DIR / "derived"

AUDIT_DIR = Path(
    os.environ.get(
        "REPRO_AUDIT_DIR",
        "/kaggle/working/submission_critical_repair_audit",
    )
)

PA_COUNT_DIR = Path(
    os.environ.get(
        "REPRO_PA_COUNT_DIR",
        str(AUDIT_DIR / "pa_exact_original_cell12a_count"),
    )
)

FINAL_PACKAGE = Path(
    os.environ.get(
        "REPRO_FINAL_PACKAGE",
        str(BASE_DIR / "final_manuscript_package_q1_mechanism"),
    )
)

REPRO_DIR = (
    FINAL_PACKAGE
    / "reproducibility"
)

ENV_PACKAGE_DIR = (
    REPRO_DIR
    / "environment"
)

PA_PACKAGE_DIR = (
    REPRO_DIR
    / "pa_exact_count"
)

AUDIT_PACKAGE_DIR = (
    REPRO_DIR
    / "final_audit"
)

for directory in [
    AUDIT_DIR,
    REPRO_DIR,
    ENV_PACKAGE_DIR,
    PA_PACKAGE_DIR,
    AUDIT_PACKAGE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

RUN_UTC = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

TOLERANCE = 5e-6

print("RUN_UTC:", RUN_UTC)
print("BASE_DIR:", BASE_DIR)
print("AUDIT_DIR:", AUDIT_DIR)
print("FINAL_PACKAGE:", FINAL_PACKAGE)
print("GPU: not used")
print("NEW_SCIENTIFIC_INFERENCE: False")


# ============================================================
# Locked values
# ============================================================
EXPECTED = {
    "gbif_rows": 143_355,
    "gbif_state_counts": {
        "Pennsylvania": 52_472,
        "Virginia": 46_754,
        "North Carolina": 44_129,
    },
    "gbif_years": [
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
    ],
    "matched_plots": {
        "PA": 2_199,
        "VA": 3_068,
        "NC": 2_516,
    },
    "formal_species": {
        "PA": 15,
        "VA": 11,
        "NC": 17,
    },
    "species_region_rows": 43,
    "unique_species_clusters": 19,
    "state_context": {
        "PA": {
            "raw_median_abs_error": 10.415667,
            "corrected_median_abs_error": 8.801564,
            "median_error_reduction": 0.154969,
            "species_improved_fraction": 0.4,
            "median_gain_km_decade": -1.184776,
        },
        "VA": {
            "raw_median_abs_error": 9.743634,
            "corrected_median_abs_error": 16.732955,
            "median_error_reduction": -0.717322,
            "species_improved_fraction": 0.181818,
            "median_gain_km_decade": -4.849790,
        },
        "NC": {
            "raw_median_abs_error": 5.566042,
            "corrected_median_abs_error": 7.447582,
            "median_error_reduction": -0.338039,
            "species_improved_fraction": 0.411765,
            "median_gain_km_decade": -0.287391,
        },
    },
    "state_feature": {
        "PA_cell_jaccard_rho": 0.717857,
        "VA_cell_jaccard_rho": 0.327273,
        "NC_cell_jaccard_rho": 0.571078,
        "NC_cell_jaccard_holm_p": 0.019598,
    },
    "confirmation_meta_rho": 0.490433,
    "confirmation_meta_p": 0.005918,
    "threshold_primary": {
        "stratified_rank_rho": 0.560363,
        "rank_permutation_p": 0.0002,
        "rank_bootstrap_low": 0.249067,
        "rank_bootstrap_high": 0.758389,
        "stratified_auc": 0.802817,
        "auc_permutation_p": 0.001,
        "auc_bootstrap_low": 0.636364,
        "auc_bootstrap_high": 0.937521,
        "partial_rank_rho": 0.476633,
        "partial_permutation_p": 0.001,
        "partial_bootstrap_low": 0.150293,
        "partial_bootstrap_high": 0.680938,
    },
    "threshold_cross_grid": {
        "stratified_rank_rho": 0.532724,
        "rank_permutation_p": 0.0008,
        "rank_bootstrap_low": 0.230810,
        "rank_bootstrap_high": 0.749312,
    },
    "cluster_models": {
        "primary_unadjusted_coefficient": 0.560144,
        "primary_unadjusted_se": 0.121216,
        "primary_unadjusted_p": 0.000106,
        "primary_adjusted_coefficient": 0.788475,
        "primary_adjusted_se": 0.151293,
        "primary_adjusted_p": 0.000029,
    },
    "cluster_bootstrap": {
        "primary_rank_rho": {
            "low": 0.233637,
            "high": 0.777111,
            "positive_fraction": 0.9986,
        },
        "partial_rank_rho": {
            "low": 0.140644,
            "high": 0.694026,
            "positive_fraction": 0.9952,
        },
        "cross_grid_rank_rho": {
            "low": 0.246467,
            "high": 0.737295,
            "positive_fraction": 0.9994,
        },
    },
    "cross_method_rho": {
        "stable_frame_raw": 0.097959,
        "equal_cell": 0.128436,
        "observer_balanced": 0.102154,
        "target_effort": 0.560363,
        "cross_method_median": 0.360038,
    },
    "cross_method_p": {
        "stable_frame_raw": 0.266367,
        "equal_cell": 0.205897,
        "observer_balanced": 0.271364,
        "target_effort": 0.0005,
        "cross_method_median": 0.012494,
    },
    "mechanism": {
        "core_replicates": 576_000,
        "marginal_replicates": 24_000,
        "total_replicates": 600_000,
        "effort_global_rho": 0.7002762425,
        "effort_permutation_p": 0.0001999600,
        "effort_bootstrap_low": 0.6856577551,
        "effort_bootstrap_high": 0.7169369019,
        "effort_monotonic_fraction": 0.9555555556,
        "effort_estimability_delta": 0.1764166667,
        "species_global_rho": -0.8415321118,
        "species_permutation_p": 0.0001999600,
        "species_bootstrap_low": -0.8765073885,
        "species_bootstrap_high": -0.8185767320,
        "species_monotonic_fraction": 1.0,
        "low_minus_high_corrected_error": 18.3070653728,
        "partial_gain_rho": 0.7052197191,
        "partial_gain_bootstrap_low": 0.6686370712,
        "partial_gain_bootstrap_high": 0.7216051213,
        "species_overlap_estimability_rho": 0.1632489579,
        "effort_overlap_corrected_error_rho": 0.1592837659,
        "marginal_effort_rho": 1.0,
        "marginal_species_rho": -1.0,
    },
    "random_seed": 20260711,
    "q1_package_manifest_rows": 39,
}


# ============================================================
# Helpers
# ============================================================
checks = []


def as_bool(
    value,
):
    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def close_enough(
    actual,
    expected,
    tolerance=TOLERANCE,
):
    try:
        actual_value = float(
            actual
        )

        expected_value = float(
            expected
        )

    except Exception:
        return False

    return bool(
        np.isfinite(
            actual_value
        )
        and np.isfinite(
            expected_value
        )
        and abs(
            actual_value
            - expected_value
        )
        <= tolerance
    )


def add_check(
    group,
    name,
    actual,
    expected,
    passed,
    source,
    note="",
):
    checks.append(
        {
            "group": group,
            "check": name,
            "actual": actual,
            "expected": expected,
            "passed": bool(
                passed
            ),
            "source": str(
                source
            ),
            "note": note,
        }
    )


def check_file(
    group,
    name,
    path,
):
    passed = bool(
        path.exists()
        and path.stat().st_size > 0
    )

    add_check(
        group=group,
        name=name,
        actual=(
            path.stat().st_size
            if passed
            else "missing"
        ),
        expected="existing non-empty file",
        passed=passed,
        source=path,
    )

    return passed


def check_numeric(
    group,
    name,
    actual,
    expected,
    source,
    tolerance=TOLERANCE,
    note="",
):
    add_check(
        group=group,
        name=name,
        actual=actual,
        expected=expected,
        passed=close_enough(
            actual,
            expected,
            tolerance=tolerance,
        ),
        source=source,
        note=note,
    )


def check_exact(
    group,
    name,
    actual,
    expected,
    source,
    note="",
):
    add_check(
        group=group,
        name=name,
        actual=actual,
        expected=expected,
        passed=(
            actual
            == expected
        ),
        source=source,
        note=note,
    )


def required_csv(
    path,
):
    if not check_file(
        "required_files",
        path.name,
        path,
    ):
        raise FileNotFoundError(
            path
        )

    return pd.read_csv(
        path
    )


def required_parquet_rows(
    path,
):
    if not check_file(
        "required_files",
        path.name,
        path,
    ):
        raise FileNotFoundError(
            path
        )

    return int(
        pq.ParquetFile(
            path
        ).metadata.num_rows
    )


def package_version(
    distribution_name,
):
    try:
        return importlib_metadata.version(
            distribution_name
        )

    except importlib_metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def sha256_file(
    path,
):
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024
                * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# ============================================================
# 1. Registered GBIF input audit
# ============================================================
gbif_manifest_path = (
    BASE_DIR
    / "manifests"
    / "registered_gbif_manifest.json"
)

if not check_file(
    "required_files",
    "registered_gbif_manifest",
    gbif_manifest_path,
):
    raise FileNotFoundError(
        gbif_manifest_path
    )

gbif_manifest = json.loads(
    gbif_manifest_path.read_text(
        encoding="utf-8"
    )
)

check_exact(
    "registered_gbif",
    "total_records",
    int(
        gbif_manifest[
            "rows"
        ]
    ),
    EXPECTED[
        "gbif_rows"
    ],
    gbif_manifest_path,
)

for state_name, expected_count in EXPECTED[
    "gbif_state_counts"
].items():
    check_exact(
        "registered_gbif",
        f"{state_name}_records",
        int(
            gbif_manifest[
                "state_counts"
            ][
                state_name
            ]
        ),
        expected_count,
        gbif_manifest_path,
    )

check_exact(
    "registered_gbif",
    "study_years",
    [
        int(
            value
        )
        for value in gbif_manifest[
            "years"
        ]
    ],
    EXPECTED[
        "gbif_years"
    ],
    gbif_manifest_path,
)

check_exact(
    "registered_gbif",
    "duplicate_gbif_ids",
    int(
        gbif_manifest[
            "duplicate_gbif_ids"
        ]
    ),
    0,
    gbif_manifest_path,
)

check_exact(
    "registered_gbif",
    "missing_coordinates",
    int(
        gbif_manifest[
            "missing_coordinates"
        ]
    ),
    0,
    gbif_manifest_path,
)


# ============================================================
# 2. Exact same-window matched physical plots
# ============================================================
pa_summary_path = (
    PA_COUNT_DIR
    / "PA_exact_matched_plot_count_summary.csv"
)

pa_summary = required_csv(
    pa_summary_path
)

pa_matched = int(
    pa_summary.iloc[
        0
    ][
        "matched_live_tree_physical_plots"
    ]
)

check_exact(
    "matched_physical_plots",
    "PA_matched_live_tree_physical_plots",
    pa_matched,
    EXPECTED[
        "matched_plots"
    ][
        "PA"
    ],
    pa_summary_path,
)

va_matched_path = (
    DERIVED_DIR
    / "va_confirmation_dataset"
    / "va_matched_physical_plots.parquet"
)

nc_matched_path = (
    DERIVED_DIR
    / "nc_final_confirmation_dataset"
    / "nc_matched_physical_plots.parquet"
)

va_matched = required_parquet_rows(
    va_matched_path
)

nc_matched = required_parquet_rows(
    nc_matched_path
)

check_exact(
    "matched_physical_plots",
    "VA_matched_live_tree_physical_plots",
    va_matched,
    EXPECTED[
        "matched_plots"
    ][
        "VA"
    ],
    va_matched_path,
)

check_exact(
    "matched_physical_plots",
    "NC_matched_live_tree_physical_plots",
    nc_matched,
    EXPECTED[
        "matched_plots"
    ][
        "NC"
    ],
    nc_matched_path,
)


# ============================================================
# 3. Formal species-region evaluation set
# ============================================================
formal_dir = (
    DERIVED_DIR
    / "registered_gbif_formal_parity"
)

formal_frames = []

for state in [
    "PA",
    "VA",
    "NC",
]:
    path = (
        formal_dir
        / f"{state}_formal_species_inputs.parquet"
    )

    if not check_file(
        "required_files",
        f"{state}_formal_species_inputs",
        path,
    ):
        raise FileNotFoundError(
            path
        )

    frame = pd.read_parquet(
        path
    )

    check_exact(
        "formal_evaluation",
        f"{state}_evaluation_species",
        int(
            len(
                frame
            )
        ),
        EXPECTED[
            "formal_species"
        ][
            state
        ],
        path,
    )

    frame = frame.copy()

    frame[
        "audit_state"
    ] = state

    formal_frames.append(
        frame
    )

formal_all = pd.concat(
    formal_frames,
    ignore_index=True,
)

species_code_column = next(
    (
        column
        for column in [
            "fia_species_code",
            "species_code",
            "SPCD",
        ]
        if column in formal_all.columns
    ),
    None,
)

if species_code_column is None:
    raise KeyError(
        "No FIA species-code column was found "
        "in the formal species-input tables."
    )

check_exact(
    "formal_evaluation",
    "species_region_rows",
    int(
        len(
            formal_all
        )
    ),
    EXPECTED[
        "species_region_rows"
    ],
    formal_dir,
)

check_exact(
    "formal_evaluation",
    "unique_fia_species_clusters",
    int(
        pd.to_numeric(
            formal_all[
                species_code_column
            ],
            errors="coerce",
        ).nunique()
    ),
    EXPECTED[
        "unique_species_clusters"
    ],
    formal_dir,
)


# ============================================================
# 4. State correction context and confirmation
# ============================================================
final_confirmation_dir = (
    DERIVED_DIR
    / "final_three_region_confirmation"
)

context_path = (
    final_confirmation_dir
    / "state_correction_context.csv"
)

feature_path = (
    final_confirmation_dir
    / "state_feature_confirmation.csv"
)

meta_path = (
    final_confirmation_dir
    / "cross_region_meta_analysis.csv"
)

decision_path = (
    final_confirmation_dir
    / "final_feature_decision.csv"
)

context_df = required_csv(
    context_path
)

feature_df = required_csv(
    feature_path
)

meta_df = required_csv(
    meta_path
)

decision_df = required_csv(
    decision_path
)

for state, metrics in EXPECTED[
    "state_context"
].items():
    row = context_df[
        context_df[
            "state_code"
        ].eq(
            state
        )
    ].iloc[
        0
    ]

    for metric, expected_value in metrics.items():
        check_numeric(
            "state_correction_context",
            f"{state}_{metric}",
            row[
                metric
            ],
            expected_value,
            context_path,
        )

cell_rows = feature_df[
    feature_df[
        "feature"
    ].eq(
        "cell_jaccard"
    )
].set_index(
    "state_code"
)

check_numeric(
    "state_feature_confirmation",
    "PA_cell_jaccard_rho",
    cell_rows.loc[
        "PA",
        "primary_rho",
    ],
    EXPECTED[
        "state_feature"
    ][
        "PA_cell_jaccard_rho"
    ],
    feature_path,
)

check_numeric(
    "state_feature_confirmation",
    "VA_cell_jaccard_rho",
    cell_rows.loc[
        "VA",
        "primary_rho",
    ],
    EXPECTED[
        "state_feature"
    ][
        "VA_cell_jaccard_rho"
    ],
    feature_path,
)

check_numeric(
    "state_feature_confirmation",
    "NC_cell_jaccard_rho",
    cell_rows.loc[
        "NC",
        "primary_rho",
    ],
    EXPECTED[
        "state_feature"
    ][
        "NC_cell_jaccard_rho"
    ],
    feature_path,
)

check_numeric(
    "state_feature_confirmation",
    "NC_cell_jaccard_Holm_p",
    cell_rows.loc[
        "NC",
        "holm_adjusted_p",
    ],
    EXPECTED[
        "state_feature"
    ][
        "NC_cell_jaccard_holm_p"
    ],
    feature_path,
)

check_exact(
    "state_feature_confirmation",
    "VA_independent_confirmation",
    as_bool(
        cell_rows.loc[
            "VA",
            "regional_confirmed",
        ]
    ),
    False,
    feature_path,
    note=(
        "Virginia retained the predicted direction but did not "
        "meet the prespecified state-level confirmation criterion."
    ),
)

check_exact(
    "state_feature_confirmation",
    "NC_independent_confirmation",
    as_bool(
        cell_rows.loc[
            "NC",
            "regional_confirmed",
        ]
    ),
    True,
    feature_path,
)

confirmation_meta = meta_df[
    meta_df[
        "feature"
    ].eq(
        "cell_jaccard"
    )
    & meta_df[
        "states"
    ].eq(
        "VA,NC"
    )
].iloc[
    0
]

check_numeric(
    "cross_region_meta",
    "VA_NC_meta_rho",
    confirmation_meta[
        "meta_rho"
    ],
    EXPECTED[
        "confirmation_meta_rho"
    ],
    meta_path,
)

check_numeric(
    "cross_region_meta",
    "VA_NC_one_sided_p",
    confirmation_meta[
        "one_sided_p"
    ],
    EXPECTED[
        "confirmation_meta_p"
    ],
    meta_path,
)

decision_cell = decision_df[
    decision_df[
        "feature"
    ].eq(
        "cell_jaccard"
    )
].iloc[
    0
]

decision_observer = decision_df[
    decision_df[
        "feature"
    ].eq(
        "abs_log_observer_growth"
    )
].iloc[
    0
]

check_exact(
    "final_feature_decision",
    "cell_jaccard_replicated_gate",
    as_bool(
        decision_cell[
            "replicated_gate"
        ]
    ),
    True,
    decision_path,
)

check_exact(
    "final_feature_decision",
    "observer_growth_replicated_gate",
    as_bool(
        decision_observer[
            "replicated_gate"
        ]
    ),
    False,
    decision_path,
)


# ============================================================
# 5. Threshold-free empirical inference
# ============================================================
threshold_dir = (
    DERIVED_DIR
    / "cell_jaccard_threshold_free_validation"
)

threshold_path = (
    threshold_dir
    / "stratified_threshold_free_inference.csv"
)

threshold_df = required_csv(
    threshold_path
)

primary_row = threshold_df[
    threshold_df[
        "outcome"
    ].eq(
        "gain_grid_0p5"
    )
].iloc[
    0
]

cross_grid_row = threshold_df[
    threshold_df[
        "outcome"
    ].eq(
        "grid_median_gain_km_decade"
    )
].iloc[
    0
]

for metric, expected_value in EXPECTED[
    "threshold_primary"
].items():
    check_numeric(
        "threshold_free_primary",
        metric,
        primary_row[
            metric
        ],
        expected_value,
        threshold_path,
    )

for metric, expected_value in EXPECTED[
    "threshold_cross_grid"
].items():
    check_numeric(
        "threshold_free_cross_grid",
        metric,
        cross_grid_row[
            metric
        ],
        expected_value,
        threshold_path,
    )


# ============================================================
# 6. FIA-species-cluster robustness
# ============================================================
cluster_dir = (
    DERIVED_DIR
    / "cell_jaccard_species_cluster_robustness"
)

model_path = (
    cluster_dir
    / "cluster_robust_models.csv"
)

bootstrap_summary_path = (
    cluster_dir
    / "species_cluster_bootstrap_summary.csv"
)

model_df = required_csv(
    model_path
)

bootstrap_summary_df = required_csv(
    bootstrap_summary_path
)

primary_models = model_df[
    model_df[
        "outcome"
    ].eq(
        "gain_grid_0p5"
    )
].copy()

primary_models[
    "adjusted_bool"
] = primary_models[
    "adjusted"
].map(
    as_bool
)

unadjusted_model = primary_models[
    ~primary_models[
        "adjusted_bool"
    ]
].iloc[
    0
]

adjusted_model = primary_models[
    primary_models[
        "adjusted_bool"
    ]
].iloc[
    0
]

cluster_model_checks = [
    (
        "primary_unadjusted_coefficient",
        unadjusted_model[
            "coefficient"
        ],
    ),
    (
        "primary_unadjusted_se",
        unadjusted_model[
            "cluster_robust_se"
        ],
    ),
    (
        "primary_unadjusted_p",
        unadjusted_model[
            "one_sided_p"
        ],
    ),
    (
        "primary_adjusted_coefficient",
        adjusted_model[
            "coefficient"
        ],
    ),
    (
        "primary_adjusted_se",
        adjusted_model[
            "cluster_robust_se"
        ],
    ),
    (
        "primary_adjusted_p",
        adjusted_model[
            "one_sided_p"
        ],
    ),
]

for name, actual_value in cluster_model_checks:
    check_numeric(
        "species_cluster_models",
        name,
        actual_value,
        EXPECTED[
            "cluster_models"
        ][
            name
        ],
        model_path,
    )

bootstrap_lookup = bootstrap_summary_df.set_index(
    "metric"
)

for metric, expected_values in EXPECTED[
    "cluster_bootstrap"
].items():
    for column, expected_value in expected_values.items():
        check_numeric(
            "species_cluster_bootstrap",
            f"{metric}_{column}",
            bootstrap_lookup.loc[
                metric,
                column,
            ],
            expected_value,
            bootstrap_summary_path,
        )


# ============================================================
# 7. Cross-method boundary
# ============================================================
cross_method_dir = (
    DERIVED_DIR
    / "cross_method_correction_benchmark_trend"
)

method_path = (
    cross_method_dir
    / "method_trend_inference.csv"
)

method_df = required_csv(
    method_path
).set_index(
    "method"
)

for method, expected_rho in EXPECTED[
    "cross_method_rho"
].items():
    check_numeric(
        "cross_method_boundary",
        f"{method}_rho",
        method_df.loc[
            method,
            "stratified_rho",
        ],
        expected_rho,
        method_path,
    )

    check_numeric(
        "cross_method_boundary",
        f"{method}_permutation_p",
        method_df.loc[
            method,
            "rho_permutation_p",
        ],
        EXPECTED[
            "cross_method_p"
        ][
            method
        ],
        method_path,
    )

main_methods = [
    "stable_frame_raw",
    "equal_cell",
    "observer_balanced",
    "target_effort",
]

n_methods_auc_ge_065 = int(
    (
        method_df.loc[
            main_methods,
            "stratified_auc",
        ]
        >= 0.65
    ).sum()
)

generalization_gate_passed = bool(
    n_methods_auc_ge_065
    >= 2
)

check_exact(
    "cross_method_boundary",
    "generalization_gate_passed",
    generalization_gate_passed,
    False,
    method_path,
    note=(
        "Cell Jaccard is retained as a target-effort-specific "
        "diagnostic, not a universal correction rule."
    ),
)


# ============================================================
# 8. Frozen 600,000-replicate mechanism confirmation
# ============================================================
mechanism_dir = (
    DERIVED_DIR
    / "full_orthogonal_support_phase_confirmation"
)

mechanism_summary_path = (
    mechanism_dir
    / "full_phase_confirmation_summary.csv"
)

core_path = (
    mechanism_dir
    / "full_core_phase_replicates.parquet"
)

marginal_path = (
    mechanism_dir
    / "full_marginal_phase_replicates.parquet"
)

negative_control_path = (
    mechanism_dir
    / "negative_control_summary.csv"
)

plan_path = (
    DERIVED_DIR
    / "full_orthogonal_phase_diagram_plan"
    / "full_orthogonal_phase_confirmation_plan.json"
)

mechanism_summary = required_csv(
    mechanism_summary_path
).iloc[
    0
]

core_rows = required_parquet_rows(
    core_path
)

marginal_rows = required_parquet_rows(
    marginal_path
)

negative_control_df = required_csv(
    negative_control_path
)

if not check_file(
    "required_files",
    "full_orthogonal_phase_confirmation_plan",
    plan_path,
):
    raise FileNotFoundError(
        plan_path
    )

plan = json.loads(
    plan_path.read_text(
        encoding="utf-8"
    )
)

check_exact(
    "mechanism_design",
    "core_replicates",
    core_rows,
    EXPECTED[
        "mechanism"
    ][
        "core_replicates"
    ],
    core_path,
)

check_exact(
    "mechanism_design",
    "marginal_replicates",
    marginal_rows,
    EXPECTED[
        "mechanism"
    ][
        "marginal_replicates"
    ],
    marginal_path,
)

check_exact(
    "mechanism_design",
    "total_replicates",
    core_rows
    + marginal_rows,
    EXPECTED[
        "mechanism"
    ][
        "total_replicates"
    ],
    mechanism_dir,
)

check_exact(
    "mechanism_design",
    "random_seed",
    int(
        plan[
            "random_seed"
        ]
    ),
    EXPECTED[
        "random_seed"
    ],
    plan_path,
)

mechanism_numeric_fields = [
    "effort_global_rho",
    "effort_permutation_p",
    "effort_bootstrap_low",
    "effort_bootstrap_high",
    "effort_monotonic_fraction",
    "effort_estimability_delta",
    "species_global_rho",
    "species_permutation_p",
    "species_bootstrap_low",
    "species_bootstrap_high",
    "species_monotonic_fraction",
    "low_minus_high_corrected_error",
    "partial_gain_rho",
    "partial_gain_bootstrap_low",
    "partial_gain_bootstrap_high",
    "species_overlap_estimability_rho",
    "effort_overlap_corrected_error_rho",
    "marginal_effort_rho",
    "marginal_species_rho",
]

for field in mechanism_numeric_fields:
    check_numeric(
        "mechanism_confirmation",
        field,
        mechanism_summary[
            field
        ],
        EXPECTED[
            "mechanism"
        ][
            field
        ],
        mechanism_summary_path,
        tolerance=1e-8,
    )

mechanism_boolean_fields = [
    "calibration_pass",
    "axis_specificity_pass",
    "marginal_curves_pass",
    "negative_controls_pass",
    "effort_axis_pass",
    "species_axis_pass",
    "partial_gain_pass",
    "passed",
]

for field in mechanism_boolean_fields:
    check_exact(
        "mechanism_confirmation",
        field,
        as_bool(
            mechanism_summary[
                field
            ]
        ),
        True,
        mechanism_summary_path,
    )

negative_controls_all_passed = bool(
    negative_control_df[
        "passed"
    ].map(
        as_bool
    ).all()
)

check_exact(
    "mechanism_confirmation",
    "all_negative_control_rows_passed",
    negative_controls_all_passed,
    True,
    negative_control_path,
)


# ============================================================
# 9. Final Q1 mechanism package
# ============================================================
package_manifest_path = (
    FINAL_PACKAGE
    / "PACKAGE_MANIFEST.csv"
)

package_manifest_df = required_csv(
    package_manifest_path
)

check_exact(
    "final_package",
    "original_package_manifest_rows",
    int(
        len(
            package_manifest_df
        )
    ),
    EXPECTED[
        "q1_package_manifest_rows"
    ],
    package_manifest_path,
)

check_file(
    "final_package",
    "mechanism_insertion_map",
    FINAL_PACKAGE
    / "MANUSCRIPT_INSERTION_MAP.txt",
)

check_file(
    "final_package",
    "q1_mechanism_readme",
    FINAL_PACKAGE
    / "README_Q1_MECHANISM_UPDATE.md",
)


# ============================================================
# 10. Build audit table before environment capture
# ============================================================
checks_df = pd.DataFrame(
    checks
)

all_scientific_checks_passed = bool(
    checks_df[
        "passed"
    ].all()
)

locked_audit_path = (
    AUDIT_DIR
    / "locked_result_audit.csv"
)

checks_df.to_csv(
    locked_audit_path,
    index=False,
)

print(
    "\nLOCKED RESULT AUDIT"
)

display(
    checks_df
)

failed_checks_df = checks_df[
    ~checks_df[
        "passed"
    ]
].copy()

if not failed_checks_df.empty:
    print(
        "\nFAILED CHECKS"
    )

    display(
        failed_checks_df
    )


# ============================================================
# 11. Capture successful reproduction environment
# ============================================================
core_distributions = [
    (
        "numpy",
        "numpy",
    ),
    (
        "pandas",
        "pandas",
    ),
    (
        "scipy",
        "scipy",
    ),
    (
        "statsmodels",
        "statsmodels",
    ),
    (
        "scikit-learn",
        "sklearn",
    ),
    (
        "matplotlib",
        "matplotlib",
    ),
    (
        "pyarrow",
        "pyarrow",
    ),
    (
        "requests",
        "requests",
    ),
    (
        "tqdm",
        "tqdm",
    ),
    (
        "Pillow",
        "PIL",
    ),
    (
        "ipython",
        "IPython",
    ),
]

version_rows = []

for distribution_name, import_name in tqdm(
    core_distributions,
    desc="Capturing core package versions",
    unit="package",
):
    version_rows.append(
        {
            "distribution": distribution_name,
            "import_name": import_name,
            "version": package_version(
                distribution_name
            ),
        }
    )

versions_df = pd.DataFrame(
    version_rows
)

software_versions_path = (
    AUDIT_DIR
    / "software_versions_core.csv"
)

versions_df.to_csv(
    software_versions_path,
    index=False,
)

requirements_core_path = (
    AUDIT_DIR
    / "requirements_core.txt"
)

requirements_core_path.write_text(
    "\n".join(
        f"{row['distribution']}=={row['version']}"
        for row in version_rows
    )
    + "\n",
    encoding="utf-8",
)

freeze_result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "freeze",
    ],
    capture_output=True,
    text=True,
    check=False,
)

requirements_full_path = (
    AUDIT_DIR
    / "requirements_full.txt"
)

requirements_full_path.write_text(
    freeze_result.stdout,
    encoding="utf-8",
)

try:
    import psutil

    memory = psutil.virtual_memory()

    memory_total_bytes = int(
        memory.total
    )

except Exception:
    memory_total_bytes = None

environment_info = {
    "captured_utc": RUN_UTC,
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "cpu_count_logical": os.cpu_count(),
    "memory_total_bytes": memory_total_bytes,
    "kaggle_environment": True,
    "gpu_used_for_analysis": False,
    "random_seed_mechanism": EXPECTED[
        "random_seed"
    ],
    "registered_gbif_doi": "10.15468/dl.sm6ygu",
    "registered_gbif_download_key": (
        "0032735-260623161305970"
    ),
    "full_mechanism_replicates": (
        core_rows
        + marginal_rows
    ),
    "all_scientific_checks_passed": (
        all_scientific_checks_passed
    ),
    "environment_scope": (
        "This is the Kaggle environment that successfully reproduced "
        "the frozen full confirmation and final audit."
    ),
    "historical_snapshot_of_every_exploratory_run": False,
    "old_archive_claimed_to_contain_these_files": False,
}

environment_info_path = (
    AUDIT_DIR
    / "environment_info.json"
)

environment_info_path.write_text(
    json.dumps(
        environment_info,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# 12. Reproduction summary
# ============================================================
reproduction_summary = {
    "run_utc": RUN_UTC,
    "status": (
        "passed"
        if all_scientific_checks_passed
        else "needs_review"
    ),
    "n_checks": int(
        len(
            checks_df
        )
    ),
    "n_passed": int(
        checks_df[
            "passed"
        ].sum()
    ),
    "n_failed": int(
        (
            ~checks_df[
                "passed"
            ]
        ).sum()
    ),
    "failed_checks": failed_checks_df[
        "check"
    ].tolist(),
    "gbif_records": EXPECTED[
        "gbif_rows"
    ],
    "matched_physical_plots": {
        "PA": pa_matched,
        "VA": va_matched,
        "NC": nc_matched,
    },
    "species_region_rows": int(
        len(
            formal_all
        )
    ),
    "unique_fia_species_clusters": int(
        pd.to_numeric(
            formal_all[
                species_code_column
            ],
            errors="coerce",
        ).nunique()
    ),
    "threshold_free_primary_rho": float(
        primary_row[
            "stratified_rank_rho"
        ]
    ),
    "threshold_free_auc": float(
        primary_row[
            "stratified_auc"
        ]
    ),
    "mechanism_effort_rho": float(
        mechanism_summary[
            "effort_global_rho"
        ]
    ),
    "mechanism_species_rho": float(
        mechanism_summary[
            "species_global_rho"
        ]
    ),
    "mechanism_low_minus_high_error": float(
        mechanism_summary[
            "low_minus_high_corrected_error"
        ]
    ),
    "mechanism_total_replicates": int(
        core_rows
        + marginal_rows
    ),
    "environment_files": {
        "software_versions_core": str(
            software_versions_path
        ),
        "requirements_core": str(
            requirements_core_path
        ),
        "requirements_full": str(
            requirements_full_path
        ),
        "environment_info": str(
            environment_info_path
        ),
    },
}

summary_json_path = (
    AUDIT_DIR
    / "final_reproduction_summary.json"
)

summary_json_path.write_text(
    json.dumps(
        reproduction_summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# 13. Copy small reproducibility materials into final package
# ============================================================
files_to_copy = [
    (
        software_versions_path,
        ENV_PACKAGE_DIR
        / software_versions_path.name,
    ),
    (
        requirements_core_path,
        ENV_PACKAGE_DIR
        / requirements_core_path.name,
    ),
    (
        requirements_full_path,
        ENV_PACKAGE_DIR
        / requirements_full_path.name,
    ),
    (
        environment_info_path,
        ENV_PACKAGE_DIR
        / environment_info_path.name,
    ),
    (
        pa_summary_path,
        PA_PACKAGE_DIR
        / pa_summary_path.name,
    ),
    (
        PA_COUNT_DIR
        / "CELL38_PA_EXACT_COUNT_COMPLETE.json",
        PA_PACKAGE_DIR
        / "CELL38_PA_EXACT_COUNT_COMPLETE.json",
    ),
    (
        locked_audit_path,
        AUDIT_PACKAGE_DIR
        / locked_audit_path.name,
    ),
    (
        summary_json_path,
        AUDIT_PACKAGE_DIR
        / summary_json_path.name,
    ),
    (
        gbif_manifest_path,
        AUDIT_PACKAGE_DIR
        / gbif_manifest_path.name,
    ),
    (
        plan_path,
        AUDIT_PACKAGE_DIR
        / plan_path.name,
    ),
]

copied_files = []

for source, destination in tqdm(
    files_to_copy,
    desc="Copying reproducibility materials",
    unit="file",
):
    if not source.exists():
        raise FileNotFoundError(
            source
        )

    shutil.copy2(
        source,
        destination,
    )

    copied_files.append(
        destination
    )

repro_readme_path = (
    REPRO_DIR
    / "README_REPRODUCIBILITY.md"
)

repro_readme_path.write_text(
    f"""# Reproducibility materials

Generated: {RUN_UTC}

## Scope

These files document the Kaggle environment and saved results that
successfully reproduced the frozen 600,000-replicate confirmation.

They are **not** claimed to be an absolute historical snapshot of every
earlier exploratory run.

## Included

- Exact core package versions
- Full `pip freeze`
- Environment metadata
- Final locked-result audit
- Registered GBIF manifest
- Frozen mechanism plan
- Exact PA same-window matched-plot count reconstruction

## Locked descriptive counts

- Pennsylvania: {pa_matched:,}
- Virginia: {va_matched:,}
- North Carolina: {nc_matched:,}

## Scientific boundary

The audit verifies saved results. It performs no new feature search,
threshold selection, scientific inference, or modification of locked
statistics.
""",
    encoding="utf-8",
)

copied_files.append(
    repro_readme_path
)

repro_manifest_rows = []

for path in tqdm(
    copied_files,
    desc="Hashing copied reproducibility files",
    unit="file",
):
    repro_manifest_rows.append(
        {
            "relative_path": str(
                path.relative_to(
                    FINAL_PACKAGE
                )
            ),
            "size_bytes": int(
                path.stat().st_size
            ),
            "sha256": sha256_file(
                path
            ),
        }
    )

repro_manifest_df = pd.DataFrame(
    repro_manifest_rows
)

repro_manifest_path = (
    REPRO_DIR
    / "REPRODUCIBILITY_MANIFEST.csv"
)

repro_manifest_df.to_csv(
    repro_manifest_path,
    index=False,
)


# ============================================================
# 14. README and completion marker
# ============================================================
project_readme_path = (
    BASE_DIR
    / "README.md"
)

project_readme = (
    project_readme_path.read_text(
        encoding="utf-8"
    )
    if project_readme_path.exists()
    else (
        "# Q1 Full Reproduction — "
        "Temporal Observation Drift\n"
    )
)

project_readme_path.write_text(
    project_readme
    + f"""

## Final full-reproduction audit — {RUN_UTC}

- Total checks: {len(checks_df)}
- Passed checks: {int(checks_df['passed'].sum())}
- Failed checks: {int((~checks_df['passed']).sum())}
- All scientific checks passed: {all_scientific_checks_passed}
- PA matched plots: {pa_matched:,}
- VA matched plots: {va_matched:,}
- NC matched plots: {nc_matched:,}
- Species-region observations: {len(formal_all):,}
- Unique FIA species clusters:
  {pd.to_numeric(formal_all[species_code_column], errors='coerce').nunique():,}
- Mechanism replicates: {core_rows + marginal_rows:,}
- Environment captured from the successful Kaggle reproduction: True
- Historical snapshot of all exploratory runs: False
- GPU used: False
- New scientific inference performed by audit: False
""",
    encoding="utf-8",
)

completion_marker_path = (
    AUDIT_DIR
    / "FINAL_REPRODUCTION_AUDIT_COMPLETE.json"
)

completion_marker_path.write_text(
    json.dumps(
        {
            "status": (
                "complete"
                if all_scientific_checks_passed
                else "needs_review"
            ),
            "run_utc": RUN_UTC,
            "n_checks": int(
                len(
                    checks_df
                )
            ),
            "n_passed": int(
                checks_df[
                    "passed"
                ].sum()
            ),
            "n_failed": int(
                (
                    ~checks_df[
                        "passed"
                    ]
                ).sum()
            ),
            "all_scientific_checks_passed": (
                all_scientific_checks_passed
            ),
            "locked_result_audit": str(
                locked_audit_path
            ),
            "summary_json": str(
                summary_json_path
            ),
            "environment_info": str(
                environment_info_path
            ),
            "reproducibility_manifest": str(
                repro_manifest_path
            ),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# 15. Final status
# ============================================================
status = (
    "FINAL_FULL_REPRODUCTION_AUDIT_PASSED"
    if all_scientific_checks_passed
    else "FINAL_FULL_REPRODUCTION_AUDIT_NEEDS_REVIEW"
)

print(
    "\nCORE SOFTWARE VERSIONS"
)

display(
    versions_df
)

print(
    "\n========== RUN SUMMARY =========="
)
print(
    f"CELL_STATUS: {status}"
)
print(
    f"AUDIT_DIR: {AUDIT_DIR}"
)
print(
    f"N_CHECKS: {len(checks_df)}"
)
print(
    f"N_PASSED: {int(checks_df['passed'].sum())}"
)
print(
    f"N_FAILED: {int((~checks_df['passed']).sum())}"
)
print(
    "FAILED_CHECKS: "
    + json.dumps(
        failed_checks_df[
            "check"
        ].tolist(),
        ensure_ascii=False,
    )
)
print(
    f"ALL_SCIENTIFIC_CHECKS_PASSED: "
    f"{all_scientific_checks_passed}"
)
print(
    f"GBIF_RECORDS: {EXPECTED['gbif_rows']}"
)
print(
    f"PA_MATCHED_PLOTS: {pa_matched}"
)
print(
    f"VA_MATCHED_PLOTS: {va_matched}"
)
print(
    f"NC_MATCHED_PLOTS: {nc_matched}"
)
print(
    f"SPECIES_REGION_ROWS: {len(formal_all)}"
)
print(
    "UNIQUE_FIA_SPECIES_CLUSTERS: "
    + str(
        pd.to_numeric(
            formal_all[
                species_code_column
            ],
            errors="coerce",
        ).nunique()
    )
)
print(
    f"MECHANISM_TOTAL_REPLICATES: "
    f"{core_rows + marginal_rows}"
)
print(
    "MECHANISM_EFFORT_RHO: "
    + str(
        mechanism_summary[
            "effort_global_rho"
        ]
    )
)
print(
    "MECHANISM_SPECIES_RHO: "
    + str(
        mechanism_summary[
            "species_global_rho"
        ]
    )
)
print(
    "LOW_MINUS_HIGH_CORRECTED_ERROR: "
    + str(
        mechanism_summary[
            "low_minus_high_corrected_error"
        ]
    )
)
print(
    f"LOCKED_RESULT_AUDIT: {locked_audit_path}"
)
print(
    f"SOFTWARE_VERSIONS_CORE: {software_versions_path}"
)
print(
    f"REQUIREMENTS_CORE: {requirements_core_path}"
)
print(
    f"REQUIREMENTS_FULL: {requirements_full_path}"
)
print(
    f"ENVIRONMENT_INFO: {environment_info_path}"
)
print(
    f"FINAL_REPRODUCTION_SUMMARY: {summary_json_path}"
)
print(
    f"REPRODUCIBILITY_DIR: {REPRO_DIR}"
)
print(
    f"REPRODUCIBILITY_MANIFEST: {repro_manifest_path}"
)
print(
    f"COMPLETION_MARKER: {completion_marker_path}"
)
print(
    "README_UPDATED: True"
)
print(
    "NEXT_STEP: If the audit passes, build the final public "
    "reproducibility archive and update repository/DOI placeholders."
)
print(
    "============================================"
)
