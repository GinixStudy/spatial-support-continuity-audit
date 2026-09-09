
# Cell 22D — Repair four-level monotonicity diagnostics for Cell 22C
#
# Root cause
# ----------
# Each fixed stratum in Cell 22C contains exactly four manipulated overlap
# levels. The original safe_spearman() required at least five observations,
# so every within-stratum correlation returned NaN and both monotonicity
# fractions were incorrectly recorded as zero.
#
# This cell:
# 1. Reuses the existing 57,600 simulation replicates and scenario summary.
# 2. Recomputes four-level within-stratum monotonicity with a minimum of
#    three observations.
# 3. Checks endpoint contrasts as an additional ordered-trend diagnostic.
# 4. Runs 1,000 species-region-cluster bootstrap replicates for the two
#    global axis associations.
# 5. Reapplies the original Cell 22C trend-stage success criteria.
#
# No new simulation, threshold, feature or parameter is introduced.

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from IPython.display import display, Image

# ============================================================
# Configuration — unchanged scientific criteria from Cell 22C
# ============================================================
BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")

INPUT_DIR = (
    BASE_DIR
    / "derived"
    / "orthogonal_support_overlap_phase_diagram_trend"
)

SCENARIO_PATH = (
    INPUT_DIR
    / "orthogonal_phase_scenario_summary.csv"
)

DIAGNOSTIC_PATH = (
    INPUT_DIR
    / "orthogonal_phase_trend_diagnostic.csv"
)

OUT_DIR = (
    BASE_DIR
    / "derived"
    / "orthogonal_support_overlap_monotonicity_repair"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

N_CLUSTER_BOOTSTRAP = 1_000
RANDOM_SEED = 20260711

MIN_EFFORT_ESTIMABILITY_RHO = 0.50
MAX_EFFORT_ESTIMABILITY_P = 0.01

MAX_SPECIES_OVERLAP_ERROR_RHO = -0.30
MAX_SPECIES_OVERLAP_ERROR_P = 0.01

MIN_EFFORT_MONOTONIC_FRACTION = 0.65
MIN_SPECIES_MONOTONIC_FRACTION = 0.60

MIN_HIGH_LOW_ESTIMABILITY_DELTA = 0.12
MIN_LOW_HIGH_CORRECTED_ERROR_DELTA = 1.0

RUN_UTC = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("INPUT_DIR:", INPUT_DIR)
print("OUT_DIR:", OUT_DIR)
print("N_CLUSTER_BOOTSTRAP:", N_CLUSTER_BOOTSTRAP)
print("GPU: not used")


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


def to_bool_scalar(value):
    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(value)

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def safe_spearman_four_level(
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
        len(frame) < 3
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


def endpoint_difference(
    frame,
    x_column,
    y_column,
    direction,
):
    work = frame[
        [
            x_column,
            y_column,
        ]
    ].dropna().sort_values(
        x_column
    )

    if (
        len(work) < 2
        or work[
            x_column
        ].nunique()
        < 2
    ):
        return np.nan

    low_value = float(
        work.iloc[
            0
        ][
            y_column
        ]
    )

    high_value = float(
        work.iloc[
            -1
        ][
            y_column
        ]
    )

    if direction == "increasing":
        return (
            high_value
            - low_value
        )

    if direction == "decreasing":
        return (
            low_value
            - high_value
        )

    raise ValueError(
        f"Unknown direction: {direction}"
    )


def cluster_bootstrap_sample(
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
        ] = bootstrap_id

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
            "expected_direction_fraction": np.nan,
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
    }


def print_failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## Four-level monotonicity repair failure — {RUN_UTC}
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
        SCENARIO_PATH,
        DIAGNOSTIC_PATH,
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
            "Missing Cell 22C outputs: "
            + " | ".join(
                missing_paths
            )
        )

    scenario_df = pd.read_csv(
        SCENARIO_PATH
    )

    original_diagnostic = pd.read_csv(
        DIAGNOSTIC_PATH
    ).iloc[
        0
    ]

    scenario_df[
        "species_region_cluster"
    ] = (
        scenario_df[
            "state_code"
        ].astype(str)
        + "_"
        + scenario_df[
            "fia_species_code"
        ].astype(str)
    )

    # --------------------------------------------------------
    # Repair effort-overlap monotonicity
    # --------------------------------------------------------
    effort_group_columns = [
        "state_code",
        "fia_species_code",
        "species_overlap_requested",
        "drift_requested_km_decade",
        "sample_size",
        "width_mode",
    ]

    effort_rows = []

    effort_groups = list(
        scenario_df.groupby(
            effort_group_columns,
            dropna=False,
        )
    )

    for keys, group in tqdm(
        effort_groups,
        desc="Repairing effort-overlap monotonicity",
        unit="stratum",
    ):
        (
            rho,
            p_value,
        ) = safe_spearman_four_level(
            group[
                "effort_overlap_requested"
            ],
            group[
                "corrected_estimable_fraction"
            ],
        )

        endpoint_delta = endpoint_difference(
            group,
            "effort_overlap_requested",
            "corrected_estimable_fraction",
            "increasing",
        )

        effort_rows.append(
            {
                "state_code": keys[
                    0
                ],
                "fia_species_code": keys[
                    1
                ],
                "species_overlap_requested": keys[
                    2
                ],
                "drift_requested_km_decade": keys[
                    3
                ],
                "sample_size": keys[
                    4
                ],
                "width_mode": keys[
                    5
                ],
                "n_overlap_levels": int(
                    group[
                        "effort_overlap_requested"
                    ].nunique()
                ),
                "rho": rho,
                "p_value": p_value,
                "endpoint_delta": endpoint_delta,
                "expected_direction": bool(
                    np.isfinite(
                        rho
                    )
                    and rho > 0
                    and np.isfinite(
                        endpoint_delta
                    )
                    and endpoint_delta > 0
                ),
                "strong_expected_direction": bool(
                    np.isfinite(
                        rho
                    )
                    and rho >= 0.50
                    and np.isfinite(
                        endpoint_delta
                    )
                    and endpoint_delta > 0
                ),
            }
        )

    effort_monotonic_df = pd.DataFrame(
        effort_rows
    )

    effort_valid = effort_monotonic_df[
        effort_monotonic_df[
            "rho"
        ].notna()
    ].copy()

    effort_positive_fraction = float(
        effort_valid[
            "expected_direction"
        ].mean()
    )

    effort_strong_fraction = float(
        effort_valid[
            "strong_expected_direction"
        ].mean()
    )

    # --------------------------------------------------------
    # Repair species-overlap monotonicity
    # --------------------------------------------------------
    species_group_columns = [
        "state_code",
        "fia_species_code",
        "effort_overlap_requested",
        "drift_requested_km_decade",
        "sample_size",
        "width_mode",
    ]

    species_rows = []

    species_groups = list(
        scenario_df.groupby(
            species_group_columns,
            dropna=False,
        )
    )

    for keys, group in tqdm(
        species_groups,
        desc="Repairing species-overlap monotonicity",
        unit="stratum",
    ):
        (
            rho,
            p_value,
        ) = safe_spearman_four_level(
            group[
                "species_overlap_requested"
            ],
            group[
                "median_corrected_error"
            ],
        )

        endpoint_delta = endpoint_difference(
            group,
            "species_overlap_requested",
            "median_corrected_error",
            "decreasing",
        )

        species_rows.append(
            {
                "state_code": keys[
                    0
                ],
                "fia_species_code": keys[
                    1
                ],
                "effort_overlap_requested": keys[
                    2
                ],
                "drift_requested_km_decade": keys[
                    3
                ],
                "sample_size": keys[
                    4
                ],
                "width_mode": keys[
                    5
                ],
                "n_overlap_levels": int(
                    group[
                        "species_overlap_requested"
                    ].nunique()
                ),
                "rho": rho,
                "p_value": p_value,
                "endpoint_delta": endpoint_delta,
                "expected_direction": bool(
                    np.isfinite(
                        rho
                    )
                    and rho < 0
                    and np.isfinite(
                        endpoint_delta
                    )
                    and endpoint_delta > 0
                ),
                "strong_expected_direction": bool(
                    np.isfinite(
                        rho
                    )
                    and rho <= -0.50
                    and np.isfinite(
                        endpoint_delta
                    )
                    and endpoint_delta > 0
                ),
            }
        )

    species_monotonic_df = pd.DataFrame(
        species_rows
    )

    species_valid = species_monotonic_df[
        species_monotonic_df[
            "rho"
        ].notna()
    ].copy()

    species_positive_fraction = float(
        species_valid[
            "expected_direction"
        ].mean()
    )

    species_strong_fraction = float(
        species_valid[
            "strong_expected_direction"
        ].mean()
    )

    # --------------------------------------------------------
    # Species-region-cluster bootstrap of global axis effects
    # --------------------------------------------------------
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    effort_bootstrap = []
    species_bootstrap = []

    for _ in tqdm(
        range(
            N_CLUSTER_BOOTSTRAP
        ),
        desc="Bootstrapping global overlap-axis associations",
        unit="replicate",
    ):
        sample = cluster_bootstrap_sample(
            scenario_df,
            rng,
        )

        (
            effort_rho,
            _,
        ) = safe_spearman_four_level(
            sample[
                "effort_overlap_actual"
            ],
            sample[
                "corrected_estimable_fraction"
            ],
        )

        estimable_sample = sample.dropna(
            subset=[
                "median_corrected_error",
            ]
        )

        (
            species_rho,
            _,
        ) = safe_spearman_four_level(
            estimable_sample[
                "species_overlap_actual"
            ],
            estimable_sample[
                "median_corrected_error"
            ],
        )

        if np.isfinite(
            effort_rho
        ):
            effort_bootstrap.append(
                effort_rho
            )

        if np.isfinite(
            species_rho
        ):
            species_bootstrap.append(
                species_rho
            )

    effort_bootstrap = np.asarray(
        effort_bootstrap,
        dtype=float,
    )

    species_bootstrap = np.asarray(
        species_bootstrap,
        dtype=float,
    )

    effort_bootstrap_summary = bootstrap_summary(
        effort_bootstrap
    )

    species_bootstrap_summary = bootstrap_summary(
        species_bootstrap
    )

    effort_bootstrap_positive_fraction = float(
        np.mean(
            effort_bootstrap > 0
        )
    )

    species_bootstrap_negative_fraction = float(
        np.mean(
            species_bootstrap < 0
        )
    )

    # --------------------------------------------------------
    # Reapply original Cell 22C criteria
    # --------------------------------------------------------
    calibration_pass = to_bool_scalar(
        original_diagnostic[
            "calibration_pass"
        ]
    )

    effort_estimability_rho = float(
        original_diagnostic[
            "effort_overlap_estimability_rho"
        ]
    )

    effort_estimability_p = float(
        original_diagnostic[
            "effort_overlap_estimability_p"
        ]
    )

    species_error_rho = float(
        original_diagnostic[
            "species_overlap_corrected_error_rho"
        ]
    )

    species_error_p = float(
        original_diagnostic[
            "species_overlap_corrected_error_p"
        ]
    )

    effort_estimability_delta = float(
        original_diagnostic[
            "effort_estimability_delta"
        ]
    )

    low_minus_high_corrected_error = float(
        original_diagnostic[
            "low_minus_high_corrected_error"
        ]
    )

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
        and effort_strong_fraction
        >= MIN_EFFORT_MONOTONIC_FRACTION
        and species_strong_fraction
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
            "The redesigned trend experiment passes after correcting the "
            "four-level diagnostic bug. Before the full 200-replicate run, "
            "freeze a confirmation plan with finer overlap levels, negative "
            "controls and separate estimability/error outcomes."
        )

    else:
        status = (
            "ORTHOGONAL_SUPPORT_PHASE_MONOTONICITY_INCOMPLETE"
        )

        next_step = (
            "Do not run the full phase diagram. Inspect the repaired "
            "within-stratum tables to identify whether effort estimability "
            "or species corrected-error monotonicity remains unstable."
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    effort_monotonic_df.to_csv(
        OUT_DIR
        / "repaired_effort_monotonicity.csv",
        index=False,
    )

    species_monotonic_df.to_csv(
        OUT_DIR
        / "repaired_species_monotonicity.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "effort_overlap_estimability_rho": (
                effort_bootstrap
            )
        }
    ).to_parquet(
        OUT_DIR
        / "effort_axis_cluster_bootstrap.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "species_overlap_corrected_error_rho": (
                species_bootstrap
            )
        }
    ).to_parquet(
        OUT_DIR
        / "species_axis_cluster_bootstrap.parquet",
        index=False,
    )

    summary_df = pd.DataFrame(
        [
            {
                "calibration_pass": (
                    calibration_pass
                ),
                "effort_global_rho": (
                    effort_estimability_rho
                ),
                "effort_global_p": (
                    effort_estimability_p
                ),
                "species_global_rho": (
                    species_error_rho
                ),
                "species_global_p": (
                    species_error_p
                ),
                "effort_valid_strata": (
                    len(
                        effort_valid
                    )
                ),
                "effort_positive_fraction": (
                    effort_positive_fraction
                ),
                "effort_strong_fraction": (
                    effort_strong_fraction
                ),
                "species_valid_strata": (
                    len(
                        species_valid
                    )
                ),
                "species_positive_fraction": (
                    species_positive_fraction
                ),
                "species_strong_fraction": (
                    species_strong_fraction
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
                    effort_bootstrap_positive_fraction
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
                    species_bootstrap_negative_fraction
                ),
                "effort_estimability_delta": (
                    effort_estimability_delta
                ),
                "low_minus_high_corrected_error": (
                    low_minus_high_corrected_error
                ),
                "passed": (
                    passed
                ),
            }
        ]
    )

    summary_df.to_csv(
        OUT_DIR
        / "four_level_monotonicity_repair_summary.csv",
        index=False,
    )

    print(
        "\nFOUR-LEVEL MONOTONICITY REPAIR SUMMARY"
    )
    display(
        summary_df
    )

    print(
        "\nEFFORT-AXIS MONOTONICITY"
    )
    display(
        effort_monotonic_df
    )

    print(
        "\nSPECIES-AXIS MONOTONICITY"
    )
    display(
        species_monotonic_df
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        effort_valid[
            "rho"
        ],
        bins=20,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.axvline(
        0.5,
        linestyle=":",
    )

    plt.xlabel(
        "Within-stratum Spearman rho"
    )

    plt.ylabel(
        "Effort-overlap strata"
    )

    plt.title(
        "Effort overlap versus corrected estimability"
    )

    plt.tight_layout()

    effort_figure = (
        OUT_DIR
        / "repaired_effort_monotonicity_distribution.png"
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

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        species_valid[
            "rho"
        ],
        bins=20,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.axvline(
        -0.5,
        linestyle=":",
    )

    plt.xlabel(
        "Within-stratum Spearman rho"
    )

    plt.ylabel(
        "Species-overlap strata"
    )

    plt.title(
        "Species support overlap versus corrected error"
    )

    plt.tight_layout()

    species_figure = (
        OUT_DIR
        / "repaired_species_monotonicity_distribution.png"
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

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    append_readme(
        f"""

## Four-level monotonicity diagnostic repair — {RUN_UTC}

### Code issue
Cell 22C used a Spearman helper requiring at least five observations, while
each fixed stratum contained four manipulated overlap levels. All within-
stratum correlations were therefore returned as NaN.

### Repair
- No simulation rerun
- Minimum observations for four-level Spearman:
  3
- Ordered endpoint contrast required in the expected direction
- Species-region-cluster bootstrap replicates:
  {N_CLUSTER_BOOTSTRAP}

### Repaired results
- Effort valid strata:
  {len(effort_valid)}
- Effort expected-direction fraction:
  {effort_positive_fraction}
- Effort strong fraction:
  {effort_strong_fraction}
- Species valid strata:
  {len(species_valid)}
- Species expected-direction fraction:
  {species_positive_fraction}
- Species strong fraction:
  {species_strong_fraction}
- Effort bootstrap interval:
  [{effort_bootstrap_summary['low']},
   {effort_bootstrap_summary['high']}]
- Species bootstrap interval:
  [{species_bootstrap_summary['low']},
   {species_bootstrap_summary['high']}]

### Decision
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Boundary
The scientific criteria, simulation data and manipulated overlap levels were
not changed. This cell repairs only the invalid sample-size requirement in the
within-stratum diagnostic.
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
        "PATCH_APPLIED: Four-level Spearman now requires "
        "at least 3 observations instead of 5"
    )
    print(
        f"OUT_DIR: {OUT_DIR}"
    )
    print(
        f"CALIBRATION_PASSED: "
        f"{calibration_pass}"
    )
    print(
        f"EFFORT_GLOBAL_RHO: "
        f"{effort_estimability_rho}"
    )
    print(
        f"EFFORT_GLOBAL_P: "
        f"{effort_estimability_p}"
    )
    print(
        f"SPECIES_GLOBAL_RHO: "
        f"{species_error_rho}"
    )
    print(
        f"SPECIES_GLOBAL_P: "
        f"{species_error_p}"
    )
    print(
        f"EFFORT_VALID_STRATA: "
        f"{len(effort_valid)}"
    )
    print(
        "EFFORT_EXPECTED_DIRECTION_FRACTION: "
        f"{effort_positive_fraction}"
    )
    print(
        f"EFFORT_STRONG_FRACTION: "
        f"{effort_strong_fraction}"
    )
    print(
        f"SPECIES_VALID_STRATA: "
        f"{len(species_valid)}"
    )
    print(
        "SPECIES_EXPECTED_DIRECTION_FRACTION: "
        f"{species_positive_fraction}"
    )
    print(
        f"SPECIES_STRONG_FRACTION: "
        f"{species_strong_fraction}"
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
        f"{effort_bootstrap_positive_fraction}"
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
        f"{species_bootstrap_negative_fraction}"
    )
    print(
        f"EFFORT_ESTIMABILITY_DELTA: "
        f"{effort_estimability_delta}"
    )
    print(
        "LOW_MINUS_HIGH_CORRECTED_ERROR: "
        f"{low_minus_high_corrected_error}"
    )
    print(
        f"EFFORT_FIGURE: "
        f"{effort_figure}"
    )
    print(
        f"SPECIES_FIGURE: "
        f"{species_figure}"
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
        "CELL22D_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Save this diagnostic summary and the displayed repair summary. "
            "Do not rerun the 57,600 simulations."
        ),
    )
