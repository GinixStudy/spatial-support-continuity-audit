
# Cell 22E — Freeze the full orthogonal phase-diagram confirmation plan
#
# This cell performs NO simulation.
#
# It freezes the full confirmation design before the expensive run:
# - core factor grid;
# - targeted marginal overlap refinements;
# - 200 replicates per scenario;
# - primary and secondary outcomes;
# - negative controls;
# - cluster-bootstrap and permutation settings;
# - success/failure criteria;
# - computational boundaries.
#
# The core overlap levels are kept identical to the successful trend design.
# This avoids post-hoc redesign after seeing the trend results.
#
# Finer overlap resolution is added only through two targeted marginal curves,
# rather than expanding the entire Cartesian grid.

from pathlib import Path
from datetime import datetime, timezone
import json
import math

import pandas as pd
from IPython.display import display

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(
    "/kaggle/working/fia_temporal_observation_drift"
)

TREND_DIR = (
    BASE_DIR
    / "derived"
    / "orthogonal_support_overlap_phase_diagram_trend"
)

REPAIR_DIR = (
    BASE_DIR
    / "derived"
    / "orthogonal_support_overlap_monotonicity_repair"
)

OUT_DIR = (
    BASE_DIR
    / "derived"
    / "full_orthogonal_phase_diagram_plan"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TREND_DIAGNOSTIC_PATH = (
    TREND_DIR
    / "orthogonal_phase_trend_diagnostic.csv"
)

REPAIR_SUMMARY_PATH = (
    REPAIR_DIR
    / "four_level_monotonicity_repair_summary.csv"
)

PLAN_JSON_PATH = (
    OUT_DIR
    / "full_orthogonal_phase_confirmation_plan.json"
)

PLAN_README_PATH = (
    OUT_DIR
    / "README_FULL_PHASE_CONFIRMATION_PLAN.md"
)

RUN_UTC = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

GBIF_DOI = "10.15468/dl.sm6ygu"
RANDOM_SEED = 20260711

print("RUN_UTC:", RUN_UTC)
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")
print("SIMULATION_RUN: False")


# ============================================================
# Frozen design
# ============================================================
STATES = [
    "PA",
    "VA",
    "NC",
]

SPECIES_PER_STATE = 5
N_SPECIES_REGION_UNITS = (
    len(
        STATES
    )
    * SPECIES_PER_STATE
)

CORE_EFFORT_OVERLAPS = [
    0.20,
    0.50,
    0.80,
    1.00,
]

CORE_SPECIES_OVERLAPS = [
    0.20,
    0.50,
    0.80,
    1.00,
]

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

REPLICATES_PER_CORE_SCENARIO = 200

# Targeted marginal refinements.
# These add intermediate information without multiplying the entire grid.
MARGINAL_REFINEMENT_LEVELS = [
    0.10,
    0.35,
    0.65,
    0.90,
]

MARGINAL_FIXED_SETTINGS = {
    "species_overlap_for_effort_curve": 0.80,
    "effort_overlap_for_species_curve": 0.80,
    "drift_km_decade": 20.0,
    "sample_size": 1500,
    "width_mode": "broad",
}

REPLICATES_PER_MARGINAL_SCENARIO = 200

GRID_RESOLUTION_DEGREES = 0.50
EFFORT_SUPPORT_SIZE_CELLS = 20
SPECIES_SUPPORT_SIZE_CELLS = 12
MIN_STABLE_CELL_EFFORT = 10
MIN_STABLE_CELLS = 3

N_SPECIES_REGION_CLUSTER_BOOTSTRAP = 5_000
N_SCENARIO_LABEL_PERMUTATIONS = 5_000

# ============================================================
# Frozen outcomes
# ============================================================
PRIMARY_OUTCOMES = {
    "effort_axis": (
        "corrected_estimable"
    ),
    "species_axis": (
        "corrected_error_km_decade"
    ),
}

SECONDARY_OUTCOMES = [
    "raw_error_km_decade",
    "gain_km_decade",
    "corrected_raw_error_ratio",
    "correction_helped",
    "stable_cell_count",
]

# ============================================================
# Frozen controls
# ============================================================
NEGATIVE_CONTROLS = {
    "calibration_zero_drift": {
        "definition": (
            "Expected effort-centre drift must equal 0 "
            "when requested drift is 0."
        ),
        "success": (
            "maximum absolute calibration error <= 1e-6"
        ),
    },
    "stationary_species_centre": {
        "definition": (
            "Expected early and late species latent centres "
            "must be identical."
        ),
        "success": (
            "maximum mismatch <= 1e-6 degrees"
        ),
    },
    "permuted_species_overlap_labels": {
        "definition": (
            "Within state/species/fixed effort settings, shuffle "
            "species-overlap labels before association testing."
        ),
        "success": (
            "permutation-null association centred near 0 and "
            "observed one-sided P <= 0.01"
        ),
    },
    "axis_specificity": {
        "definition": (
            "Effort overlap is primary for estimability; species "
            "overlap is primary for corrected error."
        ),
        "success": (
            "the primary-axis absolute association exceeds the "
            "cross-axis absolute association"
        ),
    },
}

# ============================================================
# Frozen success criteria
# ============================================================
SUCCESS_CRITERIA = {
    "design_generation_fraction_min": 0.995,
    "calibration_max_error": 1e-6,

    "effort_overlap_estimability_rho_min": 0.50,
    "effort_overlap_estimability_p_max": 0.01,
    "effort_monotonic_strata_fraction_min": 0.75,
    "effort_cluster_bootstrap_positive_fraction_min": 0.99,
    "high_minus_low_effort_estimability_min": 0.12,

    "species_overlap_corrected_error_rho_max": -0.50,
    "species_overlap_corrected_error_p_max": 0.01,
    "species_monotonic_strata_fraction_min": 0.80,
    "species_cluster_bootstrap_negative_fraction_min": 0.99,
    "low_minus_high_species_corrected_error_min": 5.0,

    "partial_species_overlap_gain_rho_min": 0.30,
    "partial_species_overlap_gain_bootstrap_positive_fraction_min": 0.95,

    "marginal_effort_curve_expected_direction": True,
    "marginal_species_curve_expected_direction": True,

    "axis_specificity_required": True,
    "negative_controls_required": True,
}

# ============================================================
# Computational boundaries
# ============================================================
BOUNDARIES = [
    "Do not add states or species after this plan is frozen.",
    "Do not change the latent-distribution construction.",
    "Do not change the overlap definitions.",
    "Do not optimize a cell-Jaccard cutoff.",
    "Do not choose scenarios based on favorable results.",
    "Do not combine estimability and corrected error into one primary endpoint.",
    "Do not interpret gain alone without raw and corrected errors.",
    "Do not change success thresholds after the full run.",
]

# ============================================================
# Validate trend prerequisites
# ============================================================
required_paths = [
    TREND_DIAGNOSTIC_PATH,
    REPAIR_SUMMARY_PATH,
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
        "Missing prerequisite outputs: "
        + " | ".join(
            missing_paths
        )
    )

trend_diagnostic = pd.read_csv(
    TREND_DIAGNOSTIC_PATH
).iloc[
    0
]

repair_summary = pd.read_csv(
    REPAIR_SUMMARY_PATH
).iloc[
    0
]

trend_prerequisites = {
    "calibration_passed": bool(
        trend_diagnostic[
            "calibration_pass"
        ]
    ),
    "effort_global_rho": float(
        repair_summary[
            "effort_global_rho"
        ]
    ),
    "effort_strong_fraction": float(
        repair_summary[
            "effort_strong_fraction"
        ]
    ),
    "species_global_rho": float(
        repair_summary[
            "species_global_rho"
        ]
    ),
    "species_strong_fraction": float(
        repair_summary[
            "species_strong_fraction"
        ]
    ),
    "repair_passed": bool(
        repair_summary[
            "passed"
        ]
    ),
}

prerequisites_passed = bool(
    trend_prerequisites[
        "calibration_passed"
    ]
    and trend_prerequisites[
        "repair_passed"
    ]
    and trend_prerequisites[
        "effort_global_rho"
    ]
    >= 0.50
    and trend_prerequisites[
        "species_global_rho"
    ]
    <= -0.50
)

if not prerequisites_passed:
    raise RuntimeError(
        "Trend prerequisites did not pass; "
        "the full plan must not be frozen."
    )

# ============================================================
# Compute task scale
# ============================================================
core_scenarios_per_species_region = (
    len(
        CORE_EFFORT_OVERLAPS
    )
    * len(
        CORE_SPECIES_OVERLAPS
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
)

core_total_replicates = (
    N_SPECIES_REGION_UNITS
    * core_scenarios_per_species_region
    * REPLICATES_PER_CORE_SCENARIO
)

# One marginal effort curve and one marginal species curve.
marginal_scenarios_per_species_region = (
    2
    * len(
        MARGINAL_REFINEMENT_LEVELS
    )
)

marginal_total_replicates = (
    N_SPECIES_REGION_UNITS
    * marginal_scenarios_per_species_region
    * REPLICATES_PER_MARGINAL_SCENARIO
)

total_replicates = (
    core_total_replicates
    + marginal_total_replicates
)

scale_df = pd.DataFrame(
    [
        {
            "component": (
                "core_orthogonal_grid"
            ),
            "species_region_units": (
                N_SPECIES_REGION_UNITS
            ),
            "scenarios_per_unit": (
                core_scenarios_per_species_region
            ),
            "replicates_per_scenario": (
                REPLICATES_PER_CORE_SCENARIO
            ),
            "total_replicates": (
                core_total_replicates
            ),
        },
        {
            "component": (
                "targeted_marginal_refinements"
            ),
            "species_region_units": (
                N_SPECIES_REGION_UNITS
            ),
            "scenarios_per_unit": (
                marginal_scenarios_per_species_region
            ),
            "replicates_per_scenario": (
                REPLICATES_PER_MARGINAL_SCENARIO
            ),
            "total_replicates": (
                marginal_total_replicates
            ),
        },
        {
            "component": (
                "full_confirmation"
            ),
            "species_region_units": (
                N_SPECIES_REGION_UNITS
            ),
            "scenarios_per_unit": (
                core_scenarios_per_species_region
                + marginal_scenarios_per_species_region
            ),
            "replicates_per_scenario": (
                REPLICATES_PER_CORE_SCENARIO
            ),
            "total_replicates": (
                total_replicates
            ),
        },
    ]
)

display(
    scale_df
)

# ============================================================
# Write frozen plan
# ============================================================
plan = {
    "plan_created_utc": RUN_UTC,
    "plan_status": "FROZEN",
    "scientific_goal": (
        "Confirm that effort-frame overlap governs corrected "
        "estimability while species-specific observation-support "
        "overlap governs corrected error and transportability."
    ),
    "formal_data": {
        "gbif_doi": GBIF_DOI,
        "states": STATES,
        "species_per_state": (
            SPECIES_PER_STATE
        ),
        "species_region_units": (
            N_SPECIES_REGION_UNITS
        ),
    },
    "random_seed": RANDOM_SEED,
    "core_design": {
        "effort_overlap_levels": (
            CORE_EFFORT_OVERLAPS
        ),
        "species_overlap_levels": (
            CORE_SPECIES_OVERLAPS
        ),
        "drift_levels_km_decade": (
            DRIFT_LEVELS_KM_DECADE
        ),
        "sample_sizes": (
            SAMPLE_SIZES
        ),
        "width_modes": (
            WIDTH_MODES
        ),
        "replicates_per_scenario": (
            REPLICATES_PER_CORE_SCENARIO
        ),
    },
    "marginal_refinement": {
        "levels": (
            MARGINAL_REFINEMENT_LEVELS
        ),
        "fixed_settings": (
            MARGINAL_FIXED_SETTINGS
        ),
        "replicates_per_scenario": (
            REPLICATES_PER_MARGINAL_SCENARIO
        ),
    },
    "estimator_settings": {
        "grid_resolution_degrees": (
            GRID_RESOLUTION_DEGREES
        ),
        "effort_support_size_cells": (
            EFFORT_SUPPORT_SIZE_CELLS
        ),
        "species_support_size_cells": (
            SPECIES_SUPPORT_SIZE_CELLS
        ),
        "minimum_stable_cell_effort": (
            MIN_STABLE_CELL_EFFORT
        ),
        "minimum_stable_cells": (
            MIN_STABLE_CELLS
        ),
    },
    "inference": {
        "species_region_cluster_bootstrap": (
            N_SPECIES_REGION_CLUSTER_BOOTSTRAP
        ),
        "scenario_label_permutations": (
            N_SCENARIO_LABEL_PERMUTATIONS
        ),
    },
    "primary_outcomes": (
        PRIMARY_OUTCOMES
    ),
    "secondary_outcomes": (
        SECONDARY_OUTCOMES
    ),
    "negative_controls": (
        NEGATIVE_CONTROLS
    ),
    "success_criteria": (
        SUCCESS_CRITERIA
    ),
    "computational_boundaries": (
        BOUNDARIES
    ),
    "trend_prerequisites": (
        trend_prerequisites
    ),
    "task_scale": {
        "core_total_replicates": (
            core_total_replicates
        ),
        "marginal_total_replicates": (
            marginal_total_replicates
        ),
        "total_replicates": (
            total_replicates
        ),
    },
}

PLAN_JSON_PATH.write_text(
    json.dumps(
        plan,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

readme_text = f"""# Frozen full orthogonal support-overlap confirmation plan

Generated: {RUN_UTC}

## Scientific aim

Confirm the orthogonal roles of two support-overlap axes:

1. Effort-frame overlap controls whether target-effort correction is estimable.
2. Species-specific observation-support overlap controls corrected error and
   correction transportability.

## Why the core levels are not changed

The successful trend experiment used four overlap levels:
{CORE_EFFORT_OVERLAPS}.

Changing the entire overlap grid after seeing the trend result would create a
post-hoc design concern. The full core confirmation therefore keeps the same
factor levels and increases replication from 20 to
{REPLICATES_PER_CORE_SCENARIO}.

Additional resolution is obtained only through two targeted marginal curves at:
{MARGINAL_REFINEMENT_LEVELS}.

## Full task size

- Species-region units: {N_SPECIES_REGION_UNITS}
- Core scenarios per unit: {core_scenarios_per_species_region}
- Core replicates: {core_total_replicates:,}
- Marginal replicates: {marginal_total_replicates:,}
- Total simulation replicates: {total_replicates:,}
- Cluster bootstrap replicates:
  {N_SPECIES_REGION_CLUSTER_BOOTSTRAP:,}
- Scenario-label permutations:
  {N_SCENARIO_LABEL_PERMUTATIONS:,}

## Primary outcomes

- Effort axis:
  corrected estimability
- Species axis:
  corrected absolute error

Correction gain is secondary and must always be reported together with raw and
corrected errors.

## Negative controls

{json.dumps(NEGATIVE_CONTROLS, ensure_ascii=False, indent=2)}

## Frozen boundaries

{chr(10).join(f"- {item}" for item in BOUNDARIES)}

## Trend prerequisites

{json.dumps(trend_prerequisites, ensure_ascii=False, indent=2)}
"""

PLAN_README_PATH.write_text(
    readme_text,
    encoding="utf-8",
)

append_path = BASE_DIR / "README.md"

existing_readme = (
    append_path.read_text(
        encoding="utf-8"
    )
    if append_path.exists()
    else "# Temporal Observation Drift\n"
)

append_path.write_text(
    existing_readme
    + f"""

## Full orthogonal phase-diagram plan frozen — {RUN_UTC}

- Plan JSON: {PLAN_JSON_PATH}
- Plan README: {PLAN_README_PATH}
- Total planned simulation replicates:
  {total_replicates:,}
- Core factor levels unchanged after trend success.
- Targeted marginal overlap refinements:
  {MARGINAL_REFINEMENT_LEVELS}
- No cutoff, method selection or new empirical feature is permitted.
""",
    encoding="utf-8",
)

print(
    "\n========== RUN SUMMARY =========="
)
print(
    "CELL_STATUS: FULL_ORTHOGONAL_PHASE_PLAN_FROZEN"
)
print(
    f"OUT_DIR: {OUT_DIR}"
)
print(
    f"PLAN_JSON: {PLAN_JSON_PATH}"
)
print(
    f"PLAN_README: {PLAN_README_PATH}"
)
print(
    f"N_SPECIES_REGION_UNITS: "
    f"{N_SPECIES_REGION_UNITS}"
)
print(
    f"CORE_SCENARIOS_PER_UNIT: "
    f"{core_scenarios_per_species_region}"
)
print(
    f"CORE_TOTAL_REPLICATES: "
    f"{core_total_replicates:,}"
)
print(
    f"MARGINAL_TOTAL_REPLICATES: "
    f"{marginal_total_replicates:,}"
)
print(
    f"TOTAL_PLANNED_REPLICATES: "
    f"{total_replicates:,}"
)
print(
    f"CLUSTER_BOOTSTRAP_REPLICATES: "
    f"{N_SPECIES_REGION_CLUSTER_BOOTSTRAP:,}"
)
print(
    f"SCENARIO_LABEL_PERMUTATIONS: "
    f"{N_SCENARIO_LABEL_PERMUTATIONS:,}"
)
print(
    f"TREND_PREREQUISITES_PASSED: "
    f"{prerequisites_passed}"
)
print(
    "CORE_LEVELS_CHANGED_AFTER_TREND: False"
)
print(
    "BINARY_CUTOFF_ALLOWED: False"
)
print(
    "PLAN_STATUS: FROZEN"
)
print(
    "PASSED: True"
)
print(
    "NEXT_STEP: Run the expensive full confirmation exactly from this frozen plan."
)
print(
    "README_UPDATED: True"
)
print(
    "============================================"
)
