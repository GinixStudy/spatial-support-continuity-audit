
# Cell 06B — Resume Cell 06A after fixing the missing `state` column
#
# Root cause:
# Cell 06A saved each state's species result table without a `state` column.
# The later cross-state meta-analysis then attempted group["state"], causing:
# KeyError: 'state'
#
# This repair cell:
# 1. Reuses the completed per-state result files.
# 2. Adds the missing state column.
# 3. Rebuilds the combined four-state table.
# 4. Resumes species meta-analysis, leave-one-state-out analysis, and decision.
#
# It does NOT reread the large TREE files and does NOT rerun permutations.

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 180)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 280)

# ============================================================
# Configuration — must match Cell 06A
# ============================================================
STATES = ["WV", "PA", "VA", "KY"]
ADDED_STATES = ["PA", "VA", "KY"]

MIN_META_STATES = 3
MIN_META_SPECIES = 20
TARGET_NON_WV_MEDIAN_ABS_CORRECTION = 1.50
TARGET_META_SIGN_CONSISTENCY = 0.60
TARGET_LOO_SIGN_STABILITY = 0.75

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
OUT_DIR = BASE_DIR / "derived" / "repeated_physical_plot_control"
STATE_SUMMARY_PATH = OUT_DIR / "state_repeated_plot_summary.csv"

RUN_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

print("RUN_UTC:", RUN_UTC)
print("OUT_DIR:", OUT_DIR)


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
    values = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if (
        len(values) < 3
        or values["x"].nunique() < 2
        or values["y"].nunique() < 2
    ):
        return np.nan

    return float(values.corr(method="spearman").iloc[0, 1])


def print_failure(status, error, next_step):
    append_readme(
        f"""

## Cell 06B repair failure — {RUN_UTC}
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
    # 1. Validate completed Cell 06A outputs
    # --------------------------------------------------------
    required_paths = [
        OUT_DIR / state / "species_repeated_plot_results.parquet"
        for state in STATES
    ]

    missing_paths = [
        str(path) for path in required_paths if not path.exists()
    ]

    if not STATE_SUMMARY_PATH.exists():
        missing_paths.append(str(STATE_SUMMARY_PATH))

    if missing_paths:
        raise FileNotFoundError(
            "Cell 06A did not finish all state-level outputs: "
            + " | ".join(missing_paths)
        )

    # --------------------------------------------------------
    # 2. Load state results and add the missing column
    # --------------------------------------------------------
    state_result_tables = []
    patch_rows = []

    for state in tqdm(
        STATES,
        desc="Loading and repairing state results",
        unit="state",
    ):
        state_dir = OUT_DIR / state
        parquet_path = (
            state_dir / "species_repeated_plot_results.parquet"
        )
        csv_path = (
            state_dir / "species_repeated_plot_results.csv"
        )

        state_df = pd.read_parquet(parquet_path)

        had_state_column = "state" in state_df.columns

        if not had_state_column:
            state_df.insert(0, "state", state)
        else:
            state_df["state"] = state_df["state"].fillna(state)
            state_df.loc[
                state_df["state"].astype(str).str.strip().eq(""),
                "state",
            ] = state

        # Save corrected outputs created by our prior cell.
        state_df.to_parquet(parquet_path, index=False)
        state_df.to_csv(csv_path, index=False)

        state_result_tables.append(state_df)

        patch_rows.append(
            {
                "state": state,
                "rows": len(state_df),
                "state_column_previously_present": had_state_column,
                "unique_state_values_after_patch": (
                    ", ".join(
                        sorted(
                            state_df["state"]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )
                ),
            }
        )

    patch_df = pd.DataFrame(patch_rows)

    print("\nPATCH SUMMARY")
    display(patch_df)

    all_results_df = pd.concat(
        state_result_tables,
        ignore_index=True,
    )

    if "state" not in all_results_df.columns:
        raise KeyError(
            "The repaired combined table still lacks the state column."
        )

    state_summary_df = pd.read_csv(STATE_SUMMARY_PATH)

    all_results_df.to_parquet(
        OUT_DIR / "all_state_repeated_plot_results.parquet",
        index=False,
    )
    all_results_df.to_csv(
        OUT_DIR / "all_state_repeated_plot_results.csv",
        index=False,
    )

    print("\nFOUR-STATE REPEATED-PLOT SUMMARY")
    display(state_summary_df)

    # --------------------------------------------------------
    # 3. Cross-state species meta-summary
    # --------------------------------------------------------
    required_result_columns = [
        "state",
        "species_code",
        "scientific_name",
        "common_name",
        "strong_coverage",
        "correction_effect_km_decade",
        "material_correction",
        "meaningful_direction_reversal",
        "matched_shift_permutation_p",
        "correction_permutation_p",
    ]

    missing_columns = [
        column
        for column in required_result_columns
        if column not in all_results_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required columns in repaired results: "
            + " | ".join(missing_columns)
        )

    eligible_df = all_results_df[
        all_results_df["strong_coverage"].fillna(False)
        & all_results_df[
            "correction_effect_km_decade"
        ].notna()
    ].copy()

    meta_rows = []
    groups = eligible_df.groupby(
        "species_code",
        sort=True,
    )

    for species_code, group in tqdm(
        groups,
        total=groups.ngroups,
        desc="Building repeated-plot species meta-summary",
        unit="species",
    ):
        effects = group[
            "correction_effect_km_decade"
        ].to_numpy(dtype=float)

        positive_count = int((effects > 0).sum())
        negative_count = int((effects < 0).sum())
        nonzero_count = positive_count + negative_count

        majority_sign_fraction = (
            max(positive_count, negative_count) / nonzero_count
            if nonzero_count
            else np.nan
        )

        non_wv_group = group[group["state"] != "WV"]
        first = group.iloc[0]

        meta_rows.append(
            {
                "species_code": int(species_code),
                "scientific_name": first["scientific_name"],
                "common_name": first["common_name"],
                "n_states": int(group["state"].nunique()),
                "median_correction_km_decade": float(
                    np.median(effects)
                ),
                "median_abs_correction_km_decade": float(
                    np.median(np.abs(effects))
                ),
                "majority_sign_fraction": (
                    majority_sign_fraction
                ),
                "n_material_states": int(
                    group["material_correction"].sum()
                ),
                "n_reversal_states": int(
                    group[
                        "meaningful_direction_reversal"
                    ].sum()
                ),
                "median_matched_shift_permutation_p": float(
                    group[
                        "matched_shift_permutation_p"
                    ].median()
                ),
                "median_correction_permutation_p": float(
                    group[
                        "correction_permutation_p"
                    ].median()
                ),
                "non_wv_state_count": int(
                    non_wv_group["state"].nunique()
                ),
                "non_wv_median_correction_km_decade": (
                    float(
                        non_wv_group[
                            "correction_effect_km_decade"
                        ].median()
                    )
                    if not non_wv_group.empty
                    else np.nan
                ),
                "non_wv_median_abs_correction_km_decade": (
                    float(
                        non_wv_group[
                            "correction_effect_km_decade"
                        ].abs().median()
                    )
                    if not non_wv_group.empty
                    else np.nan
                ),
            }
        )

    species_meta_df = pd.DataFrame(meta_rows)

    if species_meta_df.empty:
        raise ValueError(
            "No species had strong repeated-plot coverage."
        )

    species_meta_df["eligible_meta_species"] = (
        species_meta_df["n_states"] >= MIN_META_STATES
    )

    species_meta_df = species_meta_df.sort_values(
        "median_abs_correction_km_decade",
        ascending=False,
    ).reset_index(drop=True)

    species_meta_df.to_parquet(
        OUT_DIR / "repeated_plot_species_meta.parquet",
        index=False,
    )
    species_meta_df.to_csv(
        OUT_DIR / "repeated_plot_species_meta.csv",
        index=False,
    )

    print("\nREPEATED-PLOT SPECIES META-SUMMARY")
    display(species_meta_df.head(50))

    eligible_meta_df = species_meta_df[
        species_meta_df["eligible_meta_species"]
    ].copy()

    n_meta_species = len(eligible_meta_df)

    majority_sign_consistency_fraction = (
        float(
            (
                eligible_meta_df[
                    "majority_sign_fraction"
                ] >= 0.75
            ).mean()
        )
        if n_meta_species
        else np.nan
    )

    non_wv_median_abs_correction = (
        float(
            eligible_meta_df[
                "non_wv_median_abs_correction_km_decade"
            ].median()
        )
        if n_meta_species
        else np.nan
    )

    # --------------------------------------------------------
    # 4. Leave-one-state-out meta stability
    # --------------------------------------------------------
    full_meta_lookup = eligible_meta_df.set_index(
        "species_code"
    )["median_correction_km_decade"]

    loo_rows = []
    loo_species_rows = []

    for held_out_state in tqdm(
        STATES,
        desc="Repeated-plot leave-one-state-out",
        unit="state",
    ):
        remaining = eligible_df[
            eligible_df["state"] != held_out_state
        ].copy()

        loo_meta = (
            remaining.groupby(
                "species_code"
            )["correction_effect_km_decade"]
            .agg(["median", "count"])
            .rename(
                columns={
                    "median": "loo_effect",
                    "count": "loo_n_states",
                }
            )
        )

        comparison = pd.DataFrame(
            {"full_effect": full_meta_lookup}
        ).join(
            loo_meta,
            how="inner",
        )

        comparison = comparison[
            comparison["loo_n_states"] >= 2
        ].copy()

        comparison["sign_stable"] = (
            np.sign(comparison["full_effect"])
            == np.sign(comparison["loo_effect"])
        )

        sign_agreement = (
            float(comparison["sign_stable"].mean())
            if len(comparison)
            else np.nan
        )

        magnitude_spearman = safe_spearman(
            comparison["full_effect"],
            comparison["loo_effect"],
        )

        loo_rows.append(
            {
                "held_out_state": held_out_state,
                "n_species_compared": len(comparison),
                "sign_agreement": sign_agreement,
                "magnitude_spearman": magnitude_spearman,
                "median_abs_change": (
                    float(
                        (
                            comparison["full_effect"]
                            - comparison["loo_effect"]
                        ).abs().median()
                    )
                    if len(comparison)
                    else np.nan
                ),
            }
        )

        for species_code, row in comparison.iterrows():
            loo_species_rows.append(
                {
                    "held_out_state": held_out_state,
                    "species_code": int(species_code),
                    "full_effect": row["full_effect"],
                    "loo_effect": row["loo_effect"],
                    "loo_n_states": int(
                        row["loo_n_states"]
                    ),
                    "sign_stable": bool(
                        row["sign_stable"]
                    ),
                }
            )

    loo_summary_df = pd.DataFrame(loo_rows)
    loo_species_df = pd.DataFrame(loo_species_rows)

    loo_summary_df.to_csv(
        OUT_DIR / "repeated_plot_loo_summary.csv",
        index=False,
    )
    loo_species_df.to_parquet(
        OUT_DIR / "repeated_plot_loo_species.parquet",
        index=False,
    )

    print("\nREPEATED-PLOT LEAVE-ONE-STATE-OUT")
    display(loo_summary_df)

    overall_loo_sign_stability = (
        float(loo_species_df["sign_stable"].mean())
        if len(loo_species_df)
        else np.nan
    )

    # --------------------------------------------------------
    # 5. Final trend decision
    # --------------------------------------------------------
    required_summary_columns = [
        "state",
        "domain_pass",
        "material_signal",
        "permutation_pass",
        "state_pass",
    ]

    missing_summary_columns = [
        column
        for column in required_summary_columns
        if column not in state_summary_df.columns
    ]

    if missing_summary_columns:
        raise KeyError(
            "Missing state-summary columns: "
            + " | ".join(missing_summary_columns)
        )

    for column in [
        "domain_pass",
        "material_signal",
        "permutation_pass",
        "state_pass",
    ]:
        if state_summary_df[column].dtype == object:
            state_summary_df[column] = (
                state_summary_df[column]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False})
            )

    n_domain_pass_states = int(
        state_summary_df["domain_pass"].fillna(False).sum()
    )
    n_material_states = int(
        state_summary_df[
            "material_signal"
        ].fillna(False).sum()
    )
    n_permutation_pass_states = int(
        state_summary_df[
            "permutation_pass"
        ].fillna(False).sum()
    )
    n_state_passes = int(
        state_summary_df["state_pass"].fillna(False).sum()
    )
    n_added_state_passes = int(
        state_summary_df[
            state_summary_df["state"].isin(ADDED_STATES)
        ]["state_pass"].fillna(False).sum()
    )

    domain_criterion = bool(
        n_domain_pass_states >= 3
    )
    state_replication_criterion = bool(
        n_state_passes >= 2
        and n_added_state_passes >= 1
    )
    meta_coverage_criterion = bool(
        n_meta_species >= MIN_META_SPECIES
    )
    non_wv_effect_criterion = bool(
        np.isfinite(non_wv_median_abs_correction)
        and non_wv_median_abs_correction
        >= TARGET_NON_WV_MEDIAN_ABS_CORRECTION
    )
    meta_consistency_criterion = bool(
        np.isfinite(
            majority_sign_consistency_fraction
        )
        and majority_sign_consistency_fraction
        >= TARGET_META_SIGN_CONSISTENCY
    )
    loo_criterion = bool(
        np.isfinite(overall_loo_sign_stability)
        and overall_loo_sign_stability
        >= TARGET_LOO_SIGN_STABILITY
    )

    passed = bool(
        domain_criterion
        and state_replication_criterion
        and meta_coverage_criterion
        and non_wv_effect_criterion
        and meta_consistency_criterion
        and loo_criterion
    )

    if passed:
        status = "GO_FOR_GBIF_OBSERVATION_DRIFT_COMPARISON"
        next_step = (
            "The effect survives repeated-physical-plot control. "
            "Proceed to a small GBIF/iNaturalist comparison for the "
            "same species and regions."
        )
    elif (
        domain_criterion
        and meta_coverage_criterion
        and non_wv_effect_criterion
        and loo_criterion
    ):
        status = "PIVOT_TO_SPECIES_SENSITIVITY_FRAMEWORK"
        next_step = (
            "A universal state-level effect is weak, but stable "
            "species-specific sensitivity remains. Model which species "
            "range structures predict correction magnitude."
        )
    elif domain_criterion and n_state_passes == 0:
        status = "ROTATING_PANEL_ARTIFACT_DOMINATES"
        next_step = (
            "The earlier fixed-grid signal does not survive exact "
            "physical-plot matching. Do not claim general observation "
            "drift from FIA annual panels."
        )
    else:
        status = "REPEATED_PLOT_CONTROL_INCONCLUSIVE"
        next_step = (
            "Repeated-plot coverage or state consistency is insufficient. "
            "Inspect the state-level matched-plot results before changing "
            "the scientific direction."
        )

    # --------------------------------------------------------
    # 6. Visual previews
    # --------------------------------------------------------
    plot_state_df = state_summary_df.sort_values(
        "state"
    ).copy()

    if "median_abs_correction_km_decade" in plot_state_df.columns:
        plt.figure(figsize=(9, 5))
        plt.bar(
            plot_state_df["state"],
            plot_state_df[
                "median_abs_correction_km_decade"
            ],
        )
        plt.axhline(
            TARGET_NON_WV_MEDIAN_ABS_CORRECTION,
            linestyle="--",
            label="Material threshold",
        )
        plt.xlabel("State")
        plt.ylabel(
            "Median absolute raw-matched correction "
            "(km/decade)"
        )
        plt.title("Repeated physical-plot control")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if not eligible_meta_df.empty:
        top_meta_df = (
            eligible_meta_df.head(20)
            .sort_values(
                "median_correction_km_decade"
            )
            .copy()
        )

        labels = top_meta_df[
            "scientific_name"
        ].fillna(
            top_meta_df["species_code"].astype(str)
        )

        plt.figure(figsize=(10, 8))
        plt.barh(
            labels,
            top_meta_df[
                "median_correction_km_decade"
            ],
        )
        plt.axvline(0)
        plt.xlabel(
            "Median correction effect "
            "(raw minus repeated-plot, km/decade)"
        )
        plt.ylabel("Species")
        plt.title(
            "Cross-state repeated-plot correction effects"
        )
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(9, 5))
    plt.bar(
        loo_summary_df["held_out_state"],
        loo_summary_df["sign_agreement"],
    )
    plt.axhline(
        TARGET_LOO_SIGN_STABILITY,
        linestyle="--",
        label="Target",
    )
    plt.ylim(0, 1.05)
    plt.xlabel("Held-out state")
    plt.ylabel("Sign agreement")
    plt.title(
        "Repeated-plot leave-one-state-out stability"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 7. README
    # --------------------------------------------------------
    top_meta_species = (
        eligible_meta_df[
            [
                "species_code",
                "scientific_name",
                "n_states",
                "median_correction_km_decade",
                "median_abs_correction_km_decade",
                "majority_sign_fraction",
                "non_wv_median_correction_km_decade",
            ]
        ]
        .head(10)
        .round(4)
        .to_dict("records")
    )

    append_readme(
        f"""

## Cell 06B repair and resumed repeated-plot analysis — {RUN_UTC}

### Code correction
Cell 06A omitted the `state` column from each state-level species result table.
Cell 06B added the state value, rebuilt the combined table, and resumed the
cross-state meta-analysis without rereading TREE data or rerunning permutations.

### Cross-state results
- Adequate repeated-plot domains: {n_domain_pass_states}/4
- Material state signals: {n_material_states}/4
- State permutation passes: {n_permutation_pass_states}/4
- Full state passes: {n_state_passes}/4
- Added-state passes: {n_added_state_passes}/3
- Meta species represented in at least {MIN_META_STATES} states:
  {n_meta_species}
- Median non-WV absolute correction:
  {non_wv_median_abs_correction} km/decade
- Fraction with at least 75% state-sign agreement:
  {majority_sign_consistency_fraction}
- Leave-one-state-out sign stability:
  {overall_loo_sign_stability}

### Decision
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}
"""
    )

    # --------------------------------------------------------
    # 8. Compact output block
    # --------------------------------------------------------
    state_compact_columns = [
        column
        for column in [
            "state",
            "early_years",
            "late_years",
            "n_matched_physical_plots",
            "median_plot_interval_years",
            "n_strong_species",
            "median_abs_correction_km_decade",
            "meaningful_reversal_fraction",
            "attenuation_fraction",
            "permutation_p_median_abs_correction",
            "permutation_p_reversal_fraction",
            "state_pass",
        ]
        if column in state_summary_df.columns
    ]

    state_compact = (
        state_summary_df[state_compact_columns]
        .round(6)
        .to_dict("records")
    )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"PATCH_APPLIED: Added missing state column to all four state result tables")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATES: {STATES}")
    print(
        "STATE_RESULTS: "
        + json.dumps(
            state_compact,
            ensure_ascii=False,
        )
    )
    print(
        f"N_DOMAIN_PASS_STATES: "
        f"{n_domain_pass_states}"
    )
    print(
        f"N_MATERIAL_SIGNAL_STATES: "
        f"{n_material_states}"
    )
    print(
        f"N_PERMUTATION_PASS_STATES: "
        f"{n_permutation_pass_states}"
    )
    print(
        f"N_FULL_STATE_PASSES: "
        f"{n_state_passes}"
    )
    print(
        f"N_ADDED_STATE_PASSES: "
        f"{n_added_state_passes}"
    )
    print(f"N_META_SPECIES: {n_meta_species}")
    print(
        f"NON_WV_MEDIAN_ABS_CORRECTION_KM_DECADE: "
        f"{non_wv_median_abs_correction}"
    )
    print(
        f"META_MAJORITY_SIGN_CONSISTENCY_FRACTION: "
        f"{majority_sign_consistency_fraction}"
    )
    print(
        f"OVERALL_LOO_SIGN_STABILITY: "
        f"{overall_loo_sign_stability}"
    )
    print(
        "LOO_SUMMARY: "
        + json.dumps(
            loo_summary_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "TOP_META_SPECIES: "
        + json.dumps(
            top_meta_species,
            ensure_ascii=False,
        )
    )
    print(
        f"DOMAIN_CRITERION: "
        f"{domain_criterion}"
    )
    print(
        f"STATE_REPLICATION_CRITERION: "
        f"{state_replication_criterion}"
    )
    print(
        f"META_COVERAGE_CRITERION: "
        f"{meta_coverage_criterion}"
    )
    print(
        f"NON_WV_EFFECT_CRITERION: "
        f"{non_wv_effect_criterion}"
    )
    print(
        f"META_CONSISTENCY_CRITERION: "
        f"{meta_consistency_criterion}"
    )
    print(f"LOO_CRITERION: {loo_criterion}")
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except Exception as exc:
    print_failure(
        "CELL06B_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        "Save this diagnostic summary and the last displayed output. "
        "Do not rerun the large TREE scans.",
    )
