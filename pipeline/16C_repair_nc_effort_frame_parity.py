
# Cell 16C — Repair NC formal parity using the locked effort-frame flag
#
# Root-cause hypothesis
# ---------------------
# Cell 16B treated every registered NC taxon as part of the target-effort frame.
# In NC, the locked registered union contains both:
#   - ecological benchmark taxa;
#   - observation-effort-frame taxa.
#
# Some taxa may be benchmark-only. They must contribute to the evaluated species
# estimates, but must NOT enter the target-effort denominator unless
# is_effort_frame_taxon == True.
#
# PA and VA were unaffected because their registered union was effectively nested.
#
# This cell:
# 1. Reuses the formal NC thinned data from Cell 16B.
# 2. Restores the locked NC effort-frame flag.
# 3. Recomputes the 0.25°, 0.5° and 1.0° correction gains.
# 4. Compares them with the temporary API table.
# 5. Replaces the NC formal species-input table only if exact parity is restored.
#
# PA and VA are not rerun.

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display

# ============================================================
# Locked configuration
# ============================================================
EARLY_YEARS = list(range(2013, 2019))
LATE_YEARS = list(range(2020, 2026))
EARLY_MID_YEAR = float(np.mean(EARLY_YEARS))
LATE_MID_YEAR = float(np.mean(LATE_YEARS))

GRID_RESOLUTIONS = [0.25, 0.50, 1.00]
PRIMARY_GRID_RESOLUTION = 0.50
MIN_TARGET_EFFORT_PER_CELL_PERIOD = 10
PARITY_TOLERANCE = 1e-6

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
PARITY_DIR = (
    BASE_DIR
    / "derived"
    / "registered_gbif_formal_parity"
)
NC_PLAN_PATH = (
    BASE_DIR
    / "derived"
    / "nc_final_confirmation_dataset"
    / "nc_locked_selected_taxa.csv"
)
NC_THINNED_PATH = (
    PARITY_DIR
    / "NC_formal_thinned.parquet"
)
SELECTED_INPUTS_PATH = (
    PARITY_DIR
    / "formal_parity_selected_inputs.csv"
)
NC_FORMAL_OUTPUT = (
    PARITY_DIR
    / "NC_formal_species_inputs.parquet"
)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("NC_PLAN_PATH:", NC_PLAN_PATH)
print("NC_THINNED_PATH:", NC_THINNED_PATH)
print("OUT_DIR:", PARITY_DIR)
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


def resolve_column(columns, candidates):
    lookup = {
        str(column).strip().lower(): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def shift_km_decade(early_latitude, late_latitude):
    if (
        not np.isfinite(early_latitude)
        or not np.isfinite(late_latitude)
    ):
        return np.nan

    return float(
        (late_latitude - early_latitude)
        * 111.32
        * 10
        / (LATE_MID_YEAR - EARLY_MID_YEAR)
    )


def set_jaccard(first, second):
    union = first | second

    if not union:
        return np.nan

    return float(
        len(first & second)
        / len(union)
    )


def build_grid_results(
    thinned,
    evaluation_codes,
    effort_codes,
    fia_reference,
    resolution,
):
    work = thinned.copy()

    work["grid_lat_index"] = np.floor(
        work["decimalLatitude"] / resolution
    ).astype("int32")

    work["grid_lon_index"] = np.floor(
        work["decimalLongitude"] / resolution
    ).astype("int32")

    work["grid_id"] = (
        work["grid_lat_index"].astype(str)
        + "_"
        + work["grid_lon_index"].astype(str)
    )

    effort_frame = work[
        work["fia_species_code"].isin(
            effort_codes
        )
    ].copy()

    effort_df = (
        effort_frame.groupby(
            ["period", "grid_id"],
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
        effort_df.pivot(
            index="grid_id",
            columns="period",
            values="target_effort_records",
        )
        .fillna(0)
    )

    for period in ["early", "late"]:
        if period not in effort_wide.columns:
            effort_wide[period] = 0

    stable_cells = set(
        effort_wide[
            (
                effort_wide["early"]
                >= MIN_TARGET_EFFORT_PER_CELL_PERIOD
            )
            & (
                effort_wide["late"]
                >= MIN_TARGET_EFFORT_PER_CELL_PERIOD
            )
        ].index
    )

    stable_effort = effort_df[
        effort_df["grid_id"].isin(
            stable_cells
        )
    ].copy()

    pooled_cell_latitude = (
        effort_frame[
            effort_frame["grid_id"].isin(
                stable_cells
            )
        ]
        .groupby("grid_id")[
            "decimalLatitude"
        ]
        .mean()
        .to_dict()
    )

    benchmark = work[
        work["fia_species_code"].isin(
            evaluation_codes
        )
    ].copy()

    raw_period = (
        benchmark.groupby(
            ["fia_species_code", "period"],
            as_index=False,
        )
        .agg(
            raw_records=("gbifID", "size"),
            raw_latitude=("decimalLatitude", "mean"),
            raw_cells=("grid_id", "nunique"),
        )
    )

    raw_wide = raw_period.pivot(
        index="fia_species_code",
        columns="period",
        values=[
            "raw_records",
            "raw_latitude",
            "raw_cells",
        ],
    )

    raw_wide.columns = [
        f"{metric}_{period}"
        for metric, period
        in raw_wide.columns
    ]

    raw_wide = raw_wide.reset_index()

    corrected_rows = []

    for species_code in sorted(
        evaluation_codes
    ):
        species_counts = (
            benchmark[
                benchmark["fia_species_code"]
                == species_code
            ]
            .groupby(
                ["period", "grid_id"]
            )["gbifID"]
            .size()
            .rename("species_records")
            .reset_index()
        )

        merged = stable_effort.merge(
            species_counts,
            on=["period", "grid_id"],
            how="left",
        )

        merged["species_records"] = (
            merged["species_records"]
            .fillna(0)
            .astype(float)
        )

        merged["relative_occurrence_rate"] = (
            merged["species_records"]
            / merged["target_effort_records"]
        )

        merged["cell_latitude"] = (
            merged["grid_id"].map(
                pooled_cell_latitude
            )
        )

        for period in ["early", "late"]:
            period_frame = merged[
                merged["period"] == period
            ]

            weight_sum = float(
                period_frame[
                    "relative_occurrence_rate"
                ].sum()
            )

            corrected_latitude = (
                float(
                    np.average(
                        period_frame[
                            "cell_latitude"
                        ],
                        weights=period_frame[
                            "relative_occurrence_rate"
                        ],
                    )
                )
                if (
                    weight_sum > 0
                    and period_frame[
                        "cell_latitude"
                    ].notna().all()
                )
                else np.nan
            )

            corrected_rows.append(
                {
                    "fia_species_code": species_code,
                    "period": period,
                    "corrected_latitude": corrected_latitude,
                    "stable_records": float(
                        period_frame[
                            "species_records"
                        ].sum()
                    ),
                    "stable_positive_cells": int(
                        (
                            period_frame[
                                "species_records"
                            ] > 0
                        ).sum()
                    ),
                }
            )

    corrected = pd.DataFrame(
        corrected_rows
    )

    corrected_wide = corrected.pivot(
        index="fia_species_code",
        columns="period",
        values=[
            "corrected_latitude",
            "stable_records",
            "stable_positive_cells",
        ],
    )

    corrected_wide.columns = [
        f"{metric}_{period}"
        for metric, period
        in corrected_wide.columns
    ]

    corrected_wide = corrected_wide.reset_index()

    result = (
        raw_wide.merge(
            corrected_wide,
            on="fia_species_code",
            how="outer",
        )
        .merge(
            fia_reference[
                [
                    "fia_species_code",
                    "fia_shift_km_decade",
                ]
            ],
            on="fia_species_code",
            how="inner",
        )
    )

    result["raw_shift_km_decade"] = [
        shift_km_decade(
            row.get(
                "raw_latitude_early",
                np.nan,
            ),
            row.get(
                "raw_latitude_late",
                np.nan,
            ),
        )
        for _, row in result.iterrows()
    ]

    result["corrected_shift_km_decade"] = [
        shift_km_decade(
            row.get(
                "corrected_latitude_early",
                np.nan,
            ),
            row.get(
                "corrected_latitude_late",
                np.nan,
            ),
        )
        for _, row in result.iterrows()
    ]

    result["raw_abs_error"] = (
        result["raw_shift_km_decade"]
        - result["fia_shift_km_decade"]
    ).abs()

    result["corrected_abs_error"] = (
        result["corrected_shift_km_decade"]
        - result["fia_shift_km_decade"]
    ).abs()

    result["gain"] = (
        result["raw_abs_error"]
        - result["corrected_abs_error"]
    )

    summary = {
        "grid_resolution": resolution,
        "n_effort_taxa": len(effort_codes),
        "n_effort_cells": int(
            effort_frame["grid_id"].nunique()
        ),
        "n_stable_cells": len(stable_cells),
        "n_species": int(
            result["fia_species_code"].nunique()
        ),
    }

    return result, summary


def build_features(thinned, evaluation_codes):
    work = thinned[
        thinned["fia_species_code"].isin(
            evaluation_codes
        )
    ].copy()

    work["feature_lat_index"] = np.floor(
        work["decimalLatitude"]
        / PRIMARY_GRID_RESOLUTION
    ).astype("int32")

    work["feature_lon_index"] = np.floor(
        work["decimalLongitude"]
        / PRIMARY_GRID_RESOLUTION
    ).astype("int32")

    work["feature_grid_id"] = (
        work["feature_lat_index"].astype(str)
        + "_"
        + work["feature_lon_index"].astype(str)
    )

    rows = []

    for species_code in sorted(
        evaluation_codes
    ):
        species = work[
            work["fia_species_code"]
            == species_code
        ]

        early = species[
            species["period"] == "early"
        ]

        late = species[
            species["period"] == "late"
        ]

        early_cells = set(
            early["feature_grid_id"].dropna()
        )
        late_cells = set(
            late["feature_grid_id"].dropna()
        )

        early_observers = int(
            early["observer_label"].nunique(
                dropna=True
            )
        )
        late_observers = int(
            late["observer_label"].nunique(
                dropna=True
            )
        )

        rows.append(
            {
                "fia_species_code": species_code,
                "cell_jaccard": set_jaccard(
                    early_cells,
                    late_cells,
                ),
                "abs_log_observer_growth": abs(
                    np.log(
                        (late_observers + 1)
                        / (early_observers + 1)
                    )
                ),
                "early_cells": len(early_cells),
                "late_cells": len(late_cells),
                "early_observers": early_observers,
                "late_observers": late_observers,
            }
        )

    return pd.DataFrame(rows)


def compare_metric(
    formal,
    temporary,
    metric,
):
    merged = formal[
        ["fia_species_code", metric]
    ].merge(
        temporary[
            ["fia_species_code", metric]
        ],
        on="fia_species_code",
        how="inner",
        suffixes=(
            "_formal",
            "_temp",
        ),
    )

    differences = (
        pd.to_numeric(
            merged[
                f"{metric}_formal"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            merged[
                f"{metric}_temp"
            ],
            errors="coerce",
        )
    ).abs().dropna()

    maximum = (
        float(differences.max())
        if len(differences)
        else np.nan
    )

    return {
        "metric": metric,
        "n_compared": len(differences),
        "max_abs_difference": maximum,
        "median_abs_difference": (
            float(differences.median())
            if len(differences)
            else np.nan
        ),
        "passed": bool(
            len(differences)
            == len(formal)
            and np.isfinite(maximum)
            and maximum
            <= PARITY_TOLERANCE
        ),
    }


def failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## NC formal parity repair failure — {RUN_UTC}
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
    print(f"OUT_DIR: {PARITY_DIR}")
    print("PASSED: False")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")


# ============================================================
# Main
# ============================================================
try:
    required_paths = [
        NC_PLAN_PATH,
        NC_THINNED_PATH,
        SELECTED_INPUTS_PATH,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing inputs: "
            + " | ".join(missing)
        )

    selected_inputs = pd.read_csv(
        SELECTED_INPUTS_PATH
    )

    nc_input_row = selected_inputs[
        selected_inputs[
            "state_code"
        ].eq("NC")
    ]

    if nc_input_row.empty:
        raise ValueError(
            "NC input row was not found in selected-input manifest."
        )

    temporary_path = Path(
        nc_input_row.iloc[0][
            "temporary_table"
        ]
    )

    fia_reference_path = Path(
        nc_input_row.iloc[0][
            "fia_reference"
        ]
    )

    if not temporary_path.exists():
        raise FileNotFoundError(
            f"Temporary NC table missing: {temporary_path}"
        )

    if not fia_reference_path.exists():
        raise FileNotFoundError(
            f"NC FIA reference missing: {fia_reference_path}"
        )

    print("TEMPORARY_NC_TABLE:", temporary_path)
    print("NC_FIA_REFERENCE:", fia_reference_path)

    nc_plan = pd.read_csv(
        NC_PLAN_PATH
    )

    code_column = resolve_column(
        nc_plan.columns,
        [
            "fia_species_code",
            "species_code",
        ],
    )

    effort_column = resolve_column(
        nc_plan.columns,
        [
            "is_effort_frame_taxon",
            "effort_frame_taxon",
        ],
    )

    benchmark_column = resolve_column(
        nc_plan.columns,
        [
            "is_benchmark_taxon",
            "benchmark_taxon",
        ],
    )

    if code_column is None:
        raise KeyError(
            "NC plan has no FIA species-code column."
        )

    if effort_column is None:
        raise KeyError(
            "NC plan has no is_effort_frame_taxon column."
        )

    nc_plan[
        "fia_species_code"
    ] = pd.to_numeric(
        nc_plan[
            code_column
        ],
        errors="coerce",
    )

    nc_plan = nc_plan.dropna(
        subset=[
            "fia_species_code",
        ]
    ).copy()

    nc_plan[
        "fia_species_code"
    ] = nc_plan[
        "fia_species_code"
    ].astype(int)

    nc_plan[
        "is_effort_frame_taxon"
    ] = to_bool(
        nc_plan[
            effort_column
        ]
    )

    nc_plan[
        "is_benchmark_taxon"
    ] = (
        to_bool(
            nc_plan[
                benchmark_column
            ]
        )
        if benchmark_column
        else False
    )

    union_codes = set(
        nc_plan[
            "fia_species_code"
        ]
    )

    effort_codes = set(
        nc_plan.loc[
            nc_plan[
                "is_effort_frame_taxon"
            ],
            "fia_species_code",
        ]
    )

    benchmark_codes = set(
        nc_plan.loc[
            nc_plan[
                "is_benchmark_taxon"
            ],
            "fia_species_code",
        ]
    )

    benchmark_only_codes = (
        benchmark_codes
        - effort_codes
    )

    plan_summary_df = pd.DataFrame(
        [
            {
                "union_taxa": len(
                    union_codes
                ),
                "effort_frame_taxa": len(
                    effort_codes
                ),
                "benchmark_taxa": len(
                    benchmark_codes
                ),
                "benchmark_only_taxa": len(
                    benchmark_only_codes
                ),
                "benchmark_only_codes": (
                    json.dumps(
                        sorted(
                            benchmark_only_codes
                        )
                    )
                ),
            }
        ]
    )

    print(
        "\nNC LOCKED PLAN SUMMARY"
    )
    display(
        plan_summary_df
    )

    temporary_df = pd.read_parquet(
        temporary_path
    )

    temporary_df[
        "fia_species_code"
    ] = pd.to_numeric(
        temporary_df[
            "fia_species_code"
        ],
        errors="coerce",
    )

    temporary_df = temporary_df.dropna(
        subset=[
            "fia_species_code",
        ]
    ).copy()

    temporary_df[
        "fia_species_code"
    ] = temporary_df[
        "fia_species_code"
    ].astype(int)

    evaluation_codes = set(
        temporary_df[
            "fia_species_code"
        ]
    )

    fia_reference = pd.read_parquet(
        fia_reference_path
    )

    if (
        "fia_species_code"
        not in fia_reference.columns
        and "species_code"
        in fia_reference.columns
    ):
        fia_reference = fia_reference.rename(
            columns={
                "species_code": (
                    "fia_species_code"
                )
            }
        )

    fia_reference[
        "fia_species_code"
    ] = pd.to_numeric(
        fia_reference[
            "fia_species_code"
        ],
        errors="coerce",
    )

    fia_reference = fia_reference.dropna(
        subset=[
            "fia_species_code",
            "fia_shift_km_decade",
        ]
    ).copy()

    fia_reference[
        "fia_species_code"
    ] = fia_reference[
        "fia_species_code"
    ].astype(int)

    thinned = pd.read_parquet(
        NC_THINNED_PATH
    )

    thinned[
        "fia_species_code"
    ] = pd.to_numeric(
        thinned[
            "fia_species_code"
        ],
        errors="coerce",
    )

    thinned = thinned.dropna(
        subset=[
            "fia_species_code",
        ]
    ).copy()

    thinned[
        "fia_species_code"
    ] = thinned[
        "fia_species_code"
    ].astype(int)

    formal_table = pd.DataFrame(
        {
            "fia_species_code": sorted(
                evaluation_codes
            )
        }
    )

    grid_summary_rows = []

    for resolution in tqdm(
        GRID_RESOLUTIONS,
        desc="Recomputing NC with locked effort frame",
        unit="grid",
    ):
        (
            grid_result,
            grid_summary,
        ) = build_grid_results(
            thinned=thinned,
            evaluation_codes=(
                evaluation_codes
            ),
            effort_codes=effort_codes,
            fia_reference=fia_reference,
            resolution=resolution,
        )

        tag = str(
            resolution
        ).replace(
            ".",
            "p",
        )

        formal_table = formal_table.merge(
            grid_result[
                [
                    "fia_species_code",
                    "raw_shift_km_decade",
                    "corrected_shift_km_decade",
                    "fia_shift_km_decade",
                    "raw_abs_error",
                    "corrected_abs_error",
                    "gain",
                ]
            ].rename(
                columns={
                    "raw_shift_km_decade": (
                        f"raw_shift_grid_{tag}"
                    ),
                    "corrected_shift_km_decade": (
                        f"corrected_shift_grid_{tag}"
                    ),
                    "raw_abs_error": (
                        f"raw_error_grid_{tag}"
                    ),
                    "corrected_abs_error": (
                        f"corrected_error_grid_{tag}"
                    ),
                    "gain": (
                        f"gain_grid_{tag}"
                    ),
                }
            ),
            on="fia_species_code",
            how="left",
        )

        grid_summary_rows.append(
            grid_summary
        )

    gain_columns = [
        "gain_grid_0p25",
        "gain_grid_0p5",
        "gain_grid_1p0",
    ]

    formal_table[
        "grid_median_gain_km_decade"
    ] = formal_table[
        gain_columns
    ].median(
        axis=1,
        skipna=True,
    )

    formal_table[
        "positive_grid_fraction"
    ] = (
        formal_table[
            gain_columns
        ]
        .gt(0)
        .sum(axis=1)
        / formal_table[
            gain_columns
        ]
        .notna()
        .sum(axis=1)
        .replace(
            0,
            np.nan,
        )
    )

    features = build_features(
        thinned=thinned,
        evaluation_codes=(
            evaluation_codes
        ),
    )

    formal_table = formal_table.merge(
        features,
        on="fia_species_code",
        how="left",
    )

    parity_rows = [
        compare_metric(
            formal_table,
            temporary_df,
            metric,
        )
        for metric in [
            "cell_jaccard",
            "abs_log_observer_growth",
            "gain_grid_0p25",
            "gain_grid_0p5",
            "gain_grid_1p0",
            "grid_median_gain_km_decade",
        ]
    ]

    parity_df = pd.DataFrame(
        parity_rows
    )

    grid_summary_df = pd.DataFrame(
        grid_summary_rows
    )

    print(
        "\nREPAIRED NC GRID SUMMARY"
    )
    display(
        grid_summary_df
    )

    print(
        "\nREPAIRED NC PARITY"
    )
    display(
        parity_df
    )

    species_set_pass = bool(
        set(
            formal_table[
                "fia_species_code"
            ]
        )
        == set(
            temporary_df[
                "fia_species_code"
            ]
        )
    )

    metric_parity_pass = bool(
        parity_df[
            "passed"
        ].all()
    )

    passed = bool(
        species_set_pass
        and metric_parity_pass
    )

    if passed:
        status = (
            "NC_FORMAL_PARITY_REPAIRED"
        )
        next_step = (
            "Run the final locked PA, VA and NC confirmatory inference "
            "using the corrected formal species-input tables."
        )

        formal_table.to_parquet(
            NC_FORMAL_OUTPUT,
            index=False,
        )

        formal_table.to_csv(
            PARITY_DIR
            / "NC_formal_species_inputs.csv",
            index=False,
        )

        parity_df.to_csv(
            PARITY_DIR
            / "NC_effort_frame_repair_parity.csv",
            index=False,
        )

        grid_summary_df.to_csv(
            PARITY_DIR
            / "NC_effort_frame_repair_grid_summary.csv",
            index=False,
        )

    else:
        status = (
            "NC_EFFORT_FRAME_REPAIR_INCOMPLETE"
        )
        next_step = (
            "Do not run final inference. Compare raw shifts, corrected "
            "shifts and FIA reference components species by species."
        )

    append_readme(
        f"""

## NC formal parity effort-frame repair — {RUN_UTC}

### Root cause tested
Cell 16B used all locked NC union taxa as target-effort taxa. The NC registered
union contains benchmark and effort-frame roles that are not fully nested.

### Locked NC plan
- Union taxa: {len(union_codes)}
- Effort-frame taxa: {len(effort_codes)}
- Benchmark taxa: {len(benchmark_codes)}
- Benchmark-only taxa: {len(benchmark_only_codes)}
- Benchmark-only codes: {sorted(benchmark_only_codes)}

### Result
- Species-set parity: {species_set_pass}
- Metric parity: {metric_parity_pass}
- Status: **{status}**
- Passed: {passed}
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
        f"OUT_DIR: {PARITY_DIR}"
    )
    print(
        f"NC_UNION_TAXA: {len(union_codes)}"
    )
    print(
        f"NC_EFFORT_FRAME_TAXA: {len(effort_codes)}"
    )
    print(
        f"NC_BENCHMARK_TAXA: {len(benchmark_codes)}"
    )
    print(
        f"NC_BENCHMARK_ONLY_TAXA: "
        f"{len(benchmark_only_codes)}"
    )
    print(
        "NC_BENCHMARK_ONLY_CODES: "
        + json.dumps(
            sorted(
                benchmark_only_codes
            )
        )
    )
    print(
        "REPAIRED_GRID_SUMMARY: "
        + json.dumps(
            grid_summary_df.to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "REPAIRED_PARITY: "
        + json.dumps(
            parity_df.round(
                10
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        f"SPECIES_SET_PASSED: "
        f"{species_set_pass}"
    )
    print(
        f"METRIC_PARITY_PASSED: "
        f"{metric_parity_pass}"
    )
    print(
        f"PARITY_TOLERANCE: "
        f"{PARITY_TOLERANCE}"
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
        "CELL16C_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the displayed NC locked-plan summary. "
            "Do not run final inference."
        ),
    )
