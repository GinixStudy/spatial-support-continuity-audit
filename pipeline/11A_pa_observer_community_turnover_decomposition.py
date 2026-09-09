
# Cell 11A — PA observer-community turnover and activity-weighting decomposition
#
# Scientific conclusion entering this cell
# ----------------------------------------
# The personal-observer trajectory branch is not identifiable for PA trees:
# exact observer overlap is very low and repeated-observer estimates are unstable.
#
# New mechanism question
# ----------------------
# How much of the early-to-late observation-footprint shift is associated with:
# 1) changes in record-volume weighting across observers;
# 2) changes in the composition of observer activity profiles;
# 3) spatial replacement of observers within comparable activity profiles?
#
# Exact additive decomposition
# ----------------------------
# Total record-weighted shift
#   = activity-weighting component
#   + observer-profile composition component
#   + within-profile spatial-turnover component
#
# This is an observation-system decomposition. It does not infer movement of
# individual people and does not identify climate-driven species redistribution.

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
# Configuration
# ============================================================
STATE = "PA"

EARLY_YEARS = list(range(2013, 2019))
LATE_YEARS = list(range(2020, 2026))
WINDOW_INTERVAL_YEARS = (
    np.mean(LATE_YEARS) - np.mean(EARLY_YEARS)
)

FOOTPRINT_GRID_RESOLUTION = 0.50

N_BOOTSTRAP = 800
N_PROFILE_PERMUTATIONS = 800
RANDOM_SEED = 20260710

MIN_COMMON_PROFILE_OBSERVER_COVERAGE = 0.80
MIN_COMMON_PROFILE_RECORD_COVERAGE = 0.80
MATERIAL_COMPONENT_KM_DECADE = 3.0
MIN_STABLE_PROFILE_SPECIFICATIONS = 2

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
INPUT_DIR = (
    BASE_DIR
    / "derived"
    / "pa_expanded_tree_observer_decomposition"
)
OUT_DIR = (
    BASE_DIR
    / "derived"
    / "pa_observer_community_turnover_decomposition"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = (
    INPUT_DIR
    / "expanded_pa_tree_thinned_occurrences.parquet"
)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("STATE:", STATE)
print("EARLY_YEARS:", EARLY_YEARS)
print("LATE_YEARS:", LATE_YEARS)
print("OUT_DIR:", OUT_DIR)
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


def normalize_observer_label(series):
    normalized = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    blocked = {
        "",
        "nan",
        "none",
        "unknown",
        "anonymous",
        "not recorded",
        "not provided",
    }

    return normalized.mask(normalized.isin(blocked))


def bin_records(values):
    return pd.cut(
        values,
        bins=[0, 1, 3, 9, 25, np.inf],
        labels=["1", "2-3", "4-9", "10-25", "26+"],
        include_lowest=True,
    ).astype("string")


def bin_breadth(values):
    return pd.cut(
        values,
        bins=[0, 1, 3, 6, 12, np.inf],
        labels=["1", "2-3", "4-6", "7-12", "13+"],
        include_lowest=True,
    ).astype("string")


def to_km_decade(latitude_difference):
    return float(
        latitude_difference
        * 111.32
        * 10
        / WINDOW_INTERVAL_YEARS
    )


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            values[valid],
            weights=weights[valid],
        )
    )


def create_profile_id(frame, columns):
    return (
        frame[columns]
        .fillna("missing")
        .astype(str)
        .agg("|".join, axis=1)
    )


def compute_decomposition(
    observer_period_df,
    profile_columns,
    fixed_common_profiles=None,
    fixed_reference_weights=None,
):
    work = observer_period_df.copy()
    work["profile_id"] = create_profile_id(
        work,
        profile_columns,
    )

    period_centers = (
        work.groupby("period")
        .apply(
            lambda group: pd.Series(
                {
                    "record_weighted_center": weighted_mean(
                        group["observer_mean_latitude"],
                        group["n_records"],
                    ),
                    "equal_observer_center": float(
                        group[
                            "observer_mean_latitude"
                        ].mean()
                    ),
                    "n_observers": int(
                        group["observer_label"].nunique()
                    ),
                    "n_records": int(
                        group["n_records"].sum()
                    ),
                }
            ),
            include_groups=False,
        )
    )

    if not {"early", "late"}.issubset(
        period_centers.index
    ):
        raise ValueError(
            "Both early and late periods are required."
        )

    profile_period_df = (
        work.groupby(
            ["period", "profile_id"],
            as_index=False,
        )
        .agg(
            profile_mean_latitude=(
                "observer_mean_latitude",
                "mean",
            ),
            profile_observers=(
                "observer_label",
                "nunique",
            ),
            profile_records=(
                "n_records",
                "sum",
            ),
        )
    )

    profile_presence = (
        profile_period_df.groupby("profile_id")[
            "period"
        ].nunique()
    )

    observed_common_profiles = set(
        profile_presence[
            profile_presence == 2
        ].index
    )

    common_profiles = (
        set(fixed_common_profiles)
        if fixed_common_profiles is not None
        else observed_common_profiles
    )

    common_profile_df = profile_period_df[
        profile_period_df["profile_id"].isin(
            common_profiles
        )
    ].copy()

    if common_profile_df.empty:
        raise ValueError(
            "No common observer-profile strata are available."
        )

    if fixed_reference_weights is None:
        pooled_counts = (
            common_profile_df.groupby(
                "profile_id"
            )["profile_observers"]
            .sum()
        )

        reference_weights = (
            pooled_counts / pooled_counts.sum()
        ).to_dict()
    else:
        reference_weights = dict(
            fixed_reference_weights
        )

    standardized_centers = {}
    observer_coverage = {}
    record_coverage = {}

    for period in ["early", "late"]:
        period_all = work[
            work["period"] == period
        ]
        period_common = common_profile_df[
            common_profile_df["period"] == period
        ].copy()

        period_common["reference_weight"] = (
            period_common["profile_id"].map(
                reference_weights
            )
        )

        period_common = period_common.dropna(
            subset=["reference_weight"]
        )

        if period_common.empty:
            standardized_centers[period] = np.nan
        else:
            normalized_weights = (
                period_common["reference_weight"]
                / period_common[
                    "reference_weight"
                ].sum()
            )

            standardized_centers[period] = float(
                np.average(
                    period_common[
                        "profile_mean_latitude"
                    ],
                    weights=normalized_weights,
                )
            )

        common_mask = period_all[
            "profile_id"
        ].isin(common_profiles)

        observer_coverage[period] = float(
            common_mask.mean()
        )

        total_records = period_all[
            "n_records"
        ].sum()

        record_coverage[period] = float(
            period_all.loc[
                common_mask,
                "n_records",
            ].sum()
            / total_records
        ) if total_records > 0 else np.nan

    record_early = period_centers.loc[
        "early",
        "record_weighted_center",
    ]
    record_late = period_centers.loc[
        "late",
        "record_weighted_center",
    ]

    observer_early = period_centers.loc[
        "early",
        "equal_observer_center",
    ]
    observer_late = period_centers.loc[
        "late",
        "equal_observer_center",
    ]

    standardized_early = standardized_centers[
        "early"
    ]
    standardized_late = standardized_centers[
        "late"
    ]

    total_latitude_change = (
        record_late - record_early
    )

    activity_component = (
        (record_late - observer_late)
        - (record_early - observer_early)
    )

    profile_composition_component = (
        (observer_late - standardized_late)
        - (observer_early - standardized_early)
    )

    within_profile_component = (
        standardized_late - standardized_early
    )

    additive_residual = (
        total_latitude_change
        - activity_component
        - profile_composition_component
        - within_profile_component
    )

    return {
        "total_shift_km_decade": to_km_decade(
            total_latitude_change
        ),
        "activity_weighting_component_km_decade": to_km_decade(
            activity_component
        ),
        "profile_composition_component_km_decade": to_km_decade(
            profile_composition_component
        ),
        "within_profile_turnover_component_km_decade": to_km_decade(
            within_profile_component
        ),
        "additive_residual_km_decade": to_km_decade(
            additive_residual
        ),
        "record_weighted_early_latitude": (
            record_early
        ),
        "record_weighted_late_latitude": (
            record_late
        ),
        "equal_observer_early_latitude": (
            observer_early
        ),
        "equal_observer_late_latitude": (
            observer_late
        ),
        "standardized_early_latitude": (
            standardized_early
        ),
        "standardized_late_latitude": (
            standardized_late
        ),
        "n_common_profiles": len(
            common_profiles
        ),
        "early_observer_coverage": (
            observer_coverage["early"]
        ),
        "late_observer_coverage": (
            observer_coverage["late"]
        ),
        "early_record_coverage": (
            record_coverage["early"]
        ),
        "late_record_coverage": (
            record_coverage["late"]
        ),
        "common_profiles": sorted(
            common_profiles
        ),
        "reference_weights": (
            reference_weights
        ),
    }


def bootstrap_decomposition(
    observer_period_df,
    profile_columns,
    observed_result,
    n_bootstrap,
    rng,
    label,
):
    early_df = observer_period_df[
        observer_period_df["period"] == "early"
    ].reset_index(drop=True)

    late_df = observer_period_df[
        observer_period_df["period"] == "late"
    ].reset_index(drop=True)

    rows = []

    for replicate in tqdm(
        range(n_bootstrap),
        desc=f"Bootstrap {label}",
        unit="replicate",
    ):
        early_sample = early_df.iloc[
            rng.integers(
                0,
                len(early_df),
                size=len(early_df),
            )
        ].copy()

        late_sample = late_df.iloc[
            rng.integers(
                0,
                len(late_df),
                size=len(late_df),
            )
        ].copy()

        sample = pd.concat(
            [early_sample, late_sample],
            ignore_index=True,
        )

        try:
            result = compute_decomposition(
                sample,
                profile_columns,
                fixed_common_profiles=(
                    observed_result[
                        "common_profiles"
                    ]
                ),
                fixed_reference_weights=(
                    observed_result[
                        "reference_weights"
                    ]
                ),
            )
        except Exception:
            continue

        rows.append(
            {
                "replicate": replicate,
                "total_shift_km_decade": (
                    result[
                        "total_shift_km_decade"
                    ]
                ),
                "activity_weighting_component_km_decade": (
                    result[
                        "activity_weighting_component_km_decade"
                    ]
                ),
                "profile_composition_component_km_decade": (
                    result[
                        "profile_composition_component_km_decade"
                    ]
                ),
                "within_profile_turnover_component_km_decade": (
                    result[
                        "within_profile_turnover_component_km_decade"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def profile_permutation_null(
    observer_period_df,
    profile_columns,
    observed_result,
    n_permutations,
    rng,
    label,
):
    work = observer_period_df.copy()
    work["profile_id"] = create_profile_id(
        work,
        profile_columns,
    )

    work = work[
        work["profile_id"].isin(
            observed_result["common_profiles"]
        )
    ].copy()

    stratum_data = []

    for profile_id, group in work.groupby(
        "profile_id"
    ):
        early_count = int(
            (group["period"] == "early").sum()
        )
        late_count = int(
            (group["period"] == "late").sum()
        )

        if early_count == 0 or late_count == 0:
            continue

        stratum_data.append(
            {
                "profile_id": profile_id,
                "latitudes": group[
                    "observer_mean_latitude"
                ].to_numpy(dtype=float),
                "early_count": early_count,
                "late_count": late_count,
                "reference_weight": observed_result[
                    "reference_weights"
                ][profile_id],
            }
        )

    observed_value = observed_result[
        "within_profile_turnover_component_km_decade"
    ]

    null_values = np.empty(
        n_permutations,
        dtype=float,
    )

    for replicate in tqdm(
        range(n_permutations),
        desc=f"Profile permutation {label}",
        unit="replicate",
    ):
        early_means = []
        late_means = []
        weights = []

        for item in stratum_data:
            values = item["latitudes"]
            permutation = rng.permutation(
                len(values)
            )

            early_indices = permutation[
                : item["early_count"]
            ]
            late_indices = permutation[
                item["early_count"] :
            ]

            early_means.append(
                float(
                    values[
                        early_indices
                    ].mean()
                )
            )
            late_means.append(
                float(
                    values[
                        late_indices
                    ].mean()
                )
            )
            weights.append(
                item["reference_weight"]
            )

        weights = np.asarray(
            weights,
            dtype=float,
        )
        weights = weights / weights.sum()

        early_center = float(
            np.average(
                early_means,
                weights=weights,
            )
        )
        late_center = float(
            np.average(
                late_means,
                weights=weights,
            )
        )

        null_values[replicate] = to_km_decade(
            late_center - early_center
        )

    empirical_p = float(
        (
            1
            + np.sum(
                np.abs(null_values)
                >= abs(observed_value)
            )
        )
        / (n_permutations + 1)
    )

    return null_values, empirical_p


def confidence_interval(
    bootstrap_df,
    column,
):
    values = pd.to_numeric(
        bootstrap_df[column],
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan, np.nan

    return (
        float(values.quantile(0.025)),
        float(values.quantile(0.975)),
    )


def print_failure(status, error, next_step):
    append_readme(
        f"""

## PA observer-community decomposition failure — {RUN_UTC}
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
    # 1. Load and prepare the observer-period table
    # --------------------------------------------------------
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Cell 10B output: {INPUT_PATH}"
        )

    occurrence_df = pd.read_parquet(
        INPUT_PATH
    )

    occurrence_df["year"] = pd.to_numeric(
        occurrence_df["year"],
        errors="coerce",
    )
    occurrence_df[
        "decimalLatitude"
    ] = pd.to_numeric(
        occurrence_df["decimalLatitude"],
        errors="coerce",
    )
    occurrence_df[
        "decimalLongitude"
    ] = pd.to_numeric(
        occurrence_df["decimalLongitude"],
        errors="coerce",
    )

    occurrence_df = occurrence_df[
        occurrence_df["year"].isin(
            EARLY_YEARS + LATE_YEARS
        )
        & occurrence_df[
            "decimalLatitude"
        ].notna()
        & occurrence_df[
            "decimalLongitude"
        ].notna()
    ].copy()

    occurrence_df["year"] = (
        occurrence_df["year"].astype(int)
    )

    occurrence_df["period"] = np.where(
        occurrence_df["year"].isin(
            EARLY_YEARS
        ),
        "early",
        "late",
    )

    occurrence_df["observer_label"] = (
        normalize_observer_label(
            occurrence_df["recordedBy"]
        )
    )

    occurrence_df = occurrence_df.dropna(
        subset=["observer_label"]
    ).copy()

    occurrence_df["footprint_lat_index"] = (
        np.floor(
            occurrence_df[
                "decimalLatitude"
            ]
            / FOOTPRINT_GRID_RESOLUTION
        ).astype("int32")
    )
    occurrence_df["footprint_lon_index"] = (
        np.floor(
            occurrence_df[
                "decimalLongitude"
            ]
            / FOOTPRINT_GRID_RESOLUTION
        ).astype("int32")
    )
    occurrence_df["footprint_cell"] = (
        occurrence_df[
            "footprint_lat_index"
        ].astype(str)
        + "_"
        + occurrence_df[
            "footprint_lon_index"
        ].astype(str)
    )

    observer_period_df = (
        occurrence_df.groupby(
            ["period", "observer_label"],
            as_index=False,
        )
        .agg(
            n_records=("gbifID", "size"),
            n_species=(
                "fia_species_code",
                "nunique",
            ),
            n_cells=(
                "footprint_cell",
                "nunique",
            ),
            n_years=("year", "nunique"),
            observer_mean_latitude=(
                "decimalLatitude",
                "mean",
            ),
            observer_mean_longitude=(
                "decimalLongitude",
                "mean",
            ),
            observer_mean_year=(
                "year",
                "mean",
            ),
        )
    )

    observer_period_df["record_bin"] = (
        bin_records(
            observer_period_df[
                "n_records"
            ]
        )
    )
    observer_period_df["species_bin"] = (
        bin_breadth(
            observer_period_df[
                "n_species"
            ]
        )
    )
    observer_period_df["cell_bin"] = (
        bin_breadth(
            observer_period_df[
                "n_cells"
            ]
        )
    )
    observer_period_df["year_bin"] = (
        bin_breadth(
            observer_period_df[
                "n_years"
            ]
        )
    )

    observer_period_df.to_parquet(
        OUT_DIR / "observer_period_profiles.parquet",
        index=False,
    )

    print("\nOBSERVER-PERIOD PROFILE SUMMARY")
    display(
        observer_period_df.groupby(
            "period",
            as_index=False,
        )
        .agg(
            n_observers=(
                "observer_label",
                "nunique",
            ),
            n_records=(
                "n_records",
                "sum",
            ),
            median_records=(
                "n_records",
                "median",
            ),
            median_species=(
                "n_species",
                "median",
            ),
            median_cells=(
                "n_cells",
                "median",
            ),
            median_years=(
                "n_years",
                "median",
            ),
            mean_latitude=(
                "observer_mean_latitude",
                "mean",
            ),
        )
    )

    # --------------------------------------------------------
    # 2. Observer entry/exit geometry
    # --------------------------------------------------------
    early_observers = set(
        observer_period_df.loc[
            observer_period_df["period"] == "early",
            "observer_label",
        ]
    )
    late_observers = set(
        observer_period_df.loc[
            observer_period_df["period"] == "late",
            "observer_label",
        ]
    )

    common_observers = (
        early_observers & late_observers
    )
    exiting_observers = (
        early_observers - late_observers
    )
    entering_observers = (
        late_observers - early_observers
    )

    turnover_rows = []

    for period, category, observer_set in [
        (
            "early",
            "common_observers",
            common_observers,
        ),
        (
            "late",
            "common_observers",
            common_observers,
        ),
        (
            "early",
            "exiting_observers",
            exiting_observers,
        ),
        (
            "late",
            "entering_observers",
            entering_observers,
        ),
    ]:
        subset = observer_period_df[
            (observer_period_df["period"] == period)
            & observer_period_df[
                "observer_label"
            ].isin(observer_set)
        ]

        turnover_rows.append(
            {
                "period": period,
                "category": category,
                "n_observers": len(subset),
                "n_records": int(
                    subset["n_records"].sum()
                ),
                "equal_observer_latitude": (
                    float(
                        subset[
                            "observer_mean_latitude"
                        ].mean()
                    )
                    if len(subset)
                    else np.nan
                ),
                "record_weighted_latitude": (
                    weighted_mean(
                        subset[
                            "observer_mean_latitude"
                        ],
                        subset["n_records"],
                    )
                    if len(subset)
                    else np.nan
                ),
            }
        )

    turnover_geometry_df = pd.DataFrame(
        turnover_rows
    )

    turnover_geometry_df.to_csv(
        OUT_DIR / "observer_turnover_geometry.csv",
        index=False,
    )

    print("\nOBSERVER TURNOVER GEOMETRY")
    display(turnover_geometry_df)

    # --------------------------------------------------------
    # 3. Multiple observer-profile specifications
    # --------------------------------------------------------
    profile_specs = {
        "activity_only": [
            "record_bin",
        ],
        "activity_space": [
            "record_bin",
            "cell_bin",
        ],
        "activity_taxa": [
            "record_bin",
            "species_bin",
        ],
        "activity_space_taxa": [
            "record_bin",
            "cell_bin",
            "species_bin",
        ],
    }

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    decomposition_rows = []
    bootstrap_outputs = {}
    permutation_outputs = {}

    for specification, profile_columns in profile_specs.items():
        print(
            "\nPROFILE SPECIFICATION:",
            specification,
            profile_columns,
        )

        observed = compute_decomposition(
            observer_period_df,
            profile_columns,
        )

        bootstrap_df = bootstrap_decomposition(
            observer_period_df,
            profile_columns,
            observed,
            N_BOOTSTRAP,
            rng,
            specification,
        )

        null_values, permutation_p = (
            profile_permutation_null(
                observer_period_df,
                profile_columns,
                observed,
                N_PROFILE_PERMUTATIONS,
                rng,
                specification,
            )
        )

        bootstrap_outputs[
            specification
        ] = bootstrap_df

        permutation_outputs[
            specification
        ] = null_values

        bootstrap_df.to_parquet(
            OUT_DIR
            / f"bootstrap_{specification}.parquet",
            index=False,
        )

        pd.DataFrame(
            {
                "within_profile_null_km_decade": (
                    null_values
                )
            }
        ).to_parquet(
            OUT_DIR
            / f"profile_permutation_{specification}.parquet",
            index=False,
        )

        row = {
            "specification": specification,
            "profile_columns": "|".join(
                profile_columns
            ),
            **{
                key: value
                for key, value in observed.items()
                if key not in {
                    "common_profiles",
                    "reference_weights",
                }
            },
            "within_profile_permutation_p": (
                permutation_p
            ),
        }

        for component in [
            "total_shift_km_decade",
            "activity_weighting_component_km_decade",
            "profile_composition_component_km_decade",
            "within_profile_turnover_component_km_decade",
        ]:
            low, high = confidence_interval(
                bootstrap_df,
                component,
            )
            row[
                f"{component}_bootstrap_low"
            ] = low
            row[
                f"{component}_bootstrap_high"
            ] = high

        decomposition_rows.append(row)

    decomposition_df = pd.DataFrame(
        decomposition_rows
    )

    decomposition_df.to_csv(
        OUT_DIR / "observer_community_decomposition.csv",
        index=False,
    )

    print("\nOBSERVER-COMMUNITY DECOMPOSITION")
    display(decomposition_df)

    # --------------------------------------------------------
    # 4. Top-1% observer-activity trimming sensitivity
    # --------------------------------------------------------
    trimmed_parts = []

    for period, group in observer_period_df.groupby(
        "period"
    ):
        threshold = float(
            group["n_records"].quantile(0.99)
        )

        trimmed = group[
            group["n_records"] <= threshold
        ].copy()

        trimmed_parts.append(trimmed)

    trimmed_df = pd.concat(
        trimmed_parts,
        ignore_index=True,
    )

    trimmed_result = compute_decomposition(
        trimmed_df,
        profile_specs["activity_space"],
    )

    main_result = decomposition_df[
        decomposition_df["specification"]
        == "activity_space"
    ].iloc[0]

    trimming_sensitivity_df = pd.DataFrame(
        [
            {
                "analysis": "all_observers",
                "total_shift_km_decade": main_result[
                    "total_shift_km_decade"
                ],
                "activity_weighting_component_km_decade": main_result[
                    "activity_weighting_component_km_decade"
                ],
                "profile_composition_component_km_decade": main_result[
                    "profile_composition_component_km_decade"
                ],
                "within_profile_turnover_component_km_decade": main_result[
                    "within_profile_turnover_component_km_decade"
                ],
            },
            {
                "analysis": "trim_top_1pct_activity",
                "total_shift_km_decade": trimmed_result[
                    "total_shift_km_decade"
                ],
                "activity_weighting_component_km_decade": trimmed_result[
                    "activity_weighting_component_km_decade"
                ],
                "profile_composition_component_km_decade": trimmed_result[
                    "profile_composition_component_km_decade"
                ],
                "within_profile_turnover_component_km_decade": trimmed_result[
                    "within_profile_turnover_component_km_decade"
                ],
            },
        ]
    )

    trimming_sensitivity_df.to_csv(
        OUT_DIR / "top_activity_trimming_sensitivity.csv",
        index=False,
    )

    print("\nTOP-ACTIVITY TRIMMING SENSITIVITY")
    display(trimming_sensitivity_df)

    # --------------------------------------------------------
    # 5. Decision
    # --------------------------------------------------------
    coverage_columns = [
        "early_observer_coverage",
        "late_observer_coverage",
        "early_record_coverage",
        "late_record_coverage",
    ]

    decomposition_df["coverage_pass"] = (
        decomposition_df[
            [
                "early_observer_coverage",
                "late_observer_coverage",
            ]
        ].min(axis=1)
        >= MIN_COMMON_PROFILE_OBSERVER_COVERAGE
    ) & (
        decomposition_df[
            [
                "early_record_coverage",
                "late_record_coverage",
            ]
        ].min(axis=1)
        >= MIN_COMMON_PROFILE_RECORD_COVERAGE
    )

    decomposition_df["activity_material"] = (
        decomposition_df[
            "activity_weighting_component_km_decade"
        ].abs()
        >= MATERIAL_COMPONENT_KM_DECADE
    )

    decomposition_df[
        "profile_composition_material"
    ] = (
        decomposition_df[
            "profile_composition_component_km_decade"
        ].abs()
        >= MATERIAL_COMPONENT_KM_DECADE
    )

    decomposition_df[
        "within_profile_material"
    ] = (
        decomposition_df[
            "within_profile_turnover_component_km_decade"
        ].abs()
        >= MATERIAL_COMPONENT_KM_DECADE
    )

    decomposition_df[
        "within_profile_supported"
    ] = (
        decomposition_df[
            "within_profile_material"
        ]
        & (
            decomposition_df[
                "within_profile_permutation_p"
            ]
            <= 0.05
        )
    )

    stable_specs = decomposition_df[
        decomposition_df["coverage_pass"]
    ].copy()

    n_stable_specs = len(
        stable_specs
    )

    activity_consistency = (
        float(
            stable_specs[
                "activity_weighting_component_km_decade"
            ].apply(np.sign).value_counts(
                normalize=True
            ).max()
        )
        if n_stable_specs
        else np.nan
    )

    profile_consistency = (
        float(
            stable_specs[
                "profile_composition_component_km_decade"
            ].apply(np.sign).value_counts(
                normalize=True
            ).max()
        )
        if n_stable_specs
        else np.nan
    )

    within_consistency = (
        float(
            stable_specs[
                "within_profile_turnover_component_km_decade"
            ].apply(np.sign).value_counts(
                normalize=True
            ).max()
        )
        if n_stable_specs
        else np.nan
    )

    activity_supported = bool(
        n_stable_specs
        >= MIN_STABLE_PROFILE_SPECIFICATIONS
        and (
            stable_specs[
                "activity_material"
            ].mean()
            >= 0.50
        )
        and activity_consistency
        >= 0.75
    )

    profile_composition_supported = bool(
        n_stable_specs
        >= MIN_STABLE_PROFILE_SPECIFICATIONS
        and (
            stable_specs[
                "profile_composition_material"
            ].mean()
            >= 0.50
        )
        and profile_consistency
        >= 0.75
    )

    within_profile_supported = bool(
        n_stable_specs
        >= MIN_STABLE_PROFILE_SPECIFICATIONS
        and (
            stable_specs[
                "within_profile_supported"
            ].mean()
            >= 0.50
        )
        and within_consistency
        >= 0.75
    )

    trimming_sign_stable = all(
        np.sign(
            trimming_sensitivity_df.loc[
                0,
                component,
            ]
        )
        == np.sign(
            trimming_sensitivity_df.loc[
                1,
                component,
            ]
        )
        for component in [
            "activity_weighting_component_km_decade",
            "profile_composition_component_km_decade",
            "within_profile_turnover_component_km_decade",
        ]
        if np.isfinite(
            trimming_sensitivity_df.loc[
                0,
                component,
            ]
        )
        and np.isfinite(
            trimming_sensitivity_df.loc[
                1,
                component,
            ]
        )
    )

    if (
        n_stable_specs
        < MIN_STABLE_PROFILE_SPECIFICATIONS
    ):
        status = (
            "OBSERVER_PROFILE_OVERLAP_INSUFFICIENT"
        )
        passed = False
        next_step = (
            "Observer-profile strata do not overlap enough between "
            "periods. Stop profile decomposition and treat turnover "
            "only as a descriptive system change."
        )
    elif (
        activity_supported
        and profile_composition_supported
        and within_profile_supported
        and trimming_sign_stable
    ):
        status = (
            "MIXED_TURNOVER_ACTIVITY_AND_SPATIAL_REPLACEMENT"
        )
        passed = True
        next_step = (
            "All three observation-system mechanisms are material. "
            "Build the species-level correction using equal-observer, "
            "profile-standardized and equal-cell estimands."
        )
    elif (
        within_profile_supported
        and trimming_sign_stable
    ):
        status = (
            "WITHIN_PROFILE_SPATIAL_TURNOVER_DOMINATES"
        )
        passed = True
        next_step = (
            "Observers with comparable activity profiles occupy "
            "different spatial centers over time. Test whether species "
            "trends change after observer-profile standardization."
        )
    elif (
        activity_supported
        or profile_composition_supported
    ):
        status = (
            "OBSERVER_ACTIVITY_OR_PROFILE_COMPOSITION_DOMINATES"
        )
        passed = True
        next_step = (
            "The footprint shift is mainly associated with observer "
            "activity weights or observer-profile composition. Build "
            "species-level sensitivity estimates using those locked "
            "standardizations."
        )
    else:
        status = (
            "TURNOVER_DESCRIPTIVE_BUT_COMPONENTS_UNSTABLE"
        )
        passed = False
        next_step = (
            "Observer turnover is large, but the component decomposition "
            "is not stable across profile definitions. Do not claim a "
            "specific turnover mechanism."
        )

    # Save decision-enriched table.
    decomposition_df.to_csv(
        OUT_DIR / "observer_community_decomposition.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Visual previews
    # --------------------------------------------------------
    component_plot_df = decomposition_df[
        [
            "specification",
            "activity_weighting_component_km_decade",
            "profile_composition_component_km_decade",
            "within_profile_turnover_component_km_decade",
        ]
    ].copy()

    x = np.arange(
        len(component_plot_df)
    )
    width = 0.24

    plt.figure(figsize=(11, 6))
    plt.bar(
        x - width,
        component_plot_df[
            "activity_weighting_component_km_decade"
        ],
        width=width,
        label="Activity weighting",
    )
    plt.bar(
        x,
        component_plot_df[
            "profile_composition_component_km_decade"
        ],
        width=width,
        label="Profile composition",
    )
    plt.bar(
        x + width,
        component_plot_df[
            "within_profile_turnover_component_km_decade"
        ],
        width=width,
        label="Within-profile spatial turnover",
    )
    plt.axhline(0)
    plt.xticks(
        x,
        component_plot_df[
            "specification"
        ],
        rotation=20,
    )
    plt.ylabel("Component contribution (km/decade)")
    plt.xlabel("Observer-profile specification")
    plt.title(
        "PA observation-footprint shift decomposition"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    turnover_plot_df = turnover_geometry_df.dropna(
        subset=["equal_observer_latitude"]
    ).copy()

    labels = (
        turnover_plot_df["period"]
        + ": "
        + turnover_plot_df["category"]
    )

    plt.figure(figsize=(10, 5))
    plt.bar(
        labels,
        turnover_plot_df[
            "equal_observer_latitude"
        ],
    )
    plt.ylabel("Equal-observer latitude center")
    plt.xlabel("Observer category")
    plt.title(
        "Spatial geometry of entering, exiting and common observers"
    )
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 7. README
    # --------------------------------------------------------
    compact_decomposition = (
        decomposition_df[
            [
                "specification",
                "total_shift_km_decade",
                "activity_weighting_component_km_decade",
                "profile_composition_component_km_decade",
                "within_profile_turnover_component_km_decade",
                "within_profile_permutation_p",
                "early_observer_coverage",
                "late_observer_coverage",
                "early_record_coverage",
                "late_record_coverage",
                "coverage_pass",
            ]
        ]
        .round(6)
        .to_dict("records")
    )

    append_readme(
        f"""

## PA observer-community turnover decomposition — {RUN_UTC}

### Reason for analysis
The personal-observer trajectory branch was closed because exact observer overlap
was only about 2% and matched-observer estimates were unstable. This cell instead
analyzes the observer community as a changing observation system.

### Data
- State: {STATE}
- Early years: {EARLY_YEARS}
- Late years: {LATE_YEARS}
- Expanded tree records: {len(occurrence_df):,}
- Observer-period rows: {len(observer_period_df):,}
- Early observers: {len(early_observers):,}
- Late observers: {len(late_observers):,}
- Common observers: {len(common_observers):,}
- Entering observers: {len(entering_observers):,}
- Exiting observers: {len(exiting_observers):,}
- Observer Jaccard:
  {len(common_observers) / len(early_observers | late_observers)}
- GPU: not used

### Additive decomposition
Total record-weighted footprint shift is decomposed into:
1. observer activity-weighting change;
2. observer-profile composition change;
3. spatial replacement within comparable observer-profile strata.

### Profile specifications
- activity only;
- activity + spatial breadth;
- activity + taxonomic breadth;
- activity + spatial breadth + taxonomic breadth.

### Robustness
- Observer-cluster bootstrap replicates: {N_BOOTSTRAP}
- Within-profile period-label permutations:
  {N_PROFILE_PERMUTATIONS}
- Top 1% observer-activity trimming sensitivity:
  {trimming_sign_stable}

### Decision
- Stable profile specifications: {n_stable_specs}
- Activity weighting supported: {activity_supported}
- Profile composition supported:
  {profile_composition_supported}
- Within-profile spatial turnover supported:
  {within_profile_supported}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
This analysis decomposes the selected tree-record footprint. Observer profiles
are defined from record count, spatial breadth and taxonomic breadth. It does not
identify individual movement, causal platform effects, or the footprint of all
iNaturalist observations. recordedBy remains an exact normalized label.
"""
    )

    # --------------------------------------------------------
    # 8. Compact output block
    # --------------------------------------------------------
    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATE: {STATE}")
    print(f"EARLY_YEARS: {EARLY_YEARS}")
    print(f"LATE_YEARS: {LATE_YEARS}")
    print(
        f"N_EXPANDED_TREE_RECORDS: "
        f"{len(occurrence_df):,}"
    )
    print(
        f"N_OBSERVER_PERIOD_ROWS: "
        f"{len(observer_period_df):,}"
    )
    print(
        f"N_EARLY_OBSERVERS: "
        f"{len(early_observers):,}"
    )
    print(
        f"N_LATE_OBSERVERS: "
        f"{len(late_observers):,}"
    )
    print(
        f"N_COMMON_OBSERVERS: "
        f"{len(common_observers):,}"
    )
    print(
        f"N_ENTERING_OBSERVERS: "
        f"{len(entering_observers):,}"
    )
    print(
        f"N_EXITING_OBSERVERS: "
        f"{len(exiting_observers):,}"
    )
    print(
        f"OBSERVER_JACCARD: "
        f"{len(common_observers) / len(early_observers | late_observers)}"
    )
    print(
        "TURNOVER_GEOMETRY: "
        + json.dumps(
            turnover_geometry_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "DECOMPOSITION_RESULTS: "
        + json.dumps(
            compact_decomposition,
            ensure_ascii=False,
        )
    )
    print(
        "TRIMMING_SENSITIVITY: "
        + json.dumps(
            trimming_sensitivity_df.round(6).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        f"N_STABLE_PROFILE_SPECIFICATIONS: "
        f"{n_stable_specs}"
    )
    print(
        f"ACTIVITY_WEIGHTING_SUPPORTED: "
        f"{activity_supported}"
    )
    print(
        f"PROFILE_COMPOSITION_SUPPORTED: "
        f"{profile_composition_supported}"
    )
    print(
        f"WITHIN_PROFILE_TURNOVER_SUPPORTED: "
        f"{within_profile_supported}"
    )
    print(
        f"TOP_ACTIVITY_TRIMMING_SIGN_STABLE: "
        f"{trimming_sign_stable}"
    )
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except Exception as exc:
    print_failure(
        "CELL11A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Send this COPY block and the last displayed output. "
            "Do not download more taxa."
        ),
    )
