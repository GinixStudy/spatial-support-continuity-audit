
# Cell 16B — Formal registered-data parity rerun for PA, VA and NC
#
# Goal
# ----
# Before final confirmatory inference, verify that the registered GBIF download
# reproduces the deterministic species-level inputs previously built from the
# temporary GBIF Search API.
#
# For each state:
# 1. Apply the locked quality filter.
# 2. Apply identical species × event-day × 0.05° spatial thinning.
# 3. Recompute raw and target-effort-standardized shifts at 0.25°, 0.5° and 1.0°.
# 4. Recompute only the two locked features:
#       cell_jaccard
#       abs_log_observer_growth
# 5. Compare every available feature/gain value with the temporary API table.
#
# No permutation or bootstrap is run here.
# Final three-region inference is allowed only if parity passes.

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 240)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 340)

# ============================================================
# Locked configuration
# ============================================================
STATES = ["PA", "VA", "NC"]

EARLY_YEARS = list(range(2013, 2019))
LATE_YEARS = list(range(2020, 2026))
EARLY_MID_YEAR = float(np.mean(EARLY_YEARS))
LATE_MID_YEAR = float(np.mean(LATE_YEARS))

THINNING_RESOLUTION = 0.05
GRID_RESOLUTIONS = [0.25, 0.50, 1.00]
PRIMARY_GRID_RESOLUTION = 0.50

MAX_COORDINATE_UNCERTAINTY_M = 10_000
MIN_TARGET_EFFORT_PER_CELL_PERIOD = 10
PARITY_TOLERANCE = 1e-6

STATE_BOUNDS = {
    "PA": {
        "lat_min": 39.5,
        "lat_max": 42.6,
        "lon_min": -81.0,
        "lon_max": -74.4,
    },
    "VA": {
        "lat_min": 36.4,
        "lat_max": 39.7,
        "lon_min": -83.8,
        "lon_max": -75.0,
    },
    "NC": {
        "lat_min": 33.7,
        "lat_max": 36.7,
        "lon_min": -84.5,
        "lon_max": -75.2,
    },
}

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
DERIVED_DIR = BASE_DIR / "derived"

FORMAL_DIR = (
    DERIVED_DIR
    / "registered_gbif_three_region_download"
    / "normalized"
)

OUT_DIR = (
    DERIVED_DIR
    / "registered_gbif_formal_parity"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORMAL_PATHS = {
    state: FORMAL_DIR / f"{state}_registered_gbif.parquet"
    for state in STATES
}

TEMP_TABLE_PREFERRED = {
    "PA": (
        DERIVED_DIR
        / "pa_correction_transportability_audit"
        / "species_transportability_features.parquet"
    ),
    "VA": (
        DERIVED_DIR
        / "va_candidate_confirmation"
        / "va_locked_candidate_species_table.parquet"
    ),
}

FIA_REFERENCE_PREFERRED = {
    "PA": (
        DERIVED_DIR
        / "pa_species_external_validation"
        / "fia_same_window_species_reference.parquet"
    ),
    "VA": (
        DERIVED_DIR
        / "va_confirmation_dataset"
        / "va_fia_same_window_reference.parquet"
    ),
    "NC": (
        DERIVED_DIR
        / "nc_final_confirmation_dataset"
        / "nc_fia_same_window_reference.parquet"
    ),
}

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("FORMAL_DIR:", FORMAL_DIR)
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


def parquet_columns(path):
    return set(
        pq.ParquetFile(path).schema.names
    )


def find_parquet_with_columns(
    state,
    preferred_paths,
    required_columns,
    path_terms,
):
    for path in preferred_paths:
        if (
            path is not None
            and path.exists()
            and required_columns.issubset(
                parquet_columns(path)
            )
        ):
            return path

    candidates = []

    for path in DERIVED_DIR.rglob("*.parquet"):
        lower_path = str(path).lower()

        if state.lower() not in lower_path:
            continue

        if not all(
            term.lower() in lower_path
            for term in path_terms
        ):
            continue

        try:
            columns = parquet_columns(path)
        except Exception:
            continue

        if required_columns.issubset(
            columns
        ):
            candidates.append(path)

    if not candidates:
        # Second-pass search: use schema and state only.
        for path in DERIVED_DIR.rglob("*.parquet"):
            lower_path = str(path).lower()

            if state.lower() not in lower_path:
                continue

            try:
                columns = parquet_columns(path)
            except Exception:
                continue

            if required_columns.issubset(
                columns
            ):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No {state} Parquet with columns "
            f"{sorted(required_columns)} was found."
        )

    candidates = sorted(
        candidates,
        key=lambda path: (
            len(str(path)),
            str(path),
        ),
    )

    return candidates[0]


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

    return normalized.mask(
        normalized.isin(blocked)
    )


def build_event_day(frame):
    if "eventDate" in frame.columns:
        event_time = pd.to_datetime(
            frame["eventDate"],
            errors="coerce",
            utc=True,
        )
    else:
        event_time = pd.Series(
            pd.NaT,
            index=frame.index,
            dtype="datetime64[ns, UTC]",
        )

    fallback_source = {
        "year": pd.to_numeric(
            frame["year"],
            errors="coerce",
        )
    }

    for column in ["month", "day"]:
        fallback_source[column] = (
            pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            if column in frame.columns
            else pd.Series(
                np.nan,
                index=frame.index,
            )
        )

    fallback = pd.to_datetime(
        pd.DataFrame(
            fallback_source
        ),
        errors="coerce",
        utc=True,
    )

    final_time = event_time.fillna(
        fallback
    )

    result = (
        final_time.dt.strftime(
            "%Y-%m-%d"
        )
        .astype("string")
    )

    missing = result.isna()

    result.loc[missing] = (
        "unknown_"
        + frame.loc[
            missing,
            "gbifID",
        ].astype(str)
    )

    return result


def set_jaccard(first, second):
    union = first | second

    if not union:
        return np.nan

    return float(
        len(first & second)
        / len(union)
    )


def shift_km_decade(
    early_latitude,
    late_latitude,
):
    if (
        not np.isfinite(
            early_latitude
        )
        or not np.isfinite(
            late_latitude
        )
    ):
        return np.nan

    interval = (
        LATE_MID_YEAR
        - EARLY_MID_YEAR
    )

    return float(
        (
            late_latitude
            - early_latitude
        )
        * 111.32
        * 10
        / interval
    )


def prepare_formal_state(
    state,
    path,
):
    frame = pd.read_parquet(path)

    for column in [
        "fia_species_code",
        "year",
        "month",
        "day",
        "decimalLatitude",
        "decimalLongitude",
        "coordinateUncertaintyInMeters",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    frame["gbifID"] = (
        frame["gbifID"]
        .astype("string")
    )

    bounds = STATE_BOUNDS[state]

    valid = (
        frame["year"].isin(
            EARLY_YEARS
            + LATE_YEARS
        )
        & frame[
            "fia_species_code"
        ].notna()
        & frame[
            "decimalLatitude"
        ].between(
            bounds["lat_min"],
            bounds["lat_max"],
        )
        & frame[
            "decimalLongitude"
        ].between(
            bounds["lon_min"],
            bounds["lon_max"],
        )
        & (
            frame[
                "coordinateUncertaintyInMeters"
            ].isna()
            | frame[
                "coordinateUncertaintyInMeters"
            ].le(
                MAX_COORDINATE_UNCERTAINTY_M
            )
        )
    )

    clean = frame.loc[
        valid
    ].copy()

    clean[
        "fia_species_code"
    ] = clean[
        "fia_species_code"
    ].astype(int)

    clean["year"] = clean[
        "year"
    ].astype(int)

    clean["event_day"] = (
        build_event_day(clean)
    )

    clean[
        "thin_lat_index"
    ] = np.floor(
        clean[
            "decimalLatitude"
        ]
        / THINNING_RESOLUTION
    ).astype("int32")

    clean[
        "thin_lon_index"
    ] = np.floor(
        clean[
            "decimalLongitude"
        ]
        / THINNING_RESOLUTION
    ).astype("int32")

    clean[
        "uncertainty_sort"
    ] = clean[
        "coordinateUncertaintyInMeters"
    ].fillna(np.inf)

    thinned = (
        clean.sort_values(
            [
                "fia_species_code",
                "event_day",
                "thin_lat_index",
                "thin_lon_index",
                "uncertainty_sort",
                "gbifID",
            ]
        )
        .drop_duplicates(
            [
                "fia_species_code",
                "event_day",
                "thin_lat_index",
                "thin_lon_index",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "uncertainty_sort",
            ]
        )
        .reset_index(drop=True)
    )

    thinned["period"] = np.where(
        thinned["year"].isin(
            EARLY_YEARS
        ),
        "early",
        "late",
    )

    thinned[
        "observer_label"
    ] = normalize_observer_label(
        thinned["recordedBy"]
    )

    return frame, clean, thinned


def normalize_fia_reference(path):
    frame = pd.read_parquet(path)

    if (
        "fia_species_code"
        not in frame.columns
        and "species_code"
        in frame.columns
    ):
        frame = frame.rename(
            columns={
                "species_code": (
                    "fia_species_code"
                )
            }
        )

    required = {
        "fia_species_code",
        "fia_shift_km_decade",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise KeyError(
            f"FIA reference {path} is missing "
            f"{sorted(missing)}"
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
            "fia_shift_km_decade",
        ]
    ).copy()

    frame[
        "fia_species_code"
    ] = frame[
        "fia_species_code"
    ].astype(int)

    return frame


def build_grid_results(
    thinned,
    evaluation_codes,
    fia_reference,
    resolution,
):
    work = thinned.copy()

    work[
        "grid_lat_index"
    ] = np.floor(
        work[
            "decimalLatitude"
        ]
        / resolution
    ).astype("int32")

    work[
        "grid_lon_index"
    ] = np.floor(
        work[
            "decimalLongitude"
        ]
        / resolution
    ).astype("int32")

    work["grid_id"] = (
        work[
            "grid_lat_index"
        ].astype(str)
        + "_"
        + work[
            "grid_lon_index"
        ].astype(str)
    )

    # Every record belongs to the locked state-specific effort frame.
    effort_df = (
        work.groupby(
            [
                "period",
                "grid_id",
            ],
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

    for period in [
        "early",
        "late",
    ]:
        if (
            period
            not in effort_wide.columns
        ):
            effort_wide[
                period
            ] = 0

    stable_cells = set(
        effort_wide[
            (
                effort_wide[
                    "early"
                ]
                >= MIN_TARGET_EFFORT_PER_CELL_PERIOD
            )
            & (
                effort_wide[
                    "late"
                ]
                >= MIN_TARGET_EFFORT_PER_CELL_PERIOD
            )
        ].index
    )

    stable_effort = effort_df[
        effort_df[
            "grid_id"
        ].isin(
            stable_cells
        )
    ].copy()

    pooled_cell_latitude = (
        work[
            work[
                "grid_id"
            ].isin(
                stable_cells
            )
        ]
        .groupby(
            "grid_id"
        )[
            "decimalLatitude"
        ]
        .mean()
        .to_dict()
    )

    benchmark = work[
        work[
            "fia_species_code"
        ].isin(
            evaluation_codes
        )
    ].copy()

    raw_period = (
        benchmark.groupby(
            [
                "fia_species_code",
                "period",
            ],
            as_index=False,
        )
        .agg(
            raw_records=(
                "gbifID",
                "size",
            ),
            raw_latitude=(
                "decimalLatitude",
                "mean",
            ),
            raw_cells=(
                "grid_id",
                "nunique",
            ),
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

    raw_wide = (
        raw_wide.reset_index()
    )

    corrected_rows = []

    for species_code in sorted(
        evaluation_codes
    ):
        species_counts = (
            benchmark[
                benchmark[
                    "fia_species_code"
                ]
                == species_code
            ]
            .groupby(
                [
                    "period",
                    "grid_id",
                ]
            )[
                "gbifID"
            ]
            .size()
            .rename(
                "species_records"
            )
            .reset_index()
        )

        merged = stable_effort.merge(
            species_counts,
            on=[
                "period",
                "grid_id",
            ],
            how="left",
        )

        merged[
            "species_records"
        ] = (
            merged[
                "species_records"
            ]
            .fillna(0)
            .astype(float)
        )

        merged[
            "relative_occurrence_rate"
        ] = (
            merged[
                "species_records"
            ]
            / merged[
                "target_effort_records"
            ]
        )

        merged[
            "cell_latitude"
        ] = merged[
            "grid_id"
        ].map(
            pooled_cell_latitude
        )

        for period in [
            "early",
            "late",
        ]:
            period_frame = merged[
                merged[
                    "period"
                ]
                == period
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
                    "fia_species_code": (
                        species_code
                    ),
                    "period": period,
                    "corrected_latitude": (
                        corrected_latitude
                    ),
                    "stable_records": float(
                        period_frame[
                            "species_records"
                        ].sum()
                    ),
                    "stable_positive_cells": int(
                        (
                            period_frame[
                                "species_records"
                            ]
                            > 0
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

    corrected_wide = (
        corrected_wide.reset_index()
    )

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

    result[
        "raw_shift_km_decade"
    ] = [
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

    result[
        "corrected_shift_km_decade"
    ] = [
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

    result[
        "raw_abs_error"
    ] = (
        result[
            "raw_shift_km_decade"
        ]
        - result[
            "fia_shift_km_decade"
        ]
    ).abs()

    result[
        "corrected_abs_error"
    ] = (
        result[
            "corrected_shift_km_decade"
        ]
        - result[
            "fia_shift_km_decade"
        ]
    ).abs()

    result["gain"] = (
        result[
            "raw_abs_error"
        ]
        - result[
            "corrected_abs_error"
        ]
    )

    summary = {
        "grid_resolution": resolution,
        "n_effort_cells": int(
            work[
                "grid_id"
            ].nunique()
        ),
        "n_stable_cells": int(
            len(stable_cells)
        ),
        "n_species": int(
            result[
                "fia_species_code"
            ].nunique()
        ),
    }

    return result, summary


def build_locked_features(
    thinned,
    evaluation_codes,
):
    work = thinned[
        thinned[
            "fia_species_code"
        ].isin(
            evaluation_codes
        )
    ].copy()

    work[
        "feature_grid_lat_index"
    ] = np.floor(
        work[
            "decimalLatitude"
        ]
        / PRIMARY_GRID_RESOLUTION
    ).astype("int32")

    work[
        "feature_grid_lon_index"
    ] = np.floor(
        work[
            "decimalLongitude"
        ]
        / PRIMARY_GRID_RESOLUTION
    ).astype("int32")

    work[
        "feature_grid_id"
    ] = (
        work[
            "feature_grid_lat_index"
        ].astype(str)
        + "_"
        + work[
            "feature_grid_lon_index"
        ].astype(str)
    )

    rows = []

    for species_code in sorted(
        evaluation_codes
    ):
        species = work[
            work[
                "fia_species_code"
            ]
            == species_code
        ]

        early = species[
            species["period"]
            == "early"
        ]

        late = species[
            species["period"]
            == "late"
        ]

        early_cells = set(
            early[
                "feature_grid_id"
            ].dropna()
        )

        late_cells = set(
            late[
                "feature_grid_id"
            ].dropna()
        )

        early_observers = int(
            early[
                "observer_label"
            ].nunique(
                dropna=True
            )
        )

        late_observers = int(
            late[
                "observer_label"
            ].nunique(
                dropna=True
            )
        )

        rows.append(
            {
                "fia_species_code": (
                    species_code
                ),
                "cell_jaccard": (
                    set_jaccard(
                        early_cells,
                        late_cells,
                    )
                ),
                "abs_log_observer_growth": abs(
                    np.log(
                        (
                            late_observers
                            + 1
                        )
                        / (
                            early_observers
                            + 1
                        )
                    )
                ),
                "early_cells": len(
                    early_cells
                ),
                "late_cells": len(
                    late_cells
                ),
                "early_observers": (
                    early_observers
                ),
                "late_observers": (
                    late_observers
                ),
            }
        )

    return pd.DataFrame(rows)


def compare_tables(
    state,
    formal,
    temporary,
):
    temporary = temporary.copy()

    temporary[
        "fia_species_code"
    ] = pd.to_numeric(
        temporary[
            "fia_species_code"
        ],
        errors="coerce",
    )

    temporary = temporary.dropna(
        subset=[
            "fia_species_code"
        ]
    ).copy()

    temporary[
        "fia_species_code"
    ] = temporary[
        "fia_species_code"
    ].astype(int)

    merged = formal.merge(
        temporary,
        on="fia_species_code",
        how="outer",
        suffixes=(
            "_formal",
            "_temp",
        ),
        indicator=True,
    )

    comparison_columns = [
        "cell_jaccard",
        "abs_log_observer_growth",
        "gain_grid_0p25",
        "gain_grid_0p5",
        "gain_grid_1p0",
        "grid_median_gain_km_decade",
    ]

    rows = []

    for column in comparison_columns:
        formal_column = (
            f"{column}_formal"
        )
        temp_column = (
            f"{column}_temp"
        )

        if (
            formal_column
            not in merged.columns
            or temp_column
            not in merged.columns
        ):
            rows.append(
                {
                    "state_code": state,
                    "metric": column,
                    "n_compared": 0,
                    "max_abs_difference": np.nan,
                    "median_abs_difference": np.nan,
                    "passed": False,
                    "note": "metric missing",
                }
            )
            continue

        values = merged[
            [
                formal_column,
                temp_column,
            ]
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()

        differences = (
            values[
                formal_column
            ]
            - values[
                temp_column
            ]
        ).abs()

        maximum = (
            float(
                differences.max()
            )
            if len(
                differences
            )
            else np.nan
        )

        rows.append(
            {
                "state_code": state,
                "metric": column,
                "n_compared": len(
                    differences
                ),
                "max_abs_difference": (
                    maximum
                ),
                "median_abs_difference": (
                    float(
                        differences.median()
                    )
                    if len(
                        differences
                    )
                    else np.nan
                ),
                "passed": bool(
                    len(
                        differences
                    )
                    == len(
                        formal
                    )
                    and np.isfinite(
                        maximum
                    )
                    and maximum
                    <= PARITY_TOLERANCE
                ),
                "note": "",
            }
        )

    species_set_pass = bool(
        set(
            formal[
                "fia_species_code"
            ]
        )
        == set(
            temporary[
                "fia_species_code"
            ]
        )
    )

    species_summary = {
        "state_code": state,
        "formal_species": len(
            formal
        ),
        "temporary_species": len(
            temporary
        ),
        "matched_species": int(
            (
                merged[
                    "_merge"
                ]
                == "both"
            ).sum()
        ),
        "formal_only_species": int(
            (
                merged[
                    "_merge"
                ]
                == "left_only"
            ).sum()
        ),
        "temporary_only_species": int(
            (
                merged[
                    "_merge"
                ]
                == "right_only"
            ).sum()
        ),
        "species_set_pass": (
            species_set_pass
        ),
    }

    return (
        pd.DataFrame(rows),
        species_summary,
    )


def failure(
    status,
    error,
    next_step,
):
    append_readme(
        f"""

## Registered GBIF parity failure — {RUN_UTC}
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
    missing_formal = [
        str(path)
        for path in FORMAL_PATHS.values()
        if not path.exists()
    ]

    if missing_formal:
        raise FileNotFoundError(
            "Missing formal Parquet files: "
            + " | ".join(
                missing_formal
            )
        )

    state_output_rows = []
    grid_summary_rows = []
    parity_rows = []
    species_parity_rows = []
    selected_path_rows = []

    state_progress = tqdm(
        STATES,
        desc="Recomputing formal three-state inputs",
        unit="state",
    )

    for state in state_progress:
        state_progress.set_postfix_str(
            state
        )

        # ----------------------------------------------------
        # Discover the temporary deterministic species table.
        # ----------------------------------------------------
        preferred_temp = [
            TEMP_TABLE_PREFERRED.get(
                state
            )
        ]

        temp_path = find_parquet_with_columns(
            state=state,
            preferred_paths=preferred_temp,
            required_columns={
                "fia_species_code",
                "cell_jaccard",
                "abs_log_observer_growth",
            },
            path_terms=[
                "candidate",
            ],
        )

        temporary_df = pd.read_parquet(
            temp_path
        )

        # Require the three grid-gain columns in the selected table.
        required_gain_columns = {
            "gain_grid_0p25",
            "gain_grid_0p5",
            "gain_grid_1p0",
        }

        if not required_gain_columns.issubset(
            set(
                temporary_df.columns
            )
        ):
            raise KeyError(
                f"Selected temporary table lacks grid gains: "
                f"{temp_path}"
            )

        evaluation_codes = set(
            pd.to_numeric(
                temporary_df[
                    "fia_species_code"
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )

        # ----------------------------------------------------
        # Discover the matching FIA reference.
        # ----------------------------------------------------
        preferred_reference = [
            FIA_REFERENCE_PREFERRED.get(
                state
            )
        ]

        reference_path = find_parquet_with_columns(
            state=state,
            preferred_paths=preferred_reference,
            required_columns={
                "fia_shift_km_decade",
            },
            path_terms=[
                "fia",
            ],
        )

        fia_reference = normalize_fia_reference(
            reference_path
        )

        selected_path_rows.append(
            {
                "state_code": state,
                "formal_path": str(
                    FORMAL_PATHS[
                        state
                    ]
                ),
                "temporary_table": str(
                    temp_path
                ),
                "fia_reference": str(
                    reference_path
                ),
                "n_evaluation_species": len(
                    evaluation_codes
                ),
            }
        )

        # ----------------------------------------------------
        # Formal quality filtering and thinning.
        # ----------------------------------------------------
        (
            raw_df,
            clean_df,
            thinned_df,
        ) = prepare_formal_state(
            state,
            FORMAL_PATHS[
                state
            ],
        )

        thinned_path = (
            OUT_DIR
            / f"{state}_formal_thinned.parquet"
        )

        thinned_df.to_parquet(
            thinned_path,
            index=False,
        )

        # ----------------------------------------------------
        # Formal grid estimates.
        # ----------------------------------------------------
        formal_table = pd.DataFrame(
            {
                "fia_species_code": sorted(
                    evaluation_codes
                )
            }
        )

        for resolution in tqdm(
            GRID_RESOLUTIONS,
            desc=f"{state} formal grid rerun",
            unit="grid",
            leave=False,
        ):
            (
                grid_result,
                grid_summary,
            ) = build_grid_results(
                thinned=thinned_df,
                evaluation_codes=(
                    evaluation_codes
                ),
                fia_reference=(
                    fia_reference
                ),
                resolution=resolution,
            )

            tag = str(
                resolution
            ).replace(
                ".",
                "p",
            )

            formal_table = (
                formal_table.merge(
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
            )

            grid_summary[
                "state_code"
            ] = state

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
            .sum(
                axis=1
            )
            / formal_table[
                gain_columns
            ]
            .notna()
            .sum(
                axis=1
            )
            .replace(
                0,
                np.nan,
            )
        )

        features = build_locked_features(
            thinned=thinned_df,
            evaluation_codes=(
                evaluation_codes
            ),
        )

        formal_table = (
            formal_table.merge(
                features,
                on="fia_species_code",
                how="left",
            )
        )

        formal_output_path = (
            OUT_DIR
            / f"{state}_formal_species_inputs.parquet"
        )

        formal_table.to_parquet(
            formal_output_path,
            index=False,
        )

        formal_table.to_csv(
            OUT_DIR
            / f"{state}_formal_species_inputs.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Deterministic parity comparison.
        # ----------------------------------------------------
        (
            state_parity,
            species_summary,
        ) = compare_tables(
            state=state,
            formal=formal_table,
            temporary=temporary_df,
        )

        parity_rows.append(
            state_parity
        )

        species_parity_rows.append(
            species_summary
        )

        state_output_rows.append(
            {
                "state_code": state,
                "raw_records": len(
                    raw_df
                ),
                "clean_records": len(
                    clean_df
                ),
                "thinned_records": len(
                    thinned_df
                ),
                "effort_taxa": int(
                    thinned_df[
                        "fia_species_code"
                    ].nunique()
                ),
                "evaluation_species": len(
                    formal_table
                ),
                "formal_species_table": str(
                    formal_output_path
                ),
                "formal_thinned_path": str(
                    thinned_path
                ),
            }
        )

    state_progress.close()

    state_output_df = pd.DataFrame(
        state_output_rows
    )

    grid_summary_df = pd.DataFrame(
        grid_summary_rows
    )

    parity_df = pd.concat(
        parity_rows,
        ignore_index=True,
    )

    species_parity_df = pd.DataFrame(
        species_parity_rows
    )

    selected_paths_df = pd.DataFrame(
        selected_path_rows
    )

    state_output_df.to_csv(
        OUT_DIR
        / "formal_state_processing_summary.csv",
        index=False,
    )

    grid_summary_df.to_csv(
        OUT_DIR
        / "formal_grid_summary.csv",
        index=False,
    )

    parity_df.to_csv(
        OUT_DIR
        / "formal_vs_temporary_metric_parity.csv",
        index=False,
    )

    species_parity_df.to_csv(
        OUT_DIR
        / "formal_vs_temporary_species_parity.csv",
        index=False,
    )

    selected_paths_df.to_csv(
        OUT_DIR
        / "formal_parity_selected_inputs.csv",
        index=False,
    )

    print(
        "\nSELECTED FORMAL AND TEMPORARY INPUTS"
    )
    display(
        selected_paths_df
    )

    print(
        "\nFORMAL STATE PROCESSING SUMMARY"
    )
    display(
        state_output_df
    )

    print(
        "\nFORMAL GRID SUMMARY"
    )
    display(
        grid_summary_df
    )

    print(
        "\nSPECIES-SET PARITY"
    )
    display(
        species_parity_df
    )

    print(
        "\nMETRIC PARITY"
    )
    display(
        parity_df
    )

    species_set_pass = bool(
        species_parity_df[
            "species_set_pass"
        ].all()
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
            "FORMAL_GBIF_RESULTS_MATCH_TEMP_API"
        )
        next_step = (
            "Run the final locked PA, VA and NC confirmatory "
            "inference from the formal species-input tables and "
            "write the cross-region conclusion."
        )
    elif species_set_pass:
        status = (
            "FORMAL_GBIF_MINOR_NUMERIC_DIFFERENCES"
        )
        next_step = (
            "Inspect the metric-difference table before final "
            "inference. Do not reuse temporary test statistics."
        )
    else:
        status = (
            "FORMAL_GBIF_SPECIES_SET_MISMATCH"
        )
        next_step = (
            "Inspect state-specific species differences and source "
            "selection before running final inference."
        )

    append_readme(
        f"""

## Formal GBIF deterministic parity rerun — {RUN_UTC}

### Purpose
Recompute the locked PA, VA and NC deterministic species inputs from the formal
registered GBIF download before final statistical inference.

### Processing
- Coordinate uncertainty:
  <= {MAX_COORDINATE_UNCERTAINTY_M:,} m or missing
- Daily-spatial thinning:
  species × event day × {THINNING_RESOLUTION} degree cell
- Correction grids:
  {GRID_RESOLUTIONS}
- Stable effort threshold:
  {MIN_TARGET_EFFORT_PER_CELL_PERIOD} records per cell-period
- Locked features:
  cell_jaccard and abs_log_observer_growth
- Parity tolerance:
  {PARITY_TOLERANCE}
- GPU: not used

### Result
- Species-set parity passed:
  {species_set_pass}
- Deterministic metric parity passed:
  {metric_parity_pass}
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
        f"OUT_DIR: {OUT_DIR}"
    )
    print(
        "STATE_PROCESSING: "
        + json.dumps(
            state_output_df.to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "GRID_SUMMARY: "
        + json.dumps(
            grid_summary_df.round(
                8
            ).to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "SPECIES_PARITY: "
        + json.dumps(
            species_parity_df.to_dict(
                "records"
            ),
            ensure_ascii=False,
        )
    )
    print(
        "METRIC_PARITY: "
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
        "CELL16B_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Save this diagnostic summary and the last displayed selected-input "
            "table. Do not run final statistical inference."
        ),
    )
