
# Cell 23E — Exact PA matched-plot count from original Cell 12A logic
#
# This is NOT a new experiment.
# It reconstructs only the descriptive PA same-window matched FIA plot count
# using the exact original Cell 12A rules.
#
# Exact original inputs:
#   recovered_q1/kaggle_working/fia_temporal_observation_drift/raw/PA_PLOT.csv
#   recovered_q1/kaggle_working/fia_temporal_observation_drift/raw/PA_TREE.csv
#
# Exact original rules:
# - physical plot = STATECD + UNITCD + COUNTYCD + PLOT
# - early window = 2013–2018
# - late window = 2020–2025
# - choose measurement nearest each window midpoint
# - PLOT_STATUS_CD == 1 when available
# - TREE STATUSCD == 1 when available
# - retain plots whose chosen early and late measurement keys both contain
#   at least one live-tree record
#
# No GBIF analysis, correction analysis, or simulation is rerun.

from pathlib import Path
from datetime import datetime, timezone
import gc
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(
    "/kaggle/working/recovered_q1/"
    "kaggle_working/fia_temporal_observation_drift"
)

RAW_DIR = BASE_DIR / "raw"

PA_PLOT_CSV = RAW_DIR / "PA_PLOT.csv"
PA_TREE_CSV = RAW_DIR / "PA_TREE.csv"

OUT_DIR = Path(
    "/kaggle/working/submission_critical_repair_audit/"
    "pa_exact_original_cell12a_count"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EARLY_YEARS = list(
    range(
        2013,
        2019,
    )
)

LATE_YEARS = list(
    range(
        2020,
        2026,
    )
)

EARLY_MID_YEAR = float(
    np.mean(
        EARLY_YEARS
    )
)

LATE_MID_YEAR = float(
    np.mean(
        LATE_YEARS
    )
)

TREE_CHUNK_SIZE = 250_000

RUN_UTC = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("BASE_DIR:", BASE_DIR)
print("PA_PLOT_CSV:", PA_PLOT_CSV)
print("PA_TREE_CSV:", PA_TREE_CSV)
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")
print("SCIENTIFIC_INFERENCE: unchanged")


# ============================================================
# Helpers copied from original Cell 12A
# ============================================================
def resolve_column(
    columns,
    candidates,
):
    lookup = {
        str(
            column
        ).upper(): str(
            column
        )
        for column in columns
    }

    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[
                candidate.upper()
            ]

    return None


def normalize_integer_component(
    series,
):
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .astype(
            "Int64"
        )
        .astype(
            "string"
        )
    )


def build_physical_plot_id(
    frame,
    state_col,
    unit_col,
    county_col,
    plot_col,
):
    return (
        normalize_integer_component(
            frame[
                state_col
            ]
        )
        + "_"
        + normalize_integer_component(
            frame[
                unit_col
            ]
        )
        + "_"
        + normalize_integer_component(
            frame[
                county_col
            ]
        )
        + "_"
        + normalize_integer_component(
            frame[
                plot_col
            ]
        )
    )


def closest_measurement_per_plot(
    plot_df,
    years,
    midpoint,
):
    subset = plot_df[
        plot_df[
            "year"
        ].isin(
            years
        )
    ].copy()

    subset[
        "distance_to_midpoint"
    ] = (
        subset[
            "year"
        ]
        - midpoint
    ).abs()

    subset = (
        subset.sort_values(
            [
                "physical_plot_id",
                "distance_to_midpoint",
                "year",
                "plot_key",
            ]
        )
        .groupby(
            "physical_plot_id",
            as_index=False,
        )
        .first()
    )

    return subset


# ============================================================
# Validate exact inputs
# ============================================================
missing = [
    str(
        path
    )
    for path in [
        PA_PLOT_CSV,
        PA_TREE_CSV,
    ]
    if not path.exists()
]

if missing:
    print(
        "\n========== RUN SUMMARY =========="
    )
    print(
        "CELL_STATUS: PA_RAW_FILES_MISSING"
    )
    print(
        "MISSING_FILES:",
        json.dumps(
            missing,
            ensure_ascii=False,
        ),
    )
    print(
        "NEXT_STEP: Locate PA_PLOT.csv and PA_TREE.csv inside the extracted archive; do not guess the PA count."
    )
    print(
        "============================================"
    )

else:
    # ========================================================
    # 1. Rebuild same-window candidate physical-plot pairs
    # ========================================================
    plot_columns = pd.read_csv(
        PA_PLOT_CSV,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    plot_key_col = resolve_column(
        plot_columns,
        [
            "CN",
            "PLOT_CN",
        ],
    )

    inventory_year_col = resolve_column(
        plot_columns,
        [
            "INVYR",
            "INVENTORY_YEAR",
        ],
    )

    latitude_col = resolve_column(
        plot_columns,
        [
            "LAT",
            "LATITUDE",
        ],
    )

    longitude_col = resolve_column(
        plot_columns,
        [
            "LON",
            "LONGITUDE",
        ],
    )

    plot_status_col = resolve_column(
        plot_columns,
        [
            "PLOT_STATUS_CD",
            "PLOT_STATUS",
        ],
    )

    state_code_col = resolve_column(
        plot_columns,
        [
            "STATECD",
            "STATE_CODE",
        ],
    )

    unit_code_col = resolve_column(
        plot_columns,
        [
            "UNITCD",
            "UNIT_CODE",
        ],
    )

    county_code_col = resolve_column(
        plot_columns,
        [
            "COUNTYCD",
            "COUNTY_CODE",
        ],
    )

    plot_number_col = resolve_column(
        plot_columns,
        [
            "PLOT",
            "PLOT_NUMBER",
            "PLOT_NBR",
        ],
    )

    required_plot_fields = [
        plot_key_col,
        inventory_year_col,
        latitude_col,
        longitude_col,
        state_code_col,
        unit_code_col,
        county_code_col,
        plot_number_col,
    ]

    if not all(
        required_plot_fields
    ):
        raise ValueError(
            "Required original Cell 12A PA PLOT fields were not resolved."
        )

    plot_usecols = list(
        dict.fromkeys(
            [
                plot_key_col,
                inventory_year_col,
                latitude_col,
                longitude_col,
                state_code_col,
                unit_code_col,
                county_code_col,
                plot_number_col,
            ]
            + (
                [
                    plot_status_col
                ]
                if plot_status_col
                is not None
                else []
            )
        )
    )

    plot_df = pd.read_csv(
        PA_PLOT_CSV,
        usecols=plot_usecols,
        low_memory=False,
        encoding_errors="replace",
        dtype={
            plot_key_col: "string"
        },
    )

    plot_df[
        "physical_plot_id"
    ] = build_physical_plot_id(
        plot_df,
        state_code_col,
        unit_code_col,
        county_code_col,
        plot_number_col,
    )

    rename_map = {
        plot_key_col: (
            "plot_key"
        ),
        inventory_year_col: (
            "year"
        ),
        latitude_col: (
            "latitude"
        ),
        longitude_col: (
            "longitude"
        ),
    }

    if plot_status_col is not None:
        rename_map[
            plot_status_col
        ] = "plot_status"

    plot_df = plot_df.rename(
        columns=rename_map
    )

    if "plot_status" not in plot_df.columns:
        plot_df[
            "plot_status"
        ] = np.nan

    plot_df[
        "plot_key"
    ] = (
        plot_df[
            "plot_key"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    for column in [
        "year",
        "latitude",
        "longitude",
        "plot_status",
    ]:
        plot_df[
            column
        ] = pd.to_numeric(
            plot_df[
                column
            ],
            errors="coerce",
        )

    valid_plot = (
        plot_df[
            "year"
        ].isin(
            EARLY_YEARS
            + LATE_YEARS
        )
        & plot_df[
            "latitude"
        ].between(
            39.5,
            42.6,
        )
        & plot_df[
            "longitude"
        ].between(
            -81.0,
            -74.4,
        )
        & plot_df[
            "plot_key"
        ].notna()
        & plot_df[
            "physical_plot_id"
        ].notna()
    )

    if plot_status_col is not None:
        valid_plot &= plot_df[
            "plot_status"
        ].eq(
            1
        )

    plot_df = (
        plot_df.loc[
            valid_plot,
            [
                "plot_key",
                "physical_plot_id",
                "year",
                "latitude",
                "longitude",
            ],
        ]
        .drop_duplicates(
            "plot_key"
        )
        .copy()
    )

    plot_df[
        "year"
    ] = plot_df[
        "year"
    ].astype(
        int
    )

    early_measurements_df = (
        closest_measurement_per_plot(
            plot_df,
            EARLY_YEARS,
            EARLY_MID_YEAR,
        )
    )

    late_measurements_df = (
        closest_measurement_per_plot(
            plot_df,
            LATE_YEARS,
            LATE_MID_YEAR,
        )
    )

    matched_plot_df = (
        early_measurements_df[
            [
                "physical_plot_id",
                "plot_key",
                "year",
                "latitude",
                "longitude",
            ]
        ]
        .rename(
            columns={
                "plot_key": (
                    "early_plot_key"
                ),
                "year": (
                    "early_year"
                ),
                "latitude": (
                    "early_latitude"
                ),
                "longitude": (
                    "early_longitude"
                ),
            }
        )
        .merge(
            late_measurements_df[
                [
                    "physical_plot_id",
                    "plot_key",
                    "year",
                    "latitude",
                    "longitude",
                ]
            ].rename(
                columns={
                    "plot_key": (
                        "late_plot_key"
                    ),
                    "year": (
                        "late_year"
                    ),
                    "latitude": (
                        "late_latitude"
                    ),
                    "longitude": (
                        "late_longitude"
                    ),
                }
            ),
            on=(
                "physical_plot_id"
            ),
            how="inner",
            validate="one_to_one",
        )
    )

    matched_plot_df = matched_plot_df[
        matched_plot_df[
            "late_year"
        ]
        > matched_plot_df[
            "early_year"
        ]
    ].copy()

    pre_live_tree_count = int(
        len(
            matched_plot_df
        )
    )

    candidate_plot_keys = set(
        matched_plot_df[
            "early_plot_key"
        ].astype(
            str
        )
    ) | set(
        matched_plot_df[
            "late_plot_key"
        ].astype(
            str
        )
    )

    # ========================================================
    # 2. Scan PA TREE for live measurement keys
    # ========================================================
    tree_columns = pd.read_csv(
        PA_TREE_CSV,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    tree_plot_key_col = resolve_column(
        tree_columns,
        [
            "PLT_CN",
            "PLOT_CN",
        ],
    )

    tree_status_col = resolve_column(
        tree_columns,
        [
            "STATUSCD",
            "STATUS_CODE",
        ],
    )

    if tree_plot_key_col is None:
        raise ValueError(
            "Required PA TREE plot-key field was not resolved."
        )

    tree_usecols = [
        tree_plot_key_col
    ]

    if tree_status_col is not None:
        tree_usecols.append(
            tree_status_col
        )

    tree_usecols = list(
        dict.fromkeys(
            tree_usecols
        )
    )

    live_measurement_keys = set()
    tree_rows_scanned = 0

    tree_reader = pd.read_csv(
        PA_TREE_CSV,
        usecols=tree_usecols,
        chunksize=TREE_CHUNK_SIZE,
        low_memory=False,
        encoding_errors="replace",
        dtype={
            tree_plot_key_col: (
                "string"
            )
        },
    )

    for chunk in tqdm(
        tree_reader,
        desc=(
            "Scanning PA TREE using original Cell 12A rules"
        ),
        unit="chunk",
    ):
        tree_rows_scanned += len(
            chunk
        )

        chunk = chunk.rename(
            columns={
                tree_plot_key_col: (
                    "plot_key"
                ),
                **(
                    {
                        tree_status_col: (
                            "status_code"
                        )
                    }
                    if tree_status_col
                    is not None
                    else {}
                ),
            }
        )

        chunk[
            "plot_key"
        ] = (
            chunk[
                "plot_key"
            ]
            .astype(
                "string"
            )
            .str.strip()
        )

        keep = chunk[
            "plot_key"
        ].isin(
            candidate_plot_keys
        )

        if (
            "status_code"
            in chunk.columns
        ):
            chunk[
                "status_code"
            ] = pd.to_numeric(
                chunk[
                    "status_code"
                ],
                errors="coerce",
            )

            keep &= chunk[
                "status_code"
            ].eq(
                1
            )

        live_work = chunk.loc[
            keep,
            [
                "plot_key"
            ],
        ].dropna(
            subset=[
                "plot_key"
            ]
        )

        if live_work.empty:
            continue

        live_measurement_keys.update(
            live_work[
                "plot_key"
            ].astype(
                str
            )
        )

    # ========================================================
    # 3. Exact original PA count
    # ========================================================
    matched_plot_df = matched_plot_df[
        matched_plot_df[
            "early_plot_key"
        ].astype(
            str
        ).isin(
            live_measurement_keys
        )
        & matched_plot_df[
            "late_plot_key"
        ].astype(
            str
        ).isin(
            live_measurement_keys
        )
    ].copy()

    matched_plot_df = matched_plot_df.reset_index(
        drop=True
    )

    n_matched_physical_plots = int(
        len(
            matched_plot_df
        )
    )

    matched_plot_path = (
        OUT_DIR
        / "PA_same_window_matched_live_tree_physical_plots.parquet"
    )

    matched_plot_df.to_parquet(
        matched_plot_path,
        index=False,
    )

    summary_df = pd.DataFrame(
        [
            {
                "state": "PA",
                "early_years": (
                    "2013-2018"
                ),
                "late_years": (
                    "2020-2025"
                ),
                "early_mid_year": (
                    EARLY_MID_YEAR
                ),
                "late_mid_year": (
                    LATE_MID_YEAR
                ),
                "candidate_pairs_before_tree_filter": (
                    pre_live_tree_count
                ),
                "matched_live_tree_physical_plots": (
                    n_matched_physical_plots
                ),
                "tree_rows_scanned": (
                    tree_rows_scanned
                ),
                "plot_source": str(
                    PA_PLOT_CSV
                ),
                "tree_source": str(
                    PA_TREE_CSV
                ),
            }
        ]
    )

    summary_path = (
        OUT_DIR
        / "PA_exact_matched_plot_count_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    display(
        summary_df
    )

    print(
        "\n========== RUN SUMMARY =========="
    )
    print(
        "CELL_STATUS: PA_MATCHED_PLOT_COUNT_RECONSTRUCTED_FROM_ORIGINAL_CELL12A"
    )
    print(
        "STATE: PA"
    )
    print(
        f"N_MATCHED_FIA_PHYSICAL_PLOTS: "
        f"{n_matched_physical_plots:,}"
    )
    print(
        f"CANDIDATE_PAIRS_BEFORE_TREE_FILTER: "
        f"{pre_live_tree_count:,}"
    )
    print(
        f"PA_TREE_ROWS_SCANNED: "
        f"{tree_rows_scanned:,}"
    )
    print(
        f"PLOT_SOURCE: {PA_PLOT_CSV}"
    )
    print(
        f"TREE_SOURCE: {PA_TREE_CSV}"
    )
    print(
        f"MATCHED_PLOT_TABLE: {matched_plot_path}"
    )
    print(
        f"SUMMARY_CSV: {summary_path}"
    )
    print(
        "COUNT_ACCEPTED_FOR_MANUSCRIPT: True"
    )
    print(
        "NEXT_STEP: Replace PA=NR in the manuscript and Table 1 with this exact reconstructed count."
    )
    print(
        "============================================"
    )
