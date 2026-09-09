
# Cell 23D2 — Reconstruct PA matched FIA plots with VA/NC calibration
#
# Scientific safety rule
# ----------------------
# The PA count is accepted only if the same code and filtering rules reproduce:
#   VA = 3,068
#   NC = 2,516
#
# Definition tested
# -----------------
# - exact physical plot = STATECD + UNITCD + COUNTYCD + PLOT
# - live tree = STATUSCD == 1
# - measurement year = MEASYEAR when available, otherwise INVYR
# - early period = 2013–2018
# - late period = 2020–2025
# - matched physical plot = at least one live-tree record in both periods
#
# Nearest-midpoint selection affects the selected measurement for species-shift
# estimation, but not the descriptive count of physical plots represented by
# live-tree measurements in both windows.
#
# No GBIF analysis, cell-Jaccard analysis, or mechanism simulation is rerun.

from pathlib import Path
from datetime import datetime, timezone
import gzip
import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from IPython.display import display

# ============================================================
# Configuration
# ============================================================
INPUT_ROOT = Path("/kaggle/input")

RECOVERED_PROJECT = Path(
    "/kaggle/working/recovered_q1/"
    "kaggle_working/fia_temporal_observation_drift"
)

OUT_DIR = Path(
    "/kaggle/working/submission_critical_repair_audit/"
    "pa_calibrated_matched_plot_reconstruction"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STATES = {
    "PA": {
        "statecd": 42,
        "locked_count": None,
    },
    "VA": {
        "statecd": 51,
        "locked_count": 3068,
    },
    "NC": {
        "statecd": 37,
        "locked_count": 2516,
    },
}

EARLY_YEARS = set(
    range(
        2013,
        2019,
    )
)

LATE_YEARS = set(
    range(
        2020,
        2026,
    )
)

CSV_CHUNK_SIZE = 250_000

ALLOWED_SUFFIXES = {
    ".csv",
    ".gz",
    ".parquet",
}

RUN_UTC = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("INPUT_ROOT:", INPUT_ROOT)
print("RECOVERED_PROJECT:", RECOVERED_PROJECT)
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")
print("GBIF_OR_SIMULATION_RERUN: False")


# ============================================================
# Helpers
# ============================================================
def append_readme(text):
    readme_path = (
        RECOVERED_PROJECT
        / "README.md"
    )

    if not readme_path.exists():
        return False

    old = readme_path.read_text(
        encoding="utf-8"
    )

    readme_path.write_text(
        old + text,
        encoding="utf-8",
    )

    return True


def normalize_name(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).lower(),
    ).strip("_")


def state_tokens(state):
    names = {
        "PA": [
            "pa",
            "pennsylvania",
            "42",
        ],
        "VA": [
            "va",
            "virginia",
            "51",
        ],
        "NC": [
            "nc",
            "north_carolina",
            "northcarolina",
            "37",
        ],
    }

    return names[
        state
    ]


def path_state_score(
    path,
    state,
):
    text = normalize_name(
        path
    )

    filename = normalize_name(
        Path(
            path
        ).name
    )

    score = 0

    if "tree" in filename:
        score += 25

    if filename.startswith(
        f"{state.lower()}_tree"
    ):
        score += 35

    if f"_{state.lower()}_tree" in (
        "_"
        + filename
    ):
        score += 20

    for token in state_tokens(
        state
    ):
        token_normalized = normalize_name(
            token
        )

        if (
            f"_{token_normalized}_"
            in f"_{text}_"
        ):
            score += 12

    for negative in [
        "seedling",
        "subplot",
        "condition",
        "cond",
        "plot",
        "survey",
        "species",
        "reference",
        "summary",
        "output",
        "derived",
    ]:
        if negative in filename:
            score -= 15

    suffix_text = str(
        path
    ).lower()

    if suffix_text.endswith(
        ".csv"
    ):
        score += 8

    elif suffix_text.endswith(
        ".csv.gz"
    ):
        score += 7

    elif suffix_text.endswith(
        ".parquet"
    ):
        score += 6

    try:
        size = Path(
            path
        ).stat().st_size

        if size > 100 * 1024 * 1024:
            score += 8

        elif size > 10 * 1024 * 1024:
            score += 4

    except Exception:
        pass

    return score


def find_tree_candidates(
    root,
    state,
):
    candidates = []

    for current_root, _, filenames in os.walk(
        root
    ):
        for filename in filenames:
            path = (
                Path(
                    current_root
                )
                / filename
            )

            lower = filename.lower()

            supported = (
                lower.endswith(
                    ".csv"
                )
                or lower.endswith(
                    ".csv.gz"
                )
                or lower.endswith(
                    ".parquet"
                )
            )

            if not supported:
                continue

            if "tree" not in lower:
                continue

            score = path_state_score(
                path,
                state,
            )

            if score <= 0:
                continue

            try:
                size = path.stat().st_size
            except Exception:
                size = 0

            candidates.append(
                {
                    "state": state,
                    "path": str(
                        path
                    ),
                    "score": int(
                        score
                    ),
                    "size_bytes": int(
                        size
                    ),
                    "size_mb": float(
                        size
                        / 1024 ** 2
                    ),
                }
            )

    frame = pd.DataFrame(
        candidates
    )

    if frame.empty:
        return frame

    return frame.sort_values(
        [
            "score",
            "size_bytes",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


def read_csv_columns(
    path,
):
    return list(
        pd.read_csv(
            path,
            nrows=0,
            low_memory=False,
        ).columns
    )


def read_parquet_columns(
    path,
):
    return list(
        pq.ParquetFile(
            path
        ).schema.names
    )


def resolve_columns(
    columns,
):
    lookup = {
        str(
            column
        ).strip().upper(): str(
            column
        )
        for column in columns
    }

    required_id = [
        "STATECD",
        "UNITCD",
        "COUNTYCD",
        "PLOT",
    ]

    missing_id = [
        column
        for column in required_id
        if column not in lookup
    ]

    if missing_id:
        raise KeyError(
            "Missing physical-plot columns: "
            + ", ".join(
                missing_id
            )
        )

    status_column = None

    for candidate in [
        "STATUSCD",
        "STATUS_CD",
        "TREE_STATUSCD",
    ]:
        if candidate in lookup:
            status_column = lookup[
                candidate
            ]
            break

    if status_column is None:
        raise KeyError(
            "No live-tree STATUSCD column found."
        )

    year_column = None

    for candidate in [
        "MEASYEAR",
        "MEAS_YEAR",
        "INVYR",
        "INV_YEAR",
    ]:
        if candidate in lookup:
            year_column = lookup[
                candidate
            ]
            break

    if year_column is None:
        raise KeyError(
            "No MEASYEAR or INVYR column found."
        )

    return {
        "STATECD": lookup[
            "STATECD"
        ],
        "UNITCD": lookup[
            "UNITCD"
        ],
        "COUNTYCD": lookup[
            "COUNTYCD"
        ],
        "PLOT": lookup[
            "PLOT"
        ],
        "STATUSCD": status_column,
        "YEAR": year_column,
    }


def make_plot_ids(
    frame,
    columns,
):
    parts = []

    for logical_name in [
        "STATECD",
        "UNITCD",
        "COUNTYCD",
        "PLOT",
    ]:
        values = pd.to_numeric(
            frame[
                columns[
                    logical_name
                ]
            ],
            errors="coerce",
        )

        parts.append(
            values
        )

    valid = np.ones(
        len(
            frame
        ),
        dtype=bool,
    )

    for values in parts:
        valid &= values.notna().to_numpy()

    if not valid.any():
        return pd.Series(
            dtype="object"
        )

    ids = (
        parts[
            0
        ][
            valid
        ].astype(
            "int64"
        ).astype(str)
        + "_"
        + parts[
            1
        ][
            valid
        ].astype(
            "int64"
        ).astype(str)
        + "_"
        + parts[
            2
        ][
            valid
        ].astype(
            "int64"
        ).astype(str)
        + "_"
        + parts[
            3
        ][
            valid
        ].astype(
            "int64"
        ).astype(str)
    )

    return ids


def update_sets_from_frame(
    frame,
    columns,
    expected_statecd,
    early_set,
    late_set,
):
    state_values = pd.to_numeric(
        frame[
            columns[
                "STATECD"
            ]
        ],
        errors="coerce",
    )

    status_values = pd.to_numeric(
        frame[
            columns[
                "STATUSCD"
            ]
        ],
        errors="coerce",
    )

    year_values = pd.to_numeric(
        frame[
            columns[
                "YEAR"
            ]
        ],
        errors="coerce",
    )

    mask = (
        state_values.eq(
            expected_statecd
        )
        & status_values.eq(
            1
        )
        & year_values.isin(
            EARLY_YEARS
            | LATE_YEARS
        )
    )

    filtered = frame.loc[
        mask,
        [
            columns[
                "STATECD"
            ],
            columns[
                "UNITCD"
            ],
            columns[
                "COUNTYCD"
            ],
            columns[
                "PLOT"
            ],
            columns[
                "YEAR"
            ],
        ],
    ].copy()

    if filtered.empty:
        return {
            "live_window_rows": 0,
            "early_rows": 0,
            "late_rows": 0,
        }

    plot_ids = make_plot_ids(
        filtered,
        columns,
    )

    filtered = filtered.loc[
        plot_ids.index
    ].copy()

    filtered[
        "_plot_id"
    ] = plot_ids

    years = pd.to_numeric(
        filtered[
            columns[
                "YEAR"
            ]
        ],
        errors="coerce",
    )

    early_ids = set(
        filtered.loc[
            years.isin(
                EARLY_YEARS
            ),
            "_plot_id",
        ]
    )

    late_ids = set(
        filtered.loc[
            years.isin(
                LATE_YEARS
            ),
            "_plot_id",
        ]
    )

    early_set.update(
        early_ids
    )

    late_set.update(
        late_ids
    )

    return {
        "live_window_rows": int(
            len(
                filtered
            )
        ),
        "early_rows": int(
            years.isin(
                EARLY_YEARS
            ).sum()
        ),
        "late_rows": int(
            years.isin(
                LATE_YEARS
            ).sum()
        ),
    }


def process_tree_file(
    path,
    state,
    expected_statecd,
):
    path = Path(
        path
    )

    lower = str(
        path
    ).lower()

    if lower.endswith(
        ".parquet"
    ):
        columns = read_parquet_columns(
            path
        )

    else:
        columns = read_csv_columns(
            path
        )

    resolved = resolve_columns(
        columns
    )

    usecols = list(
        dict.fromkeys(
            resolved.values()
        )
    )

    early_set = set()
    late_set = set()

    rows_scanned = 0
    live_window_rows = 0
    early_live_rows = 0
    late_live_rows = 0

    if lower.endswith(
        ".parquet"
    ):
        parquet_file = pq.ParquetFile(
            path
        )

        total_groups = parquet_file.num_row_groups

        for group_index in tqdm(
            range(
                total_groups
            ),
            desc=f"{state} TREE row groups",
            unit="group",
            leave=False,
        ):
            frame = parquet_file.read_row_group(
                group_index,
                columns=usecols,
            ).to_pandas()

            rows_scanned += len(
                frame
            )

            counts = update_sets_from_frame(
                frame=frame,
                columns=resolved,
                expected_statecd=(
                    expected_statecd
                ),
                early_set=early_set,
                late_set=late_set,
            )

            live_window_rows += counts[
                "live_window_rows"
            ]

            early_live_rows += counts[
                "early_rows"
            ]

            late_live_rows += counts[
                "late_rows"
            ]

    else:
        reader = pd.read_csv(
            path,
            usecols=usecols,
            chunksize=CSV_CHUNK_SIZE,
            low_memory=False,
        )

        for frame in tqdm(
            reader,
            desc=f"{state} TREE chunks",
            unit="chunk",
            leave=False,
        ):
            rows_scanned += len(
                frame
            )

            counts = update_sets_from_frame(
                frame=frame,
                columns=resolved,
                expected_statecd=(
                    expected_statecd
                ),
                early_set=early_set,
                late_set=late_set,
            )

            live_window_rows += counts[
                "live_window_rows"
            ]

            early_live_rows += counts[
                "early_rows"
            ]

            late_live_rows += counts[
                "late_rows"
            ]

    matched = (
        early_set
        & late_set
    )

    return {
        "state": state,
        "source_path": str(
            path
        ),
        "year_column": resolved[
            "YEAR"
        ],
        "status_column": resolved[
            "STATUSCD"
        ],
        "rows_scanned": int(
            rows_scanned
        ),
        "live_tree_rows_in_windows": int(
            live_window_rows
        ),
        "early_live_tree_rows": int(
            early_live_rows
        ),
        "late_live_tree_rows": int(
            late_live_rows
        ),
        "early_unique_physical_plots": int(
            len(
                early_set
            )
        ),
        "late_unique_physical_plots": int(
            len(
                late_set
            )
        ),
        "matched_live_tree_physical_plots": int(
            len(
                matched
            )
        ),
    }


# ============================================================
# Candidate discovery
# ============================================================
candidate_parts = []

for state in tqdm(
    STATES,
    desc="Locating state TREE files",
    unit="state",
):
    state_candidates = find_tree_candidates(
        INPUT_ROOT,
        state,
    )

    if not state_candidates.empty:
        candidate_parts.append(
            state_candidates
        )

all_candidates = (
    pd.concat(
        candidate_parts,
        ignore_index=True,
    )
    if candidate_parts
    else pd.DataFrame(
        columns=[
            "state",
            "path",
            "score",
            "size_bytes",
            "size_mb",
        ]
    )
)

all_candidates.to_csv(
    OUT_DIR
    / "tree_file_candidates.csv",
    index=False,
)

print("\nTOP TREE FILE CANDIDATES")

display(
    all_candidates.groupby(
        "state",
        group_keys=False,
    ).head(
        10
    )
)

selected_files = {}

for state in STATES:
    state_frame = all_candidates[
        all_candidates[
            "state"
        ]
        == state
    ].copy()

    if state_frame.empty:
        continue

    selected_files[
        state
    ] = state_frame.iloc[
        0
    ][
        "path"
    ]


# ============================================================
# Process PA, VA and NC with identical rules
# ============================================================
missing_states = [
    state
    for state in STATES
    if state not in selected_files
]

if missing_states:
    status = (
        "TREE_FILES_NOT_FOUND"
    )

    next_step = (
        "Send tree_file_candidates.csv or the visible FIA input "
        "folder structure. Do not enter a PA count manually."
    )

    print(
        "\n========== RUN SUMMARY =========="
    )
    print(
        f"CELL_STATUS: {status}"
    )
    print(
        f"MISSING_STATES: {missing_states}"
    )
    print(
        f"CANDIDATE_CSV: "
        f"{OUT_DIR / 'tree_file_candidates.csv'}"
    )
    print(
        f"NEXT_STEP: {next_step}"
    )
    print(
        "============================================"
    )

else:
    result_rows = []

    for state, config in tqdm(
        STATES.items(),
        desc="Reconstructing matched physical plots",
        unit="state",
    ):
        result = process_tree_file(
            path=selected_files[
                state
            ],
            state=state,
            expected_statecd=config[
                "statecd"
            ],
        )

        locked_count = config[
            "locked_count"
        ]

        result[
            "locked_count"
        ] = (
            locked_count
            if locked_count
            is not None
            else np.nan
        )

        result[
            "matches_locked_count"
        ] = (
            bool(
                result[
                    "matched_live_tree_physical_plots"
                ]
                == locked_count
            )
            if locked_count
            is not None
            else np.nan
        )

        result_rows.append(
            result
        )

    result_df = pd.DataFrame(
        result_rows
    )

    result_df.to_csv(
        OUT_DIR
        / "calibrated_state_matched_plot_counts.csv",
        index=False,
    )

    print(
        "\nCALIBRATED MATCHED-PLOT COUNTS"
    )

    display(
        result_df
    )

    va_pass = bool(
        result_df.loc[
            result_df[
                "state"
            ]
            == "VA",
            "matches_locked_count",
        ].iloc[
            0
        ]
    )

    nc_pass = bool(
        result_df.loc[
            result_df[
                "state"
            ]
            == "NC",
            "matches_locked_count",
        ].iloc[
            0
        ]
    )

    calibration_pass = bool(
        va_pass
        and nc_pass
    )

    pa_count = int(
        result_df.loc[
            result_df[
                "state"
            ]
            == "PA",
            "matched_live_tree_physical_plots",
        ].iloc[
            0
        ]
    )

    if calibration_pass:
        status = (
            "PA_MATCHED_PLOT_COUNT_CALIBRATED"
        )

        accepted = True

        next_step = (
            "Replace PA=NR with the calibrated PA count in the "
            "manuscript, Table 1, methods text and figure/table notes. "
            "Keep VA=3,068 and NC=2,516."
        )

    else:
        status = (
            "PA_COUNT_NOT_ACCEPTED_CALIBRATION_FAILED"
        )

        accepted = False

        next_step = (
            "Do not use the PA candidate count. Inspect the selected "
            "TREE paths, year column and live-tree rule against the "
            "original VA/NC construction code."
        )

    summary = {
        "run_utc": RUN_UTC,
        "status": status,
        "pa_candidate_count": pa_count,
        "va_reconstructed_count": int(
            result_df.loc[
                result_df[
                    "state"
                ]
                == "VA",
                "matched_live_tree_physical_plots",
            ].iloc[
                0
            ]
        ),
        "va_locked_count": 3068,
        "va_pass": va_pass,
        "nc_reconstructed_count": int(
            result_df.loc[
                result_df[
                    "state"
                ]
                == "NC",
                "matched_live_tree_physical_plots",
            ].iloc[
                0
            ]
        ),
        "nc_locked_count": 2516,
        "nc_pass": nc_pass,
        "calibration_pass": (
            calibration_pass
        ),
        "pa_count_accepted": (
            accepted
        ),
        "selected_files": (
            selected_files
        ),
        "next_step": next_step,
    }

    (
        OUT_DIR
        / "pa_matched_plot_reconstruction_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme_updated = append_readme(
        f"""

## PA same-window matched physical-plot reconstruction — {RUN_UTC}

- Definition:
  exact physical plot = STATECD + UNITCD + COUNTYCD + PLOT
- Live-tree filter:
  STATUSCD = 1
- Windows:
  2013–2018 and 2020–2025
- VA reconstructed:
  {summary['va_reconstructed_count']}
- VA locked:
  3,068
- VA calibration passed:
  {va_pass}
- NC reconstructed:
  {summary['nc_reconstructed_count']}
- NC locked:
  2,516
- NC calibration passed:
  {nc_pass}
- PA candidate:
  {pa_count}
- PA accepted:
  {accepted}
- Status:
  **{status}**
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
        f"PA_CANDIDATE_COUNT: {pa_count}"
    )
    print(
        f"PA_COUNT_ACCEPTED: {accepted}"
    )
    print(
        f"VA_RECONSTRUCTED_COUNT: "
        f"{summary['va_reconstructed_count']}"
    )
    print(
        "VA_LOCKED_COUNT: 3068"
    )
    print(
        f"VA_CALIBRATION_PASSED: {va_pass}"
    )
    print(
        f"NC_RECONSTRUCTED_COUNT: "
        f"{summary['nc_reconstructed_count']}"
    )
    print(
        "NC_LOCKED_COUNT: 2516"
    )
    print(
        f"NC_CALIBRATION_PASSED: {nc_pass}"
    )
    print(
        f"CALIBRATION_PASSED: "
        f"{calibration_pass}"
    )
    print(
        "SELECTED_TREE_FILES: "
        + json.dumps(
            selected_files,
            ensure_ascii=False,
        )
    )
    print(
        f"RESULT_CSV: "
        f"{OUT_DIR / 'calibrated_state_matched_plot_counts.csv'}"
    )
    print(
        f"SUMMARY_JSON: "
        f"{OUT_DIR / 'pa_matched_plot_reconstruction_summary.json'}"
    )
    print(
        f"NEXT_STEP: {next_step}"
    )
    print(
        f"README_UPDATED: {readme_updated}"
    )
    print(
        "============================================"
    )
