
# Cell 10A — Pennsylvania expanded tree-taxon observer coverage audit
#
# Why this step
# -------------
# Cell 09A used only 20 benchmark species. This was enough for the drift
# calibration, but it was too sparse for observer-trajectory inference:
# only 12 observer labels and 16 observer-species pairs bridged the early
# and late windows.
#
# This cell does NOT weaken repeated-observer thresholds.
# Instead it expands the observer-behaviour frame to a taxonomically defined
# group: species-level tree taxa actually present in the PA FIA TREE table.
#
# Minimal task
# ------------
# 1. Scan PA_TREE.csv in chunks and identify live tree species present in PA.
# 2. Merge FIA species names from REF_SPECIES.csv.
# 3. Exclude genus-only, spp., hybrid and unresolved labels.
# 4. Resolve accepted names in the current GBIF taxonomy.
# 5. Count PA iNaturalist records in:
#       early window: 2013–2018
#       late window : 2020–2025
# 6. Select a coverage-based expanded tree set.
#
# No occurrence coordinates are downloaded in this cell.

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import json
import re
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 140)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 260)

# ============================================================
# Configuration
# ============================================================
STATE = "PA"
STATE_NAME = "Pennsylvania"

EARLY_START_YEAR = 2013
EARLY_END_YEAR = 2018
LATE_START_YEAR = 2020
LATE_END_YEAR = 2025

TREE_CHUNK_SIZE = 250_000
MIN_FIA_LIVE_TREE_RECORDS = 100

MIN_INAT_RECORDS_EACH_WINDOW = 30
MIN_SELECTED_TAXA_GO = 40
MAX_SELECTED_TAXA = 100
MAX_ESTIMATED_DOWNLOAD_RECORDS = 300_000
GBIF_SEARCH_ACCESS_CAP = 99_000

REQUEST_PAUSE_SECONDS = 0.03

INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
INAT_DATASET_DOI = "10.15468/ab3s5x"
GBIF_API = "https://api.gbif.org/v1"

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "derived" / "pa_expanded_tree_taxon_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PA_TREE_CSV = RAW_DIR / "PA_TREE.csv"
REF_SPECIES_CSV = RAW_DIR / "REF_SPECIES.csv"

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)
ACCESS_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

print("RUN_UTC:", RUN_UTC)
print("STATE:", STATE)
print(
    "EARLY WINDOW:",
    EARLY_START_YEAR,
    "-",
    EARLY_END_YEAR,
)
print(
    "LATE WINDOW:",
    LATE_START_YEAR,
    "-",
    LATE_END_YEAR,
)
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")


# ============================================================
# Helpers
# ============================================================
class AuditStop(Exception):
    def __init__(self, status, message, next_step):
        super().__init__(message)
        self.status = status
        self.message = message
        self.next_step = next_step


def append_readme(text):
    readme_path = BASE_DIR / "README.md"
    existing = (
        readme_path.read_text(encoding="utf-8")
        if readme_path.exists()
        else "# Temporal Observation Drift — FIA Validation\n"
    )
    readme_path.write_text(existing + text, encoding="utf-8")


def resolve_column(columns, candidates):
    lookup = {str(column).upper(): str(column) for column in columns}

    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]

    return None


def build_session():
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=12,
        pool_maxsize=12,
    )

    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "TemporalObservationDriftResearch/0.1 "
                "(expanded-tree coverage audit)"
            )
        }
    )

    return session


SESSION = build_session()


def api_get(endpoint, params):
    response = SESSION.get(
        f"{GBIF_API}{endpoint}",
        params=params,
        timeout=(30, 240),
    )
    response.raise_for_status()
    time.sleep(REQUEST_PAUSE_SECONDS)
    return response.json()


def occurrence_count(
    taxon_key,
    start_year,
    end_year,
):
    payload = api_get(
        "/occurrence/search",
        {
            "datasetKey": INAT_DATASET_KEY,
            "country": "US",
            "stateProvince": STATE_NAME,
            "taxonKey": int(taxon_key),
            "year": f"{int(start_year)},{int(end_year)}",
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "occurrenceStatus": "PRESENT",
            "basisOfRecord": "HUMAN_OBSERVATION",
            "limit": 0,
        },
    )

    return int(payload.get("count", 0))


def canonical_name_from_reference(
    frame,
    genus_col,
    species_col,
    scientific_col,
):
    if genus_col is not None and species_col is not None:
        genus = (
            frame[genus_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        species = (
            frame[species_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        return (genus + " " + species).str.strip()

    if scientific_col is not None:
        return (
            frame[scientific_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.extract(
                r"^([A-Z][A-Za-z-]+\s+[a-z][A-Za-z-]+)",
                expand=False,
            )
            .fillna("")
        )

    return pd.Series(
        "",
        index=frame.index,
        dtype="string",
    )


def is_species_level_name(name):
    if name is None:
        return False

    text = str(name).strip()

    if not re.match(
        r"^[A-Z][A-Za-z-]+\s+[a-z][A-Za-z-]+$",
        text,
    ):
        return False

    lowered = text.lower()

    blocked_tokens = [
        " spp",
        " sp.",
        " hybrid",
        " unknown",
        " other",
        " x ",
    ]

    return not any(
        token in lowered
        for token in blocked_tokens
    )


def print_stop(status, error, next_step):
    append_readme(
        f"""

## PA expanded tree-taxon audit stopped — {RUN_UTC}
- Status: **{status}**
- Reason: `{error}`
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
    # 1. Validate source files
    # --------------------------------------------------------
    missing_files = [
        str(path)
        for path in [
            PA_TREE_CSV,
            REF_SPECIES_CSV,
        ]
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing prior FIA files: "
            + " | ".join(missing_files)
        )

    # --------------------------------------------------------
    # 2. Resolve TREE columns and scan live species counts
    # --------------------------------------------------------
    tree_columns = pd.read_csv(
        PA_TREE_CSV,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    tree_species_col = resolve_column(
        tree_columns,
        ["SPCD", "SPECIES_CODE"],
    )
    tree_status_col = resolve_column(
        tree_columns,
        ["STATUSCD", "STATUS_CODE"],
    )
    tree_year_col = resolve_column(
        tree_columns,
        ["INVYR", "INVENTORY_YEAR"],
    )

    if tree_species_col is None:
        raise ValueError(
            "PA TREE species-code field was not found."
        )

    tree_usecols = [tree_species_col]

    if tree_status_col is not None:
        tree_usecols.append(tree_status_col)

    if tree_year_col is not None:
        tree_usecols.append(tree_year_col)

    tree_usecols = list(dict.fromkeys(tree_usecols))

    species_counter = defaultdict(int)
    tree_rows_scanned = 0
    live_rows_retained = 0

    tree_reader = pd.read_csv(
        PA_TREE_CSV,
        usecols=tree_usecols,
        chunksize=TREE_CHUNK_SIZE,
        low_memory=False,
        encoding_errors="replace",
    )

    for chunk in tqdm(
        tree_reader,
        desc="Scanning PA FIA TREE chunks",
        unit="chunk",
    ):
        tree_rows_scanned += len(chunk)

        chunk[tree_species_col] = pd.to_numeric(
            chunk[tree_species_col],
            errors="coerce",
        )

        keep = chunk[tree_species_col].notna()

        if tree_status_col is not None:
            chunk[tree_status_col] = pd.to_numeric(
                chunk[tree_status_col],
                errors="coerce",
            )
            keep &= chunk[tree_status_col].eq(1)

        if tree_year_col is not None:
            chunk[tree_year_col] = pd.to_numeric(
                chunk[tree_year_col],
                errors="coerce",
            )
            keep &= chunk[tree_year_col].ge(2000)

        work = chunk.loc[
            keep,
            tree_species_col,
        ].astype(int)

        live_rows_retained += len(work)

        counts = work.value_counts()

        for species_code, count in counts.items():
            species_counter[int(species_code)] += int(count)

    fia_count_df = pd.DataFrame(
        [
            {
                "species_code": species_code,
                "fia_live_tree_records": count,
            }
            for species_code, count
            in species_counter.items()
        ]
    )

    fia_count_df = fia_count_df[
        fia_count_df["fia_live_tree_records"]
        >= MIN_FIA_LIVE_TREE_RECORDS
    ].copy()

    # --------------------------------------------------------
    # 3. Resolve names from REF_SPECIES
    # --------------------------------------------------------
    reference_columns = pd.read_csv(
        REF_SPECIES_CSV,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    ref_species_col = resolve_column(
        reference_columns,
        ["SPCD", "SPECIES_CODE"],
    )
    genus_col = resolve_column(
        reference_columns,
        ["GENUS"],
    )
    species_epithet_col = resolve_column(
        reference_columns,
        ["SPECIES"],
    )
    scientific_col = resolve_column(
        reference_columns,
        [
            "SCIENTIFIC_NAME",
            "SCIENTIFIC_NAME_W_AUTHOR",
        ],
    )
    common_col = resolve_column(
        reference_columns,
        ["COMMON_NAME", "COMMONNAME"],
    )

    if ref_species_col is None:
        raise ValueError(
            "REF_SPECIES code field was not found."
        )

    ref_usecols = [
        ref_species_col,
    ]

    for column in [
        genus_col,
        species_epithet_col,
        scientific_col,
        common_col,
    ]:
        if column is not None:
            ref_usecols.append(column)

    ref_usecols = list(dict.fromkeys(ref_usecols))

    reference_df = pd.read_csv(
        REF_SPECIES_CSV,
        usecols=ref_usecols,
        low_memory=False,
        encoding_errors="replace",
    )

    reference_df[ref_species_col] = pd.to_numeric(
        reference_df[ref_species_col],
        errors="coerce",
    )

    reference_df = reference_df.dropna(
        subset=[ref_species_col]
    ).copy()

    reference_df[ref_species_col] = (
        reference_df[ref_species_col].astype(int)
    )

    reference_df["canonical_name"] = (
        canonical_name_from_reference(
            reference_df,
            genus_col,
            species_epithet_col,
            scientific_col,
        )
    )

    reference_df["common_name"] = (
        reference_df[common_col]
        if common_col is not None
        else pd.NA
    )

    reference_df = (
        reference_df[
            [
                ref_species_col,
                "canonical_name",
                "common_name",
            ]
        ]
        .rename(
            columns={
                ref_species_col: "species_code",
            }
        )
        .drop_duplicates("species_code")
    )

    candidate_df = fia_count_df.merge(
        reference_df,
        on="species_code",
        how="left",
    )

    candidate_df["species_level_name"] = (
        candidate_df["canonical_name"].map(
            is_species_level_name
        )
    )

    candidate_df = candidate_df[
        candidate_df["species_level_name"]
    ].copy()

    candidate_df = candidate_df.sort_values(
        "fia_live_tree_records",
        ascending=False,
    ).reset_index(drop=True)

    candidate_df.to_csv(
        OUT_DIR / "pa_fia_species_candidates.csv",
        index=False,
    )

    if len(candidate_df) < 20:
        raise AuditStop(
            "TOO_FEW_FIA_SPECIES_CANDIDATES",
            (
                f"Only {len(candidate_df)} species-level "
                "FIA taxa were found."
            ),
            (
                "Review REF_SPECIES name fields before "
                "querying GBIF."
            ),
        )

    print("\nPA FIA SPECIES CANDIDATES")
    display(candidate_df)

    # --------------------------------------------------------
    # 4. Resolve current accepted GBIF names
    # --------------------------------------------------------
    taxonomy_rows = []

    for row in tqdm(
        candidate_df.itertuples(index=False),
        total=len(candidate_df),
        desc="Resolving expanded tree taxa in GBIF",
        unit="taxon",
    ):
        payload = api_get(
            "/species/match",
            {
                "name": row.canonical_name,
                "kingdom": "Plantae",
                "verbose": "true",
            },
        )

        usage_key = payload.get("usageKey")
        accepted_key = payload.get(
            "acceptedUsageKey"
        )

        selected_key = (
            accepted_key
            if accepted_key is not None
            else usage_key
        )

        match_type = payload.get("matchType")
        rank = payload.get("rank")
        confidence = payload.get("confidence")

        accepted_name = (
            payload.get("acceptedScientificName")
            or payload.get("scientificName")
            or payload.get("canonicalName")
        )

        resolution_pass = bool(
            selected_key is not None
            and match_type in {
                "EXACT",
                "FUZZY",
            }
            and str(rank).upper()
            in {
                "SPECIES",
                "SUBSPECIES",
                "VARIETY",
            }
            and (
                confidence is None
                or float(confidence) >= 80
            )
        )

        taxonomy_rows.append(
            {
                "species_code": int(
                    row.species_code
                ),
                "fia_canonical_name": (
                    row.canonical_name
                ),
                "common_name": row.common_name,
                "fia_live_tree_records": int(
                    row.fia_live_tree_records
                ),
                "gbif_taxon_key": (
                    int(selected_key)
                    if selected_key is not None
                    else np.nan
                ),
                "gbif_accepted_name": accepted_name,
                "gbif_match_type": match_type,
                "gbif_rank": rank,
                "gbif_confidence": confidence,
                "gbif_status": payload.get("status"),
                "taxon_resolution_pass": (
                    resolution_pass
                ),
                "taxon_note": payload.get("note"),
            }
        )

    taxonomy_df = pd.DataFrame(taxonomy_rows)

    taxonomy_df.to_csv(
        OUT_DIR / "expanded_gbif_taxonomy_resolution.csv",
        index=False,
    )

    print("\nEXPANDED GBIF TAXONOMY RESOLUTION")
    display(taxonomy_df)

    resolved_df = taxonomy_df[
        taxonomy_df["taxon_resolution_pass"]
        & taxonomy_df["gbif_taxon_key"].notna()
    ].copy()

    resolved_df["gbif_taxon_key"] = (
        resolved_df["gbif_taxon_key"].astype(int)
    )

    # Avoid downloading the same accepted taxon twice if FIA synonyms
    # resolve to one GBIF accepted key.
    resolved_df = (
        resolved_df.sort_values(
            "fia_live_tree_records",
            ascending=False,
        )
        .drop_duplicates(
            "gbif_taxon_key",
            keep="first",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 5. Count iNaturalist records in early and late windows
    # --------------------------------------------------------
    count_rows = []

    for row in tqdm(
        resolved_df.itertuples(index=False),
        total=len(resolved_df),
        desc="Counting PA iNaturalist early/late records",
        unit="taxon",
    ):
        early_count = occurrence_count(
            row.gbif_taxon_key,
            EARLY_START_YEAR,
            EARLY_END_YEAR,
        )

        late_count = occurrence_count(
            row.gbif_taxon_key,
            LATE_START_YEAR,
            LATE_END_YEAR,
        )

        count_rows.append(
            {
                "species_code": int(
                    row.species_code
                ),
                "fia_canonical_name": (
                    row.fia_canonical_name
                ),
                "gbif_accepted_name": (
                    row.gbif_accepted_name
                ),
                "common_name": row.common_name,
                "gbif_taxon_key": int(
                    row.gbif_taxon_key
                ),
                "fia_live_tree_records": int(
                    row.fia_live_tree_records
                ),
                "early_count": early_count,
                "late_count": late_count,
                "minimum_window_count": min(
                    early_count,
                    late_count,
                ),
                "total_window_count": (
                    early_count + late_count
                ),
                "early_exceeds_search_cap": (
                    early_count
                    > GBIF_SEARCH_ACCESS_CAP
                ),
                "late_exceeds_search_cap": (
                    late_count
                    > GBIF_SEARCH_ACCESS_CAP
                ),
            }
        )

    count_df = pd.DataFrame(count_rows)

    count_df["coverage_pass"] = (
        count_df["early_count"]
        >= MIN_INAT_RECORDS_EACH_WINDOW
    ) & (
        count_df["late_count"]
        >= MIN_INAT_RECORDS_EACH_WINDOW
    )

    count_df = count_df.sort_values(
        [
            "coverage_pass",
            "minimum_window_count",
            "total_window_count",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    count_df.to_csv(
        OUT_DIR / "expanded_tree_early_late_counts.csv",
        index=False,
    )

    print("\nEXPANDED TREE EARLY/LATE COUNTS")
    display(count_df)

    # --------------------------------------------------------
    # 6. Select a count-defined expanded observer frame
    # --------------------------------------------------------
    eligible_df = count_df[
        count_df["coverage_pass"]
        & ~count_df[
            "early_exceeds_search_cap"
        ]
        & ~count_df[
            "late_exceeds_search_cap"
        ]
    ].copy()

    eligible_df = eligible_df.sort_values(
        [
            "minimum_window_count",
            "total_window_count",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    selected_rows = []
    cumulative_records = 0

    for row in eligible_df.to_dict("records"):
        if len(selected_rows) >= MAX_SELECTED_TAXA:
            break

        proposed_total = (
            cumulative_records
            + int(row["total_window_count"])
        )

        if (
            selected_rows
            and proposed_total
            > MAX_ESTIMATED_DOWNLOAD_RECORDS
        ):
            continue

        selected_rows.append(row)
        cumulative_records = proposed_total

    selected_df = pd.DataFrame(selected_rows)

    if not selected_df.empty:
        selected_df["selection_rank"] = np.arange(
            1,
            len(selected_df) + 1,
        )

    selected_df.to_csv(
        OUT_DIR / "selected_expanded_tree_taxa.csv",
        index=False,
    )

    n_selected_taxa = len(selected_df)
    estimated_download_records = int(
        selected_df["total_window_count"].sum()
    ) if n_selected_taxa else 0

    estimated_pages = int(
        np.ceil(
            selected_df["early_count"] / 300
        ).sum()
        + np.ceil(
            selected_df["late_count"] / 300
        ).sum()
    ) if n_selected_taxa else 0

    unresolved_count = int(
        (~taxonomy_df["taxon_resolution_pass"]).sum()
    )

    passed = bool(
        n_selected_taxa >= MIN_SELECTED_TAXA_GO
        and estimated_download_records
        <= MAX_ESTIMATED_DOWNLOAD_RECORDS
    )

    if passed:
        status = "GO_FOR_EXPANDED_PA_TREE_DOWNLOAD"
        next_step = (
            "Download only the selected expanded PA tree taxa, "
            "then repeat the observer-turnover and matched-observer "
            "decomposition using the broader taxonomic frame."
        )
    elif len(eligible_df) >= 20:
        status = "EXPANDED_FRAME_PARTIALLY_FEASIBLE"
        next_step = (
            "The expanded set improves coverage but remains smaller "
            "than the target. Download the selected set only if the "
            "estimated repeated-observer gain justifies it."
        )
    else:
        status = "EXPANDED_TREE_FRAME_INSUFFICIENT"
        next_step = (
            "Do not download more coordinates. A PA tree-only "
            "observer-trajectory analysis is unlikely to become stable; "
            "consider all vascular plants or a different region."
        )

    # --------------------------------------------------------
    # 7. Visual previews
    # --------------------------------------------------------
    plot_df = (
        count_df[
            count_df["coverage_pass"]
        ]
        .head(30)
        .sort_values(
            "minimum_window_count"
        )
    )

    if not plot_df.empty:
        labels = plot_df[
            "gbif_accepted_name"
        ].fillna(
            plot_df["fia_canonical_name"]
        )

        y = np.arange(len(plot_df))

        plt.figure(figsize=(11, 9))
        plt.barh(
            y,
            plot_df["early_count"],
            label="2013–2018",
        )
        plt.barh(
            y,
            plot_df["late_count"],
            left=plot_df["early_count"],
            label="2020–2025",
        )
        plt.yticks(y, labels)
        plt.xlabel("iNaturalist coordinate records")
        plt.ylabel("Tree taxon")
        plt.title(
            "PA expanded tree taxa: early and late coverage"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 8. README
    # --------------------------------------------------------
    top_selected = (
        selected_df[
            [
                "species_code",
                "gbif_accepted_name",
                "early_count",
                "late_count",
                "total_window_count",
            ]
        ]
        .head(15)
        .to_dict("records")
        if n_selected_taxa
        else []
    )

    append_readme(
        f"""

## PA expanded tree-taxon observer coverage audit — {RUN_UTC}

### Reason
The 20-species benchmark frame produced only 12 repeated observer labels and
16 repeated observer-species pairs. The observer mechanism therefore could not
be inferred without broadening the target group.

### Expanded frame definition
- Source species pool: live species-level taxa present in PA FIA TREE records
- FIA minimum live-tree records: {MIN_FIA_LIVE_TREE_RECORDS}
- Early iNaturalist window: {EARLY_START_YEAR}–{EARLY_END_YEAR}
- Late iNaturalist window: {LATE_START_YEAR}–{LATE_END_YEAR}
- Minimum records per taxon in each window:
  {MIN_INAT_RECORDS_EACH_WINDOW}
- Selection uses record coverage only, not spatial trend direction
- Maximum selected taxa: {MAX_SELECTED_TAXA}
- Maximum estimated coordinate records:
  {MAX_ESTIMATED_DOWNLOAD_RECORDS:,}
- GBIF dataset key: {INAT_DATASET_KEY}
- Dataset DOI: {INAT_DATASET_DOI}
- Access date: {ACCESS_DATE}
- GPU: not used

### Results
- PA TREE rows scanned: {tree_rows_scanned:,}
- Live rows retained: {live_rows_retained:,}
- Species-level FIA candidates: {len(candidate_df)}
- Reliably resolved unique GBIF taxa: {len(resolved_df)}
- Unresolved/failed taxa: {unresolved_count}
- Taxa passing early/late record threshold: {len(eligible_df)}
- Selected taxa: {n_selected_taxa}
- Estimated records for selected windows:
  {estimated_download_records:,}
- Estimated API pages: {estimated_pages}
- Status: **{status}**
- Passed: {passed}
- Next step: {next_step}

### Interpretation boundary
This audit expands only the observer-behaviour frame. The original 20-species
set remains the locked ecological benchmark for the semi-synthetic calibration.
No occurrence coordinates or spatial trends were downloaded or estimated here.
"""
    )

    # --------------------------------------------------------
    # 9. Compact output
    # --------------------------------------------------------
    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"STATE: {STATE}")
    print(
        f"EARLY_WINDOW: "
        f"{EARLY_START_YEAR}-{EARLY_END_YEAR}"
    )
    print(
        f"LATE_WINDOW: "
        f"{LATE_START_YEAR}-{LATE_END_YEAR}"
    )
    print(
        f"PA_TREE_ROWS_SCANNED: "
        f"{tree_rows_scanned:,}"
    )
    print(
        f"PA_LIVE_TREE_ROWS_RETAINED: "
        f"{live_rows_retained:,}"
    )
    print(
        f"N_FIA_SPECIES_LEVEL_CANDIDATES: "
        f"{len(candidate_df)}"
    )
    print(
        f"N_RESOLVED_UNIQUE_GBIF_TAXA: "
        f"{len(resolved_df)}"
    )
    print(
        f"N_UNRESOLVED_OR_FAILED_TAXA: "
        f"{unresolved_count}"
    )
    print(
        f"N_EARLY_LATE_ELIGIBLE_TAXA: "
        f"{len(eligible_df)}"
    )
    print(
        f"N_SELECTED_EXPANDED_TAXA: "
        f"{n_selected_taxa}"
    )
    print(
        f"ESTIMATED_DOWNLOAD_RECORDS: "
        f"{estimated_download_records:,}"
    )
    print(
        f"ESTIMATED_API_PAGES: "
        f"{estimated_pages:,}"
    )
    print(
        "TOP_SELECTED_TAXA: "
        + json.dumps(
            top_selected,
            ensure_ascii=False,
        )
    )
    print(
        "SUCCESS_CRITERIA: >=40 taxa with >=30 records "
        "in both windows and estimated total <=300,000 records"
    )
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except AuditStop as exc:
    print_stop(
        exc.status,
        exc.message,
        exc.next_step,
    )

except requests.HTTPError as exc:
    response = exc.response
    status_code = (
        response.status_code
        if response is not None
        else "UNKNOWN"
    )
    url = (
        response.url
        if response is not None
        else "UNKNOWN"
    )

    print_stop(
        "GBIF_HTTP_FAILED",
        f"HTTP {status_code} for {url}",
        (
            "Retry once. Existing FIA scans need not be repeated "
            "if a later cache is added."
        ),
    )

except requests.RequestException as exc:
    print_stop(
        "GBIF_CONNECTION_FAILED",
        repr(exc),
        (
            "Confirm Kaggle Internet is enabled and retry "
            "this audit cell."
        ),
    )

except Exception as exc:
    print_stop(
        "CELL10A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        (
            "Save this diagnostic summary and the last displayed table. "
            "Do not rerun earlier downloads."
        ),
    )
