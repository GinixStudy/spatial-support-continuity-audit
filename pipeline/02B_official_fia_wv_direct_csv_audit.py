
# Cell 02B — Corrected official FIA download and WV coverage audit
# Replaces Cell 02A.
#
# Change:
# FIA state data are individual CSV files:
#   WV_PLOT.csv
#   WV_TREE.csv
# They are not contained in WV.zip.
#
# Goal:
# Confirm that current official WV FIA data contain enough multi-year plot and
# live-tree records for a small temporal trend validation.
#
# This cell does NOT estimate migration speed yet.

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import json
import zipfile
import warnings

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 180)
pd.set_option("display.width", 180)

# ============================================================
# Configuration
# ============================================================
STATE = "WV"
MIN_YEAR = 2000
CHUNK_SIZE = 250_000

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
RAW_DIR = BASE_DIR / "raw"
DERIVED_DIR = BASE_DIR / "derived"

RAW_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://apps.fs.usda.gov/fia/datamart/CSV"
PLOT_URL = f"{BASE_URL}/{STATE}_PLOT.csv"
TREE_URL = f"{BASE_URL}/{STATE}_TREE.csv"
REF_URL = f"{BASE_URL}/FIADB_REFERENCE.zip"

PLOT_CSV = RAW_DIR / f"{STATE}_PLOT.csv"
TREE_CSV = RAW_DIR / f"{STATE}_TREE.csv"
REF_ZIP = RAW_DIR / "FIADB_REFERENCE.zip"
REF_SPECIES_CSV = RAW_DIR / "REF_SPECIES.csv"

RUN_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

print("RUN_UTC:", RUN_UTC)
print("STATE:", STATE)
print("PLOT_URL:", PLOT_URL)
print("TREE_URL:", TREE_URL)
print("REF_URL:", REF_URL)
print("OUT_DIR:", BASE_DIR)


# ============================================================
# Helpers
# ============================================================
def human_bytes(value):
    value = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024


def append_readme(text):
    readme_path = BASE_DIR / "README.md"
    existing = (
        readme_path.read_text(encoding="utf-8")
        if readme_path.exists()
        else "# Temporal Observation Drift — FIA Validation\n"
    )
    readme_path.write_text(existing + text, encoding="utf-8")


def stream_download(url, destination, label):
    """Download one file with a progress bar; reuse a completed file."""
    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"Reusing: {destination.name} "
            f"({human_bytes(destination.stat().st_size)})"
        )
        return {
            "url": url,
            "path": str(destination),
            "status": "reused",
            "size_bytes": destination.stat().st_size,
        }

    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    headers = {"User-Agent": "Mozilla/5.0 FIA-research-audit/1.0"}

    with requests.get(
        url,
        stream=True,
        timeout=(30, 600),
        allow_redirects=True,
        headers=headers,
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(temp_path, "wb") as handle, tqdm(
            total=total if total > 0 else None,
            desc=label,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)
                    progress.update(len(block))

    temp_path.replace(destination)

    return {
        "url": url,
        "path": str(destination),
        "status": "downloaded",
        "size_bytes": destination.stat().st_size,
    }


def extract_ref_species(zip_path, destination):
    """Extract only REF_SPECIES.csv with a progress bar."""
    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"Reusing: {destination.name} "
            f"({human_bytes(destination.stat().st_size)})"
        )
        return

    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name for name in archive.namelist()
            if Path(name).name.upper() == "REF_SPECIES.CSV"
        ]

        if not members:
            raise FileNotFoundError(
                "REF_SPECIES.csv was not found inside FIADB_REFERENCE.zip"
            )

        member = members[0]
        info = archive.getinfo(member)

        with archive.open(member) as source, open(destination, "wb") as target, tqdm(
            total=info.file_size,
            desc="Extracting REF_SPECIES",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
                progress.update(len(block))


def read_columns(path):
    return pd.read_csv(
        path,
        nrows=0,
        low_memory=False,
        encoding_errors="replace",
    ).columns.tolist()


def resolve_column(columns, candidates):
    lookup = {str(column).upper(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    return None


def print_failure(status, error, next_step):
    append_readme(
        f"""

## WV official FIA audit failure — {RUN_UTC}
- Status: **{status}**
- Error: `{error}`
- Next step: {next_step}
"""
    )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"ERROR: {error}")
    print(f"PLOT_URL: {PLOT_URL}")
    print(f"TREE_URL: {TREE_URL}")
    print(f"REF_URL: {REF_URL}")
    print("PASSED: False")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")


# ============================================================
# Main
# ============================================================
try:
    # --------------------------------------------------------
    # 1. Download current official files
    # --------------------------------------------------------
    manifest = []

    for url, path, label in [
        (PLOT_URL, PLOT_CSV, f"Downloading {STATE}_PLOT"),
        (TREE_URL, TREE_CSV, f"Downloading {STATE}_TREE"),
        (REF_URL, REF_ZIP, "Downloading FIA references"),
    ]:
        manifest.append(stream_download(url, path, label))

    manifest_df = pd.DataFrame(manifest)
    manifest_df["size"] = manifest_df["size_bytes"].map(human_bytes)
    manifest_df["download_date_utc"] = RUN_UTC

    print("\nDOWNLOAD MANIFEST")
    display(
        manifest_df[
            ["url", "status", "size", "download_date_utc"]
        ]
    )

    manifest_df.to_csv(
        DERIVED_DIR / f"{STATE}_download_manifest.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Extract the species reference table
    # --------------------------------------------------------
    extract_ref_species(REF_ZIP, REF_SPECIES_CSV)

    # --------------------------------------------------------
    # 3. Resolve real fields
    # --------------------------------------------------------
    plot_columns = read_columns(PLOT_CSV)
    tree_columns = read_columns(TREE_CSV)
    ref_columns = read_columns(REF_SPECIES_CSV)

    plot_fields = {
        "plot_key": resolve_column(plot_columns, ["CN", "PLOT_CN"]),
        "inventory_year": resolve_column(
            plot_columns, ["INVYR", "INVENTORY_YEAR"]
        ),
        "latitude": resolve_column(plot_columns, ["LAT", "LATITUDE"]),
        "longitude": resolve_column(plot_columns, ["LON", "LONGITUDE"]),
        "elevation": resolve_column(plot_columns, ["ELEV", "ELEVATION"]),
        "state_code": resolve_column(
            plot_columns, ["STATECD", "STATE_CODE"]
        ),
    }

    tree_fields = {
        "plot_key": resolve_column(tree_columns, ["PLT_CN", "PLOT_CN"]),
        "inventory_year": resolve_column(
            tree_columns, ["INVYR", "INVENTORY_YEAR"]
        ),
        "species_code": resolve_column(
            tree_columns, ["SPCD", "SPECIES_CODE"]
        ),
        "status_code": resolve_column(
            tree_columns, ["STATUSCD", "STATUS_CODE"]
        ),
        "diameter": resolve_column(tree_columns, ["DIA", "DBH"]),
    }

    ref_fields = {
        "species_code": resolve_column(
            ref_columns, ["SPCD", "SPECIES_CODE"]
        ),
        "common_name": resolve_column(
            ref_columns, ["COMMON_NAME", "COMMONNAME"]
        ),
        "genus": resolve_column(ref_columns, ["GENUS"]),
        "species": resolve_column(ref_columns, ["SPECIES"]),
        "scientific_name": resolve_column(
            ref_columns, ["SCIENTIFIC_NAME", "SCIENTIFIC_NAME_W_AUTHOR"]
        ),
    }

    mapping_rows = []

    for table_name, mapping in [
        ("PLOT", plot_fields),
        ("TREE", tree_fields),
        ("REF_SPECIES", ref_fields),
    ]:
        for role, field in mapping.items():
            mapping_rows.append(
                {
                    "table": table_name,
                    "role": role,
                    "resolved_field": field,
                }
            )

    mapping_df = pd.DataFrame(mapping_rows)

    print("\nFIELD MAPPING")
    display(mapping_df)

    mapping_df.to_csv(
        DERIVED_DIR / f"{STATE}_field_mapping.csv",
        index=False,
    )

    required_fields_ready = all(
        [
            plot_fields["plot_key"],
            plot_fields["inventory_year"],
            plot_fields["latitude"],
            plot_fields["longitude"],
            tree_fields["plot_key"],
            tree_fields["inventory_year"],
            tree_fields["species_code"],
            ref_fields["species_code"],
        ]
    )

    if not required_fields_ready:
        raise ValueError(
            "One or more required PLOT/TREE/REF_SPECIES fields were not found."
        )

    # --------------------------------------------------------
    # 4. Scan PLOT in chunks
    # --------------------------------------------------------
    plot_usecols = list(
        dict.fromkeys(
            field for field in plot_fields.values() if field is not None
        )
    )

    year_stats = defaultdict(
        lambda: {
            "plot_rows": 0,
            "valid_coordinate_rows": 0,
            "lat_sum": 0.0,
            "lon_sum": 0.0,
            "elev_sum": 0.0,
            "elev_n": 0,
        }
    )

    plot_rows_total = 0

    plot_reader = pd.read_csv(
        PLOT_CSV,
        usecols=plot_usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding_errors="replace",
    )

    for chunk in tqdm(
        plot_reader,
        desc="Scanning PLOT chunks",
        unit="chunk",
    ):
        plot_rows_total += len(chunk)

        year_col = plot_fields["inventory_year"]
        lat_col = plot_fields["latitude"]
        lon_col = plot_fields["longitude"]
        elev_col = plot_fields["elevation"]

        chunk[year_col] = pd.to_numeric(chunk[year_col], errors="coerce")
        chunk[lat_col] = pd.to_numeric(chunk[lat_col], errors="coerce")
        chunk[lon_col] = pd.to_numeric(chunk[lon_col], errors="coerce")

        chunk = chunk.dropna(subset=[year_col]).copy()
        chunk[year_col] = chunk[year_col].astype(int)

        for year, group in chunk.groupby(year_col):
            stats = year_stats[int(year)]
            stats["plot_rows"] += len(group)

            valid = (
                group[lat_col].between(24, 50)
                & group[lon_col].between(-130, -60)
            )
            coord_group = group.loc[valid]

            stats["valid_coordinate_rows"] += len(coord_group)
            stats["lat_sum"] += coord_group[lat_col].sum()
            stats["lon_sum"] += coord_group[lon_col].sum()

            if elev_col is not None:
                elevation = pd.to_numeric(
                    coord_group[elev_col],
                    errors="coerce",
                )
                stats["elev_sum"] += elevation.sum()
                stats["elev_n"] += int(elevation.notna().sum())

    plot_summary_rows = []

    for year in sorted(year_stats):
        stats = year_stats[year]
        coord_n = stats["valid_coordinate_rows"]

        plot_summary_rows.append(
            {
                "year": year,
                "plot_rows": stats["plot_rows"],
                "valid_coordinate_rows": coord_n,
                "mean_latitude": (
                    stats["lat_sum"] / coord_n
                    if coord_n else np.nan
                ),
                "mean_longitude": (
                    stats["lon_sum"] / coord_n
                    if coord_n else np.nan
                ),
                "mean_elevation": (
                    stats["elev_sum"] / stats["elev_n"]
                    if stats["elev_n"] else np.nan
                ),
            }
        )

    plot_summary_df = pd.DataFrame(plot_summary_rows)

    plot_summary_df.to_parquet(
        DERIVED_DIR / f"{STATE}_plot_year_summary.parquet",
        index=False,
    )
    plot_summary_df.to_csv(
        DERIVED_DIR / f"{STATE}_plot_year_summary.csv",
        index=False,
    )

    print("\nPLOT YEAR SUMMARY")
    display(plot_summary_df.tail(30))

    # --------------------------------------------------------
    # 5. Load species reference
    # --------------------------------------------------------
    ref_usecols = list(
        dict.fromkeys(
            field for field in ref_fields.values() if field is not None
        )
    )

    ref_df = pd.read_csv(
        REF_SPECIES_CSV,
        usecols=ref_usecols,
        low_memory=False,
        encoding_errors="replace",
    )

    ref_code_col = ref_fields["species_code"]
    ref_df[ref_code_col] = pd.to_numeric(
        ref_df[ref_code_col],
        errors="coerce",
    )
    ref_df = ref_df.dropna(subset=[ref_code_col]).copy()
    ref_df[ref_code_col] = ref_df[ref_code_col].astype(int)

    if ref_fields["scientific_name"] is not None:
        ref_df["scientific_name_final"] = ref_df[
            ref_fields["scientific_name"]
        ].astype("string")
    elif ref_fields["genus"] is not None and ref_fields["species"] is not None:
        ref_df["scientific_name_final"] = (
            ref_df[ref_fields["genus"]].fillna("").astype(str).str.strip()
            + " "
            + ref_df[ref_fields["species"]].fillna("").astype(str).str.strip()
        ).str.strip()
    else:
        ref_df["scientific_name_final"] = pd.NA

    if ref_fields["common_name"] is not None:
        ref_df["common_name_final"] = ref_df[
            ref_fields["common_name"]
        ]
    else:
        ref_df["common_name_final"] = pd.NA

    ref_df = (
        ref_df[
            [
                ref_code_col,
                "scientific_name_final",
                "common_name_final",
            ]
        ]
        .rename(
            columns={
                ref_code_col: "species_code",
                "scientific_name_final": "scientific_name",
                "common_name_final": "common_name",
            }
        )
        .drop_duplicates("species_code")
    )

    # --------------------------------------------------------
    # 6. Scan TREE in chunks
    # --------------------------------------------------------
    tree_usecols = list(
        dict.fromkeys(
            field for field in tree_fields.values() if field is not None
        )
    )

    species_year_counts = defaultdict(int)
    tree_rows_total = 0
    retained_tree_rows = 0

    tree_reader = pd.read_csv(
        TREE_CSV,
        usecols=tree_usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding_errors="replace",
    )

    for chunk in tqdm(
        tree_reader,
        desc="Scanning TREE chunks",
        unit="chunk",
    ):
        tree_rows_total += len(chunk)

        year_col = tree_fields["inventory_year"]
        species_col = tree_fields["species_code"]
        status_col = tree_fields["status_code"]

        chunk[year_col] = pd.to_numeric(
            chunk[year_col],
            errors="coerce",
        )
        chunk[species_col] = pd.to_numeric(
            chunk[species_col],
            errors="coerce",
        )

        valid = (
            chunk[year_col].notna()
            & chunk[species_col].notna()
        )

        if status_col is not None:
            chunk[status_col] = pd.to_numeric(
                chunk[status_col],
                errors="coerce",
            )
            valid &= chunk[status_col].eq(1)

        work = chunk.loc[
            valid,
            [year_col, species_col],
        ].copy()

        retained_tree_rows += len(work)

        work[year_col] = work[year_col].astype(int)
        work[species_col] = work[species_col].astype(int)

        counts = work.value_counts([species_col, year_col])

        for (species_code, year), count in counts.items():
            species_year_counts[
                (int(species_code), int(year))
            ] += int(count)

    species_year_df = pd.DataFrame(
        [
            {
                "species_code": species_code,
                "year": year,
                "tree_records": count,
            }
            for (species_code, year), count
            in species_year_counts.items()
        ]
    )

    species_year_df = species_year_df[
        species_year_df["year"] >= MIN_YEAR
    ].copy()

    species_year_df = species_year_df.merge(
        ref_df,
        on="species_code",
        how="left",
    )

    species_year_df.to_parquet(
        DERIVED_DIR / f"{STATE}_species_year_tree_records.parquet",
        index=False,
    )

    species_coverage_df = (
        species_year_df.groupby(
            ["species_code", "scientific_name", "common_name"],
            dropna=False,
            as_index=False,
        )
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_years=("year", "nunique"),
            total_tree_records=("tree_records", "sum"),
            median_records_per_year=("tree_records", "median"),
            min_records_per_year=("tree_records", "min"),
        )
        .sort_values(
            ["n_years", "total_tree_records"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    species_coverage_df.to_parquet(
        DERIVED_DIR / f"{STATE}_species_coverage.parquet",
        index=False,
    )
    species_coverage_df.to_csv(
        DERIVED_DIR / f"{STATE}_species_coverage.csv",
        index=False,
    )

    print("\nTOP SPECIES BY MULTI-YEAR COVERAGE")
    display(species_coverage_df.head(40))

    # --------------------------------------------------------
    # 7. Preliminary Go / No-Go
    # --------------------------------------------------------
    candidate_species_df = species_coverage_df[
        (species_coverage_df["n_years"] >= 10)
        & (species_coverage_df["median_records_per_year"] >= 30)
    ].copy()

    latest_plot_year = (
        int(plot_summary_df["year"].max())
        if not plot_summary_df.empty
        else None
    )

    n_recent_plot_years = (
        plot_summary_df.loc[
            plot_summary_df["year"] >= MIN_YEAR,
            "year",
        ].nunique()
    )

    n_candidate_species = len(candidate_species_df)

    passed = bool(
        latest_plot_year is not None
        and latest_plot_year >= 2020
        and n_recent_plot_years >= 15
        and n_candidate_species >= 20
    )

    status = (
        "GO_FOR_WV_PLOT_TREE_JOIN"
        if passed
        else "TREND_COVERAGE_REVIEW_REQUIRED"
    )

    # --------------------------------------------------------
    # 8. Visual previews
    # --------------------------------------------------------
    recent_plot_df = plot_summary_df[
        plot_summary_df["year"] >= MIN_YEAR
    ].copy()

    if not recent_plot_df.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(
            recent_plot_df["year"],
            recent_plot_df["valid_coordinate_rows"],
            marker="o",
        )
        plt.xlabel("Inventory year")
        plt.ylabel("Plot rows with usable public coordinates")
        plt.title(f"FIA {STATE}: yearly usable plot records")
        plt.tight_layout()
        plt.show()

    if not species_coverage_df.empty:
        preview = (
            species_coverage_df.head(20)
            .sort_values("total_tree_records")
            .copy()
        )

        labels = preview["scientific_name"].fillna(
            preview["species_code"].astype(str)
        )

        plt.figure(figsize=(10, 7))
        plt.barh(labels, preview["total_tree_records"])
        plt.xlabel(f"Live-tree records since {MIN_YEAR}")
        plt.ylabel("Species")
        plt.title(f"FIA {STATE}: species coverage")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 9. README
    # --------------------------------------------------------
    append_readme(
        f"""

## Corrected official FIA WV audit — {RUN_UTC}

### Data source
- PLOT: `{PLOT_URL}`
- TREE: `{TREE_URL}`
- Species reference: `{REF_URL}`
- Download date: {RUN_UTC}

### Configuration
- State: {STATE}
- Minimum analysis year: {MIN_YEAR}
- Chunk size: {CHUNK_SIZE:,}
- Live-tree filter: STATUSCD = 1 when available
- GPU: not used

### Results
- PLOT rows scanned: {plot_rows_total:,}
- TREE rows scanned: {tree_rows_total:,}
- Retained live-tree rows: {retained_tree_rows:,}
- Latest PLOT inventory year: {latest_plot_year}
- Recent PLOT years: {n_recent_plot_years}
- Preliminary candidate species: {n_candidate_species}
- Status: **{status}**

### Interpretation boundary
This cell evaluates data coverage only. FIA record counts are not direct
population abundance estimates. No migration speed, climate effect, or
observation-drift effect has yet been estimated.
"""
    )

    top_candidates = (
        candidate_species_df[
            [
                "species_code",
                "scientific_name",
                "n_years",
                "median_records_per_year",
            ]
        ]
        .head(10)
        .to_dict("records")
    )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {BASE_DIR}")
    print(f"STATE: {STATE}")
    print(f"PLOT_URL: {PLOT_URL}")
    print(f"TREE_URL: {TREE_URL}")
    print(f"PLOT_FILE_SIZE: {human_bytes(PLOT_CSV.stat().st_size)}")
    print(f"TREE_FILE_SIZE: {human_bytes(TREE_CSV.stat().st_size)}")
    print(f"PLOT_ROWS_SCANNED: {plot_rows_total:,}")
    print(f"TREE_ROWS_SCANNED: {tree_rows_total:,}")
    print(f"LIVE_TREE_ROWS_RETAINED: {retained_tree_rows:,}")
    print(f"LATEST_PLOT_YEAR: {latest_plot_year}")
    print(f"N_RECENT_PLOT_YEARS: {n_recent_plot_years}")
    print(f"N_SPECIES_SINCE_{MIN_YEAR}: {len(species_coverage_df)}")
    print(f"N_PRELIMINARY_CANDIDATE_SPECIES: {n_candidate_species}")
    print(
        "TOP_CANDIDATES: "
        + json.dumps(top_candidates, ensure_ascii=False)
    )
    print(
        "SUCCESS_CRITERIA: latest year >= 2020; "
        ">=15 years since 2000; >=20 species with "
        ">=10 years and median >=30 live-tree records/year"
    )
    print(f"PASSED: {passed}")
    print(
        "NEXT_STEP: "
        + (
            "Join selected TREE records to PLOT coordinates and calculate raw annual latitude/elevation centers."
            if passed
            else "Review actual year and species coverage before expanding to more states."
        )
    )
    print("README_UPDATED: True")
    print("============================================")

except requests.HTTPError as exc:
    response = exc.response
    status_code = response.status_code if response is not None else "UNKNOWN"
    failing_url = response.url if response is not None else "UNKNOWN"

    print_failure(
        "DOWNLOAD_HTTP_FAILED",
        f"HTTP {status_code} for {failing_url}",
        "Send this COPY block back. The URL pattern or server response must be checked.",
    )

except requests.RequestException as exc:
    print_failure(
        "DOWNLOAD_CONNECTION_FAILED",
        repr(exc),
        "Confirm Internet is ON and retry once. If it repeats, send this block back.",
    )

except Exception as exc:
    print_failure(
        "PROCESSING_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        "Send this COPY block and the last displayed table back.",
    )
