
# Cell 03A — WV species annual centers and first observation-system drift screen
#
# Hypothesis
# Some apparent annual movement of tree-species occurrence centers is explained
# by annual movement of the FIA sampled-plot center.
#
# Minimal calculation
# 1. Join live TREE records to PLOT coordinates.
# 2. Collapse multiple trees to one species-presence record per plot-year.
# 3. Estimate annual species latitude/elevation centers.
# 4. Compare raw center slopes with slopes relative to the annual all-plot center.
#
# Important
# - This is a trend screen, not the final bias correction.
# - FIA tree counts are NOT treated as abundance weights.
# - No GPU is used.

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display
from scipy.stats import linregress, theilslopes

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 220)
pd.set_option("display.width", 220)

# ============================================================
# Configuration
# ============================================================
STATE = "WV"
MIN_YEAR = 2000
TREE_CHUNK_SIZE = 250_000

MIN_SPECIES_YEARS = 10
MIN_OCCUPIED_PLOTS_PER_YEAR = 10
STRONG_COVERAGE_YEARS = 15

# Preliminary signal thresholds only; not final paper thresholds.
MATERIAL_LAT_DIFF_KM_DECADE = 2.0
MIN_DIRECTION_REVERSAL_FRACTION = 0.10

BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
RAW_DIR = BASE_DIR / "raw"
DERIVED_DIR = BASE_DIR / "derived"
CENTER_DIR = DERIVED_DIR / "wv_center_screen"
CENTER_DIR.mkdir(parents=True, exist_ok=True)

PLOT_CSV = RAW_DIR / f"{STATE}_PLOT.csv"
TREE_CSV = RAW_DIR / f"{STATE}_TREE.csv"
COVERAGE_PARQUET = DERIVED_DIR / f"{STATE}_species_coverage.parquet"

RUN_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
RANDOM_SEED = 20260710

print("RUN_UTC:", RUN_UTC)
print("STATE:", STATE)
print("PLOT_CSV:", PLOT_CSV)
print("TREE_CSV:", TREE_CSV)
print("COVERAGE_PARQUET:", COVERAGE_PARQUET)
print("OUT_DIR:", CENTER_DIR)


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


def resolve_column(columns, candidates):
    lookup = {str(column).upper(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    return None


def safe_quantile(series, q):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(q))


def fit_slope(years, values):
    data = pd.DataFrame(
        {
            "year": pd.to_numeric(years, errors="coerce"),
            "value": pd.to_numeric(values, errors="coerce"),
        }
    ).dropna()

    data = data.drop_duplicates("year").sort_values("year")

    if len(data) < MIN_SPECIES_YEARS:
        return {
            "n_years": len(data),
            "first_year": (
                int(data["year"].min()) if len(data) else np.nan
            ),
            "last_year": (
                int(data["year"].max()) if len(data) else np.nan
            ),
            "ols_slope_per_year": np.nan,
            "ols_p_value": np.nan,
            "theil_sen_slope_per_year": np.nan,
        }

    ols = linregress(data["year"], data["value"])
    ts_slope = theilslopes(data["value"], data["year"])[0]

    return {
        "n_years": len(data),
        "first_year": int(data["year"].min()),
        "last_year": int(data["year"].max()),
        "ols_slope_per_year": float(ols.slope),
        "ols_p_value": float(ols.pvalue),
        "theil_sen_slope_per_year": float(ts_slope),
    }


def species_label(row):
    name = row.get("scientific_name")
    if pd.notna(name) and str(name).strip():
        return str(name)
    return f"SPCD {int(row['species_code'])}"


def print_failure(status, error, next_step):
    append_readme(
        f"""

## WV center-screen failure — {RUN_UTC}
- Status: **{status}**
- Error: `{error}`
- Next step: {next_step}
"""
    )

    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"ERROR: {error}")
    print(f"OUT_DIR: {CENTER_DIR}")
    print("PASSED: False")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")


# ============================================================
# Main
# ============================================================
try:
    # --------------------------------------------------------
    # 1. Validate previous outputs
    # --------------------------------------------------------
    required_files = [PLOT_CSV, TREE_CSV, COVERAGE_PARQUET]
    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Missing files from Cell 02B: " + " | ".join(missing_files)
        )

    # --------------------------------------------------------
    # 2. Select the species that passed the coverage screen
    # --------------------------------------------------------
    coverage_df = pd.read_parquet(COVERAGE_PARQUET)

    selected_species_df = coverage_df[
        (coverage_df["n_years"] >= 10)
        & (coverage_df["median_records_per_year"] >= 30)
    ].copy()

    selected_species_df = (
        selected_species_df
        .sort_values(
            ["n_years", "total_tree_records"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    selected_species_codes = set(
        pd.to_numeric(
            selected_species_df["species_code"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    if not selected_species_codes:
        raise ValueError("No species passed the Cell 02B coverage criteria.")

    selected_species_df.to_csv(
        CENTER_DIR / "selected_species.csv",
        index=False,
    )

    print("\nSELECTED SPECIES")
    display(selected_species_df.head(50))

    # --------------------------------------------------------
    # 3. Resolve PLOT and TREE fields
    # --------------------------------------------------------
    plot_columns = pd.read_csv(
        PLOT_CSV,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    tree_columns = pd.read_csv(
        TREE_CSV,
        nrows=0,
        encoding_errors="replace",
    ).columns.tolist()

    plot_fields = {
        "plot_key": resolve_column(plot_columns, ["CN", "PLOT_CN"]),
        "inventory_year": resolve_column(
            plot_columns, ["INVYR", "INVENTORY_YEAR"]
        ),
        "latitude": resolve_column(plot_columns, ["LAT", "LATITUDE"]),
        "longitude": resolve_column(plot_columns, ["LON", "LONGITUDE"]),
        "elevation": resolve_column(plot_columns, ["ELEV", "ELEVATION"]),
        "plot_status": resolve_column(
            plot_columns, ["PLOT_STATUS_CD", "PLOT_STATUS"]
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

    required_fields = [
        plot_fields["plot_key"],
        plot_fields["inventory_year"],
        plot_fields["latitude"],
        plot_fields["longitude"],
        tree_fields["plot_key"],
        tree_fields["inventory_year"],
        tree_fields["species_code"],
    ]

    if not all(required_fields):
        raise ValueError(
            f"Required field mapping failed. "
            f"PLOT={plot_fields}; TREE={tree_fields}"
        )

    field_mapping_df = pd.DataFrame(
        [
            {"table": "PLOT", "role": role, "field": field}
            for role, field in plot_fields.items()
        ]
        + [
            {"table": "TREE", "role": role, "field": field}
            for role, field in tree_fields.items()
        ]
    )

    print("\nFIELD MAPPING")
    display(field_mapping_df)

    field_mapping_df.to_csv(
        CENTER_DIR / "field_mapping.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Read the small PLOT table
    # --------------------------------------------------------
    plot_usecols = list(
        dict.fromkeys(
            field for field in plot_fields.values() if field is not None
        )
    )

    plot_df = pd.read_csv(
        PLOT_CSV,
        usecols=plot_usecols,
        low_memory=False,
        encoding_errors="replace",
        dtype={plot_fields["plot_key"]: "string"},
    )

    plot_df = plot_df.rename(
        columns={
            plot_fields["plot_key"]: "plot_key",
            plot_fields["inventory_year"]: "plot_year",
            plot_fields["latitude"]: "latitude",
            plot_fields["longitude"]: "longitude",
        }
    )

    if plot_fields["elevation"] is not None:
        plot_df = plot_df.rename(
            columns={plot_fields["elevation"]: "elevation"}
        )
    else:
        plot_df["elevation"] = np.nan

    if plot_fields["plot_status"] is not None:
        plot_df = plot_df.rename(
            columns={plot_fields["plot_status"]: "plot_status"}
        )
    else:
        plot_df["plot_status"] = np.nan

    for column in [
        "plot_year", "latitude", "longitude",
        "elevation", "plot_status"
    ]:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        )

    plot_df["plot_key"] = plot_df["plot_key"].astype("string").str.strip()

    valid_plot = (
        plot_df["plot_year"].ge(MIN_YEAR)
        & plot_df["latitude"].between(24, 50)
        & plot_df["longitude"].between(-130, -60)
        & plot_df["plot_key"].notna()
    )

    if plot_fields["plot_status"] is not None:
        valid_plot &= plot_df["plot_status"].eq(1)

    plot_df = plot_df.loc[
        valid_plot,
        [
            "plot_key", "plot_year",
            "latitude", "longitude", "elevation"
        ],
    ].copy()

    plot_duplicate_keys = int(plot_df["plot_key"].duplicated().sum())

    # CN should be unique. Keep the first only to prevent an accidental
    # many-to-many join if the source contains duplicates.
    plot_df = plot_df.drop_duplicates("plot_key", keep="first")

    # Annual observation-system center: every sampled plot receives one vote.
    observation_year_df = (
        plot_df.groupby("plot_year", as_index=False)
        .agg(
            n_sampled_plots=("plot_key", "nunique"),
            plot_mean_latitude=("latitude", "mean"),
            plot_median_latitude=("latitude", "median"),
            plot_mean_longitude=("longitude", "mean"),
            plot_mean_elevation=("elevation", "mean"),
            plot_median_elevation=("elevation", "median"),
        )
        .rename(columns={"plot_year": "year"})
        .sort_values("year")
        .reset_index(drop=True)
    )

    observation_year_df.to_parquet(
        CENTER_DIR / "annual_observation_system_center.parquet",
        index=False,
    )
    observation_year_df.to_csv(
        CENTER_DIR / "annual_observation_system_center.csv",
        index=False,
    )

    print("\nANNUAL OBSERVATION-SYSTEM CENTER")
    display(observation_year_df)

    # Join lookup remains small.
    plot_lookup = plot_df.rename(
        columns={"plot_year": "plot_inventory_year"}
    )

    # --------------------------------------------------------
    # 5. Chunked TREE scan and coordinate join
    # --------------------------------------------------------
    tree_usecols = list(
        dict.fromkeys(
            field for field in tree_fields.values() if field is not None
        )
    )

    matched_parts = []
    tree_rows_scanned = 0
    live_selected_rows = 0
    matched_rows = 0
    unmatched_rows = 0
    year_mismatch_rows = 0

    tree_reader = pd.read_csv(
        TREE_CSV,
        usecols=tree_usecols,
        chunksize=TREE_CHUNK_SIZE,
        low_memory=False,
        encoding_errors="replace",
        dtype={tree_fields["plot_key"]: "string"},
    )

    for chunk in tqdm(
        tree_reader,
        desc="Joining TREE chunks to PLOT",
        unit="chunk",
    ):
        tree_rows_scanned += len(chunk)

        rename_map = {
            tree_fields["plot_key"]: "plot_key",
            tree_fields["inventory_year"]: "tree_year",
            tree_fields["species_code"]: "species_code",
        }

        if tree_fields["status_code"] is not None:
            rename_map[tree_fields["status_code"]] = "status_code"
        if tree_fields["diameter"] is not None:
            rename_map[tree_fields["diameter"]] = "diameter"

        chunk = chunk.rename(columns=rename_map)

        chunk["plot_key"] = (
            chunk["plot_key"].astype("string").str.strip()
        )
        chunk["tree_year"] = pd.to_numeric(
            chunk["tree_year"],
            errors="coerce",
        )
        chunk["species_code"] = pd.to_numeric(
            chunk["species_code"],
            errors="coerce",
        )

        keep = (
            chunk["tree_year"].ge(MIN_YEAR)
            & chunk["species_code"].isin(selected_species_codes)
            & chunk["plot_key"].notna()
        )

        if "status_code" in chunk.columns:
            chunk["status_code"] = pd.to_numeric(
                chunk["status_code"],
                errors="coerce",
            )
            keep &= chunk["status_code"].eq(1)

        work = chunk.loc[
            keep,
            ["plot_key", "tree_year", "species_code"],
        ].copy()

        live_selected_rows += len(work)

        if work.empty:
            continue

        work["tree_year"] = work["tree_year"].astype(int)
        work["species_code"] = work["species_code"].astype(int)

        joined = work.merge(
            plot_lookup,
            on="plot_key",
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        matched = joined["_merge"].eq("both")
        matched_rows += int(matched.sum())
        unmatched_rows += int((~matched).sum())

        joined = joined.loc[matched].drop(columns="_merge")

        year_mismatch = (
            joined["tree_year"] != joined["plot_inventory_year"]
        )
        year_mismatch_rows += int(year_mismatch.sum())

        # Use only internally consistent joins.
        joined = joined.loc[~year_mismatch].copy()
        joined = joined.rename(columns={"tree_year": "year"})

        matched_parts.append(
            joined[
                [
                    "species_code", "plot_key", "year",
                    "latitude", "longitude", "elevation"
                ]
            ]
        )

    if not matched_parts:
        raise ValueError("No selected live-tree records matched PLOT coordinates.")

    matched_tree_df = pd.concat(
        matched_parts,
        ignore_index=True,
    )

    # Core methodological choice:
    # multiple individual trees of one species at one plot-year become one
    # species-presence record, avoiding abundance-like weighting.
    species_plot_presence_df = (
        matched_tree_df
        .drop_duplicates(
            ["species_code", "plot_key", "year"]
        )
        .reset_index(drop=True)
    )

    species_plot_presence_df.to_parquet(
        CENTER_DIR / "species_plot_year_presence.parquet",
        index=False,
    )

    # Free the individual-tree joined table before aggregation.
    del matched_tree_df, matched_parts

    # --------------------------------------------------------
    # 6. Annual species centers
    # --------------------------------------------------------
    grouped = species_plot_presence_df.groupby(
        ["species_code", "year"],
        sort=True,
    )

    center_rows = []

    for (species_code, year), group in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Computing species-year centers",
        unit="species-year",
    ):
        center_rows.append(
            {
                "species_code": int(species_code),
                "year": int(year),
                "n_occupied_plots": int(group["plot_key"].nunique()),
                "mean_latitude": float(group["latitude"].mean()),
                "median_latitude": float(group["latitude"].median()),
                "latitude_p10": safe_quantile(group["latitude"], 0.10),
                "latitude_p90": safe_quantile(group["latitude"], 0.90),
                "mean_longitude": float(group["longitude"].mean()),
                "mean_elevation": float(group["elevation"].mean()),
                "median_elevation": float(group["elevation"].median()),
                "elevation_p10": safe_quantile(group["elevation"], 0.10),
                "elevation_p90": safe_quantile(group["elevation"], 0.90),
                "n_elevation_plots": int(group["elevation"].notna().sum()),
            }
        )

    species_year_center_df = pd.DataFrame(center_rows)

    taxonomy_df = selected_species_df[
        ["species_code", "scientific_name", "common_name"]
    ].drop_duplicates("species_code")

    species_year_center_df = (
        species_year_center_df
        .merge(taxonomy_df, on="species_code", how="left")
        .merge(observation_year_df, on="year", how="left")
    )

    species_year_center_df["relative_mean_latitude"] = (
        species_year_center_df["mean_latitude"]
        - species_year_center_df["plot_mean_latitude"]
    )

    species_year_center_df["relative_mean_elevation"] = (
        species_year_center_df["mean_elevation"]
        - species_year_center_df["plot_mean_elevation"]
    )

    species_year_center_df["latitude_offset_km"] = (
        species_year_center_df["relative_mean_latitude"] * 111.32
    )

    species_year_center_df = species_year_center_df.sort_values(
        ["species_code", "year"]
    ).reset_index(drop=True)

    species_year_center_df.to_parquet(
        CENTER_DIR / "species_annual_centers.parquet",
        index=False,
    )
    species_year_center_df.to_csv(
        CENTER_DIR / "species_annual_centers.csv",
        index=False,
    )

    print("\nSPECIES ANNUAL CENTER PREVIEW")
    display(species_year_center_df.head(40))

    # --------------------------------------------------------
    # 7. Observation-system and species trend slopes
    # --------------------------------------------------------
    plot_lat_fit = fit_slope(
        observation_year_df["year"],
        observation_year_df["plot_mean_latitude"],
    )
    plot_elev_fit = fit_slope(
        observation_year_df["year"],
        observation_year_df["plot_mean_elevation"],
    )

    valid_center_df = species_year_center_df[
        species_year_center_df["n_occupied_plots"]
        >= MIN_OCCUPIED_PLOTS_PER_YEAR
    ].copy()

    slope_rows = []

    grouped_species = valid_center_df.groupby(
        "species_code",
        sort=True,
    )

    for species_code, group in tqdm(
        grouped_species,
        total=grouped_species.ngroups,
        desc="Fitting species center trends",
        unit="species",
    ):
        raw_lat = fit_slope(group["year"], group["mean_latitude"])
        relative_lat = fit_slope(
            group["year"], group["relative_mean_latitude"]
        )
        raw_elev = fit_slope(group["year"], group["mean_elevation"])
        relative_elev = fit_slope(
            group["year"], group["relative_mean_elevation"]
        )

        taxon = group.iloc[0]

        raw_lat_km_decade = (
            raw_lat["theil_sen_slope_per_year"] * 111.32 * 10
            if pd.notna(raw_lat["theil_sen_slope_per_year"])
            else np.nan
        )

        corrected_lat_km_decade = (
            relative_lat["theil_sen_slope_per_year"] * 111.32 * 10
            if pd.notna(relative_lat["theil_sen_slope_per_year"])
            else np.nan
        )

        raw_elev_decade = (
            raw_elev["theil_sen_slope_per_year"] * 10
            if pd.notna(raw_elev["theil_sen_slope_per_year"])
            else np.nan
        )

        corrected_elev_decade = (
            relative_elev["theil_sen_slope_per_year"] * 10
            if pd.notna(relative_elev["theil_sen_slope_per_year"])
            else np.nan
        )

        slope_rows.append(
            {
                "species_code": int(species_code),
                "scientific_name": taxon["scientific_name"],
                "common_name": taxon["common_name"],
                "n_valid_years": raw_lat["n_years"],
                "first_year": raw_lat["first_year"],
                "last_year": raw_lat["last_year"],
                "median_occupied_plots_per_year": float(
                    group["n_occupied_plots"].median()
                ),
                "raw_latitude_km_decade": raw_lat_km_decade,
                "plot_centered_latitude_km_decade": corrected_lat_km_decade,
                "latitude_change_due_to_centering_km_decade": (
                    raw_lat_km_decade - corrected_lat_km_decade
                    if pd.notna(raw_lat_km_decade)
                    and pd.notna(corrected_lat_km_decade)
                    else np.nan
                ),
                "raw_elevation_native_units_decade": raw_elev_decade,
                "plot_centered_elevation_native_units_decade": (
                    corrected_elev_decade
                ),
                "elevation_change_due_to_centering_native_units_decade": (
                    raw_elev_decade - corrected_elev_decade
                    if pd.notna(raw_elev_decade)
                    and pd.notna(corrected_elev_decade)
                    else np.nan
                ),
                "raw_latitude_ols_p": raw_lat["ols_p_value"],
                "centered_latitude_ols_p": relative_lat["ols_p_value"],
            }
        )

    species_slope_df = pd.DataFrame(slope_rows)

    species_slope_df["latitude_direction_reversal"] = (
        np.sign(species_slope_df["raw_latitude_km_decade"])
        != np.sign(
            species_slope_df["plot_centered_latitude_km_decade"]
        )
    ) & (
        species_slope_df[
            [
                "raw_latitude_km_decade",
                "plot_centered_latitude_km_decade",
            ]
        ].notna().all(axis=1)
    )

    species_slope_df["material_latitude_difference"] = (
        species_slope_df[
            "latitude_change_due_to_centering_km_decade"
        ].abs()
        >= MATERIAL_LAT_DIFF_KM_DECADE
    )

    species_slope_df = species_slope_df.sort_values(
        "latitude_change_due_to_centering_km_decade",
        key=lambda s: s.abs(),
        ascending=False,
    ).reset_index(drop=True)

    species_slope_df.to_parquet(
        CENTER_DIR / "species_center_slopes.parquet",
        index=False,
    )
    species_slope_df.to_csv(
        CENTER_DIR / "species_center_slopes.csv",
        index=False,
    )

    print("\nSPECIES RAW VS PLOT-CENTERED SLOPES")
    display(species_slope_df.head(42))

    # --------------------------------------------------------
    # 8. Coverage and signal decision
    # --------------------------------------------------------
    valid_species_slope_df = species_slope_df[
        species_slope_df["n_valid_years"] >= STRONG_COVERAGE_YEARS
    ].copy()

    n_strong_species = len(valid_species_slope_df)
    n_material_difference = int(
        valid_species_slope_df[
            "material_latitude_difference"
        ].sum()
    )
    n_direction_reversal = int(
        valid_species_slope_df[
            "latitude_direction_reversal"
        ].sum()
    )

    direction_reversal_fraction = (
        n_direction_reversal / n_strong_species
        if n_strong_species else 0.0
    )

    median_abs_lat_difference = (
        float(
            valid_species_slope_df[
                "latitude_change_due_to_centering_km_decade"
            ].abs().median()
        )
        if n_strong_species else np.nan
    )

    join_match_rate = (
        matched_rows / live_selected_rows
        if live_selected_rows else 0.0
    )

    year_mismatch_rate = (
        year_mismatch_rows / matched_rows
        if matched_rows else np.nan
    )

    coverage_pass = bool(
        join_match_rate >= 0.95
        and n_strong_species >= 20
    )

    preliminary_signal = bool(
        (
            pd.notna(median_abs_lat_difference)
            and median_abs_lat_difference
            >= MATERIAL_LAT_DIFF_KM_DECADE
        )
        or direction_reversal_fraction
        >= MIN_DIRECTION_REVERSAL_FRACTION
    )

    if not coverage_pass:
        status = "JOIN_OR_COVERAGE_REVIEW_REQUIRED"
        passed = False
        next_step = (
            "Inspect join losses and annual occupied-plot coverage "
            "before any ecological interpretation."
        )
    elif preliminary_signal:
        status = "GO_FOR_WV_MATCHED_FIXED_DOMAIN_TEST"
        passed = True
        next_step = (
            "Run matched-grid/fixed-domain and year-permutation tests "
            "before adding neighboring states."
        )
    else:
        status = "COVERAGE_OK_WV_SIGNAL_WEAK"
        passed = True
        next_step = (
            "Do not reject the mechanism from one small state. "
            "Add adjacent Appalachian states for the same trend screen."
        )

    # --------------------------------------------------------
    # 9. Visual previews
    # --------------------------------------------------------
    observation_plot_df = observation_year_df[
        observation_year_df["year"] >= MIN_YEAR
    ].copy()

    plt.figure(figsize=(10, 5))
    plt.plot(
        observation_plot_df["year"],
        observation_plot_df["plot_mean_latitude"],
        marker="o",
    )
    plt.xlabel("Inventory year")
    plt.ylabel("Mean sampled-plot latitude")
    plt.title(f"FIA {STATE}: annual observation-system latitude center")
    plt.tight_layout()
    plt.show()

    plot_species_df = (
        valid_species_slope_df.head(15)
        .sort_values(
            "latitude_change_due_to_centering_km_decade"
        )
        .copy()
    )

    if not plot_species_df.empty:
        labels = plot_species_df.apply(species_label, axis=1)
        y = np.arange(len(plot_species_df))

        plt.figure(figsize=(11, 8))
        plt.scatter(
            plot_species_df["raw_latitude_km_decade"],
            y,
            label="Raw",
        )
        plt.scatter(
            plot_species_df[
                "plot_centered_latitude_km_decade"
            ],
            y,
            label="Relative to annual plot center",
        )

        for i, row in plot_species_df.reset_index(drop=True).iterrows():
            plt.plot(
                [
                    row["raw_latitude_km_decade"],
                    row["plot_centered_latitude_km_decade"],
                ],
                [i, i],
            )

        plt.yticks(y, labels)
        plt.axvline(0)
        plt.xlabel("Latitude-center trend (km/decade)")
        plt.ylabel("Species")
        plt.title(
            f"FIA {STATE}: raw vs simple plot-centered trends"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 10. README
    # --------------------------------------------------------
    observation_lat_km_decade = (
        plot_lat_fit["theil_sen_slope_per_year"] * 111.32 * 10
        if pd.notna(plot_lat_fit["theil_sen_slope_per_year"])
        else np.nan
    )

    observation_elev_decade = (
        plot_elev_fit["theil_sen_slope_per_year"] * 10
        if pd.notna(plot_elev_fit["theil_sen_slope_per_year"])
        else np.nan
    )

    append_readme(
        f"""

## WV annual center and observation-system drift screen — {RUN_UTC}

### Hypothesis
Annual movement of the FIA sampled-plot center can alter apparent species
latitude and elevation center trends.

### Configuration
- State: {STATE}
- Minimum year: {MIN_YEAR}
- Selected species from Cell 02B: {len(selected_species_codes)}
- TREE chunk size: {TREE_CHUNK_SIZE:,}
- Minimum occupied plots per species-year: {MIN_OCCUPIED_PLOTS_PER_YEAR}
- Minimum valid years for strong coverage: {STRONG_COVERAGE_YEARS}
- Latitude conversion: 111.32 km per degree
- Elevation kept in native FIA source units pending unit verification
- Random seed reserved for later permutation tests: {RANDOM_SEED}
- GPU: not used

### Methodological choice
Multiple live trees of the same species in one plot-year were collapsed to one
plot-level presence. Individual tree counts were not used as abundance weights.

### Key results
- TREE rows scanned: {tree_rows_scanned:,}
- Selected live-tree rows: {live_selected_rows:,}
- Coordinate join match rate: {join_match_rate:.4f}
- Tree/plot inventory-year mismatch rate among matches: {year_mismatch_rate:.4f}
- Plot-year species-presence records: {len(species_plot_presence_df):,}
- Species with at least {STRONG_COVERAGE_YEARS} usable years: {n_strong_species}
- Annual sampled-plot latitude-center trend: {observation_lat_km_decade:.3f} km/decade
- Annual sampled-plot elevation-center trend: {observation_elev_decade:.3f} native units/decade
- Median absolute raw-vs-centered latitude difference: {median_abs_lat_difference:.3f} km/decade
- Species with material latitude difference: {n_material_difference}
- Species with latitude direction reversal: {n_direction_reversal}
- Direction-reversal fraction: {direction_reversal_fraction:.3f}
- Status: **{status}**

### Interpretation boundary
The annual all-plot centering is a diagnostic adjustment, not a final correction.
FIA public coordinates are spatially perturbed, annual panels may differ, tree
presence is not population abundance, and taxonomy has not yet been harmonized
to current accepted names. No climate attribution is made.
"""
    )

    top_changed = (
        valid_species_slope_df[
            [
                "species_code",
                "scientific_name",
                "n_valid_years",
                "raw_latitude_km_decade",
                "plot_centered_latitude_km_decade",
                "latitude_change_due_to_centering_km_decade",
                "latitude_direction_reversal",
            ]
        ]
        .head(10)
        .round(4)
        .to_dict("records")
    )

    # --------------------------------------------------------
    # 11. Compact output block
    # --------------------------------------------------------
    print("\n========== RUN SUMMARY ==========")
    print(f"CELL_STATUS: {status}")
    print(f"OUT_DIR: {CENTER_DIR}")
    print(f"STATE: {STATE}")
    print(f"N_SELECTED_SPECIES: {len(selected_species_codes)}")
    print(f"TREE_ROWS_SCANNED: {tree_rows_scanned:,}")
    print(f"LIVE_SELECTED_TREE_ROWS: {live_selected_rows:,}")
    print(f"MATCHED_TREE_ROWS: {matched_rows:,}")
    print(f"UNMATCHED_TREE_ROWS: {unmatched_rows:,}")
    print(f"JOIN_MATCH_RATE: {join_match_rate:.6f}")
    print(f"YEAR_MISMATCH_ROWS: {year_mismatch_rows:,}")
    print(f"YEAR_MISMATCH_RATE: {year_mismatch_rate:.6f}")
    print(
        f"SPECIES_PLOT_YEAR_PRESENCE_ROWS: "
        f"{len(species_plot_presence_df):,}"
    )
    print(f"PLOT_DUPLICATE_KEYS_BEFORE_DEDUP: {plot_duplicate_keys:,}")
    print(f"N_STRONG_COVERAGE_SPECIES: {n_strong_species}")
    print(
        f"OBSERVATION_LAT_CENTER_KM_DECADE: "
        f"{observation_lat_km_decade:.6f}"
    )
    print(
        f"OBSERVATION_ELEV_CENTER_NATIVE_DECADE: "
        f"{observation_elev_decade:.6f}"
    )
    print(
        f"MEDIAN_ABS_RAW_CENTERED_LAT_DIFF_KM_DECADE: "
        f"{median_abs_lat_difference:.6f}"
    )
    print(f"N_MATERIAL_LAT_DIFFERENCE: {n_material_difference}")
    print(f"N_LAT_DIRECTION_REVERSAL: {n_direction_reversal}")
    print(
        f"LAT_DIRECTION_REVERSAL_FRACTION: "
        f"{direction_reversal_fraction:.6f}"
    )
    print(
        "TOP_CHANGED_SPECIES: "
        + json.dumps(top_changed, ensure_ascii=False)
    )
    print(
        "COVERAGE_SUCCESS: join >=95% and >=20 species with "
        f">={STRONG_COVERAGE_YEARS} valid annual centers"
    )
    print(
        "PRELIMINARY_SIGNAL: median absolute raw-centered latitude "
        f"difference >= {MATERIAL_LAT_DIFF_KM_DECADE} km/decade "
        f"or reversal fraction >= {MIN_DIRECTION_REVERSAL_FRACTION}"
    )
    print(f"COVERAGE_PASSED: {coverage_pass}")
    print(f"PRELIMINARY_SIGNAL_PASSED: {preliminary_signal}")
    print(f"PASSED: {passed}")
    print(f"NEXT_STEP: {next_step}")
    print("README_UPDATED: True")
    print("============================================")

except Exception as exc:
    print_failure(
        "CELL03A_FAILED",
        f"{type(exc).__name__}: {repr(exc)}",
        "Send this COPY block and the last displayed output. "
        "Do not rerun Cell 02B unless a required file is missing.",
    )
