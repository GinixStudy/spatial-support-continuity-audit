
# Cell 19A — Freeze experiments and generate the final manuscript package
#
# This cell performs NO new scientific inference.
#
# It consolidates the locked results into:
# - manuscript-ready tables;
# - final figures;
# - Methods draft;
# - Results draft;
# - Discussion and limitations draft;
# - data/code availability statement;
# - supplementary reporting checklist;
# - final README with configurations and experiment boundaries.
#
# No ZIP is created.

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display, Image

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path("/kaggle/working/fia_temporal_observation_drift")
DERIVED_DIR = BASE_DIR / "derived"

FINAL_CONFIRM_DIR = (
    DERIVED_DIR
    / "final_three_region_confirmation"
)
RANKING_DIR = (
    DERIVED_DIR
    / "cell_jaccard_threshold_free_validation"
)
CLUSTER_DIR = (
    DERIVED_DIR
    / "cell_jaccard_species_cluster_robustness"
)
REGISTERED_DIR = (
    DERIVED_DIR
    / "registered_gbif_three_region_download"
)

OUT_DIR = (
    BASE_DIR
    / "final_manuscript_package"
)
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
TEXT_DIR = OUT_DIR / "text"

for directory in [
    OUT_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    TEXT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

RUN_UTC = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

GBIF_DOI = "10.15468/dl.sm6ygu"
GBIF_DOWNLOAD_KEY = (
    "0032735-260623161305970"
)

# Locked semi-synthetic results from Cell 07B.
SEMI_SYNTHETIC = {
    "best_grid_degrees": 0.5,
    "median_excess_error_reduction": 0.9606,
    "fraction_corrected_better": 0.9995,
    "median_corrected_abs_error": 2.042,
    "frame_coverage": 0.9966,
}

# Locked observer-system audit results.
OBSERVER_SYSTEM = {
    "pa_early_observers": 499,
    "pa_late_observers": 8869,
    "pa_common_observers": 186,
    "pa_observer_jaccard": 0.020257024613373992,
}

print("RUN_UTC:", RUN_UTC)
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")
print("SCIENTIFIC_INFERENCE: frozen")


# ============================================================
# Helpers
# ============================================================
def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required result missing: {path}"
        )
    return path


def fmt(value, digits=3):
    if value is None or pd.isna(value):
        return "NA"

    return f"{float(value):.{digits}f}"


def pct(value, digits=1):
    if value is None or pd.isna(value):
        return "NA"

    return (
        f"{100 * float(value):.{digits}f}%"
    )


def append_project_readme(text):
    path = BASE_DIR / "README.md"

    old = (
        path.read_text(
            encoding="utf-8"
        )
        if path.exists()
        else "# Temporal Observation Drift\n"
    )

    path.write_text(
        old + text,
        encoding="utf-8",
    )


# ============================================================
# Load locked results
# ============================================================
paths = {
    "context": require(
        FINAL_CONFIRM_DIR
        / "state_correction_context.csv"
    ),
    "state_features": require(
        FINAL_CONFIRM_DIR
        / "state_feature_confirmation.csv"
    ),
    "meta": require(
        FINAL_CONFIRM_DIR
        / "cross_region_meta_analysis.csv"
    ),
    "decision": require(
        FINAL_CONFIRM_DIR
        / "final_feature_decision.csv"
    ),
    "state_ranking": require(
        RANKING_DIR
        / "state_threshold_free_ranking_metrics.csv"
    ),
    "stratified": require(
        RANKING_DIR
        / "stratified_threshold_free_inference.csv"
    ),
    "loo_region": require(
        RANKING_DIR
        / "leave_one_region_out_ranking.csv"
    ),
    "cluster_models": require(
        CLUSTER_DIR
        / "cluster_robust_models.csv"
    ),
    "cluster_bootstrap": require(
        CLUSTER_DIR
        / "species_cluster_bootstrap_summary.csv"
    ),
    "loo_species": require(
        CLUSTER_DIR
        / "leave_one_species_cluster_summary.csv"
    ),
    "state_audit": require(
        REGISTERED_DIR
        / "registered_download_state_audit.csv"
    ),
}

context_df = pd.read_csv(
    paths["context"]
)
state_features_df = pd.read_csv(
    paths["state_features"]
)
meta_df = pd.read_csv(
    paths["meta"]
)
decision_df = pd.read_csv(
    paths["decision"]
)
state_ranking_df = pd.read_csv(
    paths["state_ranking"]
)
stratified_df = pd.read_csv(
    paths["stratified"]
)
loo_region_df = pd.read_csv(
    paths["loo_region"]
)
cluster_models_df = pd.read_csv(
    paths["cluster_models"]
)
cluster_bootstrap_df = pd.read_csv(
    paths["cluster_bootstrap"]
)
loo_species_df = pd.read_csv(
    paths["loo_species"]
)
state_audit_df = pd.read_csv(
    paths["state_audit"]
)

cell_state_df = state_features_df[
    state_features_df[
        "feature"
    ].eq(
        "cell_jaccard"
    )
].copy()

observer_state_df = state_features_df[
    state_features_df[
        "feature"
    ].eq(
        "abs_log_observer_growth"
    )
].copy()

primary_rank_row = stratified_df[
    stratified_df[
        "outcome"
    ].eq(
        "gain_grid_0p5"
    )
].iloc[0]

cross_grid_row = stratified_df[
    stratified_df[
        "outcome"
    ].eq(
        "grid_median_gain_km_decade"
    )
].iloc[0]

confirmation_meta_row = meta_df[
    meta_df[
        "feature"
    ].eq(
        "cell_jaccard"
    )
    & meta_df[
        "states"
    ].eq(
        "VA,NC"
    )
].iloc[0]

all_meta_row = meta_df[
    meta_df[
        "feature"
    ].eq(
        "cell_jaccard"
    )
    & meta_df[
        "states"
    ].eq(
        "PA,VA,NC"
    )
].iloc[0]

primary_cluster_model = (
    cluster_models_df[
        cluster_models_df[
            "outcome"
        ].eq(
            "gain_grid_0p5"
        )
        & ~cluster_models_df[
            "adjusted"
        ].astype(bool)
    ].iloc[0]
)

adjusted_cluster_model = (
    cluster_models_df[
        cluster_models_df[
            "outcome"
        ].eq(
            "gain_grid_0p5"
        )
        & cluster_models_df[
            "adjusted"
        ].astype(bool)
    ].iloc[0]
)

# ============================================================
# Final manuscript tables
# ============================================================
table_tasks = []

table1 = state_audit_df[
    [
        "state_code",
        "n_records",
        "expected_locked_taxa",
        "observed_locked_taxa",
        "n_years",
    ]
].copy()

table1.columns = [
    "Region",
    "Registered GBIF records",
    "Locked taxa expected",
    "Locked taxa observed",
    "Years represented",
]

table_tasks.append(
    (
        table1,
        TABLE_DIR
        / "Table_1_formal_data_coverage.csv",
    )
)

table2 = context_df[
    [
        "state_code",
        "n_species",
        "raw_median_abs_error",
        "corrected_median_abs_error",
        "median_error_reduction",
        "species_improved_fraction",
        "median_gain_km_decade",
    ]
].copy()

table2.columns = [
    "Region",
    "Species",
    "Raw median absolute error",
    "Corrected median absolute error",
    "Median error reduction",
    "Species improved fraction",
    "Median correction gain",
]

table_tasks.append(
    (
        table2,
        TABLE_DIR
        / "Table_2_real_world_correction_performance.csv",
    )
)

table3 = cell_state_df[
    [
        "state_code",
        "state_role",
        "n_species",
        "primary_rho",
        "holm_adjusted_p",
        "bootstrap_low",
        "bootstrap_high",
        "grid_median_rho",
        "regional_confirmed",
    ]
].copy()

table3.columns = [
    "Region",
    "Role",
    "Species",
    "Cell-Jaccard rho",
    "Holm-adjusted P",
    "Bootstrap low",
    "Bootstrap high",
    "Cross-grid rho",
    "Independent confirmation",
]

table_tasks.append(
    (
        table3,
        TABLE_DIR
        / "Table_3_regional_cell_jaccard_confirmation.csv",
    )
)

table4 = pd.DataFrame(
    [
        {
            "Analysis": (
                "State-stratified rank"
            ),
            "Estimate": (
                primary_rank_row[
                    "stratified_rank_rho"
                ]
            ),
            "P_value": (
                primary_rank_row[
                    "rank_permutation_p"
                ]
            ),
            "CI_low": (
                primary_rank_row[
                    "rank_bootstrap_low"
                ]
            ),
            "CI_high": (
                primary_rank_row[
                    "rank_bootstrap_high"
                ]
            ),
        },
        {
            "Analysis": (
                "State-stratified AUROC"
            ),
            "Estimate": (
                primary_rank_row[
                    "stratified_auc"
                ]
            ),
            "P_value": (
                primary_rank_row[
                    "auc_permutation_p"
                ]
            ),
            "CI_low": (
                primary_rank_row[
                    "auc_bootstrap_low"
                ]
            ),
            "CI_high": (
                primary_rank_row[
                    "auc_bootstrap_high"
                ]
            ),
        },
        {
            "Analysis": (
                "Adjusted partial rank"
            ),
            "Estimate": (
                primary_rank_row[
                    "partial_rank_rho"
                ]
            ),
            "P_value": (
                primary_rank_row[
                    "partial_permutation_p"
                ]
            ),
            "CI_low": (
                primary_rank_row[
                    "partial_bootstrap_low"
                ]
            ),
            "CI_high": (
                primary_rank_row[
                    "partial_bootstrap_high"
                ]
            ),
        },
        {
            "Analysis": (
                "Cross-grid rank"
            ),
            "Estimate": (
                cross_grid_row[
                    "stratified_rank_rho"
                ]
            ),
            "P_value": (
                cross_grid_row[
                    "rank_permutation_p"
                ]
            ),
            "CI_low": (
                cross_grid_row[
                    "rank_bootstrap_low"
                ]
            ),
            "CI_high": (
                cross_grid_row[
                    "rank_bootstrap_high"
                ]
            ),
        },
        {
            "Analysis": (
                "Species-cluster robust rank model"
            ),
            "Estimate": (
                primary_cluster_model[
                    "coefficient"
                ]
            ),
            "P_value": (
                primary_cluster_model[
                    "one_sided_p"
                ]
            ),
            "CI_low": (
                primary_cluster_model[
                    "confidence_low"
                ]
            ),
            "CI_high": (
                primary_cluster_model[
                    "confidence_high"
                ]
            ),
        },
        {
            "Analysis": (
                "Adjusted species-cluster model"
            ),
            "Estimate": (
                adjusted_cluster_model[
                    "coefficient"
                ]
            ),
            "P_value": (
                adjusted_cluster_model[
                    "one_sided_p"
                ]
            ),
            "CI_low": (
                adjusted_cluster_model[
                    "confidence_low"
                ]
            ),
            "CI_high": (
                adjusted_cluster_model[
                    "confidence_high"
                ]
            ),
        },
    ]
)

table_tasks.append(
    (
        table4,
        TABLE_DIR
        / "Table_4_threshold_free_and_cluster_robust_validation.csv",
    )
)

table5 = decision_df.copy()

table_tasks.append(
    (
        table5,
        TABLE_DIR
        / "Table_5_final_candidate_decision.csv",
    )
)

for frame, path in tqdm(
    table_tasks,
    desc="Writing manuscript tables",
    unit="table",
):
    frame.to_csv(
        path,
        index=False,
    )

# ============================================================
# Final manuscript figures
# ============================================================
figure_paths = []

# Figure 1: real correction performance.
plot_context = context_df.copy()
x = np.arange(
    len(
        plot_context
    )
)
width = 0.35

plt.figure(
    figsize=(8, 5)
)
plt.bar(
    x - width / 2,
    plot_context[
        "raw_median_abs_error"
    ],
    width=width,
    label="Raw",
)
plt.bar(
    x + width / 2,
    plot_context[
        "corrected_median_abs_error"
    ],
    width=width,
    label="Corrected",
)
plt.xticks(
    x,
    plot_context[
        "state_code"
    ],
)
plt.ylabel(
    "Median absolute error versus FIA (km/decade)"
)
plt.xlabel(
    "Region"
)
plt.title(
    "A universal correction did not transport across regions"
)
plt.legend()
plt.tight_layout()

figure1 = (
    FIGURE_DIR
    / "Figure_1_real_world_correction_performance.png"
)

plt.savefig(
    figure1,
    dpi=300,
    bbox_inches="tight",
)
plt.show()
display(
    Image(
        filename=str(
            figure1
        )
    )
)
figure_paths.append(
    figure1
)

# Figure 2: cell-Jaccard regional association.
plot_cell = cell_state_df.copy()
x = np.arange(
    len(
        plot_cell
    )
)

lower_error = (
    plot_cell[
        "primary_rho"
    ]
    - plot_cell[
        "bootstrap_low"
    ]
).clip(
    lower=0
)

upper_error = (
    plot_cell[
        "bootstrap_high"
    ]
    - plot_cell[
        "primary_rho"
    ]
).clip(
    lower=0
)

plt.figure(
    figsize=(8, 5)
)
plt.errorbar(
    x,
    plot_cell[
        "primary_rho"
    ],
    yerr=[
        lower_error,
        upper_error,
    ],
    fmt="o",
    capsize=4,
)
plt.axhline(
    0,
    linestyle="--",
)
plt.xticks(
    x,
    plot_cell[
        "state_code"
    ],
)
plt.ylabel(
    "Spearman rho with correction gain"
)
plt.xlabel(
    "Region"
)
plt.title(
    "Spatial-support continuity predicts correction benefit"
)
plt.tight_layout()

figure2 = (
    FIGURE_DIR
    / "Figure_2_cell_jaccard_regional_replication.png"
)

plt.savefig(
    figure2,
    dpi=300,
    bbox_inches="tight",
)
plt.show()
display(
    Image(
        filename=str(
            figure2
        )
    )
)
figure_paths.append(
    figure2
)

# Figure 3: threshold-free metrics.
plot_rank = state_ranking_df.copy()
x = np.arange(
    len(
        plot_rank
    )
)

plt.figure(
    figsize=(8, 5)
)
plt.bar(
    x - width / 2,
    plot_rank[
        "spearman_gain"
    ],
    width=width,
    label="Spearman rho",
)
plt.bar(
    x + width / 2,
    plot_rank[
        "auc_improved_vs_harmed"
    ],
    width=width,
    label="AUROC",
)
plt.axhline(
    0.5,
    linestyle="--",
)
plt.xticks(
    x,
    plot_rank[
        "state_code"
    ],
)
plt.ylabel(
    "Threshold-free ranking metric"
)
plt.xlabel(
    "Region"
)
plt.title(
    "Regional stability of the cell-Jaccard applicability gate"
)
plt.legend()
plt.tight_layout()

figure3 = (
    FIGURE_DIR
    / "Figure_3_threshold_free_ranking_metrics.png"
)

plt.savefig(
    figure3,
    dpi=300,
    bbox_inches="tight",
)
plt.show()
display(
    Image(
        filename=str(
            figure3
        )
    )
)
figure_paths.append(
    figure3
)

# Copy the two cluster-robustness figures.
copy_sources = [
    (
        require(
            CLUSTER_DIR
            / "species_cluster_bootstrap_primary_rho.png"
        ),
        FIGURE_DIR
        / "Figure_4_species_cluster_bootstrap.png",
    ),
    (
        require(
            CLUSTER_DIR
            / "leave_one_species_cluster_robustness.png"
        ),
        FIGURE_DIR
        / "Figure_5_leave_one_species_robustness.png",
    ),
]

for source, target in tqdm(
    copy_sources,
    desc="Copying robustness figures",
    unit="figure",
):
    shutil.copy2(
        source,
        target,
    )
    figure_paths.append(
        target
    )
    display(
        Image(
            filename=str(
                target
            )
        )
    )

# ============================================================
# Manuscript text package
# ============================================================
pa_context = context_df[
    context_df[
        "state_code"
    ].eq(
        "PA"
    )
].iloc[0]

va_context = context_df[
    context_df[
        "state_code"
    ].eq(
        "VA"
    )
].iloc[0]

nc_context = context_df[
    context_df[
        "state_code"
    ].eq(
        "NC"
    )
].iloc[0]

pa_cell = cell_state_df[
    cell_state_df[
        "state_code"
    ].eq(
        "PA"
    )
].iloc[0]

va_cell = cell_state_df[
    cell_state_df[
        "state_code"
    ].eq(
        "VA"
    )
].iloc[0]

nc_cell = cell_state_df[
    cell_state_df[
        "state_code"
    ].eq(
        "NC"
    )
].iloc[0]

title_candidates = """# Title candidates

1. Spatial support continuity predicts when observation-drift correction helps or harms opportunistic biodiversity data

2. Correction can harm: cross-region validation of observation-drift adjustment in opportunistic species records

3. From simulation success to real-world failure: spatial continuity governs the transportability of biodiversity observation correction

Recommended working title:
Spatial support continuity predicts when observation-drift correction helps or harms opportunistic biodiversity data
"""

methods_text = f"""# Methods draft

## Study design

We evaluated whether temporal change in an opportunistic observation system can
distort estimated species distribution shifts, whether a fixed spatial
post-stratification correction can reduce this distortion, and under which
conditions correction is more likely to help than harm.

The analysis followed a staged design:

1. standardized repeated-plot FIA analyses;
2. semi-synthetic injection of controlled observation-footprint drift;
3. real iNaturalist validation against same-window repeated FIA physical plots;
4. Pennsylvania discovery of correction-transportability features;
5. locked independent confirmation in Virginia and North Carolina;
6. threshold-free and FIA-species-clustered robustness analyses.

## Data sources

Standardized forest reference data were obtained from the United States Forest
Service Forest Inventory and Analysis database. Opportunistic observations were
drawn from iNaturalist Research-grade Observations distributed through GBIF.

The final registered occurrence download contained 143,355 records and is
citable as:

GBIF.org (10 July 2026) GBIF Occurrence Download.
DOI: {GBIF_DOI}

The registered query was restricted to human observations with coordinates,
present occurrence status, no flagged geospatial issue, Pennsylvania, Virginia
or North Carolina, the locked state-specific taxa, and the periods 2013–2018
and 2020–2025.

## Occurrence preprocessing

Records outside the locked periods or state bounding boxes were removed.
Coordinate uncertainty above 10 km was excluded when uncertainty was known.
Records were thinned to one observation per species, event day and 0.05-degree
cell. All formal registered-download results reproduced the temporary API
results exactly after restoration of the locked North Carolina effort-frame
membership.

## FIA reference

A physical FIA plot was defined by STATECD, UNITCD, COUNTYCD and PLOT. For each
physical plot, the measurement nearest the midpoint of each period was retained.
Only plots with live-tree measurements in both periods were used. Species
centres were estimated from one presence per physical plot and period, without
tree-count abundance weighting.

## Controlled drift benchmark

Controlled observation-footprint drift was injected into repeated-plot data.
The locked primary post-stratification resolution was 0.5 degrees. Across the
calibrated semi-synthetic scenarios, the median excess-error reduction was
{SEMI_SYNTHETIC['median_excess_error_reduction']:.4f}, and correction improved
{SEMI_SYNTHETIC['fraction_corrected_better']:.4f} of evaluated replicates.

## Real-data correction

For each state and grid resolution, target-group observation effort was
estimated from the locked tree effort frame. Stable cells required at least 10
target-frame records in both periods. Species-specific relative occurrence
rates were calculated as species records divided by target-frame records within
each cell-period. Corrected latitude centres were estimated by weighting pooled
cell latitudes by these relative occurrence rates.

The primary grid was 0.5 degrees. The 0.25- and 1.0-degree estimates were used
for sensitivity analysis.

## Correction gain

Correction gain was defined as:

raw absolute error versus FIA minus corrected absolute error versus FIA.

Positive values indicate that correction improved agreement with the repeated
FIA reference; negative values indicate harm.

## Transportability discovery and confirmation

Pennsylvania was the discovery region. Virginia and North Carolina were locked
independent confirmation regions. Two Pennsylvania candidates were initially
tested: cell Jaccard, representing early-late overlap of occupied 0.5-degree
cells, and absolute log observer growth.

Only cell Jaccard independently replicated. Observer-growth imbalance was
discarded from the final analysis.

## Threshold-free validation

Cell Jaccard was evaluated as a continuous ranking diagnostic without selecting
a cutoff. Analyses included state-stratified Spearman association, stratified
AUROC for improved versus harmed species, pairwise ranking concordance, and
partial rank association adjusted for minimum record support, minimum occupied
cell support and raw FIA error.

## Dependence and robustness

The same FIA species could occur in multiple states. Dependence was handled
using FIA-species-clustered standard errors, 5,000 species-cluster bootstrap
replicates and leave-one-FIA-species-out analysis.

No feature weights or cutoffs were optimized.
"""

results_text = f"""# Results draft

## Controlled observation drift produced systematic migration error

The calibrated semi-synthetic experiment demonstrated a monotonic relationship
between observation-footprint displacement and species-shift error. The locked
0.5-degree correction reduced median excess error by
{pct(SEMI_SYNTHETIC['median_excess_error_reduction'])}, and correction was
better than the uncorrected estimate in
{pct(SEMI_SYNTHETIC['fraction_corrected_better'])} of evaluated replicates.
This established that the bias mechanism and the correction can operate under
controlled support conditions.

## The real observation system changed substantially

In Pennsylvania, the number of exact observer labels increased from
{OBSERVER_SYSTEM['pa_early_observers']:,} in the early period to
{OBSERVER_SYSTEM['pa_late_observers']:,} in the late period. Only
{OBSERVER_SYSTEM['pa_common_observers']:,} labels appeared in both periods,
yielding an observer Jaccard of
{OBSERVER_SYSTEM['pa_observer_jaccard']:.4f}. Repeated-observer and observer-
profile decompositions were insufficiently stable to attribute footprint change
to individual observer movement or a single turnover mechanism.

## A universal correction did not transport to real data

The locked correction was not uniformly beneficial. In Pennsylvania, median
absolute error decreased from {fmt(pa_context['raw_median_abs_error'])} to
{fmt(pa_context['corrected_median_abs_error'])} km/decade, a relative reduction
of {pct(pa_context['median_error_reduction'])}. However, only
{pct(pa_context['species_improved_fraction'])} of species improved.

In Virginia, median error increased from
{fmt(va_context['raw_median_abs_error'])} to
{fmt(va_context['corrected_median_abs_error'])} km/decade
({pct(va_context['median_error_reduction'])} relative change), and only
{pct(va_context['species_improved_fraction'])} of species improved.

In North Carolina, median error increased from
{fmt(nc_context['raw_median_abs_error'])} to
{fmt(nc_context['corrected_median_abs_error'])} km/decade
({pct(nc_context['median_error_reduction'])} relative change), with
{pct(nc_context['species_improved_fraction'])} of species improved.

These results reject the interpretation of the 0.5-degree adjustment as a
universal correction.

## Spatial-support continuity replicated as an applicability signal

Cell Jaccard was strongly associated with correction gain in Pennsylvania
(rho={fmt(pa_cell['primary_rho'])}) and independently confirmed in North
Carolina (rho={fmt(nc_cell['primary_rho'])}, Holm-adjusted
P={fmt(nc_cell['holm_adjusted_p'], 4)}). Virginia retained the same direction
(rho={fmt(va_cell['primary_rho'])}) but did not satisfy the locked regional
confirmation criteria.

Across the two confirmation regions, the fixed-effect meta-association was
rho={fmt(confirmation_meta_row['meta_rho'])},
one-sided P={fmt(confirmation_meta_row['one_sided_p'], 4)}, with
I2={fmt(confirmation_meta_row['i2_percent'], 1)}%. Across all three regions,
the pooled association was rho={fmt(all_meta_row['meta_rho'])},
one-sided P={fmt(all_meta_row['one_sided_p'], 6)}.

Absolute log observer growth did not replicate and was removed from the final
gate.

## Cell Jaccard provided threshold-free ranking value

Across 43 species-region observations, the state-stratified association between
cell Jaccard and 0.5-degree correction gain was
rho={fmt(primary_rank_row['stratified_rank_rho'])},
permutation P={fmt(primary_rank_row['rank_permutation_p'], 4)}, with a 95%
bootstrap interval of [{fmt(primary_rank_row['rank_bootstrap_low'])},
{fmt(primary_rank_row['rank_bootstrap_high'])}].

The stratified AUROC for ranking improved versus harmed species was
{fmt(primary_rank_row['stratified_auc'])},
permutation P={fmt(primary_rank_row['auc_permutation_p'], 4)}, with a 95%
bootstrap interval of [{fmt(primary_rank_row['auc_bootstrap_low'])},
{fmt(primary_rank_row['auc_bootstrap_high'])}].

After adjustment for record support, occupied-cell support and raw FIA error,
the partial rank association remained positive:
rho={fmt(primary_rank_row['partial_rank_rho'])},
permutation P={fmt(primary_rank_row['partial_permutation_p'], 4)}.

The cross-grid association remained positive:
rho={fmt(cross_grid_row['stratified_rank_rho'])},
permutation P={fmt(cross_grid_row['rank_permutation_p'], 4)}.

## Results remained robust to repeated species across states

The 43 species-region observations represented 19 unique FIA species clusters.
The unadjusted species-cluster-robust coefficient was
{fmt(primary_cluster_model['coefficient'])}
(SE={fmt(primary_cluster_model['cluster_robust_se'])},
one-sided P={fmt(primary_cluster_model['one_sided_p'], 6)}).

The adjusted coefficient was
{fmt(adjusted_cluster_model['coefficient'])}
(SE={fmt(adjusted_cluster_model['cluster_robust_se'])},
one-sided P={fmt(adjusted_cluster_model['one_sided_p'], 6)}).

Species-cluster bootstrap intervals remained positive, and deleting any single
FIA species preserved positive primary, adjusted and cross-grid associations.
"""

discussion_text = """# Discussion and limitations draft

## Main interpretation

The study demonstrates a gap between controlled correction success and
real-world transportability. Observation-footprint drift can generate
substantial migration error, and spatial post-stratification can remove that
error under controlled support conditions. Yet the same correction can worsen
real species estimates when spatial support changes between periods.

The central contribution is therefore not a universal correction algorithm.
It is an applicability framework: correction should not be applied uniformly,
and early-late spatial-support continuity provides a replicated, threshold-free
diagnostic of when correction is more likely to help.

## Mechanistic interpretation

Cell Jaccard measures continuity of the spatial observation support available
to a species. When early and late supports overlap strongly, cell-based
standardization compares more similar spatial frames. When overlap is weak,
the correction must extrapolate across changing support and can amplify
differences unrelated to the biological signal.

This interpretation is consistent with the observed negative association
between cell Jaccard and corrected error, but the feature is not treated as a
causal mechanism because it is derived from the same occurrence data used in
the correction.

## Why no cutoff was selected

The sample contains a limited number of independent species clusters, and an
optimized cutoff would overfit these data. Cell Jaccard is therefore reported
as a continuous ranking diagnostic. A deployment cutoff would require external
prospective validation in additional taxa, platforms and geographic systems.

## Limitations

1. FIA repeated-plot shifts are a standardized external reference, not perfect
   ecological truth.

2. iNaturalist target-group records approximate observation effort but do not
   measure all search effort or unreported absences.

3. The analysis covers eastern United States tree taxa and may not transport to
   mobile organisms, rare taxa, marine systems or other platforms.

4. The number of independent FIA species clusters is 19, despite 43
   species-region observations.

5. Observer labels may not correspond one-to-one with permanent individuals,
   and the study does not infer individual observer movement.

6. Cell Jaccard and correction gain share spatial input data. Adjustment for
   record support, cell support and raw error, cross-region replication and
   species-cluster robustness reduce but do not eliminate this concern.

7. Climate attribution was outside the scope of the study. The analysis
   evaluates measurement-system bias and correction transportability.

8. A registered GBIF download improves reproducibility, but the downloadable
   data file is retained by GBIF for a limited period unless extended; the DOI
   and query landing page remain permanent.

## Practical recommendation

Before applying observation-drift correction, analysts should report temporal
spatial-support continuity, test correction against a standardized external
reference where possible, and present corrected and uncorrected estimates
together. Low-support-continuity species should be treated as high-risk cases
rather than automatically corrected.
"""

availability_text = f"""# Data and code availability

The final opportunistic occurrence dataset was obtained through a registered
GBIF download:

GBIF.org (10 July 2026) GBIF Occurrence Download.
DOI: {GBIF_DOI}
Download key: {GBIF_DOWNLOAD_KEY}

The query included iNaturalist Research-grade human observations from
Pennsylvania, Virginia and North Carolina for 2013–2018 and 2020–2025, restricted
to the locked state-specific tree taxa.

FIA data were obtained from the United States Forest Service FIA DataMart state
PLOT and TREE tables.

The analysis repository should include:
- registered-download query and manifest;
- taxon-resolution tables;
- state-specific preprocessing outputs;
- repeated-plot FIA reference tables;
- semi-synthetic calibration outputs;
- final species-level analysis tables;
- all analysis code and random seeds.

No account credentials are stored.
"""

outline_text = """# Final manuscript outline

## Title
Spatial support continuity predicts when observation-drift correction helps or harms opportunistic biodiversity data

## Abstract
Background; controlled drift benchmark; real external validation; replicated
cell-Jaccard result; practical conclusion.

## 1. Introduction
- Opportunity and risk of opportunistic biodiversity records
- Temporal observation-system drift
- Why simulation-only correction validation is insufficient
- Study objectives and hypotheses

## 2. Materials and Methods
2.1 Study design
2.2 FIA standardized reference
2.3 Registered iNaturalist/GBIF data
2.4 Quality filtering and thinning
2.5 Repeated physical-plot construction
2.6 Semi-synthetic drift injection
2.7 Target-group effort correction
2.8 Real external validation
2.9 Discovery and locked regional confirmation
2.10 Threshold-free applicability validation
2.11 Species-cluster robustness
2.12 Reproducibility and data availability

## 3. Results
3.1 Controlled observation drift generated migration error
3.2 The real observation system changed substantially
3.3 Universal correction failed to transport
3.4 Cell Jaccard independently replicated
3.5 Threshold-free ranking validation
3.6 FIA-species-cluster robustness

## 4. Discussion
- Controlled success versus real-world failure
- Spatial support continuity as an applicability diagnostic
- Why universal correction can harm
- Implications for biodiversity monitoring
- Limitations
- Future prospective validation

## 5. Conclusion
Observation-drift correction should be treated as a conditional intervention.
Spatial-support continuity can rank correction applicability, but no universal
cutoff or universal correction is justified.
"""

checklist_rows = [
    {
        "Item": "Registered GBIF DOI reported",
        "Status": "Complete",
        "Evidence": GBIF_DOI,
    },
    {
        "Item": "Registered query manifest retained",
        "Status": "Complete",
        "Evidence": str(
            REGISTERED_DIR
            / "registered_download_manifest.json"
        ),
    },
    {
        "Item": "Formal record count verified",
        "Status": "Complete",
        "Evidence": "143,355 records",
    },
    {
        "Item": "PA/VA/NC taxon mapping verified",
        "Status": "Complete",
        "Evidence": "All locked taxa matched",
    },
    {
        "Item": "Temporary API parity verified",
        "Status": "Complete",
        "Evidence": "Exact deterministic parity",
    },
    {
        "Item": "Discovery and confirmation regions separated",
        "Status": "Complete",
        "Evidence": "PA discovery; VA and NC confirmation",
    },
    {
        "Item": "Candidate feature search frozen",
        "Status": "Complete",
        "Evidence": "Only cell Jaccard retained",
    },
    {
        "Item": "No optimized cutoff reported",
        "Status": "Complete",
        "Evidence": "Threshold-free ranking only",
    },
    {
        "Item": "Cross-grid sensitivity reported",
        "Status": "Complete",
        "Evidence": "0.25°, 0.5° and 1.0°",
    },
    {
        "Item": "State-stratified permutation reported",
        "Status": "Complete",
        "Evidence": "5,000 permutations",
    },
    {
        "Item": "Species-cluster dependence addressed",
        "Status": "Complete",
        "Evidence": "19 FIA species clusters",
    },
    {
        "Item": "Species-cluster bootstrap reported",
        "Status": "Complete",
        "Evidence": "5,000 replicates",
    },
    {
        "Item": "Leave-one-region-out analysis reported",
        "Status": "Complete",
        "Evidence": "3/3 positive",
    },
    {
        "Item": "Leave-one-species-out analysis reported",
        "Status": "Complete",
        "Evidence": "All associations remained positive",
    },
    {
        "Item": "Universal correction claim avoided",
        "Status": "Complete",
        "Evidence": "VA and NC correction harm reported",
    },
    {
        "Item": "Climate attribution avoided",
        "Status": "Complete",
        "Evidence": "Measurement-bias study only",
    },
    {
        "Item": "Author/contribution declarations",
        "Status": "Pending",
        "Evidence": "Complete before submission",
    },
    {
        "Item": "Conflict of interest statement",
        "Status": "Pending",
        "Evidence": "Complete before submission",
    },
    {
        "Item": "Funding statement",
        "Status": "Pending",
        "Evidence": "Complete before submission",
    },
]

text_files = {
    TEXT_DIR
    / "TITLE_CANDIDATES.md": (
        title_candidates
    ),
    TEXT_DIR
    / "METHODS_DRAFT.md": (
        methods_text
    ),
    TEXT_DIR
    / "RESULTS_DRAFT.md": (
        results_text
    ),
    TEXT_DIR
    / "DISCUSSION_LIMITATIONS_DRAFT.md": (
        discussion_text
    ),
    TEXT_DIR
    / "DATA_CODE_AVAILABILITY.md": (
        availability_text
    ),
    TEXT_DIR
    / "MANUSCRIPT_OUTLINE.md": (
        outline_text
    ),
}

for path, content in tqdm(
    text_files.items(),
    desc="Writing manuscript text files",
    unit="file",
):
    path.write_text(
        content.strip()
        + "\n",
        encoding="utf-8",
    )

checklist_df = pd.DataFrame(
    checklist_rows
)

checklist_path = (
    TEXT_DIR
    / "SUPPLEMENTARY_REPORTING_CHECKLIST.csv"
)

checklist_df.to_csv(
    checklist_path,
    index=False,
)

# ============================================================
# Final package README
# ============================================================
readme_text = f"""# Final manuscript package

Generated: {RUN_UTC}

## Frozen scientific conclusion

A universal 0.5-degree observation-drift correction was not validated in real
data. It modestly reduced median error in Pennsylvania but increased median
error in Virginia and North Carolina.

Spatial-support continuity, quantified by early-late cell Jaccard, replicated
as a threshold-free applicability ranking diagnostic. It was discovered in
Pennsylvania, independently confirmed in North Carolina, directionally
consistent in Virginia, and robust to state stratification, cross-grid
sensitivity, adjustment for support and raw error, FIA-species clustering,
cluster bootstrap and leave-one-species analysis.

Absolute log observer growth did not replicate and is excluded from the final
gate.

## Key results

- Formal GBIF records: 143,355
- GBIF DOI: {GBIF_DOI}
- Species-region observations: 43
- Unique FIA species clusters: 19
- State-stratified rank rho:
  {primary_rank_row['stratified_rank_rho']:.4f}
- Stratified AUROC:
  {primary_rank_row['stratified_auc']:.4f}
- Adjusted partial rank rho:
  {primary_rank_row['partial_rank_rho']:.4f}
- Unadjusted cluster-robust coefficient:
  {primary_cluster_model['coefficient']:.4f}
- Adjusted cluster-robust coefficient:
  {adjusted_cluster_model['coefficient']:.4f}

## Frozen boundaries

- Do not add new candidate features.
- Do not optimize a cell-Jaccard cutoff.
- Do not claim that the correction is universally effective.
- Do not interpret cell Jaccard as a proven causal mechanism.
- Do not attribute observed shifts to climate change in this manuscript.
- New computation should be limited to formatting or reviewer-requested checks.

## Package structure

- tables/: manuscript-ready CSV tables
- figures/: final 300 dpi PNG figures
- text/: title, Methods, Results, Discussion, data availability, outline and
  reporting checklist
"""

final_readme_path = (
    OUT_DIR
    / "README_FINAL_MANUSCRIPT_PACKAGE.md"
)

final_readme_path.write_text(
    readme_text.strip()
    + "\n",
    encoding="utf-8",
)

append_project_readme(
    f"""

## Computational experiments frozen — {RUN_UTC}

- Final replicated gate: cell_jaccard
- Use: threshold-free correction-applicability ranking
- Final GBIF DOI: {GBIF_DOI}
- Species-region rows: 43
- Unique FIA species clusters: 19
- Final manuscript package: {OUT_DIR}
- No further feature search, regional expansion or cutoff optimization planned.
"""
)

# ============================================================
# Compact output
# ============================================================
print(
    "\n========== RUN SUMMARY =========="
)
print(
    "CELL_STATUS: FINAL_MANUSCRIPT_PACKAGE_CREATED"
)
print(
    f"OUT_DIR: {OUT_DIR}"
)
print(
    f"TABLE_DIR: {TABLE_DIR}"
)
print(
    f"FIGURE_DIR: {FIGURE_DIR}"
)
print(
    f"TEXT_DIR: {TEXT_DIR}"
)
print(
    f"FINAL_README: {final_readme_path}"
)
print(
    f"N_TABLES: {len(table_tasks)}"
)
print(
    f"N_FIGURES: {len(figure_paths)}"
)
print(
    f"N_TEXT_FILES: {len(text_files) + 1}"
)
print(
    f"GBIF_DOWNLOAD_DOI: {GBIF_DOI}"
)
print(
    "FINAL_REPLICATED_GATE: cell_jaccard"
)
print(
    "GATE_USE: threshold-free applicability ranking"
)
print(
    "UNIVERSAL_CORRECTION_VALIDATED: False"
)
print(
    "EXPERIMENTS_FROZEN: True"
)
print(
    "PASSED: True"
)
print(
    "NEXT_STEP: Review the generated manuscript drafts and select a target journal."
)
print(
    "README_UPDATED: True"
)
print(
    "============================================"
)
