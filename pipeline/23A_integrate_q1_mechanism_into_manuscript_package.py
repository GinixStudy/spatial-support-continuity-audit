# Cell 23A — Integrate the confirmed two-axis mechanism into the manuscript package
# No new experiment. No ZIP. The original manuscript package is preserved.

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil

import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display

BASE = Path('/kaggle/working/fia_temporal_observation_drift')
SOURCE_PACKAGE = BASE / 'final_manuscript_package'
MECH = BASE / 'derived' / 'full_orthogonal_support_phase_confirmation'
PLAN = BASE / 'derived' / 'full_orthogonal_phase_diagram_plan' / 'full_orthogonal_phase_confirmation_plan.json'
OUT = BASE / 'final_manuscript_package_q1_mechanism'
RUN_UTC = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

SUMMARY = MECH / 'full_phase_confirmation_summary.csv'
PERM = MECH / 'scenario_label_permutation_summary.csv'
BOOT = MECH / 'species_region_cluster_bootstrap_summary.csv'
NEG = MECH / 'negative_control_summary.csv'
MARG_E = MECH / 'marginal_effort_overlap_curve.csv'
MARG_S = MECH / 'marginal_species_overlap_curve.csv'
RESULT_TXT = MECH / 'manuscript_ready_full_phase_result.txt'
FIGURES = [
    'full_estimability_phase_drift_0.png',
    'full_corrected_error_phase_drift_0.png',
    'full_estimability_phase_drift_20.png',
    'full_corrected_error_phase_drift_20.png',
    'full_estimability_phase_drift_40.png',
    'full_corrected_error_phase_drift_40.png',
    'full_marginal_effort_curve.png',
    'full_marginal_species_curve.png',
    'full_permutation_negative_controls.png',
]

print('RUN_UTC:', RUN_UTC)
print('OUT:', OUT)
print('GPU: not used')
print('NEW_EXPERIMENT: False')


def as_bool(value):
    return str(value).strip().lower() in {'true', '1', 'yes'}


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + '\n', encoding='utf-8')


def copy_package(source, destination):
    files = [p for p in source.rglob('*') if p.is_file()]
    for src in tqdm(files, desc='Copying original manuscript package', unit='file'):
        dst = destination / src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


required = [SOURCE_PACKAGE, SUMMARY, PERM, BOOT, NEG, MARG_E, MARG_S, PLAN, RESULT_TXT]
required += [MECH / name for name in FIGURES]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError('Missing required files: ' + ' | '.join(missing))

summary_df = pd.read_csv(SUMMARY)
if len(summary_df) != 1 or not as_bool(summary_df.iloc[0]['passed']):
    raise RuntimeError('The frozen full confirmation did not pass.')

plan = json.loads(PLAN.read_text(encoding='utf-8'))
if plan.get('plan_status') != 'FROZEN':
    raise RuntimeError('The full confirmation plan is not frozen.')

s = summary_df.iloc[0]
effort_rho = float(s['effort_global_rho'])
effort_p = float(s['effort_permutation_p'])
effort_low = float(s['effort_bootstrap_low'])
effort_high = float(s['effort_bootstrap_high'])
effort_boot = float(s['effort_bootstrap_positive_fraction'])
effort_mono = float(s['effort_monotonic_fraction'])
effort_delta = float(s['effort_estimability_delta'])

species_rho = float(s['species_global_rho'])
species_p = float(s['species_permutation_p'])
species_low = float(s['species_bootstrap_low'])
species_high = float(s['species_bootstrap_high'])
species_boot = float(s['species_bootstrap_negative_fraction'])
species_mono = float(s['species_monotonic_fraction'])
error_delta = float(s['low_minus_high_corrected_error'])

partial_rho = float(s['partial_gain_rho'])
partial_low = float(s['partial_gain_bootstrap_low'])
partial_high = float(s['partial_gain_bootstrap_high'])
partial_boot = float(s['partial_gain_bootstrap_positive_fraction'])

cross_estimability = float(s['species_overlap_estimability_rho'])
cross_error = float(s['effort_overlap_corrected_error_rho'])
zero_error = float(s['zero_drift_max_error'])
drift_error = float(s['expected_drift_max_error'])
centre_error = float(s['species_centre_mismatch_max'])

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)
copy_package(SOURCE_PACKAGE, OUT)

TEXT_DIR = OUT / 'mechanism_text'
TABLE_DIR = OUT / 'tables' / 'two_axis_mechanism'
FIG_DIR = OUT / 'figures' / 'two_axis_mechanism'
TEXT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

methods = f'''
## Controlled two-axis support-overlap experiment

We conducted a preregistered semi-synthetic factorial experiment using the
formal 0.5-degree spatial frames from Pennsylvania, Virginia, and North
Carolina. Fifteen species-region units were fixed before the full analysis.
The latent species distribution was held stationary between periods, so the
true ecological shift was zero.

Two support dimensions were manipulated independently. Effort-frame overlap
described the common spatial support available for target-effort
standardization and was treated as the primary determinant of whether the
corrected estimator was computable. Species-specific observation-support
overlap described continuity in the spatial support of the focal species and
was treated as the primary determinant of corrected shift error.

The core design crossed four effort-overlap levels, four species-support
overlap levels, three expected observation-footprint drift levels (0, 20, and
40 km per decade), two target-frame sample sizes (500 and 1,500 records per
period), and two species-width settings. Each core scenario was replicated 200
times, yielding 576,000 core replicates. Two prespecified marginal refinement
curves added 24,000 replicates, for 600,000 total replicates.

Effort-centre drift was generated by exponential tilting and calibrated before
sampling. Early and late latent species profiles were constrained to have
identical expected centres. The primary outcome for the effort axis was
corrected-estimator availability. The primary outcome for the species axis was
absolute corrected shift error. Raw error, correction gain, corrected-to-raw
error ratio, correction-help probability, and stable-cell count were secondary
outcomes.

Inference used 5,000 within-scenario label permutations for each primary axis
and 5,000 species-region-cluster bootstrap replicates. Prespecified negative
controls assessed zero-drift calibration, stationary latent species centres,
permutation-null centring, and axis specificity. No binary overlap cutoff was
selected.
'''

results = f'''
## Two support dimensions governed distinct correction outcomes

All 600,000 prespecified replicates were generated successfully. Calibration
errors were negligible: maximum zero-drift error was {zero_error:.3e}, maximum
expected drift calibration error was {drift_error:.3e}, and maximum latent
species-centre mismatch was {centre_error:.3e} degrees.

Effort-frame overlap strongly predicted whether target-effort correction was
estimable (Spearman rho = {effort_rho:.3f}; one-sided permutation
P = {effort_p:.6f}). The species-region-cluster bootstrap interval was
[{effort_low:.3f}, {effort_high:.3f}], with the positive direction retained in
{effort_boot:.1%} of bootstrap replicates. The expected strong monotonic
direction occurred in {effort_mono:.1%} of fixed-factor strata. High versus low
effort overlap increased correction availability by {effort_delta:.3f}.

Species-specific observation-support overlap independently predicted corrected
shift error (Spearman rho = {species_rho:.3f}; one-sided permutation
P = {species_p:.6f}). The cluster-bootstrap interval was
[{species_low:.3f}, {species_high:.3f}], with the negative direction retained in
{species_boot:.1%} of bootstrap replicates. The expected strong monotonic
direction occurred in {species_mono:.1%} of fixed-factor strata. Median
corrected error was {error_delta:.2f} km per decade higher under low than high
species-support overlap.

After adjustment for raw error, correction availability, effort overlap, drift,
sample size, width setting, region, and species-region cluster, species-support
overlap remained positively associated with correction gain (partial rank
rho = {partial_rho:.3f}; bootstrap interval [{partial_low:.3f},
{partial_high:.3f}]; positive fraction = {partial_boot:.1%}).

The two dimensions were axis-specific. Species overlap had only a weak
association with correction availability (rho = {cross_estimability:.3f}), and
effort overlap had only a weak association with corrected error
(rho = {cross_error:.3f}). Both marginal curves were strictly monotonic in the
prespecified direction, and all negative controls passed.
'''

discussion = f'''
## A two-axis support principle explains correction success and failure

The controlled results separate two problems that are often treated as one.
Effort-frame overlap is an estimability condition: without sufficient common
sampling support, target-effort correction cannot be computed reliably.
Species-specific observation-support overlap is a transportability condition:
even when correction is estimable, low continuity in the focal species'
spatial support produces substantially larger corrected errors.

This distinction explains why a correction that succeeds under controlled
sampling shift may fail or cause harm in opportunistic observations. A common
effort frame is necessary but not sufficient; the focal species signal must
also remain spatially comparable across periods. The real-data cell-Jaccard
diagnostic should therefore be interpreted as an applicability-ranking signal,
not a universal correction rule or binary threshold.

Correction gain remains secondary because it depends partly on the magnitude
of raw error. Corrected error and estimator availability are the primary
quantities. The controlled experiment complements the FIA-GBIF comparison by
providing a mechanism for heterogeneous real-data outcomes.

The experiment uses tree-derived spatial frames from three eastern United
States regions. Cross-taxon validation with eBird and the North American
Breeding Bird Survey remains the strongest test of broader generality. Until
then, the two-axis principle should be presented as a strongly supported
mechanism within the evaluated biodiversity-observation setting, not as a
universal law across all taxa and platforms.
'''

abstract_note = f'''
## Abstract result sentence

In a preregistered 600,000-replicate semi-synthetic experiment, effort-frame
overlap predicted whether correction was estimable (rho = {effort_rho:.3f}),
whereas species-specific support overlap independently predicted corrected
error (rho = {species_rho:.3f}); both associations passed permutation tests,
species-region-cluster bootstrapping, marginal-curve confirmation, and negative
controls.

## Abstract conclusion sentence

These findings support a two-axis applicability principle: common effort
support determines whether observation-drift correction can be estimated,
while species-level support continuity determines whether the corrected result
is transportable and accurate.

## Recommended title

Spatial support continuity predicts when observation-drift correction helps or
harms opportunistic biodiversity data

## Alternative mechanism-forward title

Two dimensions of spatial support govern the estimability and transportability
of observation-drift correction
'''

write_text(TEXT_DIR / '01_methods_two_axis_support_principle.txt', methods)
write_text(TEXT_DIR / '02_results_two_axis_support_principle.txt', results)
write_text(TEXT_DIR / '03_discussion_two_axis_support_principle.txt', discussion)
write_text(TEXT_DIR / '04_abstract_and_title_update.txt', abstract_note)

# Tables
sources = {
    'Table_M1_full_phase_confirmation_summary.csv': SUMMARY,
    'Table_M2_label_permutation_summary.csv': PERM,
    'Table_M3_cluster_bootstrap_summary.csv': BOOT,
    'Table_M4_negative_controls.csv': NEG,
    'Table_M5_marginal_effort_curve.csv': MARG_E,
    'Table_M6_marginal_species_curve.csv': MARG_S,
}
for name, src in tqdm(sources.items(), desc='Copying mechanism tables', unit='table'):
    shutil.copy2(src, TABLE_DIR / name)

# Figures
for i, name in enumerate(tqdm(FIGURES, desc='Copying mechanism figures', unit='figure'), start=1):
    shutil.copy2(MECH / name, FIG_DIR / f'Figure_M{i}_{name}')

insertion_map = '''
MANUSCRIPT INSERTION MAP

1. Methods
Insert mechanism_text/01_methods_two_axis_support_principle.txt after the
real-data FIA-GBIF validation methods.

2. Results
Insert mechanism_text/02_results_two_axis_support_principle.txt after the
real-data cell-Jaccard validation results.

3. Discussion
Insert mechanism_text/03_discussion_two_axis_support_principle.txt immediately
after the paragraph stating that universal correction was not validated.

4. Abstract and title
Use mechanism_text/04_abstract_and_title_update.txt.

5. Main figures
Use the drift-20 corrected-error surface and estimability surface as the main
mechanism figures. Use both marginal curves as compact confirmation figures.
Keep drift-0, drift-40 and permutation-null figures in the supplement.

6. Allowed claims
- Effort-frame overlap governed corrected-estimator availability.
- Species-specific support overlap governed corrected error.
- The two axes were distinct.
- Correction gain remained secondary but consistent after adjustment.

7. Disallowed claims
- A universal binary cell-Jaccard threshold.
- Proof across all taxa or platforms.
- A claim that target-effort correction always improves estimates.
- Treating this semi-synthetic confirmation as cross-taxon validation.
'''
write_text(OUT / 'MANUSCRIPT_INSERTION_MAP.txt', insertion_map)

readme = f'''
# Q1 mechanism manuscript package

Generated: {RUN_UTC}

The original manuscript package was copied and extended. The original directory
was not modified.

## Confirmed mechanism

1. Effort-frame overlap controls correction estimability.
2. Species-specific observation-support overlap controls corrected error and
   transportability.

## Frozen evidence

- GBIF DOI: {plan['formal_data']['gbif_doi']}
- Species-region units: {plan['formal_data']['species_region_units']}
- Core replicates: {plan['task_scale']['core_total_replicates']:,}
- Marginal replicates: {plan['task_scale']['marginal_total_replicates']:,}
- Total replicates: {plan['task_scale']['total_replicates']:,}
- Label permutations per axis: {plan['inference']['scenario_label_permutations']:,}
- Species-region-cluster bootstraps: {plan['inference']['species_region_cluster_bootstrap']:,}

## Main effects

- Effort overlap versus estimability: rho = {effort_rho:.4f}
- Species overlap versus corrected error: rho = {species_rho:.4f}
- Partial species overlap versus gain: rho = {partial_rho:.4f}
- Low-minus-high corrected error: {error_delta:.4f} km/decade

## Boundary

Tree mechanism experiments are frozen. Do not add simulations, optimize a
cutoff, change the frozen design, or select favorable scenarios. The next
independent extension is eBird-BBS cross-taxon validation.
'''
write_text(OUT / 'README_Q1_MECHANISM_UPDATE.md', readme)

# Manifest
manifest = []
for path in tqdm([p for p in OUT.rglob('*') if p.is_file()], desc='Building manifest', unit='file'):
    manifest.append({
        'relative_path': str(path.relative_to(OUT)),
        'size_bytes': int(path.stat().st_size),
    })
manifest_df = pd.DataFrame(manifest).sort_values('relative_path')
manifest_path = OUT / 'PACKAGE_MANIFEST.csv'
manifest_df.to_csv(manifest_path, index=False)
display(manifest_df)

# Project README
project_readme_path = BASE / 'README.md'
project_readme = project_readme_path.read_text(encoding='utf-8') if project_readme_path.exists() else '# Temporal Observation Drift\n'
project_readme_path.write_text(
    project_readme
    + f'''\n\n## Q1 mechanism manuscript package — {RUN_UTC}\n\n'''
      f'''- Package: {OUT}\n'''
      f'''- Full phase confirmation: passed\n'''
      f'''- Tree mechanism experiments: frozen\n'''
      f'''- Next independent extension: eBird-BBS cross-taxon validation\n''',
    encoding='utf-8',
)

print('\n========== RUN SUMMARY ==========')
print('CELL_STATUS: Q1_MECHANISM_MANUSCRIPT_PACKAGE_READY')
print(f'OUT_PACKAGE: {OUT}')
print('ORIGINAL_PACKAGE_PRESERVED: True')
print('FULL_CONFIRMATION_PASSED: True')
print(f'EFFORT_GLOBAL_RHO: {effort_rho}')
print(f'SPECIES_GLOBAL_RHO: {species_rho}')
print(f'PARTIAL_GAIN_RHO: {partial_rho}')
print(f'LOW_MINUS_HIGH_CORRECTED_ERROR: {error_delta}')
print(f'MECHANISM_TEXT_DIR: {TEXT_DIR}')
print(f'MECHANISM_TABLE_DIR: {TABLE_DIR}')
print(f'MECHANISM_FIGURE_DIR: {FIG_DIR}')
print(f'INSERTION_MAP: {OUT / "MANUSCRIPT_INSERTION_MAP.txt"}')
print(f'PACKAGE_README: {OUT / "README_Q1_MECHANISM_UPDATE.md"}')
print(f'PACKAGE_MANIFEST: {manifest_path}')
print(f'N_PACKAGE_FILES: {len(manifest_df)}')
print('TREE_MECHANISM_EXPERIMENTS_FROZEN: True')
print('NEXT_STEP: Review the integrated manuscript sections and wait for eBird EBD access for independent cross-taxon validation.')
print('README_UPDATED: True')
print('============================================')
