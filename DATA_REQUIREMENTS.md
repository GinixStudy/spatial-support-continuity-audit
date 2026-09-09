# Data requirements

The extracted scientific code is Kaggle-native. A full raw-data run requires:

1. Internet access to the USFS FIA DataMart and GBIF API for the download cells.
2. The registered GBIF download DOI `10.15468/dl.sm6ygu`, mounted at
   `/kaggle/input/datasets/nanjide/20260711gbif` (143,355 locked records).
3. Writable `/kaggle/working/fia_temporal_observation_drift`.
4. For the PA descriptive-count repair, the extracted legacy archive at
   `/kaggle/working/recovered_q1/kaggle_working/fia_temporal_observation_drift`,
   containing `raw/PA_PLOT.csv` and `raw/PA_TREE.csv`.

The old `20260711q1test.tar` is a frozen historical snapshot. Do not overwrite
it. Use it with `verify_frozen_results.py` or extract it for the PA repair.
