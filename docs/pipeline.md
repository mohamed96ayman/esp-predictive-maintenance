# Pipeline and fitting boundaries

The supported path is SPE telemetry CSVs → harmonization → fixed sensor thresholds → bounded filling → aggregation → BASIC features → windows → saved normalization → Rich-15 → classifier.

## Preparation

The loader merges redundant intake aliases, validates timestamps and labels, and keeps only the seven public physical sensor families. Units follow the source: amperes, Hz, psi, °F, and source vibration units. No voltage channel or production feature enters the BASIC representation.

Operating zeros in current, frequency, pressures, and temperatures become missing; zero vibration remains valid. A well is screened out if any of the six checked sensors exceeds 30% missingness before threshold cleaning. Fixed engineering bounds mask implausible values; differential pressure is recomputed after cleaning and bounded to [-100, 6000] psi.

The extracted execution path uses forward fill up to **3 samples**, backward fill up to **1 sample**, and an elapsed-time fill eligibility limit of `20 × --max-gap` minutes (default 5). Fill propagation stays within a well. Aggregation uses 20-minute bins within continuous blocks separated by gaps over six hours. Means, sample standard deviations, minima, and maxima are retained for the seven sensors, plus mean differential pressure. Three-hour rolling standard deviation and one-step differences of each of the eight means complete 45 channels. These settings reflect the extracted execution path verified in the reproduction report.

Windowing uses 504 retained samples and a hop of nine. Nominal cadence is 20 minutes; timestamps may span longer periods when source blocks contain gaps. A window passes if no channel exceeds 30% missingness, with the archived exception retaining event-containing and ≤168-hour pre-event windows. This label-aware exception defines the benchmark population and is not an online input-quality rule. Labels and state flags never enter the 45 model channels.

## Normalization and classifier fitting

`prepare --reference-stats models/normalization.json` replays the saved normalization, including its per-well references, global fallback, standard-deviation floor, zero replacement, and [-10, 10] clipping. The artifact's behavior is preserved explicitly. This is distinct from the classifier's `StandardScaler`, which is fitted independently within every LOWO training fold. The saved upstream normalization and fold-specific classifier standardizer have distinct fitting scopes.

Omitting `--reference-stats` fits normalization on the newly prepared population. It is a new preparation, with its own identity, and must be evaluated separately. API requests carry their preparation identity, preventing accidental use of differently normalized windows with the bundled model.

Rich-15 is deterministic and window-local. Its block order is last, mean, population standard deviation, last-minus-first, last-24-hour mean, last-24-hour standard deviation, last-48-hour mean, late-versus-early 24-hour shift, volatility ratio, minimum, maximum, and percentiles 10/25/75/90. Each block contains all 45 channels: **statistic-major order**. Last-24 and last-48 refer to the final 72 and 144 prepared samples.

Classifier scaling, fitting, class balancing, and per-well sample weighting exclude the held-out well. No held-out labels choose the threshold. The fixed model uses depth 2, seven leaves, learning rate 0.02, 500 iterations, L2 10, minimum leaf size 100, balanced class weights, and random seed 42. The existing configuration is reproduced rather than retuned in this repository. Historical feature/configuration selection on this fleet is development evidence, not a new independent holdout.

## Artifacts

A prepared directory contains `windows.npy`, `metadata.parquet`, `features.json`, `normalization.json`, and `preparation.json`. Features are extracted in bounded chunks and read through memory maps. The evaluation reader also accepts the original public BASIC artifact filenames.

Evaluation saves `metrics.json` and `oof.npz` under the chosen run directory. Training saves `champion.joblib`, `manifest.json`, and `normalization.json` under its output directory. The served artifact is a final refit; training-set fit metrics are not substituted for LOWO performance.
