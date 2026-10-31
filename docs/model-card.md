# Model and input contract

**Task:** flag proximity to an ESP failure within a 168-hour horizon on the SPE Ecuador public cohort.

**Model:** a scikit-learn `StandardScaler` followed by `HistGradientBoostingClassifier`, fitted with equal well contributions and balanced class weights. The bundled model is an all-data refit on 13,398 selected windows from 32 wells. Its generalization estimate comes from a separate 30-well LOWO experiment. The score is not independently probability-calibrated.

**Input:** one `504 × 45` normalized BASIC window in the exact channel order published by `/model-info`; `preprocessing_id` must match. Rows run oldest to newest. Values must be finite numbers representable in float32. Raw readings, 675-dimensional summary vectors, strings, booleans, labels, and unknown fields do not satisfy this endpoint's schema.

**Output:** `failure_risk` between zero and one; a boolean `alert` using the fixed 0.5 threshold; horizon, threshold, and model version. No RUL estimate or failure-mode diagnosis is returned.

**Example provenance:** `examples/request.json` contains generated sinusoidal numbers in the prepared input space. It is an interface demonstration, not a real SPE window, and its response must not be interpreted as a benchmark result.

**Artifact validation:** a manifest records channel and statistic order, dependency versions, model checksum, normalization checksum, preprocessing identity, hyperparameters, and training-population counts. Startup refuses incompatible bundles. Load only the repository's own trusted local model artifacts; the endpoint does not accept uploaded model files.

**Limitations:** one public fleet, heterogeneous well behavior, and repeated development on that fleet. See the README's five weak held-out wells. Failure-risk detection does not establish precise failure-time estimation. Historical window eligibility and post-repair negative selection are documented in the pipeline notes.
