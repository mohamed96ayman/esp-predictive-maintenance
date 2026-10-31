# Reproduction record

Verification date: 2026-10-31. Platform: Windows, Python 3.11.5, NumPy 2.4.3, pandas 2.3.3, scikit-learn 1.8.0, SciPy 1.17.1, joblib 1.5.3.

| Check | Observed result |
|---|---|
| Raw SPE CSV reconstruction with saved normalization | 30,798 × 504 × 45 windows |
| Full metadata comparison | Every column and row exactly equal |
| Full window comparison | All 698,498,640 values exactly equal; maximum absolute difference 0 |
| Shared Rich-15 extraction | Matches the archived 675-feature cache |
| Complete raw-route LOWO | All 13,319 probabilities, labels, well IDs, and row indices exactly equal |
| Primary metrics | AUC 0.9463615022; AP 0.8385818351; F1 0.8205457464 |
| Confusion matrix | `[[11482,229],[330,1278]]` |
| Fold boundary audits | No held-out well in classifier fitting; zero train/validation row overlap |
| Automated package tests | 23 passed |
| Actual HTTP smoke test | Synthetic prepared window accepted; valid risk response |

The raw-route comparison **explicitly replays archived fitted normalization statistics**. It verifies reconstruction and inference through that saved transformation; it does not establish independent parity for a fresh normalization fit. The included bundle carries that transformation, so the README's reproduction command names it explicitly. Omitting it creates a new preparation and requires its own evaluation.

The classifier result is therefore reproduced on this platform for the documented replay path. The SmallLSTM regression hierarchy remains archived thesis evidence; these commands do not retrain it. The IQR result is also retained as archived sensitivity evidence rather than rerun.

See [preprocessing comparison](../evidence/preprocessing_parity.json), [classifier reproduction](../evidence/classification_reproduction.json), and [all per-well metrics](../evidence/per_well.csv). No telemetry is included in these reports.

The native API was verified locally. GitHub Actions also passed all 23 tests, built both Docker targets, started both services with Compose, and sent the example request successfully. A fresh public GitHub clone loaded the model and predicted in a serving-only environment with no MLflow installed. Docker Desktop is absent on the local Windows machine, so container execution was verified on the Linux CI runner. See [verification status](../evidence/verification.json).
