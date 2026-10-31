"""Frozen classifier specification; no machine-specific configuration."""
import hashlib
import json

HORIZON_HOURS = 168
THRESHOLD = 0.5
WINDOW_STEPS = 504
STRIDE_STEPS = 9
SEED = 42
SENSORS = (
    "current", "frequency", "intake_pressure", "discharge_pressure",
    "intake_temp", "motor_temp", "vibration_rms",
)
SOURCE_FEATURES = [f"{s}_{stat}" for s in SENSORS for stat in ("mean", "std", "min", "max")]
SOURCE_FEATURES += ["delta_p_mean"]
SOURCE_FEATURES += [f"{s}_{stat}" for s in (*SENSORS, "delta_p") for stat in ("rstd3h", "d1")]
STATISTICS = (
    "last", "mean", "std", "delta", "last24_mean", "last24_std", "last48_mean",
    "shift24", "volatility_ratio", "min", "max", "p10", "p25", "p75", "p90",
)
TABULAR_FEATURES = [f"{feature}__{stat}" for stat in STATISTICS for feature in SOURCE_FEATURES]
HYPERPARAMETERS = {
    "max_iter": 500, "max_depth": 2, "learning_rate": 0.02,
    "l2_regularization": 10.0, "min_samples_leaf": 100, "max_leaf_nodes": 7,
    "random_state": SEED, "class_weight": "balanced",
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()
