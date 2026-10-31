"""Validated local artifact loading and stateless inference."""
from pathlib import Path
import joblib
import numpy as np
from .config import HORIZON_HOURS, SOURCE_FEATURES, TABULAR_FEATURES, THRESHOLD, WINDOW_STEPS, fingerprint
from .features import rich15
from .io import read_json, sha256, versions


class RiskModel:
    def __init__(self, model_dir):
        self.directory = Path(model_dir)
        self.manifest = read_json(self.directory / "manifest.json")
        artifact = self.directory / "champion.joblib"
        if sha256(artifact) != self.manifest["artifact_sha256"]:
            raise ValueError("Model artifact checksum mismatch")
        if sha256(self.directory / "normalization.json") != self.manifest["normalization_sha256"]:
            raise ValueError("Normalization artifact checksum mismatch")
        if self.manifest["tabular_features"] != TABULAR_FEATURES:
            raise ValueError("Rich-15 feature order mismatch")
        if self.manifest["source_features"] != SOURCE_FEATURES:
            raise ValueError("Model feature manifest mismatch")
        if self.manifest["window_steps"] != WINDOW_STEPS:
            raise ValueError("Model window dimensions mismatch")
        for package, version in versions().items():
            if self.manifest["dependencies"].get(package) != version:
                raise ValueError(f"Incompatible {package} version; use the pinned environment")
        bundle = joblib.load(artifact)
        identity = "basic-" + fingerprint({"features": SOURCE_FEATURES,
            "normalization": bundle["normalization"], "window_steps": WINDOW_STEPS,
            "input_stage": "normalized_BASIC_window"})[:16]
        if identity != self.manifest["preprocessing_id"]:
            raise ValueError("Preprocessing identity mismatch")
        self.pipeline = bundle["pipeline"]
        if self.pipeline.n_features_in_ != 675:
            raise ValueError("Expected a 675-feature classifier")
        if self.manifest["horizon_hours"] != HORIZON_HOURS or self.manifest["threshold"] != THRESHOLD:
            raise ValueError("Unsupported prediction target")

    def validate(self, samples, feature_names, preprocessing_id):
        if feature_names != SOURCE_FEATURES:
            raise ValueError("feature_names must match the ordered model schema")
        if preprocessing_id != self.manifest["preprocessing_id"]:
            raise ValueError("preprocessing_id does not match this model")
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.asarray(samples, dtype=np.float32)
        if x.shape != (WINDOW_STEPS, len(SOURCE_FEATURES)) or not np.isfinite(x).all():
            raise ValueError("samples must be a finite numeric 504 x 45 window")
        return x

    def predict(self, samples, feature_names, preprocessing_id):
        window = self.validate(samples, feature_names, preprocessing_id)
        risk = float(self.pipeline.predict_proba(rich15(window[None]))[0, 1])
        return {"failure_risk": risk, "horizon_hours": HORIZON_HOURS,
                "threshold": THRESHOLD, "alert": risk >= THRESHOLD,
                "model_version": self.manifest["model_version"]}

    def info(self):
        return {k: self.manifest[k] for k in (
            "model_version", "preprocessing_id", "source_features", "window_steps",
            "horizon_hours", "threshold", "training_role", "input_stage",
        )}
