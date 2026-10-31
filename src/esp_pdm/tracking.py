"""Explicit MLflow tracking; inference does not import this module."""
from contextlib import contextmanager
import os
from pathlib import Path


class Tracker:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.mlflow = None
        if enabled:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow
            self.mlflow = mlflow
            directory = Path("runs").resolve()
            directory.mkdir(exist_ok=True)
            uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///" + (directory / "mlflow.db").as_posix())
            mlflow.set_tracking_uri(uri)
            name = "SPE-ESP-168h"
            experiment = mlflow.get_experiment_by_name(name)
            if experiment is None and uri.startswith("sqlite:"):
                mlflow.create_experiment(name, artifact_location=(directory / "mlartifacts").as_uri())
            mlflow.set_experiment(name)

    @contextmanager
    def run(self, name, nested=False, tags=None):
        if not self.enabled:
            yield None
            return
        with self.mlflow.start_run(run_name=name, nested=nested, tags=tags) as run:
            yield run.info.run_id

    def params(self, values):
        if self.enabled:
            self.mlflow.log_params(values)

    def metrics(self, values):
        if self.enabled:
            self.mlflow.log_metrics({k: v for k, v in values.items()
                                     if isinstance(v, (int, float)) and v is not None})

    def artifact(self, path):
        if self.enabled:
            self.mlflow.log_artifact(str(path))

    def model(self, pipeline, example):
        if self.enabled:
            from mlflow.models import infer_signature
            self.mlflow.sklearn.log_model(
                pipeline, name="champion",
                signature=infer_signature(example, pipeline.predict_proba(example)),
                input_example=example, pyfunc_predict_fn="predict_proba",
                skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
            )
