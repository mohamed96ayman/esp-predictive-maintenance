"""Frozen-model LOWO evaluation and a separate all-data deployment refit."""
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .config import (HORIZON_HOURS, HYPERPARAMETERS, SEED, SOURCE_FEATURES,
                     TABULAR_FEATURES, THRESHOLD, fingerprint)
from .dataset import Dataset, well_weights
from .io import sha256, versions, write_json
from .tracking import Tracker


def metrics(y, probability):
    prediction = probability >= THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "n_positive": int(y.sum()),
        "auc": float(roc_auc_score(y, probability)) if both else None,
        "ap": float(average_precision_score(y, probability)) if y.sum() else None,
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "negative_fpr": float(fp / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def make_pipeline():
    return Pipeline([("scaler", StandardScaler()),
                     ("classifier", HistGradientBoostingClassifier(**HYPERPARAMETERS))])


def fit_pipeline(x, y, wells):
    model = make_pipeline()
    model.fit(x, y, classifier__sample_weight=well_weights(wells))
    return model


def evaluate(data, out, track=True, threads=8, max_folds=None, reference=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(data, out / "cache")
    tracker = Tracker(track)
    rows = []
    predictions = []
    started = time.time()
    with tracker.run("LOWO", tags={"role": "held_out_evaluation", "protocol": "LOWO"}) as run_id:
        tracker.params({**HYPERPARAMETERS, "horizon_hours": HORIZON_HOURS,
                        "threshold": THRESHOLD, "preprocessing_id": dataset.preprocessing_id,
                        "feature_count": 675, "seed": SEED})
        with threadpool_limits(limits=threads):
            for well in dataset.eval_wells[:max_folds]:
                train = dataset.wells != well
                val = dataset.wells == well
                train_indices, val_indices = dataset.index[train], dataset.index[val]
                if well in set(dataset.wells[train]) or np.intersect1d(train_indices, val_indices).size:
                    raise ValueError("Outer well or rows crossed the LOWO boundary")
                model = fit_pipeline(np.asarray(dataset.features[train_indices], dtype=np.float32),
                                     dataset.y[train], dataset.wells[train])
                probability = model.predict_proba(np.asarray(dataset.features[val_indices], dtype=np.float32))[:, 1]
                score = metrics(dataset.y[val], probability)
                fold = {"well": well, "n_train": int(train.sum()),
                        "n_train_wells": len(set(dataset.wells[train])),
                        "held_out_well_in_training": False, "row_overlap": 0,
                        "scaler_fit_scope": "outer_training_wells", **score}
                rows.append(fold)
                predictions.append((dataset.y[val], probability, dataset.wells[val], val_indices))
                with tracker.run(well, nested=True):
                    tracker.params({"held_out_well": well, "n_train": int(train.sum()),
                                    "scaler_fit_scope": "outer_training_wells"})
                    tracker.metrics(score)
                print(f"LOWO {len(rows)}/{min(len(dataset.eval_wells), max_folds or len(dataset.eval_wells))} "
                      f"{well}: F1={score['f1']:.4f}", flush=True)
        if not predictions:
            raise ValueError("No eligible LOWO folds: each held-out well needs 3 positives and 3 negatives")
        y, prob, wells, indices = [np.concatenate([p[i] for p in predictions]) for i in range(4)]
        pooled = metrics(y, prob)
        f1s = np.array([r["f1"] for r in rows])
        summary = {"role": "held_out_evaluation", "protocol": "LOWO", "n_folds": len(rows),
                   "preprocessing_id": dataset.preprocessing_id, "mlflow_run_id": run_id,
                   "pooled": pooled, "per_well": rows,
                   "well_f1": {"mean": float(f1s.mean()), "sd_ddof0": float(f1s.std()),
                               "median": float(np.median(f1s)),
                               "q25": float(np.quantile(f1s, .25)), "q75": float(np.quantile(f1s, .75)),
                               "below_0_05": int((f1s < .05).sum())},
                   "dependencies": versions(), "duration_seconds": time.time()-started}
        np.savez_compressed(out / "oof.npz", y_true=y, prob=prob, well=wells, original_index=indices)
        if reference:
            with np.load(reference, allow_pickle=False) as ref:
                aligned = (np.array_equal(y, ref["y_true"]) and np.array_equal(wells, ref["well"]) and
                           np.array_equal(indices, ref["original_index"]))
                summary["reference_comparison"] = {
                    "rows_labels_wells_exact": aligned,
                    "probabilities_exact": bool(aligned and np.array_equal(prob, ref["prob"])),
                    "max_probability_difference": float(np.max(np.abs(prob-ref["prob"]))) if aligned else None,
                }
        write_json(out / "metrics.json", summary)
        tracker.metrics(pooled)
        tracker.metrics({"macro_f1": float(f1s.mean()), "well_f1_sd": float(f1s.std())})
        tracker.artifact(out / "metrics.json")
        tracker.artifact(out / "oof.npz")
    return summary


def train(data, out, track=True, threads=8):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(data, Path("runs") / "training_cache")
    tracker = Tracker(track)
    x = np.asarray(dataset.features[dataset.index], dtype=np.float32)
    with tracker.run("all-data-refit", tags={"role": "deployment_refit"}) as run_id:
        tracker.params({**HYPERPARAMETERS, "threshold": THRESHOLD, "horizon_hours": HORIZON_HOURS,
                        "n_training": len(dataset.y), "preprocessing_id": dataset.preprocessing_id})
        with threadpool_limits(limits=threads):
            pipeline = fit_pipeline(x, dataset.y, dataset.wells)
        bundle = {"pipeline": pipeline, "normalization": dataset.normalization}
        artifact = out / "champion.joblib"
        joblib.dump(bundle, artifact, compress=3)
        normalized = {k: v for k, v in dataset.normalization.items()
                      if k not in ("timestamp", "source_path", "artifacts")}
        write_json(out / "normalization.json", normalized)
        identity = fingerprint({"preprocessing_id": dataset.preprocessing_id,
                                "hp": HYPERPARAMETERS, "artifact": sha256(artifact)})[:12]
        manifest = {"schema_version": 1, "model_version": f"basic675-{identity}",
                    "training_role": "all-data refit; new-well performance estimated by LOWO",
                    "input_stage": "normalized_BASIC_window", "window_steps": 504,
                    "source_features": SOURCE_FEATURES, "tabular_features": TABULAR_FEATURES,
                    "preprocessing_id": dataset.preprocessing_id,
                    "horizon_hours": HORIZON_HOURS, "threshold": THRESHOLD,
                    "hyperparameters": HYPERPARAMETERS, "dependencies": versions(),
                    "n_training_windows": len(dataset.y), "n_training_wells": len(set(dataset.wells)),
                    "n_training_positive": int(dataset.y.sum()), "mlflow_run_id": run_id,
                    "artifact_sha256": sha256(artifact), "normalization_sha256": sha256(out / "normalization.json")}
        write_json(out / "manifest.json", manifest)
        tracker.model(pipeline, x[:2])
        tracker.artifact(artifact)
        tracker.artifact(out / "manifest.json")
        tracker.artifact(out / "normalization.json")
    return manifest


def export_request(data, row, model_dir, output):
    from .model import RiskModel
    out = Path(output)
    dataset = Dataset(data, Path("runs") / "request_cache")
    model = RiskModel(model_dir)
    if dataset.preprocessing_id != model.manifest["preprocessing_id"]:
        raise ValueError("Dataset preprocessing does not match model")
    if row < 0 or row >= len(dataset.windows):
        raise ValueError("Window index out of range")
    request = {"preprocessing_id": dataset.preprocessing_id, "feature_names": SOURCE_FEATURES,
               "samples": np.asarray(dataset.windows[row], dtype=np.float32).tolist()}
    model.predict(**{ "samples": request["samples"], "feature_names": request["feature_names"],
                     "preprocessing_id": request["preprocessing_id"]})
    write_json(out, request)
    return {"output": out.name}
