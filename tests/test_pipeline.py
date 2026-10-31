import numpy as np
import pandas as pd
import pytest
from esp_pdm.config import SOURCE_FEATURES
from esp_pdm.dataset import classification_index, well_weights
from esp_pdm.io import write_json
from esp_pdm.preprocessing import basic_features, harmonize, impute, scan_well
from esp_pdm.training import evaluate


def test_postrepair_negatives_require_mature_followup_and_do_not_label_other_wells():
    meta = pd.DataFrame({
        "well_id": ["A"]*5 + ["B"],
        "ts_start": pd.to_datetime(["2020-01-01", "2020-01-04", "2020-01-06", "2020-01-07", "2020-01-15", "2020-01-01"]),
        "ts_end": pd.to_datetime(["2020-01-02", "2020-01-05", "2020-01-07", "2020-01-08", "2020-01-16", "2020-01-02"]),
        "hours_to_event": [200., 24., np.nan, np.nan, np.nan, np.nan],
    })
    index, labels, wells = classification_index(meta)
    np.testing.assert_array_equal(index, [0, 1, 3])
    np.testing.assert_array_equal(labels, [0, 1, 0])
    assert set(wells) == {"A"}


def test_well_weights_give_equal_total_weight():
    wells = np.asarray(["A"]*2 + ["B"]*7)
    weights = well_weights(wells)
    assert weights[wells == "A"].sum() == pytest.approx(weights[wells == "B"].sum())
    assert weights.sum() == pytest.approx(len(wells))


def test_imputation_and_basic_differences_do_not_cross_well_boundaries():
    frame = pd.DataFrame({"well_id": ["A"]*3 + ["B"]*3,
                          "ts": pd.date_range("2020-01-01", periods=6, freq="20min")})
    sensors = ["current", "frequency", "intake_pressure", "discharge_pressure",
               "intake_temp", "motor_temp", "vibration_rms", "delta_p"]
    for sensor in sensors:
        frame[sensor] = [100., 110., np.nan, np.nan, 1000., 1100.]
    filled = impute(frame)
    assert filled.loc[2, "current"] == 110
    assert np.isnan(filled.loc[3, "current"])
    aggregated = pd.DataFrame({"well_id": frame.well_id})
    for sensor in sensors:
        aggregated[sensor+"_mean"] = filled[sensor]
    result = basic_features(aggregated)
    assert np.isnan(result.loc[3, "current_d1"])


def test_failure_windows_are_retained_with_missing_channels():
    frame = pd.DataFrame(np.nan, index=range(513), columns=SOURCE_FEATURES)
    frame["ts"] = pd.date_range("2020-01-01", periods=len(frame), freq="20min")
    frame["failure_event_any"] = 0
    frame.loc[512, "failure_event_any"] = 1
    frame["manual_off_frac"] = 0
    frame["op_on_frac"] = 1
    scanned = scan_well(frame)
    assert len(scanned[1]) == 2
    assert scanned[3][0] == pytest.approx(3)


def test_raw_schema_rejects_missing_labels():
    frame = pd.DataFrame({"WELL_ID": ["A"], "DATE": ["2020-01-01"], "CURRENT": [30],
                          "FREQUENCY": [50], "PRESS_INT": [100], "PRESS_DESC": [500],
                          "TEMP_INT": [200], "TEMP_MOT": [250], "VIBRATION": [.1]})
    with pytest.raises(ValueError, match="WellFailure"):
        harmonize(frame)


def test_small_lowo_run_is_grouped_and_fold_scalers_exclude_the_target(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    directory.mkdir()
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(18, 504, 45)).astype(np.float32)
    np.save(directory / "windows.npy", windows)
    rows = []
    for wi, well in enumerate(["A", "B", "C"]):
        for i in range(6):
            end = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i*3)
            rows.append({"well_id": well, "ts_start": end-pd.Timedelta(hours=168),
                         "ts_end": end, "hours_to_event": 24. if i < 3 else 200.})
    pd.DataFrame(rows).to_parquet(directory / "metadata.parquet", index=False)
    write_json(directory / "features.json", SOURCE_FEATURES)
    write_json(directory / "normalization.json", {"feature_names": SOURCE_FEATURES})
    from esp_pdm import training
    original = training.fit_pipeline
    fitted = []
    def capture(x, y, wells):
        model = original(x, y, wells)
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, x.mean(axis=0, dtype=np.float64))
        fitted.append(set(wells))
        return model
    monkeypatch.setattr(training, "fit_pipeline", capture)
    result = evaluate(directory, tmp_path / "results", track=False, threads=1)
    assert result["n_folds"] == 3
    assert fitted == [{"B", "C"}, {"A", "C"}, {"A", "B"}]
    assert all(not f["held_out_well_in_training"] and f["row_overlap"] == 0 for f in result["per_well"])


def test_directory_loader_excludes_production_tables(tmp_path):
    from esp_pdm.preprocessing import load_csvs
    pd.DataFrame({"WELL_ID": ["A"], "DATE": ["2020-01-01"],
                  "CURRENT": [30.], "FREQUENCY": [50.]}).to_csv(tmp_path / "telemetry.csv", index=False)
    pd.DataFrame({"WELL_ID": ["A"], "DATE": ["2020-01-01"],
                  "OIL_RATE": [100.]}).to_csv(tmp_path / "production.csv", index=False)
    frame, inputs = load_csvs(tmp_path)
    assert len(frame) == 1
    assert [item["file"] for item in inputs] == ["telemetry.csv"]
