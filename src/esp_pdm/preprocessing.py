"""Public SPE threshold/BASIC preprocessing, with explicit saved normalization."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from .config import SENSORS, SOURCE_FEATURES, STRIDE_STEPS, WINDOW_STEPS, fingerprint
from .io import read_json, sha256, write_json

COLUMN_MAP = {
    "WELL_ID": "well_id", "DATE": "ts", "CURRENT": "current", "FREQUENCY": "frequency",
    "PRESS_INT": "intake_pressure", "PRESS_DESC": "discharge_pressure",
    "TEMP_INT": "intake_temp", "TEMP_MOT": "motor_temp", "VIBRATION": "vibration_rms",
}
BOUNDS = {
    "current": (10, 300), "frequency": (25, 150), "intake_pressure": (0, 5000),
    "discharge_pressure": (0, 10500), "intake_temp": (40, 500),
    "motor_temp": (175, 500), "vibration_rms": (0, 10),
}


def load_csvs(source):
    source = Path(source)
    paths = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    frames = []
    inputs = []
    for path in paths:
        header = pd.read_csv(path, nrows=0).columns
        if not {"WELL_ID", "DATE", "CURRENT", "FREQUENCY"}.issubset(header):
            if source.is_dir():
                continue
            raise ValueError("SPE telemetry must contain WELL_ID, DATE, CURRENT, and FREQUENCY")
        frame = pd.read_csv(path, low_memory=False)
        for main, alternative in (("PRESS_INT", "PRESS_INTK"), ("TEMP_INT", "TEMP_INTK")):
            if alternative in frame:
                if main not in frame:
                    frame[main] = frame[alternative]
                else:
                    frame[main] = frame[main].fillna(frame[alternative])
        frames.append(frame)
        inputs.append({"file": path.name, "sha256": sha256(path)})
    if not frames:
        raise ValueError("No SPE telemetry CSVs found; point --input to the extracted Train directory")
    return pd.concat(frames, ignore_index=True), inputs


def harmonize(raw, max_bad_fraction=.30):
    frame = raw.rename(columns=COLUMN_MAP).copy()
    missing = set(SENSORS) | {"ts", "well_id"}
    missing -= set(frame.columns)
    if missing:
        raise ValueError(f"Missing SPE columns: {sorted(missing)}")
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce").astype("datetime64[ns]")
    frame["well_id"] = frame["well_id"].astype(str).str.strip()
    frame = frame.dropna(subset=["ts"]).sort_values(["well_id", "ts"]).reset_index(drop=True)
    for sensor in SENSORS:
        frame[sensor] = pd.to_numeric(frame[sensor], errors="coerce")
    if "WellFailure" not in frame:
        raise ValueError("WellFailure labels are required to prepare the evaluation dataset")
    labels = frame["WellFailure"].fillna("").astype(str).str.lower().str.strip()
    frame["failure_event"] = labels.eq("yes").astype(np.int8)
    frame["manual_off"] = labels.str.contains("manual off", regex=False).astype(np.int8)
    frame["op_on"] = ((frame.failure_event == 0) & (frame.manual_off == 0)).astype(np.int8)
    frame.loc[(frame.vibration_rms == 65535) | (frame.vibration_rms > 1e4), "vibration_rms"] = np.nan
    operating = frame.op_on == 1
    frame["zero_imputed"] = np.int8(0)
    for sensor in SENSORS[:-1]:
        bad = frame[sensor].abs().lt(1e-2) & operating
        frame.loc[bad, sensor] = np.nan
        frame.loc[bad, "zero_imputed"] = np.int8(1)
    checked = list(SENSORS[:-1])
    fractions = frame[checked].isna().groupby(frame.well_id).mean()
    rejected = fractions.index[(fractions > max_bad_fraction).any(axis=1)].tolist()
    frame = frame.loc[~frame.well_id.isin(rejected)].reset_index(drop=True)
    for sensor, (low, high) in BOUNDS.items():
        frame.loc[(frame[sensor] < low) | (frame[sensor] > high), sensor] = np.nan
    frame["delta_p"] = frame.discharge_pressure - frame.intake_pressure
    frame.loc[(frame.delta_p < -100) | (frame.delta_p > 6000), "delta_p"] = np.nan
    columns = ["ts", "well_id", *SENSORS, "delta_p", "failure_event", "manual_off", "op_on", "zero_imputed"]
    return frame[columns].copy(), rejected


def impute(frame, max_gap=5):
    """Bounded per-well ffill (3) / bfill (1), with elapsed-time gap eligibility."""
    frame = frame.copy()
    minutes = frame.ts.astype("int64").to_numpy() / 1e9 / 60
    for sensor in (*SENSORS, "delta_p"):
        observed = frame[sensor].notna()
        observed_at = pd.Series(np.where(observed, minutes, np.nan), index=frame.index)
        previous = observed_at.groupby(frame.well_id, sort=False).ffill().to_numpy()
        elapsed = np.where(observed, 0, np.nan_to_num(minutes - previous, nan=10080))
        fillable = ~observed & (elapsed <= 20 * max_gap)
        filled = frame.groupby("well_id", sort=False)[sensor].transform(
            lambda s: s.ffill(limit=3).bfill(limit=1))
        frame.loc[fillable, sensor] = filled.loc[fillable]
    return frame


def aggregate(frame):
    frame = frame.sort_values(["well_id", "ts"]).reset_index(drop=True)
    gaps = frame.groupby("well_id").ts.diff().gt(pd.Timedelta(hours=6))
    changes = frame.well_id.ne(frame.well_id.shift())
    frame["block"] = (gaps | changes).cumsum()
    aggregation = {sensor: ["mean", "std", "min", "max"] for sensor in SENSORS}
    aggregation["delta_p"] = ["mean"]
    flags = ["failure_event", "manual_off", "op_on", "zero_imputed"]
    aggregation.update({flag: ["max", "min", "mean"] for flag in flags})
    grouped = frame.set_index("ts").groupby(["well_id", "block"]).resample("20min")
    result = grouped.agg(aggregation).reset_index()
    names = []
    for base, stat in result.columns:
        if not stat:
            names.append(base)
        elif base in flags:
            names.append(base + "_" + {"max": "any", "min": "all", "mean": "frac"}[stat])
        else:
            names.append(base + "_" + stat)
    result.columns = names
    return result.drop(columns="block").sort_values(["well_id", "ts"]).reset_index(drop=True)


def basic_features(frame):
    frame = frame.copy()
    for sensor in (*SENSORS, "delta_p"):
        means = frame.groupby("well_id", sort=False)[sensor + "_mean"]
        frame[sensor + "_rstd3h"] = means.transform(lambda s: s.rolling(9, min_periods=1).std())
        frame[sensor + "_d1"] = means.diff()
    return frame


def scan_well(group):
    if len(group) < WINDOW_STEPS:
        return None
    values = group[SOURCE_FEATURES].to_numpy(dtype=np.float32)
    starts = np.arange(0, len(group) - WINDOW_STEPS + 1, STRIDE_STEPS)
    ends = starts + WINDOW_STEPS - 1
    cumulative = np.zeros((len(group) + 1, len(SOURCE_FEATURES)), dtype=np.int32)
    cumulative[1:] = np.cumsum(np.isnan(values), axis=0)
    missing = ((cumulative[ends + 1] - cumulative[starts]).astype(np.float32) / WINDOW_STEPS).max(axis=1)
    events = group.failure_event_any.fillna(0).to_numpy() > 0
    stops = group.manual_off_frac.fillna(0).to_numpy() > .5
    operating = group.op_on_frac.fillna(1).to_numpy() > .5
    event_indices = np.flatnonzero(events)
    positions = np.searchsorted(event_indices, ends, side="left")
    hte = np.full(len(starts), np.nan, dtype=np.float32)
    valid = positions < len(event_indices)
    timestamps = group.ts.astype("int64").to_numpy()
    hte[valid] = (timestamps[event_indices[positions[valid]]] - timestamps[ends[valid]]) / 3.6e12
    event_prefix = np.r_[0, np.cumsum(events)]
    stop_prefix = np.r_[0, np.cumsum(stops)]
    on_prefix = np.r_[0, np.cumsum(operating)]
    has_event = (event_prefix[ends+1] - event_prefix[starts]) > 0
    off = ((stop_prefix[ends+1] - stop_prefix[starts]) > 0) | (
        (on_prefix[ends+1] - on_prefix[starts]) / WINDOW_STEPS < .5)
    keep = (missing <= .30) | has_event | (np.isfinite(hte) & (hte <= 168))
    return values, starts[keep], ends[keep], hte[keep], off[keep], has_event[keep]


def make_windows(frame, out):
    frame = frame.copy()
    for column in frame.select_dtypes(include="float64"):
        frame[column] = frame[column].astype(np.float32)
    scans = []
    total = 0
    for well, group in frame.groupby("well_id", sort=True):
        group = group.sort_values("ts").reset_index(drop=True)
        scan = scan_well(group)
        if scan is not None and len(scan[1]):
            scans.append((well, group.ts, scan))
            total += len(scan[1])
    if total == 0:
        raise ValueError("No windows survived preparation; need at least 504 samples per well")
    windows = np.lib.format.open_memmap(out / "windows.npy", mode="w+", dtype=np.float32,
                                       shape=(total, WINDOW_STEPS, len(SOURCE_FEATURES)))
    metadata = []
    cursor = 0
    for well, timestamps, (values, starts, ends, hte, off, has_event) in scans:
        for first in range(0, len(starts), 128):
            selection = starts[first:first+128, None] + np.arange(WINDOW_STEPS)[None, :]
            windows[cursor+first:cursor+first+len(selection)] = values[selection]
        metadata.append(pd.DataFrame({"well_id": well, "ts_start": timestamps.iloc[starts].to_numpy(),
                                      "ts_end": timestamps.iloc[ends].to_numpy(), "missing_frac": np.nan,
                                      "window_off": off.astype(np.int8), "window_has_event": has_event.astype(np.int8),
                                      "hours_to_event": hte}))
        cursor += len(starts)
    windows.flush()
    return windows, pd.concat(metadata, ignore_index=True)


def window_stats(windows):
    """Match the archive's float32 moment calculation for a selected window population."""
    flattened = np.asarray(windows).reshape(-1, len(SOURCE_FEATURES))
    finite = np.isfinite(flattened)
    count = finite.sum(axis=0)
    sums = np.nansum(np.where(finite, flattened, 0.0), axis=0)
    squares = np.nansum(np.where(finite, flattened * flattened, 0.0), axis=0)
    mean = np.zeros(len(SOURCE_FEATURES), dtype=np.float32)
    good = count > 0
    mean[good] = sums[good] / count[good]
    variance = np.zeros_like(mean)
    good = count > 1
    variance[good] = np.maximum(squares[good] / count[good] - mean[good]**2, 0)
    std = np.ones_like(mean)
    std[good] = np.sqrt(variance[good])
    return np.nan_to_num(mean).tolist(), np.nan_to_num(std, nan=1).tolist()


def normalize_windows(windows, meta, reference=None):
    if reference:
        stats = read_json(reference)
        if stats["feature_names"] != SOURCE_FEATURES:
            raise ValueError("Normalization channels do not match BASIC schema")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, std = window_stats(windows)
            global_std = np.maximum(std, .1)
            perwell = {}
            for well in sorted(set(meta.well_id)):
                mean_w, std_w = window_stats(windows[meta.well_id.to_numpy() == well])
                perwell[well] = {"mu": mean_w, "sd": np.maximum(std_w, np.maximum(.05 * global_std, .1)).tolist()}
            stats = {"version": "v3-fixed-fast", "feature_names": SOURCE_FEATURES,
                     "gap_flag_indices": [], "sd_floor_frac": .05, "min_absolute_sd": .1,
                     "global": {"mu": mean, "sd": global_std.tolist()}, "perwell": perwell}
    if stats["feature_names"] != SOURCE_FEATURES:
        raise ValueError("Normalization channels do not match BASIC schema")
    for well in sorted(set(meta.well_id)):
        indices = np.flatnonzero(meta.well_id.to_numpy() == well)
        selected = stats.get("perwell", {}).get(well, stats["global"])
        mean = np.asarray(selected["mu"], dtype=np.float32)
        global_std = np.asarray(stats["global"]["sd"], dtype=np.float32)
        std = np.maximum(selected["sd"], np.maximum(.05 * global_std, .1)).astype(np.float32)
        for first in range(0, len(indices), 128):
            rows = indices[first:first+128]
            chunk = np.array(windows[rows], dtype=np.float32, copy=True)
            np.nan_to_num(chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            chunk -= mean[None, None, :]
            chunk /= std[None, None, :]
            np.clip(chunk, -10, 10, out=chunk)
            windows[rows] = chunk
    windows.flush()
    return stats


def prepare(source, output, reference_stats=None, max_gap=5):
    out = Path(output)
    if (out / "windows.npy").exists():
        raise ValueError("Output already contains windows; select a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    raw, inputs = load_csvs(source)
    print(f"Loaded {len(raw):,} rows", flush=True)
    harmonized, rejected = harmonize(raw)
    del raw
    print(f"Retained {harmonized.well_id.nunique()} wells after quality screening", flush=True)
    canonical = basic_features(aggregate(impute(harmonized, max_gap)))
    del harmonized
    print(f"Aggregated {len(canonical):,} rows; constructing windows", flush=True)
    windows, meta = make_windows(canonical, out)
    del canonical
    print(f"Normalizing {len(windows):,} windows", flush=True)
    stats = normalize_windows(windows, meta, reference_stats)
    meta.to_parquet(out / "metadata.parquet", index=False)
    write_json(out / "features.json", SOURCE_FEATURES)
    write_json(out / "normalization.json", stats)
    manifest = {"input_files": inputs, "window_shape": list(windows.shape),
                "source_features": SOURCE_FEATURES, "rejected_wells": rejected,
                "max_gap_samples": max_gap, "ffill_samples": 3, "bfill_samples": 1,
                "window_hop": STRIDE_STEPS, "normalization_reference_supplied": bool(reference_stats),
                "normalization_scope": "prepared population; per-well reference with global fallback",
                "normalization_fingerprint": fingerprint(stats),
                "reproduction_status": "not compared; see the separate parity report"}
    write_json(out / "preparation.json", manifest)
    return manifest
