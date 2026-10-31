"""Prepared-dataset reader and the archived classifier's cohort rules."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import HORIZON_HOURS, SOURCE_FEATURES, fingerprint
from .features import rich15
from .io import read_json


def classification_index(meta):
    h = meta["hours_to_event"].to_numpy(dtype=np.float64)
    wells = meta["well_id"].astype(str).to_numpy()
    end = meta["ts_end"].to_numpy(dtype="datetime64[ns]")
    start = meta["ts_start"].to_numpy(dtype="datetime64[ns]")
    original = np.flatnonzero(~np.isnan(h))
    repaired = []
    for well in np.unique(wells):
        rows = np.flatnonzero(wells == well)
        valid = rows[~np.isnan(h[rows])]
        unknown = rows[np.isnan(h[rows])]
        if not len(valid) or not len(unknown):
            continue
        nearest = valid[np.argmin(h[valid])]
        event = end[nearest] + np.timedelta64(int(h[nearest] * 3600), "s")
        candidates = [int(i) for i in unknown if start[i] > event and
                      end[i] + np.timedelta64(HORIZON_HOURS * 3600, "s") <= end[rows].max()]
        positives = int(np.nansum(h[rows] <= HORIZON_HOURS))
        negatives = int(np.nansum(h[rows] > HORIZON_HOURS))
        budget = max(0, 10 * max(positives, 1) - negatives)
        repaired.extend(candidates[:budget])
    index = np.concatenate([original, np.asarray(repaired, dtype=np.int64)])
    y = np.concatenate([(h[original] <= HORIZON_HOURS).astype(np.int32),
                        np.zeros(len(repaired), dtype=np.int32)])
    return index, y, wells[index]


def well_weights(wells):
    unique, counts = np.unique(wells, return_counts=True)
    lookup = {w: 1.0 / n for w, n in zip(unique, counts)}
    factor = len(wells) / sum(lookup[w] for w in wells)
    return np.asarray([lookup[w] * factor for w in wells], dtype=np.float32)


class Dataset:
    def __init__(self, directory, cache_dir):
        directory = Path(directory)
        legacy = (directory / "public_X.npy").exists()
        self.windows = np.load(directory / ("public_X.npy" if legacy else "windows.npy"),
                               mmap_mode="r", allow_pickle=False)
        names = read_json(directory / ("public_features.json" if legacy else "features.json"))
        if names != SOURCE_FEATURES or self.windows.shape[1:] != (504, 45):
            raise ValueError("Dataset is not the 45-channel BASIC window representation")
        self.meta = pd.read_parquet(directory / ("public_meta.parquet" if legacy else "metadata.parquet"))
        for column in ("ts_start", "ts_end"):
            self.meta[column] = pd.to_datetime(self.meta[column])
        if len(self.meta) != len(self.windows):
            raise ValueError("Window/metadata row counts differ")
        norm_path = directory / ("public_norm_stats.json" if legacy else "normalization.json")
        self.normalization = {k: v for k, v in read_json(norm_path).items()
                              if k not in ("timestamp", "source_path", "artifacts")}
        self.preprocessing_id = "basic-" + fingerprint({
            "features": names, "normalization": self.normalization,
            "window_steps": 504, "input_stage": "normalized_BASIC_window",
        })[:16]
        cached = directory / ("public_rich15.npy" if legacy else "rich15.npy")
        if cached.exists():
            self.features = np.load(cached, mmap_mode="r", allow_pickle=False)
        else:
            cached = Path(cache_dir) / "rich15.npy"
            cached.parent.mkdir(parents=True, exist_ok=True)
            out = np.lib.format.open_memmap(cached, mode="w+", dtype=np.float32,
                                           shape=(len(self.windows), 675))
            for first in range(0, len(self.windows), 128):
                last = min(first + 128, len(self.windows))
                out[first:last] = rich15(self.windows[first:last])
            out.flush()
            del out
            self.features = np.load(cached, mmap_mode="r", allow_pickle=False)
        if self.features.shape != (len(self.windows), 675) or not np.isfinite(self.features).all():
            raise ValueError("Invalid Rich-15 feature matrix")
        # A sparse deterministic feature replay detects stale or differently ordered caches.
        sample = np.linspace(0, len(self.windows)-1, min(7, len(self.windows)), dtype=int)
        if not np.array_equal(rich15(self.windows[sample]), self.features[sample]):
            raise ValueError("Cached Rich-15 features do not match the shared extractor")
        self.index, self.y, self.wells = classification_index(self.meta)
        self.eval_wells = [w for w in sorted(set(self.wells)) if
                           int(self.y[self.wells == w].sum()) >= 3 and
                           int((self.y[self.wells == w] == 0).sum()) >= 3]
