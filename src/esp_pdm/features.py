"""One feature extractor shared by evaluation, model export, and inference."""
import warnings
import numpy as np
from .config import SOURCE_FEATURES, WINDOW_STEPS


def rich15(windows):
    """Return float32 statistics in statistic-major order, as in the archived run."""
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (WINDOW_STEPS, len(SOURCE_FEATURES)):
        raise ValueError("Expected windows with shape (n, 504, 45)")
    c = np.nan_to_num(x, nan=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(x, axis=1)
        recent_mean = np.nanmean(x[:, -72:], axis=1)
        recent_std = np.nanstd(x[:, -72:], axis=1)
        first_mean = np.nanmean(x[:, :72], axis=1)
        percentiles = [np.nanpercentile(x, q, axis=1) for q in (10, 25, 75, 90)]
        result = np.concatenate([
            c[:, -1], np.nanmean(x, axis=1), std, c[:, -1] - c[:, 0],
            recent_mean, recent_std, np.nanmean(x[:, -144:], axis=1),
            np.nan_to_num(recent_mean - first_mean, nan=0.0),
            np.nan_to_num(recent_std / (std + 1e-8), nan=0.0),
            np.nanmin(x, axis=1), np.nanmax(x, axis=1),
            *[np.nan_to_num(p, nan=0.0) for p in percentiles],
        ], axis=1)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
