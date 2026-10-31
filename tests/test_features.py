import numpy as np
from esp_pdm.config import SOURCE_FEATURES, TABULAR_FEATURES
from esp_pdm.features import rich15


def test_statistics_are_in_statistic_major_order():
    ramp = np.arange(504, dtype=np.float32)
    window = np.stack([ramp + channel * 1000 for channel in range(45)], axis=1)
    result = rich15(window[None])[0]
    assert result.shape == (675,)
    np.testing.assert_array_equal(result[:45], window[-1])
    np.testing.assert_allclose(result[45:90], window.mean(axis=0))
    np.testing.assert_allclose(result[3*45:4*45], np.full(45, 503))
    np.testing.assert_allclose(result[11*45:12*45], np.percentile(window, 10, axis=0), rtol=1e-6)
    assert TABULAR_FEATURES[45] == SOURCE_FEATURES[0] + "__mean"


def test_degenerate_and_missing_channels_produce_finite_features():
    window = np.ones((1, 504, 45), dtype=np.float32)
    window[:, :, 0] = np.nan
    result = rich15(window)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[0, np.arange(15)*45], 0)
