"""Tests for the bootstrap confidence interval helper."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import bootstrap_confidence_interval


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.01, scale=0.02, size=400)
    result = bootstrap_confidence_interval(sample, statistic="mean", n_resamples=500)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]
    assert result["n"] == 400


def test_bootstrap_ci_handles_empty_input():
    result = bootstrap_confidence_interval(np.array([]), statistic="mean")
    assert result["n"] == 0
    assert np.isnan(result["point"])
    assert np.isnan(result["ci_low"])


def test_bootstrap_ci_supports_block_size():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0.0, 1.0, size=200))
    result = bootstrap_confidence_interval(
        series, statistic="mean", n_resamples=300, block_size=10, random_state=42
    )
    assert result["ci_low"] < result["ci_high"]
    assert result["n"] == 200


def test_bootstrap_ci_rejects_invalid_args():
    sample = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(sample, confidence=0.0)
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(sample, confidence=1.0)
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(sample, n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(sample, statistic="variance")


def test_bootstrap_ci_drops_nans():
    sample = np.array([0.1, np.nan, 0.2, np.nan, 0.3])
    result = bootstrap_confidence_interval(sample, n_resamples=200)
    assert result["n"] == 3
