"""Tests for the state-distribution drift monitor."""

import numpy as np
import pandas as pd
import pytest

from src.drift import (
    DEFAULT_KL_THRESHOLD,
    drift_alert,
    kl_divergence,
    state_frequency_distribution,
)


def test_state_frequency_distribution_smooths_and_normalises():
    states = pd.Series([0, 0, 1, 1, 2])
    dist = state_frequency_distribution(states, n_states=4, alpha=1e-6)
    assert dist.shape == (4,)
    np.testing.assert_allclose(dist.sum(), 1.0, atol=1e-9)
    assert dist[3] > 0.0  # Laplace floor keeps the unseen state non-zero.


def test_state_frequency_distribution_handles_empty_series():
    dist = state_frequency_distribution(pd.Series([], dtype=float), n_states=3, alpha=0.0)
    np.testing.assert_allclose(dist, [1.0 / 3, 1.0 / 3, 1.0 / 3])


def test_state_frequency_distribution_ignores_out_of_range_values():
    dist = state_frequency_distribution(pd.Series([-1, 0, 1, 5]), n_states=3, alpha=0.0)
    np.testing.assert_allclose(dist, [0.5, 0.5, 0.0])


def test_state_frequency_distribution_rejects_invalid_n_states():
    with pytest.raises(ValueError):
        state_frequency_distribution(pd.Series([0]), n_states=1)


def test_kl_divergence_zero_for_identical_distributions():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_divergence_positive_when_shifted():
    p = np.array([0.7, 0.1, 0.1, 0.1])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    assert kl_divergence(p, q) > 0.1


def test_kl_divergence_shape_mismatch_raises():
    with pytest.raises(ValueError):
        kl_divergence(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


def test_drift_alert_flags_when_distribution_shifts():
    rng = np.random.default_rng(0)
    train = pd.Series(rng.integers(0, 4, size=2000))
    drift = pd.Series(np.zeros(500, dtype=int))  # all in state 0
    result = drift_alert(train, drift, n_states=4, threshold=DEFAULT_KL_THRESHOLD)
    assert result["alert"] is True
    assert result["kl_divergence"] > DEFAULT_KL_THRESHOLD
    assert result["n_training"] == 2000
    assert result["n_current"] == 500


def test_drift_alert_silent_when_distributions_match():
    rng = np.random.default_rng(1)
    train = pd.Series(rng.integers(0, 4, size=5000))
    current = pd.Series(rng.integers(0, 4, size=2000))
    result = drift_alert(train, current, n_states=4)
    assert result["alert"] is False
    assert result["kl_divergence"] < DEFAULT_KL_THRESHOLD
