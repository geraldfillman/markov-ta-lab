"""Tests for visible Markov chain utilities."""

import numpy as np
import pandas as pd

from src.markov import (
    estimate_transition_matrix,
    forecast_state_probs,
    stationary_distribution,
    walkforward_forecasts,
)


def test_transition_matrix_shape_and_rows_sum_to_one():
    states = pd.Series([0, 1, 1, 2, 0, 1])

    matrix = estimate_transition_matrix(states, n_states=3)

    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(matrix[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(matrix[1], [0.0, 0.5, 0.5])


def test_sparse_rows_are_valid_with_laplace_smoothing():
    states = pd.Series([0, 1, 0])

    matrix = estimate_transition_matrix(states, n_states=3, alpha=1.0)

    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(matrix[2], [1 / 3, 1 / 3, 1 / 3])


def test_empty_sparse_rows_without_smoothing_self_loop():
    states = pd.Series([0, 1, 0])

    matrix = estimate_transition_matrix(states, n_states=3, alpha=0.0)

    np.testing.assert_allclose(matrix[2], [0.0, 0.0, 1.0])


def test_multistep_forecast_is_probability_vector():
    matrix = np.array(
        [
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )

    probs = forecast_state_probs(matrix, current_state=0, horizon=2)

    np.testing.assert_allclose(probs, [0.5, 0.5])
    np.testing.assert_allclose(probs.sum(), 1.0)
    assert (probs >= 0).all()


def test_stationary_distribution_sums_to_one():
    matrix = np.array(
        [
            [0.9, 0.1],
            [0.5, 0.5],
        ]
    )

    dist = stationary_distribution(matrix)

    np.testing.assert_allclose(dist @ matrix, dist)
    np.testing.assert_allclose(dist.sum(), 1.0)


def test_walkforward_forecasts_use_prior_window_only():
    index = pd.date_range("2024-01-02", periods=6, freq="B", name="Date")
    states = pd.Series([0, 1, 1, 2, 0, 1], index=index, name="state")

    forecasts = walkforward_forecasts(states, n_states=3, lookback=3, horizons=(1, 2))

    assert forecasts.index.tolist() == index[3:].tolist()
    assert forecasts.loc[index[3], "current_state"] == 2
    np.testing.assert_allclose(
        forecasts.loc[index[3], ["prob_h1_state0", "prob_h1_state1", "prob_h1_state2"]].to_numpy(dtype=float),
        [0.0, 0.0, 1.0],
    )
    assert {"prob_h2_state0", "prob_h2_state1", "prob_h2_state2"}.issubset(forecasts.columns)
