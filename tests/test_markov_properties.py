"""Property-based invariants for the visible Markov chain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from src.markov import (
    estimate_transition_matrix,
    forecast_state_probs,
    stationary_distribution,
)


@settings(max_examples=50, deadline=None)
@given(
    states=st.lists(st.integers(min_value=0, max_value=5), min_size=2, max_size=400),
    alpha=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_transition_matrix_rows_sum_to_one(states: list[int], alpha: float) -> None:
    series = pd.Series(states, dtype="int64")
    matrix = estimate_transition_matrix(series, n_states=6, alpha=alpha)
    row_sums = matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


@settings(max_examples=40, deadline=None)
@given(
    states=st.lists(st.integers(min_value=0, max_value=4), min_size=2, max_size=300),
    current_state=st.integers(min_value=0, max_value=4),
    horizon=st.integers(min_value=1, max_value=20),
)
def test_forecast_state_probs_is_probability_vector(
    states: list[int], current_state: int, horizon: int
) -> None:
    series = pd.Series(states, dtype="int64")
    matrix = estimate_transition_matrix(series, n_states=5, alpha=0.01)
    probs = forecast_state_probs(matrix, current_state=current_state, horizon=horizon)
    assert (probs >= 0.0).all()
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert probs.shape == (5,)


@settings(max_examples=30, deadline=None)
@given(
    states=st.lists(st.integers(min_value=0, max_value=3), min_size=20, max_size=400),
)
def test_stationary_distribution_is_fixed_point(states: list[int]) -> None:
    series = pd.Series(states, dtype="int64")
    matrix = estimate_transition_matrix(series, n_states=4, alpha=0.05)
    distribution = stationary_distribution(matrix)
    next_distribution = distribution @ matrix
    assert distribution.sum() == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(next_distribution, distribution, atol=1e-6)
