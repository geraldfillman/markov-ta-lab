"""Tests for src/stationarity.py — rolling transition matrices and stability."""

import numpy as np
import pandas as pd
import pytest

from src.stationarity import (
    flag_unstable_states,
    probability_stability_per_state,
    rolling_transition_matrix,
    stability_score,
)

N = 12


# ---------------------------------------------------------------------------
# rolling_transition_matrix
# ---------------------------------------------------------------------------


def _make_states(pattern, n_repeats=1):
    seq = list(pattern) * n_repeats
    return pd.Series(seq, index=pd.date_range("2024-01-01", periods=len(seq)))


def test_rtm_row_sums_to_one():
    states = _make_states([0, 1, 2, 3, 4, 5], n_repeats=5)
    mats = rolling_transition_matrix(states, window=10, n_states=N)
    assert len(mats) > 0
    for mat in mats.values():
        row_sums = mat.sum(axis=1)
        # rows that have observations must sum to 1
        observed_rows = row_sums > 0
        np.testing.assert_allclose(row_sums[observed_rows], 1.0, atol=1e-9)


def test_rtm_unobserved_rows_are_zero():
    states = _make_states([0, 1], n_repeats=10)
    mats = rolling_transition_matrix(states, window=10, n_states=N)
    for mat in mats.values():
        # states 2-11 never appear, rows should be zero
        for i in range(2, N):
            assert mat[i].sum() == pytest.approx(0.0)


def test_rtm_constant_state_yields_self_transition():
    """A window of all-same state should have P[s,s] == 1."""
    states = pd.Series([5] * 20, index=pd.date_range("2024-01-01", periods=20))
    mats = rolling_transition_matrix(states, window=10, n_states=N)
    assert len(mats) > 0
    for mat in mats.values():
        assert mat[5, 5] == pytest.approx(1.0)


def test_rtm_empty_input():
    states = pd.Series([], dtype=int, index=pd.DatetimeIndex([]))
    mats = rolling_transition_matrix(states, window=5)
    assert mats == {}


def test_rtm_window_too_small():
    states = _make_states([0, 1, 2])
    mats = rolling_transition_matrix(states, window=1)
    assert mats == {}


def test_rtm_step_parameter():
    states = _make_states([0, 1, 2, 3], n_repeats=5)
    mats_step1 = rolling_transition_matrix(states, window=8, n_states=N, step=1)
    mats_step2 = rolling_transition_matrix(states, window=8, n_states=N, step=2)
    assert len(mats_step1) > len(mats_step2)


# ---------------------------------------------------------------------------
# stability_score
# ---------------------------------------------------------------------------


def test_stability_score_identical_matrices_is_one():
    mat = np.eye(N) / N
    dates = pd.date_range("2024-01-01", periods=3)
    matrices = {d: mat.copy() for d in dates}
    scores = stability_score(matrices)
    np.testing.assert_allclose(scores.values, 1.0, atol=1e-9)


def test_stability_score_stationary_process_near_one():
    states = _make_states([0, 1, 2, 3], n_repeats=20)
    mats = rolling_transition_matrix(states, window=8, n_states=N, step=4)
    scores = stability_score(mats)
    assert (scores >= 0.8).all()


def test_stability_score_flipping_law_drops():
    """Process that changes its transition law mid-sequence should produce a lower score."""
    idx_stable = pd.date_range("2024-01-01", periods=40)
    # First half: 0->1->0->1..., second half: 2->3->2->3...
    seq_a = [0, 1] * 20
    seq_b = [2, 3] * 20
    states_a = pd.Series(seq_a, index=idx_stable)
    idx_b = pd.date_range("2024-03-12", periods=40)
    states_b = pd.Series(seq_b, index=idx_b)
    states = pd.concat([states_a, states_b])
    mats = rolling_transition_matrix(states, window=20, n_states=N, step=10)
    scores = stability_score(mats)
    assert scores.min() < 0.95


def test_stability_score_fewer_than_two_matrices():
    dates = pd.date_range("2024-01-01", periods=1)
    mats = {dates[0]: np.eye(N)}
    scores = stability_score(mats)
    assert scores.empty


def test_stability_score_values_in_range():
    states = _make_states([0, 1, 2, 3, 4, 5], n_repeats=6)
    mats = rolling_transition_matrix(states, window=10, n_states=N, step=2)
    scores = stability_score(mats)
    assert (scores >= 0).all() and (scores <= 1).all()


# ---------------------------------------------------------------------------
# probability_stability_per_state
# ---------------------------------------------------------------------------


def test_per_state_stability_in_range():
    states = _make_states([0, 1, 2, 3], n_repeats=10)
    mats = rolling_transition_matrix(states, window=8, n_states=N, step=2)
    for s in range(N):
        scores = probability_stability_per_state(mats, s)
        if not scores.empty:
            assert (scores >= 0).all() and (scores <= 1).all()


def test_per_state_stability_constant_is_one():
    mat = np.zeros((N, N))
    mat[3, 3] = 1.0
    dates = pd.date_range("2024-01-01", periods=3)
    matrices = {d: mat.copy() for d in dates}
    scores = probability_stability_per_state(matrices, state_id=3)
    np.testing.assert_allclose(scores.values, 1.0, atol=1e-9)


def test_per_state_stability_empty():
    scores = probability_stability_per_state({}, state_id=0)
    assert scores.empty


# ---------------------------------------------------------------------------
# flag_unstable_states
# ---------------------------------------------------------------------------


def test_flag_unstable_states_below_threshold():
    # Build matrices where state 0 is very unstable, state 1 is stable
    dates = pd.date_range("2024-01-01", periods=3)
    mat_base = np.zeros((N, N))
    mat_base[1, 1] = 1.0  # stable state 1

    mat_a = mat_base.copy()
    mat_a[0, 0] = 1.0  # state 0 -> 0

    mat_b = mat_base.copy()
    mat_b[0, 1] = 1.0  # state 0 -> 1 (completely different)

    matrices = {dates[0]: mat_a, dates[1]: mat_b, dates[2]: mat_a}
    unstable = flag_unstable_states(matrices, threshold=0.9)
    assert 0 in unstable
    assert 1 not in unstable


def test_flag_unstable_states_all_stable():
    mat = np.eye(N)
    dates = pd.date_range("2024-01-01", periods=3)
    matrices = {d: mat.copy() for d in dates}
    unstable = flag_unstable_states(matrices, threshold=0.7)
    assert unstable == []


def test_flag_unstable_states_empty():
    result = flag_unstable_states({}, threshold=0.7)
    assert result == []


def test_flag_unstable_states_returns_list_of_ints():
    states = _make_states([0, 1, 2, 3], n_repeats=10)
    mats = rolling_transition_matrix(states, window=8, n_states=N, step=4)
    result = flag_unstable_states(mats, threshold=0.5)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, int)
