"""Tests for src/path_features.py — memory features."""

import numpy as np
import pandas as pd
import pytest

from src.path_features import (
    build_path_features,
    entry_velocity,
    failed_breakout_memory,
    prior_n_state_path,
    prior_state,
    state_age,
)

FAILED_BREAKOUT_ID = 9


# ---------------------------------------------------------------------------
# state_age
# ---------------------------------------------------------------------------


def test_state_age_resets_on_change():
    s = pd.Series([0, 0, 1, 1, 1, 2])
    result = state_age(s)
    assert list(result) == [1, 2, 1, 2, 3, 1]


def test_state_age_constant_run():
    n = 5
    s = pd.Series([3] * n)
    result = state_age(s)
    assert list(result) == list(range(1, n + 1))


def test_state_age_empty():
    s = pd.Series([], dtype=int)
    result = state_age(s)
    assert result.empty


def test_state_age_single_element():
    s = pd.Series([7])
    assert list(state_age(s)) == [1]


# ---------------------------------------------------------------------------
# prior_state
# ---------------------------------------------------------------------------


def test_prior_state_equals_shift1():
    s = pd.Series([0, 1, 2, 3, 4])
    result = prior_state(s, lookback=1)
    expected = s.shift(1)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_prior_state_lookback2():
    s = pd.Series([10, 11, 12, 13, 14])
    result = prior_state(s, lookback=2)
    assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[1])
    assert result.iloc[2] == 10
    assert result.iloc[4] == 12


def test_prior_state_empty():
    s = pd.Series([], dtype=int)
    result = prior_state(s)
    assert result.empty


# ---------------------------------------------------------------------------
# prior_n_state_path
# ---------------------------------------------------------------------------


def test_prior_n_state_path_basic():
    s = pd.Series([0, 1, 2, 3, 4])
    result = prior_n_state_path(s, n=3)
    # first 3 indices should be NaN (warmup)
    assert result.iloc[0] is None or (isinstance(result.iloc[0], float) and np.isnan(result.iloc[0]))
    assert result.iloc[1] is None or (isinstance(result.iloc[1], float) and np.isnan(result.iloc[1]))
    assert result.iloc[2] is None or (isinstance(result.iloc[2], float) and np.isnan(result.iloc[2]))
    # index 3: prior 3 states are [0,1,2]
    assert result.iloc[3] == "0>1>2"
    # index 4: prior 3 states are [1,2,3]
    assert result.iloc[4] == "1>2>3"


def test_prior_n_state_path_string_format():
    s = pd.Series([10, 11, 9])
    result = prior_n_state_path(s, n=2)
    assert result.iloc[2] == "11>9" or result.iloc[2] == "10>11"
    # index 2 prior-2 = [11, 9]? No: prior n means the n bars BEFORE current (exclusive)
    # i=2, values[0:2] = [10, 11] => "10>11"
    assert result.iloc[2] == "10>11"


def test_prior_n_state_path_empty():
    s = pd.Series([], dtype=int)
    result = prior_n_state_path(s, n=3)
    assert result.empty


def test_prior_n_state_path_warmup_is_none():
    s = pd.Series([1, 2, 3])
    result = prior_n_state_path(s, n=3)
    for val in result:
        assert val is None or (isinstance(val, float) and np.isnan(val))


# ---------------------------------------------------------------------------
# entry_velocity
# ---------------------------------------------------------------------------


def test_entry_velocity_defined_on_entry_bars():
    states = pd.Series([0, 0, 0, 1, 1, 1])
    prices = pd.Series([100.0, 101.0, 102.0, 105.0, 106.0, 107.0])
    result = entry_velocity(states, prices, window=3)
    # Bar 3 is entry: price[3]=105, price[0]=100 -> abs pct = 5/100 = 0.05
    assert result.iloc[3] == pytest.approx(0.05, abs=1e-8)


def test_entry_velocity_forward_fills():
    states = pd.Series([0, 0, 0, 1, 1, 2])
    prices = pd.Series([100.0, 101.0, 102.0, 106.0, 107.0, 108.0])
    result = entry_velocity(states, prices, window=3)
    # Bar 3 entry velocity forward-filled to bar 4
    assert result.iloc[4] == result.iloc[3]


def test_entry_velocity_empty():
    result = entry_velocity(pd.Series([], dtype=int), pd.Series([], dtype=float))
    assert result.empty


def test_entry_velocity_warmup_nan():
    states = pd.Series([5, 5, 5, 5, 5])
    prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = entry_velocity(states, prices, window=3)
    # First bar is "entry" but no prior price window available; bar 0: start=-3 < 0 -> NaN
    # Subsequent bars: same state, no new entry, forward-fill from NaN
    assert result.iloc[1:].isna().all()


# ---------------------------------------------------------------------------
# failed_breakout_memory
# ---------------------------------------------------------------------------


def test_failed_breakout_memory_counts_correctly():
    states = pd.Series([9, 0, 9, 0, 0])  # FBs at positions 0 and 2
    result = failed_breakout_memory(states, lookback=5)
    # bar 0: no prior -> 0
    assert result.iloc[0] == 0
    # bar 1: 1 FB in prior window (bar 0)
    assert result.iloc[1] == 1
    # bar 3: bars [0,1,2] in window: 2 FBs
    assert result.iloc[3] == 2
    # bar 4: bars [0,1,2,3] in window: 2 FBs
    assert result.iloc[4] == 2


def test_failed_breakout_memory_none_present():
    states = pd.Series([0, 1, 2, 3])
    result = failed_breakout_memory(states, lookback=10)
    assert (result == 0).all()


def test_failed_breakout_memory_empty():
    result = failed_breakout_memory(pd.Series([], dtype=int), lookback=10)
    assert result.empty


# ---------------------------------------------------------------------------
# build_path_features
# ---------------------------------------------------------------------------


def test_build_path_features_columns_and_length():
    states = pd.Series([0, 0, 1, 9, 1, 2, 3])
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    df = build_path_features(states, prices)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(states)
    for col in ["state_age", "prior_state", "prior_path", "entry_velocity", "failed_breakout_memory"]:
        assert col in df.columns


def test_build_path_features_index_aligned():
    idx = pd.date_range("2024-01-01", periods=5)
    states = pd.Series([0, 1, 0, 1, 0], index=idx)
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
    df = build_path_features(states, prices)
    assert list(df.index) == list(idx)


def test_build_path_features_empty():
    df = build_path_features(pd.Series([], dtype=int), pd.Series([], dtype=float))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
