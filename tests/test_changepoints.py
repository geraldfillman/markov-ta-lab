"""Tests for the Change-Point Agent."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ruptures")

from src.changepoints import (
    annotate_changepoints,
    changepoint_pause_signal,
    detect_changepoints,
)


def _piecewise_signal(seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    calm = rng.normal(loc=0.0, scale=0.005, size=120)
    shock = rng.normal(loc=0.0, scale=0.040, size=120)
    calm2 = rng.normal(loc=0.0, scale=0.005, size=120)
    data = np.concatenate([calm, shock, calm2])
    index = pd.date_range("2024-01-01", periods=len(data), freq="B")
    return pd.Series(data, index=index, name="returns")


def test_detect_changepoints_finds_at_least_one_break():
    series = _piecewise_signal()
    positions = detect_changepoints(series, min_size=30, penalty=2.0)
    assert positions, "Expected at least one change-point on a piecewise signal"
    assert all(0 <= position < len(series) for position in positions)


def test_detect_changepoints_returns_empty_for_short_series():
    short = pd.Series([0.0, 0.1, 0.2])
    assert detect_changepoints(short, min_size=20) == []


def test_detect_changepoints_rejects_invalid_args():
    series = _piecewise_signal()
    with pytest.raises(ValueError):
        detect_changepoints(series, min_size=0)
    with pytest.raises(ValueError):
        detect_changepoints(series, penalty=0)


def test_changepoint_pause_signal_activates_window():
    pause = changepoint_pause_signal([10, 50], n_bars=3, total_length=60)
    assert len(pause) == 60
    assert pause.iloc[10:13].all()
    assert not pause.iloc[13]
    assert pause.iloc[50:53].all()
    assert not pause.iloc[9]


def test_changepoint_pause_signal_clips_to_total_length():
    pause = changepoint_pause_signal([18], n_bars=10, total_length=20)
    assert pause.iloc[18:20].all()
    assert len(pause) == 20


def test_changepoint_pause_signal_handles_empty_list():
    pause = changepoint_pause_signal([], n_bars=5, total_length=10)
    assert not pause.any()


def test_changepoint_pause_signal_rejects_invalid_args():
    with pytest.raises(ValueError):
        changepoint_pause_signal([1], n_bars=-1, total_length=10)
    with pytest.raises(ValueError):
        changepoint_pause_signal([1], n_bars=1, total_length=-1)


def test_annotate_changepoints_returns_per_bar_frame():
    series = _piecewise_signal()
    frame = annotate_changepoints(series, min_size=30, penalty=2.0, pause_window=4)
    assert frame.index.equals(series.index)
    assert set(frame.columns) == {"is_changepoint", "pause_active"}
    assert frame["pause_active"].sum() >= frame["is_changepoint"].sum()
