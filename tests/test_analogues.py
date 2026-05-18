"""Tests for src/analogues.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analogues import (
    build_state_path_index,
    find_analogues,
    analogue_outcomes,
    apply_analogue_metadata,
)
from composite_states import StateRecord, composite_key
from conditional_markov import ConditionalForecast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_records(states: list[str]) -> list[StateRecord]:
    return [StateRecord(base_state=s) for s in states]


def make_forecast(confidence="high", warnings=(), probs=None):
    if probs is None:
        probs = {"UP": 0.6, "DOWN": 0.4}
    return ConditionalForecast(
        base_state="UP",
        conditions_requested={},
        conditions_used={},
        fallback_level=0,
        sample_count=20,
        next_state_probs=probs,
        confidence=confidence,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# build_state_path_index
# ---------------------------------------------------------------------------

def test_index_returns_dict():
    records = make_records(["A", "B", "C", "D", "E"])
    idx = build_state_path_index(records, window=3)
    assert isinstance(idx, dict)


def test_index_keys_are_tuples():
    records = make_records(["A", "B", "C", "D", "E"])
    idx = build_state_path_index(records, window=3)
    for k in idx:
        assert isinstance(k, tuple)


def test_index_values_are_lists_of_int():
    records = make_records(["A", "B", "C", "D", "E"])
    idx = build_state_path_index(records, window=3)
    for v in idx.values():
        assert isinstance(v, list)
        for i in v:
            assert isinstance(i, int)


def test_index_window_5_correct_count():
    records = make_records(["A", "B", "C", "D", "E", "F", "G"])
    idx = build_state_path_index(records, window=5)
    # 7 - 5 + 1 = 3 windows
    total = sum(len(v) for v in idx.values())
    assert total == 3


def test_index_repeated_pattern():
    records = make_records(["A", "B", "A", "B", "A", "B"])
    idx = build_state_path_index(records, window=2)
    # Pattern (A, B) should appear at indices 0, 2, 4
    key = (composite_key(StateRecord("A")), composite_key(StateRecord("B")))
    assert len(idx[key]) == 3


def test_index_empty_records():
    idx = build_state_path_index([], window=5)
    assert idx == {}


def test_index_shorter_than_window():
    records = make_records(["A", "B"])
    idx = build_state_path_index(records, window=5)
    assert idx == {}


# ---------------------------------------------------------------------------
# find_analogues
# ---------------------------------------------------------------------------

def test_find_analogues_exact_match():
    records = make_records(["A", "B", "C", "A", "B", "C"])
    idx = build_state_path_index(records, window=2)
    key = (composite_key(StateRecord("A")), composite_key(StateRecord("B")))
    result = find_analogues(key, idx, top_n=5)
    assert 0 in result
    assert 3 in result


def test_find_analogues_no_match():
    records = make_records(["A", "B", "C"])
    idx = build_state_path_index(records, window=2)
    key = (composite_key(StateRecord("X")), composite_key(StateRecord("Y")))
    result = find_analogues(key, idx, top_n=5)
    assert result == []


def test_find_analogues_top_n_respected():
    records = make_records(["A", "B"] * 10)
    idx = build_state_path_index(records, window=2)
    key = (composite_key(StateRecord("A")), composite_key(StateRecord("B")))
    result = find_analogues(key, idx, top_n=3)
    assert len(result) <= 3


def test_find_analogues_returns_list():
    idx = {}
    result = find_analogues(("X",), idx, top_n=5)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# analogue_outcomes
# ---------------------------------------------------------------------------

def test_analogue_outcomes_shape():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.01, -0.02])
    df = analogue_outcomes([0, 2], returns, horizon=3)
    assert df.shape == (2, 4)  # index + t+1, t+2, t+3


def test_analogue_outcomes_columns():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00])
    df = analogue_outcomes([0], returns, horizon=3)
    assert list(df.columns) == ["index", "t+1", "t+2", "t+3"]


def test_analogue_outcomes_values():
    returns = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4])
    df = analogue_outcomes([0], returns, horizon=3)
    assert df.iloc[0]["t+1"] == pytest.approx(0.1)
    assert df.iloc[0]["t+2"] == pytest.approx(0.2)
    assert df.iloc[0]["t+3"] == pytest.approx(0.3)


def test_analogue_outcomes_nan_at_boundary():
    returns = pd.Series([0.1, 0.2])
    df = analogue_outcomes([1], returns, horizon=3)
    # index=1, t+2 and t+3 are out of bounds
    assert pd.isna(df.iloc[0]["t+2"])
    assert pd.isna(df.iloc[0]["t+3"])


def test_analogue_outcomes_empty_indices():
    returns = pd.Series([0.1, 0.2, 0.3])
    df = analogue_outcomes([], returns, horizon=3)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# apply_analogue_metadata
# ---------------------------------------------------------------------------

def test_few_analogues_adds_warning():
    fc = make_forecast()
    fc2 = apply_analogue_metadata(fc, analogue_count=2)
    assert "few_analogues" in fc2.warnings


def test_zero_analogues_adds_warning():
    fc = make_forecast()
    fc2 = apply_analogue_metadata(fc, analogue_count=0)
    assert "few_analogues" in fc2.warnings


def test_three_analogues_no_warning():
    fc = make_forecast()
    fc2 = apply_analogue_metadata(fc, analogue_count=3)
    assert fc2 is fc  # unchanged


def test_many_analogues_no_warning():
    fc = make_forecast()
    fc2 = apply_analogue_metadata(fc, analogue_count=10)
    assert "few_analogues" not in fc2.warnings


def test_metadata_does_not_change_confidence():
    fc = make_forecast(confidence="high")
    fc2 = apply_analogue_metadata(fc, analogue_count=0)
    assert fc2.confidence == "high"


def test_metadata_does_not_change_probs():
    fc = make_forecast()
    fc2 = apply_analogue_metadata(fc, analogue_count=0)
    assert fc2.next_state_probs == fc.next_state_probs


def test_metadata_returns_new_instance_when_warning():
    fc = make_forecast()
    fc2 = apply_analogue_metadata(fc, analogue_count=1)
    assert fc2 is not fc


def test_existing_warnings_preserved():
    fc = make_forecast(warnings=("prior",))
    fc2 = apply_analogue_metadata(fc, analogue_count=0)
    assert "prior" in fc2.warnings
    assert "few_analogues" in fc2.warnings


def test_original_not_mutated():
    fc = make_forecast()
    apply_analogue_metadata(fc, analogue_count=0)
    assert "few_analogues" not in fc.warnings
