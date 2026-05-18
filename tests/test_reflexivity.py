"""Tests for src/reflexivity.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reflexivity import compute_reflexivity_score, apply_reflexivity_warning
from conditional_markov import ConditionalForecast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_forecast(confidence="high", warnings=()):
    return ConditionalForecast(
        base_state="UP",
        conditions_requested={},
        conditions_used={},
        fallback_level=0,
        sample_count=20,
        next_state_probs={"UP": 0.6, "DOWN": 0.4},
        confidence=confidence,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# compute_reflexivity_score
# ---------------------------------------------------------------------------

def test_all_zero_inputs():
    score = compute_reflexivity_score(0.0, 0.0, 0.0, 3.0)  # kurtosis=3 -> excess=0
    assert score == 0.0


def test_all_none_inputs():
    score = compute_reflexivity_score(None, None, None, None)
    assert score == 0.0


def test_all_nan_inputs():
    score = compute_reflexivity_score(float("nan"), float("nan"), float("nan"), float("nan"))
    assert score == 0.0


def test_score_in_unit_interval():
    score = compute_reflexivity_score(0.1, 5.0, 0.9, 15.0)
    assert 0.0 <= score <= 1.0


def test_max_inputs_clip_to_one():
    score = compute_reflexivity_score(100.0, 100.0, 100.0, 1000.0)
    assert score == pytest.approx(1.0)


def test_score_increases_with_volume_zscore():
    low = compute_reflexivity_score(0.0, 1.0, 0.0, 3.0)
    high = compute_reflexivity_score(0.0, 3.5, 0.0, 3.0)
    assert high > low


def test_score_increases_with_price_change():
    low = compute_reflexivity_score(0.01, 0.0, 0.0, 3.0)
    high = compute_reflexivity_score(0.04, 0.0, 0.0, 3.0)
    assert high > low


def test_high_kurtosis_increases_score():
    low = compute_reflexivity_score(0.0, 0.0, 0.0, 3.0)
    high = compute_reflexivity_score(0.0, 0.0, 0.0, 13.0)
    assert high > low


def test_negative_price_change_symmetric():
    pos = compute_reflexivity_score(0.03, 2.0, 0.5, 5.0)
    neg = compute_reflexivity_score(-0.03, 2.0, 0.5, 5.0)
    assert pos == pytest.approx(neg)


def test_partial_none_handled():
    score = compute_reflexivity_score(0.03, None, 0.7, None)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# apply_reflexivity_warning
# ---------------------------------------------------------------------------

def test_below_threshold_unchanged():
    fc = make_forecast()
    fc2 = apply_reflexivity_warning(fc, 0.5)
    assert fc2 is fc


def test_above_threshold_returns_new_instance():
    fc = make_forecast()
    fc2 = apply_reflexivity_warning(fc, 0.8)
    assert fc2 is not fc


def test_warning_added():
    fc = make_forecast()
    fc2 = apply_reflexivity_warning(fc, 0.8)
    assert "reflexivity_elevated" in fc2.warnings


def test_confidence_unchanged_above_threshold():
    for conf in ("high", "medium", "low"):
        fc = make_forecast(confidence=conf)
        fc2 = apply_reflexivity_warning(fc, 0.9)
        assert fc2.confidence == conf, f"Confidence modified for {conf}"


def test_probs_unchanged_above_threshold():
    fc = make_forecast()
    fc2 = apply_reflexivity_warning(fc, 0.9)
    assert fc2.next_state_probs == fc.next_state_probs


def test_exact_threshold_triggers():
    fc = make_forecast()
    fc2 = apply_reflexivity_warning(fc, 0.7)
    assert "reflexivity_elevated" in fc2.warnings


def test_custom_threshold():
    fc = make_forecast()
    fc2 = apply_reflexivity_warning(fc, 0.5, threshold=0.4)
    assert "reflexivity_elevated" in fc2.warnings


def test_existing_warnings_preserved():
    fc = make_forecast(warnings=("prior_warning",))
    fc2 = apply_reflexivity_warning(fc, 0.8)
    assert "prior_warning" in fc2.warnings
    assert "reflexivity_elevated" in fc2.warnings


def test_original_not_mutated():
    fc = make_forecast(confidence="high")
    apply_reflexivity_warning(fc, 0.9)
    assert fc.confidence == "high"
    assert "reflexivity_elevated" not in fc.warnings
