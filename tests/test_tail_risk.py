"""Tests for src/tail_risk.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tail_risk import compute_tail_risk_score, apply_tail_risk_override
from conditional_markov import ConditionalForecast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# compute_tail_risk_score
# ---------------------------------------------------------------------------

def test_all_zero_inputs():
    score = compute_tail_risk_score(0.0, 0.0, 0.0, 0.0)
    assert score == 0.0


def test_all_none_inputs():
    score = compute_tail_risk_score(None, None, None, None)
    assert score == 0.0


def test_all_nan_inputs():
    score = compute_tail_risk_score(float("nan"), float("nan"), float("nan"), float("nan"))
    assert score == 0.0


def test_score_in_unit_interval():
    # Extreme values
    score = compute_tail_risk_score(1.0, 10.0, 0.5, 0.5)
    assert 0.0 <= score <= 1.0


def test_maximum_score_clipped_to_one():
    score = compute_tail_risk_score(100.0, 100.0, 100.0, 100.0)
    assert score == pytest.approx(1.0)


def test_score_increases_with_volatility():
    low = compute_tail_risk_score(0.1, 0.0, 0.0, 0.0)
    high = compute_tail_risk_score(0.5, 0.0, 0.0, 0.0)
    assert high > low


def test_score_increases_with_atr_zscore():
    low = compute_tail_risk_score(0.0, 1.0, 0.0, 0.0)
    high = compute_tail_risk_score(0.0, 3.0, 0.0, 0.0)
    assert high > low


def test_score_symmetric_gap():
    pos = compute_tail_risk_score(0.0, 0.0, 0.03, 0.0)
    neg = compute_tail_risk_score(0.0, 0.0, -0.03, 0.0)
    assert pos == pytest.approx(neg)


def test_partial_none_handled():
    score = compute_tail_risk_score(0.2, None, 0.01, None)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# apply_tail_risk_override
# ---------------------------------------------------------------------------

def test_below_threshold_unchanged():
    fc = make_forecast(confidence="high")
    fc2 = apply_tail_risk_override(fc, 0.5)
    assert fc2 is fc


def test_elevated_returns_new_instance():
    fc = make_forecast(confidence="high")
    fc2 = apply_tail_risk_override(fc, 0.80)
    assert fc2 is not fc


def test_elevated_downgrades_high_to_medium():
    fc = make_forecast(confidence="high")
    fc2 = apply_tail_risk_override(fc, 0.80)
    assert fc2.confidence == "medium"


def test_elevated_downgrades_medium_to_low():
    fc = make_forecast(confidence="medium")
    fc2 = apply_tail_risk_override(fc, 0.80)
    assert fc2.confidence == "low"


def test_elevated_does_not_drop_below_low():
    fc = make_forecast(confidence="low")
    fc2 = apply_tail_risk_override(fc, 0.80)
    assert fc2.confidence == "low"


def test_elevated_adds_warning():
    fc = make_forecast()
    fc2 = apply_tail_risk_override(fc, 0.80)
    assert "tail_risk_elevated" in fc2.warnings


def test_elevated_does_not_zero_probs():
    fc = make_forecast()
    fc2 = apply_tail_risk_override(fc, 0.80)
    assert sum(fc2.next_state_probs.values()) > 0


def test_extreme_zeros_probs():
    fc = make_forecast()
    fc2 = apply_tail_risk_override(fc, 0.95)
    assert all(v == 0.0 for v in fc2.next_state_probs.values())


def test_extreme_forces_low_confidence():
    fc = make_forecast(confidence="high")
    fc2 = apply_tail_risk_override(fc, 0.95)
    assert fc2.confidence == "low"


def test_extreme_adds_warning():
    fc = make_forecast()
    fc2 = apply_tail_risk_override(fc, 0.95)
    assert "tail_risk_extreme" in fc2.warnings


def test_extreme_returns_new_instance():
    fc = make_forecast()
    fc2 = apply_tail_risk_override(fc, 0.95)
    assert fc2 is not fc


def test_overlay_never_raises_confidence():
    for score in [0.0, 0.5, 0.80, 0.95]:
        fc = make_forecast(confidence="low")
        fc2 = apply_tail_risk_override(fc, score)
        assert fc2.confidence == "low", f"Confidence raised at score={score}"


def test_exact_threshold_high_triggers():
    fc = make_forecast(confidence="high")
    fc2 = apply_tail_risk_override(fc, 0.75)
    assert fc2.confidence == "medium"


def test_exact_threshold_extreme_triggers():
    fc = make_forecast()
    fc2 = apply_tail_risk_override(fc, 0.90)
    assert all(v == 0.0 for v in fc2.next_state_probs.values())


def test_custom_thresholds():
    fc = make_forecast(confidence="high")
    fc2 = apply_tail_risk_override(fc, 0.60, threshold_high=0.55, threshold_extreme=0.95)
    assert fc2.confidence == "medium"


def test_original_not_mutated():
    fc = make_forecast(confidence="high")
    apply_tail_risk_override(fc, 0.95)
    assert fc.confidence == "high"
    assert sum(fc.next_state_probs.values()) > 0
