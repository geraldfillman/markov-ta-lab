"""Tests for src/cross_asset_stress.py"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cross_asset_stress import STRESS_LEVELS, classify_cross_asset_stress, apply_stress_overlay
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
# STRESS_LEVELS
# ---------------------------------------------------------------------------

def test_stress_levels_tuple():
    assert isinstance(STRESS_LEVELS, tuple)
    assert STRESS_LEVELS == ("CALM", "ELEVATED", "STRESS", "PANIC")


# ---------------------------------------------------------------------------
# classify_cross_asset_stress
# ---------------------------------------------------------------------------

def test_calm_all_none():
    assert classify_cross_asset_stress(None, None, None) == "CALM"


def test_calm_all_nan():
    assert classify_cross_asset_stress(float("nan"), float("nan"), float("nan")) == "CALM"


def test_calm_below_thresholds():
    assert classify_cross_asset_stress(15.0, 3.0, 1.0) == "CALM"


def test_elevated_by_vix():
    assert classify_cross_asset_stress(22.0, None, None) == "ELEVATED"


def test_elevated_by_credit_spread():
    assert classify_cross_asset_stress(None, 4.5, None) == "ELEVATED"


def test_elevated_by_dollar_zscore_negative():
    assert classify_cross_asset_stress(None, None, -1.8) == "ELEVATED"


def test_stress_by_vix():
    assert classify_cross_asset_stress(30.0, None, None) == "STRESS"


def test_stress_by_credit_spread():
    assert classify_cross_asset_stress(None, 6.0, None) == "STRESS"


def test_stress_by_dollar_zscore():
    assert classify_cross_asset_stress(None, None, 2.5) == "STRESS"


def test_panic_by_vix():
    assert classify_cross_asset_stress(41.0, None, None) == "PANIC"


def test_panic_by_credit_spread():
    assert classify_cross_asset_stress(None, 9.0, None) == "PANIC"


def test_panic_by_dollar_zscore():
    assert classify_cross_asset_stress(None, None, -3.5) == "PANIC"


def test_panic_wins_over_stress():
    # VIX=41 (PANIC) and cs=6 (STRESS): PANIC wins
    assert classify_cross_asset_stress(41.0, 6.0, None) == "PANIC"


def test_stress_wins_over_elevated():
    assert classify_cross_asset_stress(25.0, 6.0, None) == "STRESS"


def test_nan_vix_only_other_calm():
    # NaN vix, low credit spread, no dollar z -> CALM
    assert classify_cross_asset_stress(float("nan"), 1.0, 0.5) == "CALM"


def test_custom_thresholds():
    result = classify_cross_asset_stress(25.0, None, None, thresholds={"panic_vix": 20.0})
    assert result == "PANIC"


# ---------------------------------------------------------------------------
# apply_stress_overlay
# ---------------------------------------------------------------------------

def test_overlay_returns_new_instance():
    fc = make_forecast()
    fc2 = apply_stress_overlay(fc, "CALM")
    assert fc2 is not fc


def test_calm_does_not_change_confidence():
    fc = make_forecast(confidence="high")
    fc2 = apply_stress_overlay(fc, "CALM")
    assert fc2.confidence == "high"


def test_calm_adds_warning():
    fc = make_forecast()
    fc2 = apply_stress_overlay(fc, "CALM")
    assert "cross_asset_calm" in fc2.warnings


def test_elevated_does_not_change_confidence():
    fc = make_forecast(confidence="high")
    fc2 = apply_stress_overlay(fc, "ELEVATED")
    assert fc2.confidence == "high"


def test_elevated_adds_warning():
    fc = make_forecast()
    fc2 = apply_stress_overlay(fc, "ELEVATED")
    assert "cross_asset_elevated" in fc2.warnings


def test_stress_downgrades_high_to_medium():
    fc = make_forecast(confidence="high")
    fc2 = apply_stress_overlay(fc, "STRESS")
    assert fc2.confidence == "medium"


def test_stress_does_not_upgrade_medium():
    fc = make_forecast(confidence="medium")
    fc2 = apply_stress_overlay(fc, "STRESS")
    assert fc2.confidence == "medium"


def test_stress_does_not_upgrade_low():
    fc = make_forecast(confidence="low")
    fc2 = apply_stress_overlay(fc, "STRESS")
    assert fc2.confidence == "low"


def test_panic_downgrades_high_to_low():
    fc = make_forecast(confidence="high")
    fc2 = apply_stress_overlay(fc, "PANIC")
    assert fc2.confidence == "low"


def test_panic_downgrades_medium_to_low():
    fc = make_forecast(confidence="medium")
    fc2 = apply_stress_overlay(fc, "PANIC")
    assert fc2.confidence == "low"


def test_panic_keeps_low():
    fc = make_forecast(confidence="low")
    fc2 = apply_stress_overlay(fc, "PANIC")
    assert fc2.confidence == "low"


def test_overlay_never_raises_confidence():
    for stress in STRESS_LEVELS:
        fc = make_forecast(confidence="low")
        fc2 = apply_stress_overlay(fc, stress)
        assert fc2.confidence == "low", f"Confidence raised under {stress}"


def test_warnings_appended_not_replaced():
    fc = make_forecast(warnings=("existing_warning",))
    fc2 = apply_stress_overlay(fc, "PANIC")
    assert "existing_warning" in fc2.warnings
    assert "cross_asset_panic" in fc2.warnings


def test_original_forecast_unchanged():
    fc = make_forecast(confidence="high")
    apply_stress_overlay(fc, "PANIC")
    assert fc.confidence == "high"  # original not mutated
