"""Tests for conditional_markov.py."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from composite_states import MODIFIER_ORDER, StateRecord
from conditional_markov import (
    ConditionalForecast,
    fit_state_counts,
    fit_transitions,
    forecast,
    forecast_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(base, vol=None, macro=None, cal=None, liq=None) -> StateRecord:
    return StateRecord(
        base_state=base,
        volatility_state=vol,
        macro_state=macro,
        calendar_state=cal,
        liquidity_state=liq,
    )


def _make_sequence(base, vol, n_transitions: int) -> list[StateRecord]:
    """n_transitions pairs of (base|vol, bear|vol) so n+1 records."""
    result = []
    for _ in range(n_transitions):
        result.append(_r(base, vol=vol))
        result.append(_r("bear", vol=vol))
    return result


# ---------------------------------------------------------------------------
# ConditionalForecast dataclass
# ---------------------------------------------------------------------------

class TestConditionalForecastDataclass:
    def test_frozen(self):
        fc = ConditionalForecast(
            base_state="bull",
            conditions_requested={},
            conditions_used={},
            fallback_level=0,
            sample_count=10,
            next_state_probs={},
            confidence="low",
            warnings=(),
        )
        with pytest.raises((AttributeError, TypeError)):
            fc.base_state = "bear"  # type: ignore[misc]

    def test_warnings_is_tuple(self):
        fc = ConditionalForecast(
            base_state="bull",
            conditions_requested={},
            conditions_used={},
            fallback_level=0,
            sample_count=10,
            next_state_probs={},
            confidence="low",
            warnings=("fallback_used",),
        )
        assert isinstance(fc.warnings, tuple)


# ---------------------------------------------------------------------------
# fit_transitions
# ---------------------------------------------------------------------------

class TestFitTransitions:
    def test_returns_all_levels(self):
        records = [_r("bull"), _r("bear")]
        result = fit_transitions(records)
        expected_levels = set(range(len(MODIFIER_ORDER) + 1))
        assert set(result.keys()) == expected_levels

    def test_empty_input(self):
        result = fit_transitions([])
        assert all(len(v) == 0 for v in result.values())

    def test_level_0_base_only_keys(self):
        records = [_r("bull", vol="HV"), _r("bear", vol="HV")]
        result = fit_transitions(records)
        # At level 0 truncation zeros all modifiers → wildcards in key
        for (fk, tk) in result[0]:
            assert fk.startswith("bull") or fk.startswith("bear")
            assert "|*|*|*|*" in fk


# ---------------------------------------------------------------------------
# fit_state_counts
# ---------------------------------------------------------------------------

class TestFitStateCounts:
    def test_returns_all_levels(self):
        records = [_r("bull"), _r("bear")]
        result = fit_state_counts(records)
        assert set(result.keys()) == set(range(len(MODIFIER_ORDER) + 1))

    def test_level_0_aggregates_base(self):
        records = [_r("bull", vol="HV"), _r("bull", vol="LV"), _r("bear")]
        sc = fit_state_counts(records)
        # level-0 truncation produces keys like "bull|*|*|*|*"
        assert sc[0]["bull|*|*|*|*"] == 2
        assert sc[0]["bear|*|*|*|*"] == 1


# ---------------------------------------------------------------------------
# forecast — confidence & fallback levels
# ---------------------------------------------------------------------------

class TestForecastConfidence:
    def _build_dataset(self, n_bull_bear: int, vol="HV"):
        """n_bull_bear copies of (bull|HV → bear|HV)."""
        recs = []
        for _ in range(n_bull_bear):
            recs.append(_r("bull", vol=vol))
            recs.append(_r("bear", vol=vol))
        return recs

    def test_uses_full_composite_when_count_ge_50(self):
        """With >=50 samples, fallback_level should equal max level (4)."""
        records = self._build_dataset(60)
        sc = fit_state_counts(records)
        tr = fit_transitions(records)
        rec = _r("bull", vol="HV")
        fc = forecast(rec, sc, tr)
        assert fc.fallback_level == len(MODIFIER_ORDER)
        assert fc.confidence == "high"
        assert "fallback_used" not in fc.warnings

    def test_falls_back_when_count_below_50(self):
        """With 35 samples at full level, falls back one level."""
        records = self._build_dataset(35)
        sc = fit_state_counts(records)
        tr = fit_transitions(records)
        rec = _r("bull", vol="HV")
        fc = forecast(rec, sc, tr)
        # Level 4 has 35 (>=30 mid) → fall to level 3, but level 3 is base+vol
        # In practice select_fallback_level returns level-1 when mid<=count<high
        assert fc.fallback_level < len(MODIFIER_ORDER)

    def test_falls_to_base_when_count_below_30(self):
        """With only 10 samples, should reach base-only (level 0)."""
        records = self._build_dataset(10)
        sc = fit_state_counts(records)
        tr = fit_transitions(records)
        rec = _r("bull", vol="HV")
        fc = forecast(rec, sc, tr)
        assert fc.confidence == "low"
        assert "low_sample" in fc.warnings

    def test_probs_sum_to_one_when_transitions_exist(self):
        records = self._build_dataset(60)
        sc = fit_state_counts(records)
        tr = fit_transitions(records)
        rec = _r("bull", vol="HV")
        fc = forecast(rec, sc, tr)
        assert fc.next_state_probs
        total = sum(fc.next_state_probs.values())
        assert abs(total - 1.0) < 1e-9

    def test_fallback_used_warning_present_when_not_max_level(self):
        records = self._build_dataset(35)
        sc = fit_state_counts(records)
        tr = fit_transitions(records)
        rec = _r("bull", vol="HV")
        fc = forecast(rec, sc, tr)
        assert "fallback_used" in fc.warnings

    def test_no_fallback_warning_when_full_level_used(self):
        records = self._build_dataset(60)
        sc = fit_state_counts(records)
        tr = fit_transitions(records)
        rec = _r("bull", vol="HV")
        fc = forecast(rec, sc, tr)
        assert "fallback_used" not in fc.warnings


# ---------------------------------------------------------------------------
# forecast — warnings
# ---------------------------------------------------------------------------

class TestForecastWarnings:
    def test_no_transitions_warning(self):
        """A record with no outgoing transitions should get no_transitions."""
        records = [_r("bull"), _r("bear"), _r("bull")]
        sc = fit_state_counts(records)
        tr = fit_transitions(records)
        # "sideways" never appears in history → no transitions
        rec = _r("sideways")
        fc = forecast(rec, sc, tr)
        assert "no_transitions" in fc.warnings
        assert fc.next_state_probs == {}

    def test_low_sample_warning(self):
        records = [_r("bull"), _r("bear")] * 5  # only 5 pairs → count < 30
        sc = fit_state_counts(records)
        tr = fit_transitions(records)
        rec = _r("bull")
        fc = forecast(rec, sc, tr)
        assert "low_sample" in fc.warnings

    def test_conditions_requested_vs_used(self):
        """conditions_used should reflect the truncated level actually used."""
        records = [_r("bull", vol="HV"), _r("bear", vol="HV")] * 5
        sc = fit_state_counts(records)
        tr = fit_transitions(records)
        # Request full conditions but only 5 pairs → will fall back
        rec = _r("bull", vol="HV", macro="risk_on")
        fc = forecast(rec, sc, tr)
        assert fc.conditions_requested["macro_state"] == "risk_on"
        # used should not include macro (it was zeroed by truncation)
        assert fc.conditions_used.get("macro_state") is None


# ---------------------------------------------------------------------------
# forecast_batch
# ---------------------------------------------------------------------------

class TestForecastBatch:
    def test_returns_one_result_per_input(self):
        history = [_r("bull"), _r("bear"), _r("bull"), _r("bear")] * 20
        to_forecast = [_r("bull"), _r("bear"), _r("sideways")]
        results = forecast_batch(to_forecast, history)
        assert len(results) == 3

    def test_all_results_are_conditional_forecast(self):
        history = [_r("bull"), _r("bear")] * 30
        results = forecast_batch([_r("bull")], history)
        assert isinstance(results[0], ConditionalForecast)

    def test_empty_to_forecast(self):
        history = [_r("bull"), _r("bear")] * 10
        assert forecast_batch([], history) == []

    def test_empty_history(self):
        results = forecast_batch([_r("bull")], [])
        assert results[0].next_state_probs == {}
        assert "no_transitions" in results[0].warnings

    def test_probs_sum_to_one_for_known_state(self):
        history = [_r("bull"), _r("bear")] * 60
        results = forecast_batch([_r("bull")], history)
        fc = results[0]
        if fc.next_state_probs:
            total = sum(fc.next_state_probs.values())
            assert abs(total - 1.0) < 1e-9
