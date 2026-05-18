"""Tests for dashboard_panels.py."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd

from conditional_markov import ConditionalForecast
from dashboard_panels import (
    fallback_usage_panel,
    low_confidence_panel,
    top_rare_states_panel,
    coverage_by_symbol_panel,
    warnings_summary_panel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_forecast(
    fallback_level: int = 0,
    confidence: str = "high",
    sample_count: int = 100,
    warnings: tuple[str, ...] = (),
) -> ConditionalForecast:
    return ConditionalForecast(
        base_state="UP",
        conditions_requested={"vol": "HIGH"},
        conditions_used={"vol": "HIGH"},
        fallback_level=fallback_level,
        sample_count=sample_count,
        next_state_probs={"UP": 0.6, "DOWN": 0.4},
        confidence=confidence,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# fallback_usage_panel
# ---------------------------------------------------------------------------

class TestFallbackUsagePanel:
    def test_empty_returns_correct_columns(self):
        df = fallback_usage_panel([])
        assert list(df.columns) == ["fallback_level", "count", "share"]
        assert len(df) == 0

    def test_shares_sum_to_one(self):
        forecasts = [
            make_forecast(fallback_level=0),
            make_forecast(fallback_level=1),
            make_forecast(fallback_level=1),
            make_forecast(fallback_level=2),
        ]
        df = fallback_usage_panel(forecasts)
        assert abs(df["share"].sum() - 1.0) < 1e-9

    def test_counts_correct(self):
        forecasts = [
            make_forecast(fallback_level=0),
            make_forecast(fallback_level=0),
            make_forecast(fallback_level=1),
        ]
        df = fallback_usage_panel(forecasts)
        row0 = df[df["fallback_level"] == 0].iloc[0]
        assert row0["count"] == 2
        assert abs(row0["share"] - 2 / 3) < 1e-9

    def test_single_forecast(self):
        df = fallback_usage_panel([make_forecast(fallback_level=3)])
        assert df.iloc[0]["share"] == 1.0


# ---------------------------------------------------------------------------
# low_confidence_panel
# ---------------------------------------------------------------------------

class TestLowConfidencePanel:
    def test_empty_returns_correct_columns(self):
        df = low_confidence_panel([])
        assert list(df.columns) == ["confidence", "count", "share"]
        assert len(df) == 0

    def test_aggregates_mixed_confidences(self):
        forecasts = [
            make_forecast(confidence="high"),
            make_forecast(confidence="high"),
            make_forecast(confidence="medium"),
            make_forecast(confidence="low"),
        ]
        df = low_confidence_panel(forecasts)
        assert set(df["confidence"]) == {"high", "medium", "low"}
        high_row = df[df["confidence"] == "high"].iloc[0]
        assert high_row["count"] == 2
        assert abs(high_row["share"] - 0.5) < 1e-9

    def test_shares_sum_to_one(self):
        forecasts = [make_forecast(confidence=c) for c in ["high", "medium", "low", "low"]]
        df = low_confidence_panel(forecasts)
        assert abs(df["share"].sum() - 1.0) < 1e-9

    def test_single_tier(self):
        forecasts = [make_forecast(confidence="high")] * 3
        df = low_confidence_panel(forecasts)
        assert len(df) == 1
        assert df.iloc[0]["count"] == 3


# ---------------------------------------------------------------------------
# top_rare_states_panel
# ---------------------------------------------------------------------------

class TestTopRareStatesPanel:
    def test_empty_returns_correct_columns(self):
        df = top_rare_states_panel({})
        assert list(df.columns) == ["composite_key", "sample_count"]
        assert len(df) == 0

    def test_sorted_ascending(self):
        counts = {"A": 50, "B": 5, "C": 20, "D": 1}
        df = top_rare_states_panel(counts, top_n=10)
        assert list(df["sample_count"]) == sorted(df["sample_count"])

    def test_respects_top_n(self):
        counts = {str(i): i for i in range(1, 11)}
        df = top_rare_states_panel(counts, top_n=3)
        assert len(df) == 3
        assert df.iloc[0]["sample_count"] == 1

    def test_rarest_first(self):
        counts = {"rare": 2, "common": 1000}
        df = top_rare_states_panel(counts, top_n=2)
        assert df.iloc[0]["composite_key"] == "rare"

    def test_top_n_larger_than_dict(self):
        counts = {"A": 1, "B": 2}
        df = top_rare_states_panel(counts, top_n=100)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# coverage_by_symbol_panel
# ---------------------------------------------------------------------------

class TestCoverageBySymbolPanel:
    def test_empty_returns_correct_columns(self):
        df = coverage_by_symbol_panel([], [])
        assert list(df.columns) == [
            "symbol", "forecast_count", "mean_sample_count",
            "low_confidence_count", "low_confidence_share",
        ]
        assert len(df) == 0

    def test_parallel_list_semantics(self):
        forecasts = [
            make_forecast(confidence="high", sample_count=100),
            make_forecast(confidence="low", sample_count=50),
        ]
        symbols = ["AAPL", "TSLA"]
        df = coverage_by_symbol_panel(forecasts, symbols)
        assert set(df["symbol"]) == {"AAPL", "TSLA"}
        aapl = df[df["symbol"] == "AAPL"].iloc[0]
        assert aapl["forecast_count"] == 1
        assert aapl["mean_sample_count"] == 100.0
        assert aapl["low_confidence_count"] == 0
        tsla = df[df["symbol"] == "TSLA"].iloc[0]
        assert tsla["low_confidence_count"] == 1
        assert tsla["low_confidence_share"] == 1.0

    def test_multiple_forecasts_same_symbol(self):
        forecasts = [
            make_forecast(confidence="high", sample_count=100),
            make_forecast(confidence="low", sample_count=50),
            make_forecast(confidence="low", sample_count=30),
        ]
        symbols = ["AAPL", "AAPL", "AAPL"]
        df = coverage_by_symbol_panel(forecasts, symbols)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["forecast_count"] == 3
        assert abs(row["mean_sample_count"] - (100 + 50 + 30) / 3) < 1e-9
        assert row["low_confidence_count"] == 2

    def test_low_confidence_share_zero(self):
        forecasts = [make_forecast(confidence="high")] * 4
        symbols = ["X"] * 4
        df = coverage_by_symbol_panel(forecasts, symbols)
        assert df.iloc[0]["low_confidence_share"] == 0.0


# ---------------------------------------------------------------------------
# warnings_summary_panel
# ---------------------------------------------------------------------------

class TestWarningsSummaryPanel:
    def test_empty_returns_correct_columns(self):
        df = warnings_summary_panel([])
        assert list(df.columns) == ["warning", "count"]
        assert len(df) == 0

    def test_no_warnings_returns_empty(self):
        forecasts = [make_forecast(warnings=()), make_forecast(warnings=())]
        df = warnings_summary_panel(forecasts)
        assert len(df) == 0

    def test_counts_duplicates_across_forecasts(self):
        forecasts = [
            make_forecast(warnings=("low_sample",)),
            make_forecast(warnings=("low_sample", "fallback_used")),
            make_forecast(warnings=("fallback_used",)),
        ]
        df = warnings_summary_panel(forecasts)
        low = df[df["warning"] == "low_sample"].iloc[0]
        fallback = df[df["warning"] == "fallback_used"].iloc[0]
        assert low["count"] == 2
        assert fallback["count"] == 2

    def test_sorted_descending(self):
        forecasts = [
            make_forecast(warnings=("rare_warning",)),
            make_forecast(warnings=("common_warning",)),
            make_forecast(warnings=("common_warning",)),
            make_forecast(warnings=("common_warning",)),
        ]
        df = warnings_summary_panel(forecasts)
        assert df.iloc[0]["warning"] == "common_warning"
        assert df.iloc[0]["count"] == 3

    def test_single_warning_type(self):
        forecasts = [make_forecast(warnings=("x",))] * 5
        df = warnings_summary_panel(forecasts)
        assert len(df) == 1
        assert df.iloc[0]["count"] == 5
