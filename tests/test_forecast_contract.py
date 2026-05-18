"""Tests for src/forecast_contract.py (Phase F.4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make src importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from composite_states import StateRecord
from conditional_markov import ConditionalForecast
from forecast_contract import (
    FORECAST_COLUMNS,
    forecast_row,
    forecasts_to_dataframe,
    write_forecast_table,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS_ORDER = (
    "symbol",
    "date",
    "base_state",
    "conditions_requested",
    "conditions_used",
    "fallback_level",
    "sample_count",
    "probability",
    "expected_value_after_cost",
    "cost_model_used",
    "confidence",
    "historical_stability",
    "invalidation_trigger",
    "warnings",
)

PLACEHOLDER_FIELDS = (
    "expected_value_after_cost",
    "cost_model_used",
    "historical_stability",
    "invalidation_trigger",
)


def make_cf(
    base_state: str = "up",
    next_state_probs: dict[str, float] | None = None,
    confidence: str = "high",
    fallback_level: int = 0,
    sample_count: int = 100,
    warnings: tuple[str, ...] = (),
) -> ConditionalForecast:
    return ConditionalForecast(
        base_state=base_state,
        conditions_requested={"volatility": "low", "macro": "bull"},
        conditions_used={"volatility": "low", "macro": "bull"},
        fallback_level=fallback_level,
        sample_count=sample_count,
        next_state_probs=next_state_probs or {"up": 0.6, "down": 0.3, "flat": 0.1},
        confidence=confidence,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# FORECAST_COLUMNS schema tests
# ---------------------------------------------------------------------------


def test_forecast_columns_has_14_entries():
    assert len(FORECAST_COLUMNS) == 14


def test_forecast_columns_order_matches_contract():
    assert FORECAST_COLUMNS == EXPECTED_COLUMNS_ORDER


# ---------------------------------------------------------------------------
# forecast_row tests
# ---------------------------------------------------------------------------


def test_forecast_row_populates_8_fields():
    cf = make_cf()
    row = forecast_row("AAPL", "2024-01-15", cf)
    assert row["symbol"] == "AAPL"
    assert row["date"] == "2024-01-15"
    assert row["base_state"] == "up"
    assert row["conditions_requested"] == {"volatility": "low", "macro": "bull"}
    assert row["conditions_used"] == {"volatility": "low", "macro": "bull"}
    assert row["fallback_level"] == 0
    assert row["sample_count"] == 100
    assert row["confidence"] == "high"


def test_forecast_row_probability_none_when_no_next_state():
    cf = make_cf()
    row = forecast_row("AAPL", "2024-01-15", cf)
    assert row["probability"] is None


def test_forecast_row_probability_set_when_next_state_given():
    cf = make_cf(next_state_probs={"up": 0.6, "down": 0.3, "flat": 0.1})
    row = forecast_row("AAPL", "2024-01-15", cf, next_state="up")
    assert row["probability"] == pytest.approx(0.6)


def test_forecast_row_probability_returns_none_for_unknown_next_state():
    cf = make_cf(next_state_probs={"up": 0.6, "down": 0.4})
    row = forecast_row("AAPL", "2024-01-15", cf, next_state="sideways")
    assert row["probability"] is None


def test_forecast_row_placeholder_fields_are_none():
    cf = make_cf()
    row = forecast_row("AAPL", "2024-01-15", cf)
    for field in PLACEHOLDER_FIELDS:
        assert row[field] is None, f"Expected {field} to be None"


def test_forecast_row_warnings_propagated():
    cf = make_cf(warnings=("fallback_used", "low_sample"))
    row = forecast_row("AAPL", "2024-01-15", cf)
    assert row["warnings"] == ("fallback_used", "low_sample")


def test_forecast_row_contains_all_14_columns():
    cf = make_cf()
    row = forecast_row("AAPL", "2024-01-15", cf, next_state="up")
    for col in FORECAST_COLUMNS:
        assert col in row, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# forecasts_to_dataframe tests
# ---------------------------------------------------------------------------


def test_forecasts_to_dataframe_preserves_column_order():
    cf = make_cf()
    # Build row in arbitrary key order by overriding with dict comprehension
    row = forecast_row("SPY", "2024-02-01", cf, next_state="up")
    # Shuffle keys
    shuffled = dict(reversed(list(row.items())))
    df = forecasts_to_dataframe([shuffled])
    assert list(df.columns) == list(FORECAST_COLUMNS)


def test_forecasts_to_dataframe_empty_list_returns_correct_columns():
    df = forecasts_to_dataframe([])
    assert list(df.columns) == list(FORECAST_COLUMNS)
    assert len(df) == 0


def test_forecasts_to_dataframe_multiple_rows():
    cf1 = make_cf(base_state="up")
    cf2 = make_cf(base_state="down", confidence="low", fallback_level=2)
    rows = [
        forecast_row("AAPL", "2024-01-01", cf1, next_state="up"),
        forecast_row("AAPL", "2024-01-02", cf2, next_state="down"),
    ]
    df = forecasts_to_dataframe(rows)
    assert len(df) == 2
    assert list(df.columns) == list(FORECAST_COLUMNS)
    assert df.iloc[0]["base_state"] == "up"
    assert df.iloc[1]["base_state"] == "down"


def test_forecasts_to_dataframe_fills_missing_columns_with_none():
    # Row dict missing some columns (simulates a partial dict from old code)
    partial = {"symbol": "X", "date": "2024-01-01", "base_state": "flat"}
    df = forecasts_to_dataframe([partial])
    assert list(df.columns) == list(FORECAST_COLUMNS)
    assert pd.isna(df.iloc[0]["confidence"]) or df.iloc[0]["confidence"] is None


def test_forecasts_to_dataframe_drops_extra_keys():
    cf = make_cf()
    row = forecast_row("AAPL", "2024-01-01", cf)
    row["extra_debug_field"] = "should_be_dropped"
    df = forecasts_to_dataframe([row])
    assert "extra_debug_field" not in df.columns
    assert list(df.columns) == list(FORECAST_COLUMNS)


# ---------------------------------------------------------------------------
# write_forecast_table tests
# ---------------------------------------------------------------------------


def test_write_forecast_table_creates_csv(tmp_path):
    cf = make_cf()
    rows = [forecast_row("MSFT", "2024-03-01", cf, next_state="up")]
    df = forecasts_to_dataframe(rows)
    out_path = tmp_path / "sub" / "forecasts.csv"
    result = write_forecast_table(df, out_path)
    assert result.exists()
    assert result.suffix == ".csv"


def test_write_forecast_table_correct_header(tmp_path):
    cf = make_cf()
    rows = [forecast_row("MSFT", "2024-03-01", cf, next_state="up")]
    df = forecasts_to_dataframe(rows)
    out_path = tmp_path / "forecasts.csv"
    write_forecast_table(df, out_path)
    loaded = pd.read_csv(out_path)
    assert list(loaded.columns) == list(FORECAST_COLUMNS)


def test_write_forecast_table_creates_parent_dirs(tmp_path):
    cf = make_cf()
    df = forecasts_to_dataframe([forecast_row("NVDA", "2024-04-01", cf)])
    deep_path = tmp_path / "a" / "b" / "c" / "output.csv"
    write_forecast_table(df, deep_path)
    assert deep_path.exists()
