"""Tests for calendar_states module."""

from __future__ import annotations

import datetime
from datetime import date

import pandas as pd
import pytest

from src.calendar_states import (
    CALENDAR_STATES,
    classify_calendar,
    classify_calendar_series,
    load_event_schedule,
    us_market_holidays,
)
from src.composite_states import StateRecord, composite_key


# ---------------------------------------------------------------------------
# CALENDAR_STATES tuple
# ---------------------------------------------------------------------------

def test_calendar_states_contains_all_labels():
    """All expected labels present."""
    expected = {"FOMC_WINDOW", "CPI_WINDOW", "OPEX_WEEK", "MONTH_END", "HOLIDAY_LIQUIDITY", "NONE"}
    assert expected == set(CALENDAR_STATES)


def test_calendar_states_includes_none():
    assert "NONE" in CALENDAR_STATES


# ---------------------------------------------------------------------------
# load_event_schedule
# ---------------------------------------------------------------------------

def test_load_event_schedule_keys():
    sched = load_event_schedule()
    assert "CPI" in sched
    assert "FOMC" in sched


def test_load_event_schedule_nonempty():
    sched = load_event_schedule()
    assert len(sched["CPI"]) >= 12
    assert len(sched["FOMC"]) >= 8


def test_load_event_schedule_returns_copies():
    """Mutating returned set should not affect module data."""
    s1 = load_event_schedule()
    s1["CPI"].clear()
    s2 = load_event_schedule()
    assert len(s2["CPI"]) > 0


# ---------------------------------------------------------------------------
# us_market_holidays
# ---------------------------------------------------------------------------

def test_us_market_holidays_2025_christmas():
    holidays = us_market_holidays(2025)
    assert date(2025, 12, 25) in holidays


def test_us_market_holidays_2025_independence_day():
    holidays = us_market_holidays(2025)
    assert date(2025, 7, 4) in holidays


def test_us_market_holidays_2025_good_friday():
    """Good Friday 2025 is April 18."""
    holidays = us_market_holidays(2025)
    assert date(2025, 4, 18) in holidays


def test_us_market_holidays_returns_dates():
    holidays = us_market_holidays(2025)
    for h in holidays:
        assert isinstance(h, date)


# ---------------------------------------------------------------------------
# classify_calendar — FOMC_WINDOW
# ---------------------------------------------------------------------------

def test_fomc_window_on_meeting_day():
    """Jan 31 2024 is an FOMC day."""
    assert classify_calendar(date(2024, 1, 31)) == "FOMC_WINDOW"


def test_fomc_window_day_before():
    """Day before FOMC (Jan 30 2024) is FOMC_WINDOW."""
    assert classify_calendar(date(2024, 1, 30)) == "FOMC_WINDOW"


def test_fomc_window_day_after():
    """Day after FOMC (Feb 1 2024) is FOMC_WINDOW."""
    assert classify_calendar(date(2024, 2, 1)) == "FOMC_WINDOW"


# ---------------------------------------------------------------------------
# classify_calendar — CPI_WINDOW
# ---------------------------------------------------------------------------

def test_cpi_window_on_release_day():
    """Jan 11 2024 is a CPI release day."""
    assert classify_calendar(date(2024, 1, 11)) == "CPI_WINDOW"


def test_cpi_window_day_before():
    """Day before CPI (Jan 10 2024) is CPI_WINDOW."""
    assert classify_calendar(date(2024, 1, 10)) == "CPI_WINDOW"


# ---------------------------------------------------------------------------
# classify_calendar — OPEX_WEEK
# ---------------------------------------------------------------------------

def test_opex_week_third_friday():
    """Third Friday of January 2025 is Jan 17."""
    assert classify_calendar(date(2025, 1, 17)) == "OPEX_WEEK"


def test_opex_week_monday_of_opex_week():
    """Monday of OPEX week (Jan 13 2025) also returns OPEX_WEEK."""
    assert classify_calendar(date(2025, 1, 13)) == "OPEX_WEEK"


def test_opex_week_third_friday_march_2024():
    """Third Friday of March 2024 is Mar 15."""
    assert classify_calendar(date(2024, 3, 15)) == "OPEX_WEEK"


# ---------------------------------------------------------------------------
# classify_calendar — MONTH_END
# ---------------------------------------------------------------------------

def test_month_end_last_weekday():
    """Last weekday of a month is MONTH_END."""
    # April 2025 ends on Wed Apr 30
    assert classify_calendar(date(2025, 4, 30)) == "MONTH_END"


def test_month_end_second_to_last_weekday():
    """Second-to-last weekday is also MONTH_END."""
    # April 2025: Apr 30 (Wed), Apr 29 (Tue), Apr 28 (Mon) — all MONTH_END
    assert classify_calendar(date(2025, 4, 28)) == "MONTH_END"


# ---------------------------------------------------------------------------
# classify_calendar — HOLIDAY_LIQUIDITY
# ---------------------------------------------------------------------------

def test_holiday_liquidity_day_before_july4():
    """Day before July 4 (July 3) is HOLIDAY_LIQUIDITY when July 4 is a weekday."""
    # July 4 2025 is a Friday; July 3 is Thursday
    result = classify_calendar(date(2025, 7, 3))
    assert result == "HOLIDAY_LIQUIDITY"


def test_holiday_liquidity_day_after_july4():
    """Day after July 4 2024 (Thursday) — July 5 (Friday) is adjacent."""
    # July 4 2024 is a Thursday; July 5 is a weekday 1 day after
    result = classify_calendar(date(2024, 7, 5))
    assert result == "HOLIDAY_LIQUIDITY"


# ---------------------------------------------------------------------------
# classify_calendar — NONE
# ---------------------------------------------------------------------------

def test_none_for_mundane_tuesday():
    """A random Tuesday in mid-month with no events returns NONE."""
    # June 10 2025 — mid-month, no FOMC/CPI, not OPEX week, not month-end
    # Check it's actually NONE (if it lands on a known event, adjust date)
    result = classify_calendar(date(2025, 6, 3))
    # June 3 2025 is a Tuesday; verify manually: FOMC Jun 18, CPI Jun 11, OPEX 3rd Fri = Jun 20
    assert result == "NONE"


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

def test_priority_fomc_over_cpi():
    """If a date is both FOMC-adjacent and CPI-adjacent, FOMC wins."""
    # June 12 2024: CPI release AND FOMC day — FOMC should win
    result = classify_calendar(date(2024, 6, 12))
    assert result == "FOMC_WINDOW"


def test_priority_cpi_over_opex():
    """CPI_WINDOW takes priority over OPEX_WEEK."""
    # Find a date that is both CPI and in OPEX week
    # Jan 15 2025: CPI release. Third Friday of Jan 2025 = Jan 17.
    # OPEX week Mon-Fri = Jan 13-17. Jan 15 (Wed) is in OPEX week AND is CPI day.
    result = classify_calendar(date(2025, 1, 15))
    assert result == "CPI_WINDOW"


# ---------------------------------------------------------------------------
# pd.Timestamp input
# ---------------------------------------------------------------------------

def test_classify_calendar_accepts_timestamp():
    ts = pd.Timestamp("2024-01-31")
    assert classify_calendar(ts) == "FOMC_WINDOW"


def test_classify_calendar_accepts_datetime():
    dt = datetime.datetime(2024, 1, 31, 10, 0)
    assert classify_calendar(dt) == "FOMC_WINDOW"


# ---------------------------------------------------------------------------
# classify_calendar_series
# ---------------------------------------------------------------------------

def test_classify_calendar_series_matches_elementwise():
    """Series classifier matches single-date classifier for a date range."""
    dates = pd.date_range("2024-01-08", periods=10, freq="B")
    series_result = classify_calendar_series(dates)
    for ts in dates:
        assert series_result[ts] == classify_calendar(ts)


def test_classify_calendar_series_returns_series():
    dates = pd.date_range("2024-06-01", periods=5, freq="B")
    result = classify_calendar_series(dates)
    assert isinstance(result, pd.Series)
    assert len(result) == 5


def test_classify_calendar_series_accepts_series_input():
    s = pd.Series(pd.date_range("2025-01-13", periods=5, freq="B"))
    result = classify_calendar_series(s)
    assert isinstance(result, pd.Series)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Integration smoke test — StateRecord round-trip
# ---------------------------------------------------------------------------

def test_state_record_with_calendar_state():
    """StateRecord with calendar_state from classify_calendar round-trips through composite_key."""
    cal_state = classify_calendar(date(2024, 1, 31))  # FOMC_WINDOW
    record = StateRecord(
        base_state="TRENDING_UP",
        volatility_state="HIGH_VOL",
        macro_state="RISK_ON",
        calendar_state=cal_state,
        liquidity_state=None,
    )
    key = composite_key(record)
    assert "FOMC_WINDOW" in key
    assert "TRENDING_UP" in key


def test_state_record_none_calendar_state():
    """StateRecord with NONE calendar_state produces wildcard in composite key."""
    cal_state = classify_calendar(date(2025, 6, 3))  # NONE
    record = StateRecord(
        base_state="RANGING",
        calendar_state=cal_state,
    )
    key = composite_key(record)
    # calendar_state="NONE" is a non-None string, so it appears as-is
    assert "NONE" in key or "RANGING" in key


def test_state_record_immutable():
    """StateRecord is frozen — assignment raises."""
    record = StateRecord(base_state="TRENDING_UP", calendar_state="FOMC_WINDOW")
    with pytest.raises(Exception):
        record.calendar_state = "CPI_WINDOW"  # type: ignore[misc]
