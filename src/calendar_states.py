"""Calendar state classification for Markov TA Lab.

Classifies dates into market-relevant calendar regimes based on
FOMC meetings, CPI releases, OPEX weeks, month-end, and holiday proximity.
"""

from __future__ import annotations

import datetime
from datetime import date
from typing import Union

import pandas as pd

# ---------------------------------------------------------------------------
# State labels
# ---------------------------------------------------------------------------

CALENDAR_STATES: tuple[str, ...] = (
    "FOMC_WINDOW",
    "CPI_WINDOW",
    "OPEX_WEEK",
    "MONTH_END",
    "HOLIDAY_LIQUIDITY",
    "NONE",
)

# ---------------------------------------------------------------------------
# Static event schedules (2024–2026)
# ---------------------------------------------------------------------------

_CPI_DATES: set[date] = {
    # 2024 CPI release dates (BLS published)
    date(2024, 1, 11), date(2024, 2, 13), date(2024, 3, 12),
    date(2024, 4, 10), date(2024, 5, 15), date(2024, 6, 12),
    date(2024, 7, 11), date(2024, 8, 14), date(2024, 9, 11),
    date(2024, 10, 10), date(2024, 11, 13), date(2024, 12, 11),
    # 2025 CPI release dates
    date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
    date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
    date(2025, 7, 15), date(2025, 8, 12), date(2025, 9, 10),
    date(2025, 10, 15), date(2025, 11, 13), date(2025, 12, 10),
    # 2026 CPI release dates
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 15), date(2026, 5, 13), date(2026, 6, 10),
    date(2026, 7, 14), date(2026, 8, 12), date(2026, 9, 9),
    date(2026, 10, 14), date(2026, 11, 12), date(2026, 12, 9),
}

_FOMC_DATES: set[date] = {
    # 2024 FOMC meeting end dates (8 per year)
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
    date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
    date(2024, 11, 7), date(2024, 12, 18),
    # 2025 FOMC meeting end dates
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025, 11, 5), date(2025, 12, 17),
    # 2026 FOMC meeting end dates
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 11, 4), date(2026, 12, 16),
}

# ---------------------------------------------------------------------------
# US market holidays
# ---------------------------------------------------------------------------

def us_market_holidays(year: int) -> set[date]:
    """Return NYSE-observed market holidays for the given year."""
    from pandas.tseries.holiday import (
        USFederalHolidayCalendar,
    )
    cal = USFederalHolidayCalendar()
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    holidays = cal.holidays(start=start, end=end)
    # Convert to date objects
    result: set[date] = {ts.date() for ts in holidays}

    # NYSE observes Good Friday (not in USFederalHolidayCalendar)
    good_friday = _calc_good_friday(year)
    result.add(good_friday)

    # Juneteenth added in 2021
    if year >= 2021:
        juneteenth = date(year, 6, 19)
        # Observe on Monday if Sunday, Friday if Saturday
        if juneteenth.weekday() == 6:
            juneteenth = date(year, 6, 20)
        elif juneteenth.weekday() == 5:
            juneteenth = date(year, 6, 18)
        result.add(juneteenth)

    return result


def _calc_good_friday(year: int) -> date:
    """Return Good Friday date for the given year using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter_sunday = date(year, month, day)
    # Good Friday is 2 days before Easter
    return easter_sunday - datetime.timedelta(days=2)


# ---------------------------------------------------------------------------
# Event schedule loader
# ---------------------------------------------------------------------------

def load_event_schedule() -> dict[str, set[date]]:
    """Return the static CPI and FOMC event schedules."""
    return {
        "CPI": set(_CPI_DATES),
        "FOMC": set(_FOMC_DATES),
    }


# ---------------------------------------------------------------------------
# OPEX helpers
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> date:
    """Return the third Friday of the given month."""
    # Find the first Friday
    first_day = date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + datetime.timedelta(days=days_until_friday)
    third_friday = first_friday + datetime.timedelta(weeks=2)
    return third_friday


def _opex_week_range(year: int, month: int) -> tuple[date, date]:
    """Return (Monday, Friday) of the OPEX week for the given month."""
    tf = _third_friday(year, month)
    monday = tf - datetime.timedelta(days=tf.weekday())  # Monday of that week
    friday = monday + datetime.timedelta(days=4)
    return monday, friday


# ---------------------------------------------------------------------------
# Month-end helpers
# ---------------------------------------------------------------------------

def _last_3_weekdays(year: int, month: int) -> set[date]:
    """Return the last 3 weekday dates of the given calendar month."""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    result: set[date] = set()
    d = date(year, month, last_day)
    while len(result) < 3:
        if d.weekday() < 5:  # Mon-Fri
            result.add(d)
        d -= datetime.timedelta(days=1)
    return result


# ---------------------------------------------------------------------------
# Single-date classifier
# ---------------------------------------------------------------------------

def _to_date(d: Union[pd.Timestamp, datetime.date]) -> date:
    """Normalize pd.Timestamp or datetime.date to datetime.date."""
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime.datetime):
        return d.date()
    return d  # type: ignore[return-value]


def classify_calendar(d: Union[pd.Timestamp, datetime.date]) -> str:
    """Return the highest-priority calendar state for the given date."""
    dt = _to_date(d)
    one_day = datetime.timedelta(days=1)

    # --- FOMC_WINDOW: FOMC day or day before/after ---
    for fomc_day in _FOMC_DATES:
        if abs((dt - fomc_day).days) <= 1:
            return "FOMC_WINDOW"

    # --- CPI_WINDOW: CPI release day or day before ---
    for cpi_day in _CPI_DATES:
        delta = (dt - cpi_day).days
        if delta == 0 or delta == -1:  # day-of or day before
            return "CPI_WINDOW"

    # --- OPEX_WEEK: Mon-Fri of week containing third Friday ---
    opex_mon, opex_fri = _opex_week_range(dt.year, dt.month)
    if opex_mon <= dt <= opex_fri:
        return "OPEX_WEEK"

    # --- MONTH_END: last 3 weekdays of month ---
    if dt in _last_3_weekdays(dt.year, dt.month):
        return "MONTH_END"

    # --- HOLIDAY_LIQUIDITY: day adjacent to a US market holiday ---
    try:
        holidays = us_market_holidays(dt.year)
        # Also check adjacent years for New Year's Eve / Jan 2 proximity
        if dt.month == 12:
            holidays |= us_market_holidays(dt.year + 1)
        elif dt.month == 1:
            holidays |= us_market_holidays(dt.year - 1)

        for h in holidays:
            if abs((dt - h).days) == 1 and dt.weekday() < 5:
                return "HOLIDAY_LIQUIDITY"
    except Exception:
        pass

    return "NONE"


# ---------------------------------------------------------------------------
# Vectorized classifier
# ---------------------------------------------------------------------------

def classify_calendar_series(
    dates: Union[pd.DatetimeIndex, pd.Series],
) -> pd.Series:
    """Return a Series of calendar state strings for each date in `dates`."""
    if isinstance(dates, pd.DatetimeIndex):
        index = dates
        timestamps = dates
    else:
        index = dates.index
        timestamps = dates

    states = [classify_calendar(ts) for ts in timestamps]
    return pd.Series(states, index=index, dtype="object")
