"""Tests for market data source quality comparison."""

import pandas as pd

from src.source_quality import compare_ohlcv_sources, summarize_source_frames


def _frame(dates: list[str], closes: list[float], volumes: list[int]) -> pd.DataFrame:
    index = pd.to_datetime(dates)
    index.name = "Date"
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1 for value in closes],
            "Low": [value - 1 for value in closes],
            "Close": closes,
            "Volume": volumes,
        },
        index=index,
    )


def test_compare_ohlcv_sources_reports_overlap_and_differences():
    fmp = {
        "SPY": _frame(["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 101.0, 102.0], [10, 11, 12])
    }
    yfinance = {
        "SPY": _frame(["2024-01-03", "2024-01-04", "2024-01-05"], [101.5, 102.0, 103.0], [11, 99, 13])
    }

    result = compare_ohlcv_sources(fmp, yfinance, left_name="fmp", right_name="yfinance")
    row = result.loc["SPY"]

    assert row["fmp_rows"] == 3
    assert row["yfinance_rows"] == 3
    assert row["overlap_rows"] == 2
    assert row["fmp_only_rows"] == 1
    assert row["yfinance_only_rows"] == 1
    assert row["close_mismatch_rows"] == 1
    assert row["max_abs_close_diff"] == 0.5
    assert row["volume_mismatch_rows"] == 1


def test_summarize_source_frames_reports_date_ranges_and_missing_values():
    data = {
        "SPY": _frame(["2024-01-02", "2024-01-03"], [100.0, 101.0], [10, 11]),
        "EMPTY": pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
    }

    result = summarize_source_frames(data, provider="fmp")

    assert result.loc["SPY", "provider"] == "fmp"
    assert result.loc["SPY", "rows"] == 2
    assert result.loc["SPY", "first_date"] == "2024-01-02"
    assert result.loc["SPY", "last_date"] == "2024-01-03"
    assert result.loc["EMPTY", "rows"] == 0
