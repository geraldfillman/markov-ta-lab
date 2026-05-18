"""Tests for the Data Agent pipeline."""

from uuid import uuid4

import pandas as pd
import pytest

from src.data import (
    download_ohlcv,
    load_processed,
    missing_data_report,
    save_processed,
    save_raw,
)


TEST_OUTPUT_ROOT = ".test_output"


def _sample_vendor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Adj Close": [100.4, 101.4, 102.4],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"]),
    )


def test_download_ohlcv_returns_clean_sorted_required_columns():
    def downloader(symbol: str, start: str, end: str) -> pd.DataFrame:
        assert symbol == "SPY"
        assert start == "2024-01-01"
        assert end == "2024-01-05"
        return _sample_vendor_frame()

    result = download_ohlcv("SPY", "2024-01-01", "2024-01-05", downloader=downloader)

    assert list(result) == ["SPY"]
    frame = result["SPY"]
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert frame.index.name == "Date"
    assert frame.index.is_monotonic_increasing
    assert frame.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))


def test_download_ohlcv_drops_rows_missing_required_ohlcv_values():
    def downloader(symbol: str, start: str, end: str) -> pd.DataFrame:
        frame = _sample_vendor_frame()
        frame.loc[pd.Timestamp("2024-01-02"), "Close"] = None
        return frame

    result = download_ohlcv(["SPY"], "2024-01-01", "2024-01-05", downloader=downloader)

    assert len(result["SPY"]) == 2
    assert result["SPY"]["Close"].isna().sum() == 0


def test_download_ohlcv_accepts_fmp_provider(monkeypatch):
    def fmp_downloader(symbol: str, start: str, end: str) -> pd.DataFrame:
        assert symbol == "SPY"
        assert start == "2024-01-01"
        assert end == "2024-01-05"
        return _sample_vendor_frame()

    monkeypatch.setattr("src.data._download_with_fmp", fmp_downloader)

    result = download_ohlcv("SPY", "2024-01-01", "2024-01-05", provider="fmp")

    assert list(result["SPY"].columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_download_ohlcv_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported OHLCV provider"):
        download_ohlcv("SPY", "2024-01-01", "2024-01-05", provider="unknown")


def test_missing_data_report_counts_missing_values_by_symbol():
    data = {
        "SPY": pd.DataFrame({"Open": [1.0, None], "Close": [1.5, 1.6]}),
        "QQQ": pd.DataFrame({"Open": [2.0, 2.1], "Close": [None, None]}),
    }

    report = missing_data_report(data)

    assert report.loc["SPY", "rows"] == 2
    assert report.loc["SPY", "total_missing"] == 1
    assert report.loc["QQQ", "missing_Close"] == 2


def test_save_and_load_raw_and_processed_files():
    data = {"SPY": download_ohlcv("SPY", "2024-01-01", "2024-01-05", downloader=lambda *_: _sample_vendor_frame())["SPY"]}
    run_dir = f"{TEST_OUTPUT_ROOT}/{uuid4().hex}"
    raw_dir = f"{run_dir}/raw"
    processed_dir = f"{run_dir}/processed"

    save_raw(data, raw_dir)
    save_processed(data, processed_dir)
    loaded = load_processed("SPY", processed_dir)

    assert pd.io.common.file_exists(f"{raw_dir}/SPY.csv")
    assert pd.io.common.file_exists(f"{processed_dir}/SPY.parquet")
    pd.testing.assert_frame_equal(loaded, data["SPY"])
