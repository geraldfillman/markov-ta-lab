"""Tests for technical indicator features."""

import numpy as np
import pandas as pd

from src.indicators import add_indicators


def _sample_ohlcv(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2023-01-02", periods=rows, freq="B", name="Date")
    base = pd.Series(np.linspace(100.0, 130.0, rows), index=index)
    wave = pd.Series(np.sin(np.arange(rows) / 7.0), index=index)
    close = base + wave
    open_ = close.shift(1).fillna(close.iloc[0] - 0.25)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([open_, close], axis=1).min(axis=1) - 1.0
    volume = pd.Series(1_000_000 + np.arange(rows) * 1000, index=index)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def test_add_indicators_preserves_ohlcv_and_adds_expected_columns():
    result = add_indicators(_sample_ohlcv())

    expected_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_20",
        "atr_14",
        "rsi_14",
        "bb_width_20",
        "return_1d",
        "return_5d",
        "return_10d",
        "realized_vol_20",
        "volume_zscore_20",
        "dist_to_sma_20",
        "dist_to_sma_50",
        "dist_to_sma_200",
    }

    assert expected_columns.issubset(result.columns)
    assert result.index.equals(_sample_ohlcv().index)


def test_indicators_have_expected_warmup_and_value_ranges():
    result = add_indicators(_sample_ohlcv())

    assert pd.isna(result["sma_20"].iloc[18])
    assert not pd.isna(result["sma_20"].iloc[19])
    assert pd.isna(result["atr_14"].iloc[12])
    assert not pd.isna(result["atr_14"].iloc[13])
    assert result["rsi_14"].dropna().between(0, 100).all()
    assert (result["bb_width_20"].dropna() >= 0).all()


def test_indicators_do_not_use_future_data():
    original = _sample_ohlcv()
    changed_future = original.copy()
    changed_future.loc[changed_future.index[-1], "Close"] = 10_000.0

    original_result = add_indicators(original)
    changed_result = add_indicators(changed_future)
    compare_until = original.index[-2]

    pd.testing.assert_series_equal(
        original_result.loc[:compare_until, "sma_20"],
        changed_result.loc[:compare_until, "sma_20"],
    )
    pd.testing.assert_series_equal(
        original_result.loc[:compare_until, "atr_14"],
        changed_result.loc[:compare_until, "atr_14"],
    )
    pd.testing.assert_series_equal(
        original_result.loc[:compare_until, "rsi_14"],
        changed_result.loc[:compare_until, "rsi_14"],
    )


def test_distance_to_sma_is_atr_normalized():
    result = add_indicators(_sample_ohlcv())
    last = result.dropna(subset=["dist_to_sma_20"]).iloc[-1]
    expected = (last["Close"] - last["sma_20"]) / last["atr_14"]

    assert np.isclose(last["dist_to_sma_20"], expected)
