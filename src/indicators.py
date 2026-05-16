"""Indicator Agent - technical features for state labeling."""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return an OHLCV dataframe enriched with timestamp-aligned indicators."""
    _validate_ohlcv(df)

    result = df.copy()
    close = result["Close"]
    high = result["High"]
    low = result["Low"]
    volume = result["Volume"]

    result["sma_20"] = close.rolling(20, min_periods=20).mean()
    result["sma_50"] = close.rolling(50, min_periods=50).mean()
    result["sma_200"] = close.rolling(200, min_periods=200).mean()
    result["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()

    result["atr_14"] = average_true_range(high, low, close, window=14)
    result["rsi_14"] = relative_strength_index(close, window=14)
    result["bb_width_20"] = bollinger_band_width(close, window=20, num_std=2.0)

    result["return_1d"] = close.pct_change(1)
    result["return_5d"] = close.pct_change(5)
    result["return_10d"] = close.pct_change(10)
    result["realized_vol_20"] = result["return_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
    result["volume_zscore_20"] = zscore(volume, window=20)

    for window in (20, 50, 200):
        sma = result[f"sma_{window}"]
        result[f"dist_to_sma_{window}"] = (close - sma) / result["atr_14"]

    return result


def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Compute rolling Average True Range using current and prior bars only."""
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(window, min_periods=window).mean()


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI with simple rolling average gains and losses."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.rolling(window, min_periods=window).mean()
    average_loss = loss.rolling(window, min_periods=window).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(average_loss != 0, 100.0)
    rsi = rsi.where(average_gain != 0, 0.0)

    return rsi


def bollinger_band_width(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Compute Bollinger Band width as a fraction of the moving average."""
    middle = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std

    return (upper - lower) / middle


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score using the rolling sample mean and std."""
    rolling_mean = series.rolling(window, min_periods=window).mean()
    rolling_std = series.rolling(window, min_periods=window).std()

    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def _validate_ohlcv(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
