"""Volatility Agent – volatility forecasts and filters.

Responsibilities (from playbook §3.11):
- ATR percentile states.
- Realised volatility states.
- GARCH-style volatility models (optional).
- Volatility-adjusted stop width and position size.

Volatility states:
    LOW_VOL, NORMAL_VOL, HIGH_VOL, VOL_COMPRESSION, VOL_EXPANSION
"""

import pandas as pd
import numpy as np


def classify_vol_state(
    df: pd.DataFrame,
    lookback: int = 63,
    delta_window: int = 21,
    compression_threshold: float = -0.25,
    expansion_threshold: float = 0.25,
) -> pd.Series:
    """Classify each bar into a volatility state.

    Level states (0-2) are assigned from the rolling-percentile rank of the
    selected volatility series. Dynamic states (3=compression, 4=expansion)
    override the level state when the recent change in volatility percentile
    is large in magnitude, capturing regime transitions independent of the
    absolute level.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ATR or realised volatility columns.
    lookback : int
        Rolling window for percentile calculation.
    delta_window : int
        Window over which to measure change in percentile rank.
    compression_threshold : float
        Percentile-rank delta below this triggers VOL_COMPRESSION (state 3).
    expansion_threshold : float
        Percentile-rank delta above this triggers VOL_EXPANSION (state 4).

    Returns
    -------
    pd.Series
        Integer volatility state label in {0,1,2,3,4}.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if delta_window < 1:
        raise ValueError("delta_window must be at least 1")

    vol = _select_volatility_series(df)
    rolling_rank = vol.rolling(lookback, min_periods=lookback).rank(pct=True)
    rank_delta = rolling_rank - rolling_rank.shift(delta_window)

    states = pd.Series(1, index=df.index, name="vol_state", dtype="int64")
    states.loc[rolling_rank <= (1.0 / 3.0)] = 0
    states.loc[rolling_rank >= (2.0 / 3.0)] = 2
    states.loc[rank_delta <= compression_threshold] = 3
    states.loc[rank_delta >= expansion_threshold] = 4
    return states


def fit_garch(returns: pd.Series) -> dict:
    """Fit a GARCH(1,1) model and return forecasted volatility."""
    raise NotImplementedError


def vol_adjusted_position_size(
    portfolio_value: float,
    risk_per_trade: float,
    atr: float,
    atr_multiple: float = 2.0,
) -> float:
    """Compute position size inversely proportional to volatility.

    size = (portfolio_value * risk_per_trade) / (atr * atr_multiple)
    """
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    if risk_per_trade <= 0:
        raise ValueError("risk_per_trade must be positive")
    if atr <= 0:
        raise ValueError("atr must be positive")
    if atr_multiple <= 0:
        raise ValueError("atr_multiple must be positive")

    return float((portfolio_value * risk_per_trade) / (atr * atr_multiple))


def _select_volatility_series(df: pd.DataFrame) -> pd.Series:
    for column in ("realized_vol_20", "atr_14", "bb_width_20"):
        if column in df.columns:
            return df[column].astype(float)
    raise ValueError("df must include realized_vol_20, atr_14, or bb_width_20")
