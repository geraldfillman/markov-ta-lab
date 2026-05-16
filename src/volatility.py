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
) -> pd.Series:
    """Classify each bar into a volatility state.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ATR or realised volatility columns.
    lookback : int
        Rolling window for percentile calculation.

    Returns
    -------
    pd.Series
        Integer volatility state label.
    """
    raise NotImplementedError("Implement in Volatility Agent phase")


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
    raise NotImplementedError
