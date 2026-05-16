"""Indicator Agent – technical features for state labeling.

Responsibilities (from playbook §3.3):
- Moving averages (SMA/EMA).
- ATR (Average True Range).
- RSI.
- ADX (optional).
- Bollinger Band width.
- Rolling returns.
- Realized volatility.
- Volume z-score.
- Distance-to-moving-average features.

Rules:
- No future data is used.
- Every feature is aligned to the correct timestamp.
- NaN warm-up periods are documented.
"""

import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV dataframe enriched with technical features.

    Added columns:
    - sma_20, sma_50, sma_200, ema_20
    - atr_14
    - rsi_14
    - bb_width_20 (Bollinger Band width)
    - return_1d, return_5d, return_10d
    - realized_vol_20 (annualized)
    - volume_zscore_20
    - dist_to_sma_20, dist_to_sma_50, dist_to_sma_200 (ATR-normalised)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Open, High, Low, Close, Volume columns.

    Returns
    -------
    pd.DataFrame
        Original columns plus computed indicators.
    """
    raise NotImplementedError("Implement in Indicator Agent phase")
