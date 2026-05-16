"""Support/Resistance Agent – ATR-normalised level detection.

Responsibilities (from playbook §3.4):
- Detect swing highs/lows.
- Build ATR-normalized zones (not single price lines).
- Track prior weekly/monthly highs and lows.
- Ensure all levels are lagged and historical (no lookahead).

Zone width formula:
    zone = level ± (atr_mult × ATR)
"""

import pandas as pd


def detect_levels(
    df: pd.DataFrame,
    lookback: int = 126,
    atr_mult: float = 0.5,
) -> pd.DataFrame:
    """Add nearest support/resistance zones using only past data.

    Added columns:
    - nearest_support, nearest_resistance
    - support_zone_low, support_zone_high
    - resistance_zone_low, resistance_zone_high
    - dist_to_support_atr, dist_to_resistance_atr

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with at least ATR already computed.
    lookback : int
        Number of bars to search for swing pivots.
    atr_mult : float
        Half-width of the zone as a multiple of ATR.

    Returns
    -------
    pd.DataFrame
        Original columns plus level/zone features.
    """
    raise NotImplementedError("Implement in S/R Agent phase")


def plot_levels(df: pd.DataFrame, symbol: str = "") -> None:
    """Visualize price with support/resistance zones overlaid."""
    raise NotImplementedError
