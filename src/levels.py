"""Support/Resistance Agent - ATR-normalized level detection."""

import pandas as pd


REQUIRED_COLUMNS = ["High", "Low", "Close", "atr_14"]


def detect_levels(
    df: pd.DataFrame,
    lookback: int = 126,
    atr_mult: float = 0.5,
) -> pd.DataFrame:
    """Add prior-data-only support/resistance zones and ATR distances.

    The first implementation uses the prior rolling high as resistance and the
    prior rolling low as support. The prior shift is the important guardrail:
    today's bar cannot define today's level.
    """
    _validate_input(df, lookback, atr_mult)

    result = df.copy()
    atr = result["atr_14"]

    result["nearest_resistance"] = result["High"].shift(1).rolling(lookback, min_periods=lookback).max()
    result["nearest_support"] = result["Low"].shift(1).rolling(lookback, min_periods=lookback).min()

    zone_half_width = atr_mult * atr
    result["resistance_zone_low"] = result["nearest_resistance"] - zone_half_width
    result["resistance_zone_high"] = result["nearest_resistance"] + zone_half_width
    result["support_zone_low"] = result["nearest_support"] - zone_half_width
    result["support_zone_high"] = result["nearest_support"] + zone_half_width

    result["dist_to_resistance_atr"] = (result["nearest_resistance"] - result["Close"]) / atr
    result["dist_to_support_atr"] = (result["Close"] - result["nearest_support"]) / atr

    return result


def plot_levels(df: pd.DataFrame, symbol: str = "") -> None:
    """Visualize price with support/resistance zones overlaid."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot support/resistance levels") from exc

    title = f"{symbol} support/resistance zones".strip()
    ax = df["Close"].plot(figsize=(12, 6), label="Close", title=title)
    df["nearest_support"].plot(ax=ax, label="Support", linestyle="--")
    df["nearest_resistance"].plot(ax=ax, label="Resistance", linestyle="--")
    ax.fill_between(df.index, df["support_zone_low"], df["support_zone_high"], alpha=0.15)
    ax.fill_between(df.index, df["resistance_zone_low"], df["resistance_zone_high"], alpha=0.15)
    ax.legend()
    plt.show()


def _validate_input(df: pd.DataFrame, lookback: int, atr_mult: float) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required level columns: {missing}")

    if lookback < 1:
        raise ValueError("lookback must be at least 1")

    if atr_mult < 0:
        raise ValueError("atr_mult must be non-negative")
