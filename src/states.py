"""State Labeling Agent - deterministic market state classification."""

import pandas as pd

from src.config import N_STATES, STATE_LABELS


FAR_FROM_LEVEL = 0
APPROACHING_SUPPORT = 1
TOUCHING_SUPPORT = 2
SUPPORT_RECLAIM = 3
SUPPORT_BREAKDOWN = 4
APPROACHING_RESISTANCE = 5
COMPRESSION_BELOW_RESISTANCE = 6
RESISTANCE_BREAKOUT = 7
BREAKOUT_RETEST = 8
FAILED_BREAKOUT = 9
CONTINUATION = 10
CHOP_OR_NO_EDGE = 11

REQUIRED_COLUMNS = [
    "Close",
    "sma_20",
    "sma_50",
    "bb_width_20",
    "nearest_support",
    "nearest_resistance",
    "support_zone_low",
    "support_zone_high",
    "resistance_zone_low",
    "resistance_zone_high",
    "dist_to_support_atr",
    "dist_to_resistance_atr",
]

DEFAULT_CONFIG = {
    "proximity_atr": 1.0,
    "compression_bb_width": 0.06,
    "breakout_memory": 5,
}


def label_states(df: pd.DataFrame, config: dict | None = None) -> pd.Series:
    """Return one integer state label per bar.

    Rules are deterministic and ordered from highest-priority structural events
    to lower-information proximity states. Incomplete warm-up rows fall back to
    `CHOP_OR_NO_EDGE` instead of being dropped.
    """
    _validate_columns(df)
    settings = {**DEFAULT_CONFIG, **(config or {})}
    proximity_atr = settings["proximity_atr"]
    compression_bb_width = settings["compression_bb_width"]
    breakout_memory = settings["breakout_memory"]

    close = df["Close"]
    previous_close = close.shift(1)
    trend_up = (close > df["sma_20"]) & (df["sma_20"] >= df["sma_50"])
    valid = df[REQUIRED_COLUMNS].notna().all(axis=1)

    support_breakdown = close < df["support_zone_low"]
    support_reclaim = (previous_close < df["support_zone_low"].shift(1)) & (close >= df["support_zone_low"])
    touching_support = close.between(df["support_zone_low"], df["support_zone_high"], inclusive="both")
    approaching_support = (
        (df["dist_to_support_atr"] >= 0)
        & (df["dist_to_support_atr"] <= proximity_atr)
    )

    resistance_breakout = close > df["resistance_zone_high"]
    recent_breakout = resistance_breakout.astype(float).shift(1).rolling(
        breakout_memory,
        min_periods=1,
    ).max().fillna(0).astype(bool)
    failed_breakout = recent_breakout & (close < df["resistance_zone_low"])
    breakout_retest = (
        recent_breakout
        & close.between(df["resistance_zone_low"], df["resistance_zone_high"], inclusive="both")
    )
    compression_below_resistance = (
        (close < df["resistance_zone_low"])
        & (df["dist_to_resistance_atr"] >= 0)
        & (df["dist_to_resistance_atr"] <= proximity_atr)
        & (df["bb_width_20"] <= compression_bb_width)
    )
    approaching_resistance = (
        (df["dist_to_resistance_atr"] >= 0)
        & (df["dist_to_resistance_atr"] <= proximity_atr)
    )
    continuation = recent_breakout & resistance_breakout & trend_up

    states = pd.Series(FAR_FROM_LEVEL, index=df.index, name="state", dtype="int64")
    states.loc[~valid] = CHOP_OR_NO_EDGE

    _assign(states, valid & approaching_support, APPROACHING_SUPPORT)
    _assign(states, valid & approaching_resistance, APPROACHING_RESISTANCE)
    _assign(states, valid & compression_below_resistance, COMPRESSION_BELOW_RESISTANCE)
    _assign(states, valid & touching_support, TOUCHING_SUPPORT)
    _assign(states, valid & resistance_breakout, RESISTANCE_BREAKOUT)
    _assign(states, valid & continuation, CONTINUATION)
    _assign(states, valid & breakout_retest, BREAKOUT_RETEST)
    _assign(states, valid & failed_breakout, FAILED_BREAKOUT)
    _assign(states, valid & support_reclaim, SUPPORT_RECLAIM)
    _assign(states, valid & support_breakdown, SUPPORT_BREAKDOWN)

    return states


def state_frequency_report(states: pd.Series) -> pd.DataFrame:
    """Return a frequency table with count and percentage per state."""
    counts = states.astype(int).value_counts().sort_index()
    total = int(counts.sum())
    rows = []
    for state_id, count in counts.items():
        rows.append(
            {
                "state": int(state_id),
                "label": STATE_LABELS.get(int(state_id), "UNKNOWN"),
                "count": int(count),
                "percent": round((int(count) / total) * 100, 4) if total else 0.0,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["label", "count", "percent"]).rename_axis("state")

    return pd.DataFrame(rows).set_index("state")


def flag_rare_states(states: pd.Series, min_count: int = 30) -> list[int]:
    """Return state IDs with at least one observation but fewer than min_count."""
    counts = states.astype(int).value_counts()
    return sorted(int(state_id) for state_id, count in counts.items() if count < min_count)


def _assign(states: pd.Series, mask: pd.Series, state_id: int) -> None:
    states.loc[mask] = state_id


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required state columns: {missing}")

    if len(STATE_LABELS) != N_STATES:
        raise ValueError("STATE_LABELS and N_STATES are inconsistent")
