"""State Labeling Agent – discrete market state classification.

Responsibilities (from playbook §3.5):
- Label exactly one state per bar (mutually exclusive, collectively exhaustive).
- Deterministic rules based on technical features and levels.
- Track state frequencies; flag/merge rare states.

State map (initial 12 states):
    0  = FAR_FROM_LEVEL
    1  = APPROACHING_SUPPORT
    2  = TOUCHING_SUPPORT
    3  = SUPPORT_RECLAIM
    4  = SUPPORT_BREAKDOWN
    5  = APPROACHING_RESISTANCE
    6  = COMPRESSION_BELOW_RESISTANCE
    7  = RESISTANCE_BREAKOUT
    8  = BREAKOUT_RETEST
    9  = FAILED_BREAKOUT
    10 = CONTINUATION
    11 = CHOP_OR_NO_EDGE
"""

import pandas as pd


def label_states(df: pd.DataFrame, config: dict | None = None) -> pd.Series:
    """Return integer state label per bar.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain OHLCV, indicators, and level features.
    config : dict, optional
        Override default thresholds (ATR proximity, compression rules, etc.).

    Returns
    -------
    pd.Series
        Integer state label for each bar (0–11).
    """
    raise NotImplementedError("Implement in State Labeling Agent phase")


def state_frequency_report(states: pd.Series) -> pd.DataFrame:
    """Return a frequency table with count and percentage per state."""
    raise NotImplementedError


def flag_rare_states(states: pd.Series, min_count: int = 30) -> list[int]:
    """Return list of state IDs with fewer than min_count observations."""
    raise NotImplementedError
