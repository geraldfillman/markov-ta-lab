"""Change-Point Agent – regime break detection.

Responsibilities (from playbook §3.10):
- Use `ruptures` to detect changes in return/volatility behavior.
- Flag periods where prior transition matrix may be stale.
- Test whether pausing after change-points improves performance.

Rule example:
    If recent change-point detected, pause new trades for N bars
    or reduce size by 50%.
"""

import pandas as pd
import numpy as np


def detect_changepoints(
    series: pd.Series,
    model: str = "rbf",
    min_size: int = 20,
    penalty: float = 3.0,
) -> list[int]:
    """Return indices of detected change-points using ruptures.

    Parameters
    ----------
    series : pd.Series
        Signal to analyse (returns, volatility, etc.).
    model : str
        Cost model for ruptures (rbf, l2, normal, etc.).
    min_size : int
        Minimum segment length.
    penalty : float
        Penalty value for Pelt algorithm.

    Returns
    -------
    list[int]
        List of change-point indices.
    """
    raise NotImplementedError("Implement in Change-Point Agent phase")


def changepoint_pause_signal(
    changepoints: list[int],
    n_bars: int,
    total_length: int,
) -> pd.Series:
    """Return a boolean Series that is True during the pause window after each change-point."""
    raise NotImplementedError
