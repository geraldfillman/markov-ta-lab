"""Change-Point Agent — regime break detection.

Responsibilities (from playbook §3.10):
- Use `ruptures` to detect changes in return/volatility behavior.
- Flag periods where the prior transition matrix may be stale.
- Test whether pausing after change-points improves performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_changepoints(
    series: pd.Series,
    model: str = "rbf",
    min_size: int = 20,
    penalty: float = 3.0,
) -> list[int]:
    """Return positional indices of detected change-points using ruptures Pelt.

    NaNs are dropped before fitting; returned indices are positions within the
    original `series` (mapped back through `series.dropna().index`).

    The final element returned by ruptures (always `len(signal)`) is omitted —
    it is a fence-post sentinel, not a real change-point.
    """
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if penalty <= 0:
        raise ValueError("penalty must be positive")

    try:
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("ruptures is required for change-point detection") from exc

    clean = series.dropna()
    if len(clean) < min_size * 2:
        return []

    signal = clean.to_numpy(dtype=float).reshape(-1, 1)
    algorithm = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    breakpoints = algorithm.predict(pen=float(penalty))

    # ruptures appends len(signal) as a sentinel; strip it.
    if breakpoints and breakpoints[-1] == len(signal):
        breakpoints = breakpoints[:-1]

    if not breakpoints:
        return []

    clean_index_positions = series.index.get_indexer(clean.index)
    return [int(clean_index_positions[bp]) for bp in breakpoints]


def changepoint_pause_signal(
    changepoints: list[int],
    n_bars: int,
    total_length: int,
) -> pd.Series:
    """Boolean Series — True during the pause window after each change-point.

    Returns an integer-indexed Series of length `total_length`. Each detected
    change-point activates the pause window for the following `n_bars` bars
    (inclusive of the change-point bar itself).
    """
    if n_bars < 0:
        raise ValueError("n_bars must be non-negative")
    if total_length < 0:
        raise ValueError("total_length must be non-negative")

    pause = np.zeros(total_length, dtype=bool)
    for cp in changepoints:
        start = max(0, int(cp))
        end = min(total_length, start + n_bars)
        if start < end:
            pause[start:end] = True

    return pd.Series(pause, name="changepoint_pause")


def annotate_changepoints(
    series: pd.Series,
    model: str = "rbf",
    min_size: int = 20,
    penalty: float = 3.0,
    pause_window: int = 5,
) -> pd.DataFrame:
    """Convenience wrapper — detect change-points and return a per-bar annotation frame."""
    positions = detect_changepoints(series, model=model, min_size=min_size, penalty=penalty)
    pause = changepoint_pause_signal(positions, n_bars=pause_window, total_length=len(series))
    is_changepoint = np.zeros(len(series), dtype=bool)
    for cp in positions:
        if 0 <= cp < len(series):
            is_changepoint[cp] = True

    return pd.DataFrame(
        {
            "is_changepoint": is_changepoint,
            "pause_active": pause.to_numpy(),
        },
        index=series.index,
    )
