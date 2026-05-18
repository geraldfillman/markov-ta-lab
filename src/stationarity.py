"""Rolling transition matrices and stability analysis for non-stationarity detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_transition_matrix(
    states: pd.Series,
    window: int,
    n_states: int = 12,
    step: int = 1,
) -> dict[pd.Timestamp, np.ndarray]:
    """Compute empirical row-normalised transition matrix for each anchor date.

    Anchor dates are every `step` bars from index `window` onward.
    Rows with no observations are zeroed (not set to identity) to distinguish
    unseen states from self-loops.
    """
    if states.empty or window < 2:
        return {}

    values = states.to_numpy(dtype=float)
    index = states.index
    n = len(values)
    result: dict[pd.Timestamp, np.ndarray] = {}

    positions = range(window, n + 1, step)
    for end in positions:
        start = end - window
        window_vals = values[start:end]
        valid = window_vals[~np.isnan(window_vals)].astype(int)

        counts = np.zeros((n_states, n_states), dtype=float)
        if valid.size >= 2:
            src = valid[:-1]
            dst = valid[1:]
            mask = (src >= 0) & (src < n_states) & (dst >= 0) & (dst < n_states)
            np.add.at(counts, (src[mask], dst[mask]), 1.0)

        row_sums = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)
        anchor = index[end - 1]
        result[anchor] = matrix

    return result


def stability_score(matrices: dict[pd.Timestamp, np.ndarray]) -> pd.Series:
    """1 - 0.5 * mean(|P_t - P_{t-1}|.sum(axis=1)) for consecutive matrix pairs.

    Returns a Series indexed by the later anchor date; values in [0, 1].
    Higher = more stable (1 = identical).
    """
    if len(matrices) < 2:
        return pd.Series(dtype=float)

    dates = sorted(matrices.keys())
    scores: dict[pd.Timestamp, float] = {}
    for i in range(1, len(dates)):
        p_prev = matrices[dates[i - 1]]
        p_curr = matrices[dates[i]]
        diff = np.abs(p_curr - p_prev).sum(axis=1).mean()
        scores[dates[i]] = 1.0 - 0.5 * diff

    return pd.Series(scores)


def probability_stability_per_state(
    matrices: dict[pd.Timestamp, np.ndarray],
    state_id: int,
) -> pd.Series:
    """Per-anchor stability for a single state row only.

    Values in [0, 1].
    """
    if len(matrices) < 2:
        return pd.Series(dtype=float)

    dates = sorted(matrices.keys())
    scores: dict[pd.Timestamp, float] = {}
    for i in range(1, len(dates)):
        row_prev = matrices[dates[i - 1]][state_id]
        row_curr = matrices[dates[i]][state_id]
        diff = np.abs(row_curr - row_prev).sum()
        scores[dates[i]] = 1.0 - 0.5 * diff

    return pd.Series(scores)


def flag_unstable_states(
    matrices: dict[pd.Timestamp, np.ndarray],
    threshold: float = 0.7,
) -> list[int]:
    """Return state ids whose mean per-state stability is below `threshold`."""
    if len(matrices) < 2:
        return []

    dates = sorted(matrices.keys())
    n_states = next(iter(matrices.values())).shape[0]
    # Accumulate per-state L1 diffs across consecutive pairs
    total_diff = np.zeros(n_states, dtype=float)
    n_pairs = len(dates) - 1

    for i in range(1, len(dates)):
        p_prev = matrices[dates[i - 1]]
        p_curr = matrices[dates[i]]
        total_diff += np.abs(p_curr - p_prev).sum(axis=1)

    mean_stability = 1.0 - 0.5 * (total_diff / n_pairs)
    return [int(s) for s in np.where(mean_stability < threshold)[0]]
