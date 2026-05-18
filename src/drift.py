"""Drift monitor — KL divergence on state-frequency distributions.

Acceptance criterion (playbook A3.13): alert when the distribution of
recently observed states diverges from the training-window distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_KL_THRESHOLD = 0.10


def state_frequency_distribution(
    states: pd.Series,
    n_states: int,
    alpha: float = 1e-6,
) -> np.ndarray:
    """Return a length-n_states probability vector with Laplace smoothing."""
    if n_states < 2:
        raise ValueError("n_states must be at least 2")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    counts = np.full(n_states, alpha, dtype=float)
    values = states.dropna().astype(int).to_numpy()
    valid = (values >= 0) & (values < n_states)
    if valid.any():
        np.add.at(counts, values[valid], 1.0)

    total = counts.sum()
    if total <= 0.0:
        return np.full(n_states, 1.0 / n_states)
    return counts / total


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) in nats with epsilon-clipping on both arms."""
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must share shape")
    p_safe = np.clip(p_arr, eps, None)
    q_safe = np.clip(q_arr, eps, None)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def drift_alert(
    training_states: pd.Series,
    current_states: pd.Series,
    n_states: int,
    threshold: float = DEFAULT_KL_THRESHOLD,
    alpha: float = 1e-6,
) -> dict[str, float | bool | int]:
    """Compute KL(current || training) and flag drift above threshold."""
    p_train = state_frequency_distribution(training_states, n_states=n_states, alpha=alpha)
    p_now = state_frequency_distribution(current_states, n_states=n_states, alpha=alpha)
    kl = kl_divergence(p_now, p_train)
    return {
        "kl_divergence": kl,
        "threshold": float(threshold),
        "alert": bool(kl >= threshold),
        "n_training": int(training_states.dropna().shape[0]),
        "n_current": int(current_states.dropna().shape[0]),
    }
