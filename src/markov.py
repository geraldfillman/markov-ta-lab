"""Markov Agent – visible Markov Chain model.

Responsibilities (from playbook §3.6):
- Estimate transition counts → probabilities.
- Compute 1-step and multi-step state forecasts.
- Compute stationary distribution.
- Optional Laplace smoothing for sparse rows.
- Rolling / walk-forward estimation.
"""

import numpy as np
import pandas as pd


def estimate_transition_matrix(
    states: pd.Series,
    n_states: int,
    alpha: float = 0.0,
) -> np.ndarray:
    """Estimate transition matrix with optional Laplace smoothing.

    P(i, j) = Count(i → j) / Count(i → any)

    Parameters
    ----------
    states : pd.Series
        Integer state labels.
    n_states : int
        Total number of possible states.
    alpha : float
        Laplace smoothing parameter (0 = no smoothing).

    Returns
    -------
    np.ndarray
        (n_states, n_states) transition probability matrix. Rows sum to 1.
    """
    raise NotImplementedError("Implement in Markov Agent phase")


def forecast_state_probs(
    P: np.ndarray,
    current_state: int,
    horizon: int,
) -> np.ndarray:
    """Return horizon-step state probabilities from current state.

    Computed as P^horizon[current_state].
    """
    raise NotImplementedError


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of the transition matrix."""
    raise NotImplementedError


def walkforward_forecasts(
    states: pd.Series,
    n_states: int,
    lookback: int,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Run walk-forward Markov forecasts over the full series.

    At each bar, estimate the transition matrix from only the prior
    `lookback` bars and compute horizon-step forecasts.

    Returns
    -------
    pd.DataFrame
        Columns: current_state, prob_h{h}_state{s} for each horizon/state.
    """
    raise NotImplementedError
