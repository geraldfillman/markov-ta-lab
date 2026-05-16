"""HMM Regime Agent – hidden regime detection.

Responsibilities (from playbook §3.9):
- Fit Gaussian HMM to emissions (returns, vol, ATR pct, volume z, S/R dist).
- Multiple random starts.
- Label hidden states by realised behavior (not component number).
- Compare HMM regimes to visible technical states.
- Use HMM as a filter, not a replacement.

Only build after visible-state Markov model works.
"""

import pandas as pd
import numpy as np


def fit_hmm(
    emissions: pd.DataFrame,
    n_regimes: int = 3,
    n_starts: int = 10,
) -> dict:
    """Fit a Gaussian HMM and return model + regime labels.

    Parameters
    ----------
    emissions : pd.DataFrame
        Observable features (returns, vol, etc.).
    n_regimes : int
        Number of hidden states.
    n_starts : int
        Random initialisation runs (best log-likelihood wins).

    Returns
    -------
    dict
        Keys: 'model', 'regimes' (pd.Series), 'log_likelihood',
              'means', 'covars', 'transmat'.
    """
    raise NotImplementedError("Implement in HMM Regime Agent phase")


def label_regimes_by_behavior(
    regimes: pd.Series,
    returns: pd.Series,
    volatility: pd.Series,
) -> dict[int, str]:
    """Assign human-readable names to hidden regimes based on realised stats."""
    raise NotImplementedError
