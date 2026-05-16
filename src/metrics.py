"""Expected Value Agent – trade quality metrics.

Responsibilities (from playbook §3.7):
- Average forward returns by state and horizon.
- Average adverse/favourable excursion.
- Win rate, average win, average loss.
- Expected value after costs.
- Compare Markov forecast to baseline state behavior.

EV formula:
    EV = P(win) × Avg(win) − P(loss) × Avg(loss) − costs
    EV_state,horizon = mean(forward_return | current_state, horizon) − costs
"""

import pandas as pd
import numpy as np


def state_expectancy_table(
    df: pd.DataFrame,
    states: pd.Series,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Compute expected value per state per horizon, after costs.

    Returns
    -------
    pd.DataFrame
        Index: state, Columns include: horizon, mean_return, win_rate,
        avg_win, avg_loss, ev_after_costs, sample_count.
    """
    raise NotImplementedError("Implement in EV Agent phase")


def adverse_excursion(
    df: pd.DataFrame,
    entry_indices: pd.Index,
    horizon: int,
) -> pd.Series:
    """Max adverse excursion (MAE) for each entry within horizon bars."""
    raise NotImplementedError


def favorable_excursion(
    df: pd.DataFrame,
    entry_indices: pd.Index,
    horizon: int,
) -> pd.Series:
    """Max favorable excursion (MFE) for each entry within horizon bars."""
    raise NotImplementedError
