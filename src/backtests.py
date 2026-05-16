"""Backtest Agent – strategy simulation.

Responsibilities (from playbook §3.8):
- Readable prototype using backtesting.py.
- Fast parameter sweeps using vectorbt.
- Include costs, slippage, position sizing.
- Compare against baselines.
- Report performance by state, ticker, and regime.

Metrics to report:
    Total return, Annualized return, Sharpe, Sortino,
    Max drawdown, Calmar, Win rate, Avg win, Avg loss,
    Profit factor, Exposure time, Turnover, Trade count,
    Avg holding period, Best/worst trade,
    Performance by state, asset, and year.
"""

import pandas as pd
import numpy as np


def run_backtest_readable(
    df: pd.DataFrame,
    states: pd.Series,
    ev_table: pd.DataFrame,
    cost_bps: float = 10.0,
) -> dict:
    """Run a readable backtesting.py strategy on a single asset.

    Returns
    -------
    dict
        Keys: 'stats', 'trades', 'equity_curve'.
    """
    raise NotImplementedError("Implement in Backtest Agent phase")


def run_vectorbt_sweep(
    data: dict[str, pd.DataFrame],
    param_grid: dict,
) -> pd.DataFrame:
    """Run vectorised parameter sweeps across assets.

    Returns
    -------
    pd.DataFrame
        Results sorted by out-of-sample Sharpe.
    """
    raise NotImplementedError


def baseline_buy_and_hold(df: pd.DataFrame) -> dict:
    """Compute buy-and-hold benchmark metrics."""
    raise NotImplementedError


def baseline_ma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> dict:
    """Compute MA crossover benchmark metrics."""
    raise NotImplementedError
