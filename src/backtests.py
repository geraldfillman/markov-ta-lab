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

Known unimplemented surface:
    run_vectorbt_sweep is retained as a placeholder, but it is not part of
    the current walk-forward research pipeline contract.
"""

import pandas as pd
import numpy as np

from src.metrics import walkforward_state_expectancy


def run_backtest_readable(
    df: pd.DataFrame,
    states: pd.Series,
    ev_table: pd.DataFrame,
    horizon: int = 5,
    ev_threshold: float = 0.0,
    cost_bps: float = 10.0,
) -> dict:
    """Run a readable long-only fixed-horizon backtest on a single asset.

    Returns
    -------
    dict
        Keys: 'stats', 'trades', 'equity_curve', 'benchmark'.
    """
    if "Close" not in df.columns:
        raise ValueError("df must include a Close column")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    ev_col = f"ev_after_cost_{horizon}"
    if ev_col not in ev_table.columns:
        raise ValueError(f"ev_table must include {ev_col}")

    aligned = pd.DataFrame({"Close": df["Close"], "state": states.reindex(df.index)}, index=df.index)
    cost = cost_bps / 10_000.0
    trades = []
    last_exit_pos = -1

    for signal_pos in range(0, len(aligned) - horizon - 1):
        if signal_pos <= last_exit_pos:
            continue

        state = aligned.iloc[signal_pos]["state"]
        if pd.isna(state):
            continue

        state = int(state)
        if state not in ev_table.index:
            continue

        state_ev = ev_table.loc[state, ev_col]
        if pd.isna(state_ev) or float(state_ev) <= ev_threshold:
            continue

        entry_pos = signal_pos + 1
        exit_pos = entry_pos + horizon
        entry_price = float(aligned.iloc[entry_pos]["Close"])
        exit_price = float(aligned.iloc[exit_pos]["Close"])
        gross_return = (exit_price / entry_price) - 1.0
        net_return = gross_return - cost
        trades.append(
            {
                "signal_date": aligned.index[signal_pos],
                "entry_date": aligned.index[entry_pos],
                "exit_date": aligned.index[exit_pos],
                "state": state,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "holding_period": horizon,
            }
        )
        last_exit_pos = exit_pos

    trades_df = pd.DataFrame(trades)
    equity_curve = _equity_curve_from_trades(aligned.index, trades_df)
    return {
        "stats": _performance_stats(equity_curve, trades_df),
        "trades": trades_df,
        "equity_curve": equity_curve,
        "benchmark": baseline_buy_and_hold(df),
    }


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


def run_walkforward_ev_backtest(
    df: pd.DataFrame,
    states: pd.Series,
    horizon: int = 5,
    lookback: int = 252,
    min_samples: int = 5,
    ev_threshold: float = 0.0,
    cost_bps: float = 10.0,
    signal_mask: pd.Series | None = None,
) -> dict:
    """Run the long-only fixed-horizon strategy with walk-forward EV estimates.

    ``signal_mask`` is an optional boolean ``Series`` indexed compatibly with
    ``df``. When supplied, a signal is only taken on dates where the mask is
    truthy — used by the macro filter and the drift gate.
    """
    if "Close" not in df.columns:
        raise ValueError("df must include a Close column")

    aligned = pd.DataFrame({"Close": df["Close"], "state": states.reindex(df.index)}, index=df.index)
    ev = walkforward_state_expectancy(
        df,
        states,
        horizon=horizon,
        lookback=lookback,
        min_samples=min_samples,
        cost_bps=cost_bps,
    )
    cost = cost_bps / 10_000.0
    trades = []
    last_exit_pos = -1
    mask_aligned = (
        signal_mask.reindex(aligned.index).fillna(False).astype(bool)
        if signal_mask is not None
        else None
    )

    for signal_date, row in ev.iterrows():
        signal_pos = aligned.index.get_loc(signal_date)
        if signal_pos <= last_exit_pos or signal_pos + horizon + 1 >= len(aligned):
            continue
        if mask_aligned is not None and not bool(mask_aligned.iloc[signal_pos]):
            continue
        signal_ev = row["walkforward_ev"]
        if pd.isna(signal_ev) or float(signal_ev) <= ev_threshold:
            continue

        entry_pos = signal_pos + 1
        exit_pos = entry_pos + horizon
        entry_price = float(aligned.iloc[entry_pos]["Close"])
        exit_price = float(aligned.iloc[exit_pos]["Close"])
        gross_return = (exit_price / entry_price) - 1.0
        net_return = gross_return - cost
        trades.append(
            {
                "signal_date": signal_date,
                "entry_date": aligned.index[entry_pos],
                "exit_date": aligned.index[exit_pos],
                "state": int(row["state"]),
                "signal_ev": float(signal_ev),
                "train_count": int(row["train_count"]),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "holding_period": horizon,
            }
        )
        last_exit_pos = exit_pos

    trades_df = pd.DataFrame(trades)
    equity_curve = _equity_curve_from_trades(aligned.index, trades_df)
    return {
        "stats": _performance_stats(equity_curve, trades_df),
        "trades": trades_df,
        "equity_curve": equity_curve,
        "benchmark": baseline_buy_and_hold(df),
        "walkforward_ev": ev,
    }


def baseline_buy_and_hold(df: pd.DataFrame) -> dict:
    """Compute buy-and-hold benchmark metrics."""
    if "Close" not in df.columns:
        raise ValueError("df must include a Close column")
    close = df["Close"].dropna()
    if len(close) < 2:
        return {"total_return": 0.0, "bars": int(len(close))}

    return {
        "total_return": float((close.iloc[-1] / close.iloc[0]) - 1.0),
        "bars": int(len(close)),
    }


def baseline_ma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> dict:
    """Compute MA crossover benchmark metrics."""
    if fast < 1 or slow < 1:
        raise ValueError("fast and slow windows must be positive")
    if "Close" not in df.columns:
        raise ValueError("df must include a Close column")

    close = df["Close"].dropna()
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    position = (fast_ma > slow_ma).shift(1, fill_value=False).astype(float)
    returns = close.pct_change().fillna(0.0)
    strategy_returns = position * returns
    equity = (1.0 + strategy_returns).cumprod()

    return {
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "bars": int(len(close)),
        "exposure_time": float(position.mean()) if len(position) else 0.0,
    }


def baseline_breakout_fixed_horizon(
    df: pd.DataFrame,
    lookback: int = 126,
    horizon: int = 5,
    cost_bps: float = 10.0,
) -> dict:
    """Compute a simple prior-high breakout baseline with next-bar entry."""
    if "Close" not in df.columns:
        raise ValueError("df must include a Close column")
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")

    close = df["Close"].dropna()
    prior_high = close.shift(1).rolling(lookback, min_periods=lookback).max()
    breakout = close > prior_high
    cost = cost_bps / 10_000.0
    trades = []
    last_exit_pos = -1

    for signal_pos in range(0, len(close) - horizon - 1):
        if signal_pos <= last_exit_pos or not bool(breakout.iloc[signal_pos]):
            continue

        entry_pos = signal_pos + 1
        exit_pos = entry_pos + horizon
        net_return = (float(close.iloc[exit_pos]) / float(close.iloc[entry_pos])) - 1.0 - cost
        trades.append(net_return)
        last_exit_pos = exit_pos

    total_return = float(np.prod([1.0 + trade for trade in trades]) - 1.0) if trades else 0.0
    return {
        "total_return": total_return,
        "trade_count": int(len(trades)),
        "win_rate": float(np.mean([trade > 0.0 for trade in trades])) if trades else np.nan,
        "exposure_time": float((len(trades) * horizon) / len(close)) if len(close) else 0.0,
    }


def baseline_random_label_walkforward(
    df: pd.DataFrame,
    states: pd.Series,
    horizon: int = 5,
    lookback: int = 252,
    min_samples: int = 5,
    cost_bps: float = 10.0,
    seed: int = 0,
) -> dict:
    """Walk-forward backtest on **shuffled** state labels.

    Permutation-baseline check (playbook §robustness): if performance survives
    when the state column is randomly permuted, the strategy is exploiting
    label-set artefacts rather than genuine state→return persistence. The
    shuffle preserves the marginal distribution of states; only the temporal
    pairing with prices is broken.
    """
    rng = np.random.default_rng(seed)
    clean = states.dropna().astype(int).to_numpy()
    if clean.size == 0:
        shuffled_values = np.array([], dtype=int)
    else:
        shuffled_values = clean.copy()
        rng.shuffle(shuffled_values)

    shuffled = pd.Series(index=states.dropna().index, data=shuffled_values, name="state")
    shuffled = shuffled.reindex(states.index)

    return run_walkforward_ev_backtest(
        df,
        shuffled,
        horizon=horizon,
        lookback=lookback,
        min_samples=min_samples,
        cost_bps=cost_bps,
    )


def compare_backtest_to_baselines(
    df: pd.DataFrame,
    strategy_result: dict,
    ma_fast: int = 50,
    ma_slow: int = 200,
    breakout_lookback: int = 126,
    horizon: int = 5,
    cost_bps: float = 10.0,
    random_label_result: dict | None = None,
) -> pd.DataFrame:
    """Return a compact comparison table for strategy and simple baselines."""
    buy_hold = baseline_buy_and_hold(df)
    rows = [
        {"model": "state_ev_strategy", **strategy_result["stats"]},
        {"model": "buy_and_hold", **buy_hold},
        {"model": "ma_crossover", **baseline_ma_crossover(df, fast=ma_fast, slow=ma_slow)},
        {
            "model": "breakout",
            **baseline_breakout_fixed_horizon(df, lookback=breakout_lookback, horizon=horizon, cost_bps=cost_bps),
        },
    ]
    if random_label_result is not None:
        rows.append({"model": "random_label", **random_label_result["stats"]})
    comparison = pd.DataFrame(rows)
    comparison["excess_vs_buy_hold"] = comparison["total_return"] - float(buy_hold["total_return"])
    return comparison


def run_walkforward_sensitivity(
    df: pd.DataFrame,
    states: pd.Series,
    horizons: tuple[int, ...] = (5,),
    lookbacks: tuple[int, ...] = (126, 252),
    costs_bps: tuple[float, ...] = (5.0, 10.0),
    min_samples_values: tuple[int, ...] = (5, 10),
    ev_threshold: float = 0.0,
) -> pd.DataFrame:
    """Run a small walk-forward parameter sensitivity grid for one asset."""
    rows = []
    for horizon in horizons:
        for lookback in lookbacks:
            for cost_bps in costs_bps:
                for min_samples in min_samples_values:
                    result = run_walkforward_ev_backtest(
                        df,
                        states,
                        horizon=horizon,
                        lookback=lookback,
                        min_samples=min_samples,
                        ev_threshold=ev_threshold,
                        cost_bps=cost_bps,
                    )
                    rows.append(
                        {
                            "horizon": horizon,
                            "lookback": lookback,
                            "cost_bps": cost_bps,
                            "min_samples": min_samples,
                            **result["stats"],
                            "benchmark_total_return": result["benchmark"]["total_return"],
                        }
                    )

    return pd.DataFrame(rows)


def _equity_curve_from_trades(index: pd.Index, trades: pd.DataFrame) -> pd.DataFrame:
    equity = pd.Series(1.0, index=index, name="equity")
    if trades.empty:
        return equity.to_frame()

    sorted_trades = trades.sort_values("exit_date")
    exit_dates = pd.to_datetime(sorted_trades["exit_date"].to_numpy())
    cumulative = np.cumprod(1.0 + sorted_trades["net_return"].to_numpy(dtype=float))

    index_values = pd.to_datetime(index).to_numpy()
    insertion = np.searchsorted(exit_dates, index_values, side="right")
    values = np.ones(len(index_values), dtype=float)
    has_trade = insertion > 0
    values[has_trade] = cumulative[insertion[has_trade] - 1]
    equity.iloc[:] = values

    return equity.to_frame()


def _performance_stats(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> dict:
    equity = equity_curve["equity"]
    total_return = float(equity.iloc[-1] - 1.0)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    returns = equity.pct_change().dropna()
    sharpe = np.nan
    if len(returns) and not np.isclose(returns.std(ddof=0), 0.0):
        sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))

    if trades.empty:
        return {
            "total_return": total_return,
            "max_drawdown": float(drawdown.min()),
            "sharpe": sharpe,
            "trade_count": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "exposure_time": 0.0,
        }

    wins = trades.loc[trades["net_return"] > 0.0, "net_return"]
    losses = trades.loc[trades["net_return"] < 0.0, "net_return"]
    exposure_bars = trades["holding_period"].sum()
    return {
        "total_return": total_return,
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "trade_count": int(len(trades)),
        "win_rate": float((trades["net_return"] > 0.0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "exposure_time": float(exposure_bars / len(equity)) if len(equity) else 0.0,
    }
