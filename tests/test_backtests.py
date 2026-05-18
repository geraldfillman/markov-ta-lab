"""Tests for backtesting (src/backtests.py).

Key checks (from playbook §7.5):
- Transaction costs included.
- Positions are shifted correctly (no same-bar future info).
- Benchmark comparison included.
"""

import numpy as np
import pandas as pd

from src.backtests import (
    baseline_breakout_fixed_horizon,
    baseline_buy_and_hold,
    baseline_ma_crossover,
    baseline_random_label_walkforward,
    compare_backtest_to_baselines,
    run_walkforward_sensitivity,
    run_walkforward_ev_backtest,
    run_backtest_readable,
)


def _frame(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": closes},
        index=pd.date_range("2024-01-02", periods=len(closes), freq="B", name="Date"),
    )


def _ev_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["FAR_FROM_LEVEL", "APPROACHING_SUPPORT"],
            "ev_after_cost_1": [-0.01, 0.02],
        },
        index=pd.Index([0, 1], name="state"),
    )


def test_costs_included():
    """Backtest results should differ from zero-cost results."""
    df = _frame([100.0, 100.0, 110.0, 110.0])
    states = pd.Series([1, 0, 0, 0], index=df.index, name="state")

    zero_cost = run_backtest_readable(df, states, _ev_table(), horizon=1, cost_bps=0.0)
    high_cost = run_backtest_readable(df, states, _ev_table(), horizon=1, cost_bps=100.0)

    np.testing.assert_allclose(zero_cost["trades"].iloc[0]["gross_return"], 0.10)
    np.testing.assert_allclose(high_cost["trades"].iloc[0]["net_return"], 0.09)
    assert high_cost["stats"]["total_return"] < zero_cost["stats"]["total_return"]


def test_no_same_bar_signal():
    """Signals should not use same-bar close for entry."""
    df = _frame([100.0, 100.0, 110.0, 110.0])
    states = pd.Series([1, 0, 0, 0], index=df.index, name="state")

    result = run_backtest_readable(df, states, _ev_table(), horizon=1, cost_bps=0.0)
    trade = result["trades"].iloc[0]

    assert trade["signal_date"] == df.index[0]
    assert trade["entry_date"] == df.index[1]
    assert trade["exit_date"] == df.index[2]
    assert trade["entry_price"] == 100.0


def test_benchmark_exists():
    """Results dict should include benchmark comparison."""
    df = _frame([100.0, 105.0, 110.0])
    states = pd.Series([0, 0, 0], index=df.index, name="state")

    result = run_backtest_readable(df, states, _ev_table(), horizon=1, cost_bps=0.0)

    assert "benchmark" in result
    assert result["benchmark"] == baseline_buy_and_hold(df)
    np.testing.assert_allclose(result["benchmark"]["total_return"], 0.10)


def test_ma_crossover_baseline_uses_prior_signal():
    df = _frame([100.0, 100.0, 110.0, 121.0])

    result = baseline_ma_crossover(df, fast=1, slow=2)

    np.testing.assert_allclose(result["total_return"], 0.10)
    assert result["bars"] == 4
    assert result["exposure_time"] == 0.25


def test_breakout_baseline_uses_prior_high_and_next_bar_entry():
    df = _frame([100.0, 101.0, 105.0, 106.0, 112.0])

    result = baseline_breakout_fixed_horizon(df, lookback=2, horizon=1, cost_bps=0.0)

    assert result["trade_count"] == 1
    np.testing.assert_allclose(result["total_return"], (112.0 / 106.0) - 1.0)


def test_compare_backtest_to_baselines_returns_named_rows():
    df = _frame([100.0, 100.0, 110.0, 121.0])
    states = pd.Series([1, 0, 0, 0], index=df.index, name="state")
    strategy = run_backtest_readable(df, states, _ev_table(), horizon=1, cost_bps=0.0)

    comparison = compare_backtest_to_baselines(df, strategy, ma_fast=1, ma_slow=2, breakout_lookback=2, horizon=1)

    assert set(comparison["model"]) == {"state_ev_strategy", "buy_and_hold", "ma_crossover", "breakout"}
    assert "excess_vs_buy_hold" in comparison.columns


def test_walkforward_ev_backtest_uses_prior_ev_series_for_entries():
    df = _frame([100.0, 110.0, 121.0, 100.0, 90.0, 99.0])
    states = pd.Series([1, 1, 1, 1, 1, 1], index=df.index, name="state")

    result = run_walkforward_ev_backtest(
        df,
        states,
        horizon=1,
        lookback=2,
        min_samples=2,
        cost_bps=0.0,
    )

    assert result["trades"].iloc[0]["signal_date"] == df.index[3]
    assert result["trades"].iloc[0]["entry_date"] == df.index[4]
    assert result["trades"].iloc[0]["signal_ev"] > 0.0
    assert "walkforward_ev" in result


def test_walkforward_sensitivity_returns_parameter_rows():
    df = _frame([100.0, 110.0, 121.0, 100.0, 90.0, 99.0])
    states = pd.Series([1, 1, 1, 1, 1, 1], index=df.index, name="state")

    result = run_walkforward_sensitivity(
        df,
        states,
        horizons=(1,),
        lookbacks=(2,),
        costs_bps=(0.0, 10.0),
        min_samples_values=(2,),
    )

    assert len(result) == 2
    assert {"horizon", "lookback", "cost_bps", "min_samples", "total_return", "trade_count"}.issubset(result.columns)


def _trending_frame(n: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.002, 0.01, size=n)
    closes = 100.0 * np.cumprod(1.0 + returns)
    idx = pd.date_range("2023-01-02", periods=n, freq="B", name="Date")
    return pd.DataFrame({"Close": closes}, index=idx)


def test_random_label_baseline_runs_walkforward_with_shuffled_states():
    df = _trending_frame()
    rng = np.random.default_rng(11)
    states = pd.Series(rng.integers(0, 3, size=len(df)), index=df.index, name="state")

    result = baseline_random_label_walkforward(
        df, states, horizon=2, lookback=20, min_samples=3, cost_bps=0.0, seed=42
    )
    assert {"stats", "trades", "equity_curve", "benchmark"}.issubset(result.keys())
    assert "sharpe" in result["stats"]


def test_random_label_baseline_seed_is_reproducible():
    df = _trending_frame()
    rng = np.random.default_rng(3)
    states = pd.Series(rng.integers(0, 3, size=len(df)), index=df.index, name="state")

    first = baseline_random_label_walkforward(df, states, horizon=2, lookback=20, min_samples=3, seed=99)
    second = baseline_random_label_walkforward(df, states, horizon=2, lookback=20, min_samples=3, seed=99)
    assert first["stats"]["total_return"] == second["stats"]["total_return"]


def test_walkforward_ev_backtest_respects_signal_mask():
    df = _frame([100.0, 110.0, 121.0, 100.0, 90.0, 99.0, 108.9, 119.79])
    states = pd.Series([1] * len(df), index=df.index, name="state")
    block_all = pd.Series(False, index=df.index)

    masked = run_walkforward_ev_backtest(
        df, states, horizon=1, lookback=2, min_samples=2, cost_bps=0.0, signal_mask=block_all
    )
    unmasked = run_walkforward_ev_backtest(
        df, states, horizon=1, lookback=2, min_samples=2, cost_bps=0.0
    )
    assert masked["trades"].shape[0] == 0
    assert unmasked["trades"].shape[0] >= 1


def test_compare_backtest_to_baselines_includes_random_label_when_supplied():
    df = _trending_frame()
    rng = np.random.default_rng(5)
    states = pd.Series(rng.integers(0, 3, size=len(df)), index=df.index, name="state")
    strategy = run_walkforward_ev_backtest(df, states, horizon=2, lookback=20, min_samples=3, cost_bps=0.0)
    random_label = baseline_random_label_walkforward(df, states, horizon=2, lookback=20, min_samples=3, seed=0)

    comparison = compare_backtest_to_baselines(
        df, strategy, ma_fast=5, ma_slow=20, breakout_lookback=10, horizon=2,
        random_label_result=random_label,
    )
    assert "random_label" in set(comparison["model"])
