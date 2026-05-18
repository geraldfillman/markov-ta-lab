"""Tests for Expected Value Agent metrics."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.config import STATE_LABELS
from src.metrics import (
    conditioned_state_expectancy_table,
    cluster_pooled_state_expectancy_table,
    forecast_expected_value,
    save_state_expectancy_table,
    state_expectancy_table,
    universe_state_expectancy_table,
    walkforward_markov_expected_value,
    walkforward_state_expectancy,
)


def _price_frame(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": closes},
        index=pd.date_range("2024-01-02", periods=len(closes), freq="B", name="Date"),
    )


def _repo_tmp_path(name: str) -> Path:
    path = Path(".test_output") / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_state_expectancy_uses_forward_returns_and_costs():
    df = _price_frame([100.0, 110.0, 105.0, 120.0])
    states = pd.Series([1, 1, 2, 1], index=df.index, name="state")

    table = state_expectancy_table(df, states, horizons=(1,), cost_bps=10.0)

    state_one_returns = np.array([0.10, (105.0 / 110.0) - 1.0])
    expected_mean = state_one_returns.mean()

    assert table.loc[1, "label"] == STATE_LABELS[1]
    assert table.loc[1, "count_1"] == 2
    assert table.loc[1, "win_rate_1"] == 0.5
    np.testing.assert_allclose(table.loc[1, "avg_win_1"], 0.10)
    np.testing.assert_allclose(table.loc[1, "avg_loss_1"], (105.0 / 110.0) - 1.0)
    np.testing.assert_allclose(table.loc[1, "avg_forward_return_1"], expected_mean)
    np.testing.assert_allclose(table.loc[1, "ev_after_cost_1"], expected_mean - 0.001)
    assert {"ci_low_1", "ci_high_1"}.issubset(table.columns)
    assert table.loc[1, "ci_low_1"] < table.loc[1, "avg_forward_return_1"] < table.loc[1, "ci_high_1"]


def test_state_expectancy_excludes_rows_without_future_horizon():
    df = _price_frame([100.0, 105.0, 110.0])
    states = pd.Series([5, 5, 5], index=df.index, name="state")

    table = state_expectancy_table(df, states, horizons=(2,), cost_bps=0.0)

    assert table.loc[5, "count_2"] == 1
    np.testing.assert_allclose(table.loc[5, "avg_forward_return_2"], 0.10)


def test_state_expectancy_aligns_states_to_price_index():
    df = _price_frame([100.0, 102.0, 101.0])
    states = pd.Series(
        [9, 9, 4],
        index=pd.date_range("2024-01-02", periods=3, freq="B", name="Date"),
        name="state",
    )

    table = state_expectancy_table(df.iloc[:2], states, horizons=(1,), cost_bps=0.0)

    assert table.loc[9, "count_1"] == 1
    assert 4 not in table.index


def test_universe_state_expectancy_adds_symbol_dimension():
    spy = _price_frame([100.0, 110.0, 120.0])
    spy["state"] = [1, 1, 2]
    qqq = _price_frame([200.0, 190.0, 210.0])
    qqq["state"] = [2, 2, 1]

    table = universe_state_expectancy_table(
        {"SPY": spy, "QQQ": qqq},
        horizons=(1,),
        cost_bps=0.0,
    )

    assert table.index.names == ["symbol", "state"]
    assert table.loc[("SPY", 1), "count_1"] == 2
    assert table.loc[("QQQ", 2), "count_1"] == 2
    np.testing.assert_allclose(table.loc[("SPY", 1), "avg_forward_return_1"], np.mean([0.10, (120.0 / 110.0) - 1.0]))


def test_cluster_pooled_state_expectancy_groups_assets_by_cluster():
    spy = _price_frame([100.0, 110.0, 121.0])
    spy["state"] = [1, 1, 2]
    qqq = _price_frame([200.0, 220.0, 242.0])
    qqq["state"] = [1, 1, 2]
    clusters = pd.DataFrame(
        {"cluster_label": ["risk_on", "risk_on"]},
        index=pd.Index(["SPY", "QQQ"], name="symbol"),
    )

    table = cluster_pooled_state_expectancy_table(
        {"SPY": spy, "QQQ": qqq},
        clusters,
        horizons=(1,),
        cost_bps=0.0,
    )

    assert table.index.names == ["cluster_label", "state"]
    assert table.loc[("risk_on", 1), "count_1"] == 4
    np.testing.assert_allclose(table.loc[("risk_on", 1), "avg_forward_return_1"], 0.10)


def test_save_state_expectancy_table_creates_parent_directory():
    table = pd.DataFrame(
        {"label": ["APPROACHING_SUPPORT"], "count_1": [1]},
        index=pd.MultiIndex.from_tuples([("SPY", 1)], names=["symbol", "state"]),
    )
    output_root = _repo_tmp_path("state_expectancy_save")

    output_path = save_state_expectancy_table(table, output_root / "reports" / "tables" / "state_expectancy.csv")

    saved = pd.read_csv(output_path)
    assert saved.loc[0, "symbol"] == "SPY"
    assert saved.loc[0, "state"] == 1
    assert saved.loc[0, "count_1"] == 1


def test_forecast_expected_value_weights_destination_payoffs():
    probabilities = np.array([0.25, 0.75])
    expectancy = pd.DataFrame(
        {
            "label": ["FAR_FROM_LEVEL", "APPROACHING_SUPPORT"],
            "ev_after_cost_5": [0.02, -0.01],
            "count_5": [100, 50],
        },
        index=pd.Index([0, 1], name="state"),
    )

    result = forecast_expected_value(probabilities, expectancy, horizon=5)

    np.testing.assert_allclose(result["expected_value"], (0.25 * 0.02) + (0.75 * -0.01))
    assert result["horizon"] == 5
    assert result["coverage"] == 1.0
    assert result["contributions"].loc[0, "probability"] == 0.25
    assert result["contributions"].loc[1, "state_ev"] == -0.01


def test_conditioned_state_expectancy_groups_by_vol_state_and_state():
    df = _price_frame([100.0, 110.0, 100.0, 120.0])
    states = pd.Series([1, 1, 1, 1], index=df.index, name="state")
    vol_states = pd.Series([0, 0, 2, 2], index=df.index, name="vol_state")

    table = conditioned_state_expectancy_table(df, states, vol_states, horizons=(1,), cost_bps=0.0)

    assert table.index.names == ["vol_state", "state"]
    assert table.loc[(0, 1), "count_1"] == 2
    assert table.loc[(2, 1), "count_1"] == 1
    np.testing.assert_allclose(table.loc[(0, 1), "avg_forward_return_1"], np.mean([0.10, (100.0 / 110.0) - 1.0]))
    np.testing.assert_allclose(table.loc[(2, 1), "avg_forward_return_1"], 0.20)


def test_walkforward_state_expectancy_uses_only_realized_prior_outcomes():
    df = _price_frame([100.0, 110.0, 121.0, 100.0, 90.0])
    states = pd.Series([1, 1, 1, 1, 1], index=df.index, name="state")

    result = walkforward_state_expectancy(
        df,
        states,
        horizon=1,
        lookback=2,
        min_samples=2,
        cost_bps=0.0,
    )

    first_forecast_date = df.index[3]
    assert result.index[0] == first_forecast_date
    assert result.loc[first_forecast_date, "state"] == 1
    assert result.loc[first_forecast_date, "train_count"] == 2
    np.testing.assert_allclose(result.loc[first_forecast_date, "walkforward_ev"], 0.10)


def test_walkforward_state_expectancy_requires_min_samples():
    df = _price_frame([100.0, 110.0, 121.0, 100.0])
    states = pd.Series([1, 1, 1, 1], index=df.index, name="state")

    result = walkforward_state_expectancy(df, states, horizon=1, lookback=2, min_samples=3, cost_bps=0.0)

    assert result["walkforward_ev"].isna().all()


def test_walkforward_markov_expected_value_weights_prior_destination_payoffs():
    df = _price_frame([100.0, 110.0, 121.0, 133.1, 146.41])
    states = pd.Series([0, 1, 1, 1, 1], index=df.index, name="state")

    result = walkforward_markov_expected_value(
        df,
        states,
        n_states=2,
        horizon=1,
        lookback=3,
        min_samples=2,
        cost_bps=0.0,
    )

    row = result.loc[df.index[4]]
    assert row["current_state"] == 1
    assert row["coverage"] == 1.0
    np.testing.assert_allclose(row["markov_weighted_ev"], 0.10)


def test_sensitivity_stability_summary_per_symbol():
    from src.metrics import sensitivity_stability_summary

    sens = pd.DataFrame(
        [
            {"symbol": "SPY", "sharpe": 0.5},
            {"symbol": "SPY", "sharpe": 0.6},
            {"symbol": "SPY", "sharpe": 0.4},
            {"symbol": "QQQ", "sharpe": -0.1},
            {"symbol": "QQQ", "sharpe": 0.1},
        ]
    )
    summary = sensitivity_stability_summary(sens)
    assert set(summary["symbol"]) == {"SPY", "QQQ"}
    spy = summary[summary["symbol"] == "SPY"].iloc[0]
    assert spy["n_configs"] == 3
    assert spy["sharpe_median"] == pytest.approx(0.5)
    qqq = summary[summary["symbol"] == "QQQ"].iloc[0]
    assert qqq["sharpe_share_negative"] == pytest.approx(0.5)


def test_sensitivity_stability_summary_rejects_missing_columns():
    from src.metrics import sensitivity_stability_summary

    with pytest.raises(ValueError):
        sensitivity_stability_summary(pd.DataFrame({"sharpe": [1.0]}))
    with pytest.raises(ValueError):
        sensitivity_stability_summary(pd.DataFrame({"symbol": ["A"]}))


def test_bootstrap_sharpe_ci_brackets_point_estimate():
    from src.metrics import bootstrap_sharpe_ci_from_trades

    rng = np.random.default_rng(0)
    returns = rng.normal(0.002, 0.01, size=400)
    result = bootstrap_sharpe_ci_from_trades(returns, n_resamples=500, random_state=1)
    assert result["ci_low"] <= result["sharpe_point"] <= result["ci_high"]
    assert result["n"] == 400


def test_bootstrap_sharpe_ci_block_resampling_runs():
    from src.metrics import bootstrap_sharpe_ci_from_trades

    rng = np.random.default_rng(2)
    returns = rng.normal(0.001, 0.02, size=200)
    result = bootstrap_sharpe_ci_from_trades(returns, n_resamples=200, block_size=10, random_state=3)
    assert result["ci_low"] <= result["sharpe_point"] <= result["ci_high"]


def test_bootstrap_sharpe_ci_handles_small_sample():
    from src.metrics import bootstrap_sharpe_ci_from_trades

    result = bootstrap_sharpe_ci_from_trades(pd.Series([0.01]), n_resamples=10)
    assert np.isnan(result["sharpe_point"])
    assert result["n"] == 1
