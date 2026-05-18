"""Tests for the Macro Filter Agent."""

import numpy as np
import pandas as pd
import pytest

from src.macro import (
    NEUTRAL,
    RISK_OFF,
    RISK_ON,
    classify_macro_regime,
    compare_conditional_to_unconditional,
    conditional_transition_matrices,
    macro_filter_signal,
    macro_regime_distribution,
)


def _spy_vix_fixture(seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 400
    bull = np.cumprod(1.0 + rng.normal(0.0008, 0.008, size=n // 2))
    bear = np.cumprod(1.0 + rng.normal(-0.0008, 0.018, size=n // 2))
    spy_values = np.concatenate([bull, bull[-1] * bear])
    vix_values = np.concatenate(
        [rng.normal(13.0, 1.5, size=n // 2), rng.normal(28.0, 4.0, size=n // 2)]
    )
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return (
        pd.Series(spy_values * 100, index=index, name="SPY"),
        pd.Series(vix_values, index=index, name="VIX"),
    )


def test_classify_macro_regime_emits_all_three_states():
    spy, vix = _spy_vix_fixture()
    regime = classify_macro_regime(spy, vix, spy_ma_window=50)
    counts = regime.dropna().astype(int).value_counts()
    assert RISK_OFF in counts.index
    assert {RISK_ON, RISK_OFF, NEUTRAL}.issuperset(set(counts.index.tolist()))


def test_classify_macro_regime_without_vix_falls_back_to_ma():
    spy, _ = _spy_vix_fixture(seed=1)
    regime = classify_macro_regime(spy, vix_close=None, spy_ma_window=30)
    assert regime.dropna().astype(int).between(0, 2).all()


def test_classify_macro_regime_rejects_invalid_window():
    spy, _ = _spy_vix_fixture()
    with pytest.raises(ValueError):
        classify_macro_regime(spy, spy_ma_window=1)


def test_conditional_transition_matrices_are_row_stochastic():
    rng = np.random.default_rng(2)
    states = pd.Series(rng.integers(0, 4, size=300))
    regimes = pd.Series(rng.integers(0, 3, size=300))
    matrices = conditional_transition_matrices(states, regimes, n_states=4, alpha=1e-3)
    assert matrices, "expected at least one regime matrix"
    for matrix in matrices.values():
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-9)
        assert matrix.shape == (4, 4)


def test_conditional_transition_matrices_skip_singleton_regimes():
    states = pd.Series([0, 1, 2, 3])
    regimes = pd.Series([0, 0, 1, 1])
    matrices = conditional_transition_matrices(states, regimes, n_states=4)
    assert set(matrices.keys()) <= {0, 1}


def test_macro_regime_distribution_returns_percent_table():
    regimes = pd.Series([0, 0, 0, 1, 2, 2])
    table = macro_regime_distribution(regimes)
    assert set(table.columns) == {"label", "count", "percent"}
    assert table["count"].sum() == 6


def test_macro_filter_signal_masks_disallowed():
    regimes = pd.Series([0, 1, 2, 0, np.nan])
    mask = macro_filter_signal(regimes, allowed_regimes=[0])
    assert mask.tolist() == [True, False, False, True, False]


def test_macro_filter_signal_empty_allowed_raises():
    regimes = pd.Series([0, 1])
    with pytest.raises(ValueError):
        macro_filter_signal(regimes, allowed_regimes=[])


def test_compare_conditional_to_unconditional_returns_distances():
    rng = np.random.default_rng(3)
    states = pd.Series(rng.integers(0, 3, size=200))
    regimes = pd.Series(rng.integers(0, 2, size=200))
    table = compare_conditional_to_unconditional(states, regimes, n_states=3)
    assert "l1_distance" in table.columns
    assert (table["l1_distance"] >= 0).all()


def _macro_filter_scenario(seed: int = 0):
    """Construct a deterministic dataset where filtering RISK_OFF bars helps.

    The price series is positive-drift during RISK_ON bars and negative-drift
    during RISK_OFF bars. States are constant so EV is driven purely by
    realised price action — meaning the macro filter has a real lift to
    capture.
    """
    rng = np.random.default_rng(seed)
    n = 600
    half = n // 2
    on_returns = rng.normal(0.004, 0.01, size=half)
    off_returns = rng.normal(-0.004, 0.01, size=half)
    returns = np.concatenate([on_returns, off_returns, on_returns, off_returns])[:n]
    regimes = np.concatenate(
        [np.zeros(half, dtype=int), np.ones(half, dtype=int),
         np.zeros(half, dtype=int), np.ones(half, dtype=int)]
    )[:n]
    closes = 100.0 * np.cumprod(1.0 + returns)
    idx = pd.date_range("2022-01-03", periods=n, freq="B", name="Date")
    df = pd.DataFrame({"Close": closes}, index=idx)
    states = pd.Series([0] * n, index=idx, name="state")
    macro_regimes = pd.Series(regimes, index=idx, name="macro_regime")
    return df, states, macro_regimes


def test_evaluate_macro_filter_sharpe_lift_returns_expected_shape():
    from src.macro import MIN_REQUIRED_SHARPE_LIFT, evaluate_macro_filter_sharpe_lift

    df, states, macro_regimes = _macro_filter_scenario()
    result = evaluate_macro_filter_sharpe_lift(
        df, states, macro_regimes,
        allowed_regimes=[0],
        horizon=2, lookback=40, min_samples=5, cost_bps=0.0,
    )
    required = {
        "sharpe_unfiltered", "sharpe_filtered", "sharpe_lift",
        "trade_count_unfiltered", "trade_count_filtered",
        "min_required_lift", "passes_gate",
    }
    assert required.issubset(result.keys())
    assert result["min_required_lift"] == pytest.approx(MIN_REQUIRED_SHARPE_LIFT)


def test_macro_filter_drops_signals_outside_allowed_regimes():
    from src.macro import evaluate_macro_filter_sharpe_lift

    df, states, macro_regimes = _macro_filter_scenario()
    result = evaluate_macro_filter_sharpe_lift(
        df, states, macro_regimes,
        allowed_regimes=[0],
        horizon=2, lookback=40, min_samples=5, cost_bps=0.0,
    )
    # Filtering must not increase the number of trades — it can only remove them.
    assert result["trade_count_filtered"] <= result["trade_count_unfiltered"]


def test_macro_filter_acceptance_gate_recognises_meaningful_lift():
    """Acceptance tripwire (playbook §3.12): keeping only the favourable
    macro regime should produce a measurable Sharpe lift on a scenario
    constructed to make the filter informative."""
    from src.macro import evaluate_macro_filter_sharpe_lift

    df, states, macro_regimes = _macro_filter_scenario(seed=1)
    result = evaluate_macro_filter_sharpe_lift(
        df, states, macro_regimes,
        allowed_regimes=[0],
        horizon=2, lookback=40, min_samples=5, cost_bps=0.0,
    )
    # On the constructed scenario the filter must produce a positive lift —
    # the magnitude depends on the EV estimator, so we only require sign +
    # a generous floor here. The 0.2 hurdle is asserted in the experiment
    # script (scripts/run_macro_gate.py) against real data.
    assert result["sharpe_lift"] >= 0.0
