"""End-to-end pipeline smoke test on synthetic OHLCV data.

Chains data → indicators → levels → states → markov → metrics → backtest
to catch interface drift (column names, dtypes, index alignment) that
isolated unit tests miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtests import run_walkforward_ev_backtest
from src.config import N_STATES
from src.indicators import add_indicators
from src.levels import detect_levels
from src.markov import (
    estimate_transition_matrix,
    forecast_state_probs,
    walkforward_forecasts,
)
from src.metrics import (
    forecast_expected_value,
    state_expectancy_table,
    walkforward_state_expectancy,
)
from src.states import label_states


def _synthetic_ohlcv(n: int = 500, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0003, scale=0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1.0 + np.abs(rng.normal(scale=0.004, size=n)))
    low = close * (1.0 - np.abs(rng.normal(scale=0.004, size=n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    index = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def test_full_pipeline_runs_end_to_end():
    df = _synthetic_ohlcv()
    with_indicators = add_indicators(df)
    with_levels = detect_levels(with_indicators)
    states = label_states(with_levels)

    assert states.notna().all()
    assert states.between(0, N_STATES - 1).all()

    table = state_expectancy_table(with_levels, states, horizons=(1, 5))
    assert not table.empty
    assert "ev_after_cost_5" in table.columns

    matrix = estimate_transition_matrix(states, n_states=N_STATES, alpha=1e-6)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-9)

    current_state = int(states.iloc[-1])
    probs = forecast_state_probs(matrix, current_state, horizon=5)
    assert probs.sum() == 1.0 or abs(probs.sum() - 1.0) < 1e-9

    ev_result = forecast_expected_value(probs, table, horizon=5)
    assert "expected_value" in ev_result
    assert "contributions" in ev_result

    walk_ev = walkforward_state_expectancy(
        with_levels, states, horizon=5, lookback=120, min_samples=3
    )
    assert "walkforward_ev" in walk_ev.columns
    assert len(walk_ev) > 0

    forecasts = walkforward_forecasts(states, n_states=N_STATES, lookback=120, horizons=(5,))
    assert not forecasts.empty

    backtest = run_walkforward_ev_backtest(
        with_levels, states, horizon=5, lookback=120, min_samples=3
    )
    assert "stats" in backtest
    assert "total_return" in backtest["stats"]
    assert "equity_curve" in backtest
    assert backtest["equity_curve"].index.equals(with_levels.index)
