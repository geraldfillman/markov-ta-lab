"""Macro Filter Agent — conditional transition matrices.

Responsibilities (from playbook §3.12):
- Classify each bar into a macro regime (RISK_ON, RISK_OFF, NEUTRAL).
- Build per-regime transition matrices: P(state' | state, macro_regime).
- Compare conditioned forecasts to the unconditional Markov chain.
- Filter signals through macro regime to assess Sharpe lift.

Inputs are derived from broad-market context columns (SPY 200dma, VIX level,
yield-curve slope, DXY) so the same vocabulary works across asset universes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MACRO_LABELS
from src.markov import estimate_transition_matrix

RISK_ON = 0
RISK_OFF = 1
NEUTRAL = 2

MIN_REQUIRED_SHARPE_LIFT = 0.20  # Playbook §3.12 acceptance gate.


def classify_macro_regime(
    spy_close: pd.Series,
    vix_close: pd.Series | None = None,
    spy_ma_window: int = 200,
    vix_high_threshold: float = 25.0,
    vix_low_threshold: float = 15.0,
) -> pd.Series:
    """Label each bar with a macro regime.

    Rules (ordered, first match wins):
    - RISK_OFF if SPY < SPY 200dma OR VIX >= vix_high_threshold.
    - RISK_ON  if SPY > SPY 200dma AND (VIX is missing OR VIX <= vix_low_threshold).
    - NEUTRAL  otherwise.

    All inputs are aligned via reindex on the SPY index.
    """
    if spy_ma_window < 2:
        raise ValueError("spy_ma_window must be at least 2")

    spy = spy_close.astype(float).dropna()
    spy_ma = spy.rolling(spy_ma_window, min_periods=spy_ma_window).mean()
    spy_above = spy > spy_ma
    spy_below = spy < spy_ma

    if vix_close is not None:
        vix = vix_close.astype(float).reindex(spy.index).ffill()
        vix_hot = vix >= vix_high_threshold
        vix_cool = vix <= vix_low_threshold
    else:
        vix_hot = pd.Series(False, index=spy.index)
        vix_cool = pd.Series(True, index=spy.index)

    regime = pd.Series(NEUTRAL, index=spy.index, name="macro_regime", dtype="int64")
    regime.loc[spy_above & vix_cool] = RISK_ON
    regime.loc[spy_below | vix_hot] = RISK_OFF
    regime.loc[spy_ma.isna()] = pd.NA
    return regime


def conditional_transition_matrices(
    states: pd.Series,
    macro_regimes: pd.Series,
    n_states: int,
    alpha: float = 1e-6,
) -> dict[int, np.ndarray]:
    """Estimate one transition matrix per macro regime label.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping macro regime id → (n_states, n_states) transition matrix.
        Regimes with fewer than two consecutive bars are skipped.
    """
    aligned = pd.DataFrame({"state": states, "regime": macro_regimes}).dropna()
    aligned = aligned.astype({"state": int, "regime": int})
    if aligned.empty:
        return {}

    matrices: dict[int, np.ndarray] = {}
    for regime_id, group in aligned.groupby("regime"):
        if len(group) < 2:
            continue
        matrices[int(regime_id)] = estimate_transition_matrix(
            group["state"], n_states=n_states, alpha=alpha
        )
    return matrices


def macro_regime_distribution(macro_regimes: pd.Series) -> pd.DataFrame:
    """Return a small frame summarising regime frequencies and labels."""
    counts = macro_regimes.dropna().astype(int).value_counts().sort_index()
    total = int(counts.sum())
    rows = []
    for regime_id, count in counts.items():
        rows.append(
            {
                "macro_regime": int(regime_id),
                "label": MACRO_LABELS.get(int(regime_id), str(regime_id)),
                "count": int(count),
                "percent": float(count) / total * 100.0 if total else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("macro_regime")


def macro_filter_signal(
    macro_regimes: pd.Series,
    allowed_regimes: list[int],
) -> pd.Series:
    """Boolean mask — True when the current bar's macro regime is allowed."""
    if not allowed_regimes:
        raise ValueError("allowed_regimes must not be empty")
    allowed = set(int(value) for value in allowed_regimes)
    return macro_regimes.fillna(-1).astype(int).isin(allowed)


def compare_conditional_to_unconditional(
    states: pd.Series,
    macro_regimes: pd.Series,
    n_states: int,
    alpha: float = 1e-6,
) -> pd.DataFrame:
    """Compute the L1 distance between each conditional matrix and the unconditional one.

    Large distances suggest that conditioning on the macro regime carries
    meaningful information beyond the base Markov chain.
    """
    aligned_states = states.dropna()
    unconditional = estimate_transition_matrix(
        aligned_states, n_states=n_states, alpha=alpha
    )
    conditional = conditional_transition_matrices(
        states, macro_regimes, n_states=n_states, alpha=alpha
    )

    rows = []
    for regime_id, matrix in conditional.items():
        rows.append(
            {
                "macro_regime": regime_id,
                "label": MACRO_LABELS.get(regime_id, str(regime_id)),
                "l1_distance": float(np.abs(matrix - unconditional).sum()),
                "max_row_diff": float(np.abs(matrix - unconditional).sum(axis=1).max()),
            }
        )
    return pd.DataFrame(rows).set_index("macro_regime").sort_index()


def evaluate_macro_filter_sharpe_lift(
    df: pd.DataFrame,
    states: pd.Series,
    macro_regimes: pd.Series,
    allowed_regimes: list[int],
    horizon: int = 5,
    lookback: int = 252,
    min_samples: int = 5,
    cost_bps: float = 10.0,
) -> dict[str, float | bool]:
    """Compare walk-forward strategy Sharpe with and without the macro filter.

    Returns a dict including ``sharpe_unfiltered``, ``sharpe_filtered``,
    ``sharpe_lift`` (filtered − unfiltered), and a boolean ``passes_gate``
    flag indicating whether the lift meets ``MIN_REQUIRED_SHARPE_LIFT``.
    Local import of :func:`src.backtests.run_walkforward_ev_backtest` avoids
    a circular module dependency.
    """
    from src.backtests import run_walkforward_ev_backtest  # local: avoid cycles

    mask = macro_filter_signal(macro_regimes, allowed_regimes=allowed_regimes)

    unfiltered = run_walkforward_ev_backtest(
        df, states, horizon=horizon, lookback=lookback,
        min_samples=min_samples, cost_bps=cost_bps,
    )
    filtered = run_walkforward_ev_backtest(
        df, states, horizon=horizon, lookback=lookback,
        min_samples=min_samples, cost_bps=cost_bps, signal_mask=mask,
    )

    sharpe_unfiltered = float(unfiltered["stats"].get("sharpe", 0.0) or 0.0)
    sharpe_filtered = float(filtered["stats"].get("sharpe", 0.0) or 0.0)
    lift = sharpe_filtered - sharpe_unfiltered

    return {
        "sharpe_unfiltered": sharpe_unfiltered,
        "sharpe_filtered": sharpe_filtered,
        "sharpe_lift": lift,
        "trade_count_unfiltered": int(unfiltered["stats"].get("trade_count", 0) or 0),
        "trade_count_filtered": int(filtered["stats"].get("trade_count", 0) or 0),
        "min_required_lift": float(MIN_REQUIRED_SHARPE_LIFT),
        "passes_gate": bool(lift >= MIN_REQUIRED_SHARPE_LIFT),
    }
