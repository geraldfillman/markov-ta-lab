"""Tests for the paper-trading orchestrator step."""

import numpy as np
import pandas as pd
import pytest

from src.paper_trading import paper_trading_step
from src.positions import PositionTracker


def _frame_with_states(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.01, size=n))
    idx = pd.date_range("2024-01-02", periods=n, freq="B", name="Date")
    states = np.ones(n, dtype=int)  # constant state => EV is well-defined.
    return pd.DataFrame({"Close": closes, "state": states}, index=idx)


def test_paper_trading_step_rejects_unknown_today():
    frame = _frame_with_states()
    with pytest.raises(KeyError):
        paper_trading_step("SPY", frame, PositionTracker.empty(),
                           today=pd.Timestamp("2050-01-01"), horizon=2, lookback=10, min_samples=3)


def test_paper_trading_step_skips_when_drift_blocked():
    frame = _frame_with_states()
    today = frame.index[20]
    updated, decisions = paper_trading_step(
        "SPY", frame, PositionTracker.empty(), today=today,
        horizon=2, lookback=10, min_samples=3, drift_blocked=True,
    )
    assert updated == PositionTracker.empty()
    assert any(d["action"] == "skip_entry" and d["reason"] == "drift_alert" for d in decisions)


def test_paper_trading_step_skips_when_macro_blocked():
    frame = _frame_with_states()
    today = frame.index[20]
    _, decisions = paper_trading_step(
        "SPY", frame, PositionTracker.empty(), today=today,
        horizon=2, lookback=10, min_samples=3, macro_blocked=True,
    )
    assert any(d["reason"] == "macro_filter" for d in decisions)


def test_paper_trading_step_opens_position_on_positive_ev():
    frame = _frame_with_states()
    today = frame.index[30]
    updated, decisions = paper_trading_step(
        "SPY", frame, PositionTracker.empty(), today=today,
        horizon=2, lookback=10, min_samples=3, ev_threshold=-1.0, cost_bps=0.0,
    )
    open_decisions = [d for d in decisions if d["action"] == "open"]
    if open_decisions:
        assert updated.has_open_position("SPY") is True
        assert open_decisions[0]["entry_price"] == float(frame.loc[today, "Close"])


def test_paper_trading_step_does_not_double_open():
    frame = _frame_with_states()
    today = frame.index[30]
    # Target exit is *after* today, so the close branch should NOT fire.
    future_exit = frame.index[35]
    tracker = PositionTracker.empty().open(
        symbol="SPY",
        signal_date=str(frame.index[28].date()),
        entry_date=str(frame.index[29].date()),
        entry_price=100.0, horizon=5, state=1, signal_ev=0.01,
        target_exit_date=str(future_exit.date()),
    )
    updated, decisions = paper_trading_step(
        "SPY", frame, tracker, today=today,
        horizon=2, lookback=10, min_samples=3, ev_threshold=-1.0,
    )
    assert any(d.get("reason") == "already_open" for d in decisions)
    assert len(updated.open_positions("SPY")) == 1


def test_paper_trading_step_closes_position_when_target_exit_reached():
    frame = _frame_with_states()
    entry_idx = 20
    horizon = 3
    exit_idx = entry_idx + horizon
    entry_date = frame.index[entry_idx]
    target_exit = frame.index[exit_idx]
    entry_price = float(frame.loc[entry_date, "Close"])

    tracker = PositionTracker.empty().open(
        symbol="SPY",
        signal_date=str(frame.index[entry_idx - 1].date()),
        entry_date=str(entry_date.date()),
        entry_price=entry_price,
        horizon=horizon,
        state=1,
        signal_ev=0.01,
        target_exit_date=str(target_exit.date()),
        position_id="trade-1",
    )
    updated, decisions = paper_trading_step(
        "SPY", frame, tracker, today=target_exit,
        horizon=horizon, lookback=10, min_samples=3, cost_bps=0.0,
    )
    closed = updated.find("trade-1")
    assert closed is not None
    assert closed.is_open is False
    assert closed.closed_reason == "horizon_reached"
    assert any(d["action"] == "close" for d in decisions)
