"""Paper-trading orchestrator — turns one bar's state + EV into a tracker update.

This module is **research-only**. It does not place orders, contact brokers,
or simulate any market microstructure beyond fills at the day's close. The
output of :func:`paper_trading_step` is an updated :class:`PositionTracker`
plus a list of decision records explaining each open/close/skip.

Daily timing convention
-----------------------
* ``today`` is the most recent bar.
* The signal that decides today's entry was generated *yesterday* (the bar
  before ``today``) — i.e. next-bar entry, the same convention as
  :func:`src.backtests.run_walkforward_ev_backtest`.
* Entries fill at today's close; exits fill at the target-exit bar's close.
"""

from __future__ import annotations

import pandas as pd

from src.metrics import walkforward_state_expectancy
from src.positions import PositionTracker


def _iso(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def paper_trading_step(
    symbol: str,
    frame: pd.DataFrame,
    tracker: PositionTracker,
    today: pd.Timestamp,
    horizon: int = 5,
    lookback: int = 252,
    min_samples: int = 10,
    ev_threshold: float = 0.0,
    cost_bps: float = 5.0,
    drift_blocked: bool = False,
    macro_blocked: bool = False,
) -> tuple[PositionTracker, list[dict[str, object]]]:
    """Advance the tracker by one bar.

    Returns a tuple of ``(updated_tracker, decisions)``. Each decision is a
    small dict with at least an ``action`` key (``open``, ``close``,
    ``skip_entry``, ``skip``) and a ``reason`` field for skips.
    """
    if "Close" not in frame.columns or "state" not in frame.columns:
        raise ValueError("frame must include 'Close' and 'state' columns")

    today_ts = pd.Timestamp(today)
    if today_ts not in frame.index:
        raise KeyError(f"{today_ts!r} not present in frame.index")

    decisions: list[dict[str, object]] = []
    new_tracker = tracker

    # 1) Close any open positions whose target_exit_date has arrived.
    for pos in tracker.open_positions(symbol=symbol):
        target = pd.Timestamp(pos.target_exit_date)
        if target <= today_ts and target in frame.index:
            exit_price = float(frame.loc[target, "Close"])
            new_tracker = new_tracker.close(
                pos.position_id,
                exit_date=_iso(target),
                exit_price=exit_price,
                reason="horizon_reached",
                cost_bps=cost_bps,
            )
            decisions.append(
                {
                    "action": "close",
                    "position_id": pos.position_id,
                    "exit_date": _iso(target),
                    "exit_price": exit_price,
                }
            )

    # 2) Filter gates (drift, macro, already-open).
    if drift_blocked:
        decisions.append({"action": "skip_entry", "reason": "drift_alert"})
        return new_tracker, decisions
    if macro_blocked:
        decisions.append({"action": "skip_entry", "reason": "macro_filter"})
        return new_tracker, decisions
    if new_tracker.has_open_position(symbol):
        decisions.append({"action": "skip_entry", "reason": "already_open"})
        return new_tracker, decisions

    # 3) Read yesterday's signal.
    today_idx = frame.index.get_loc(today_ts)
    if today_idx < 1:
        decisions.append({"action": "skip_entry", "reason": "no_yesterday_bar"})
        return new_tracker, decisions
    yesterday = frame.index[today_idx - 1]
    history = frame.loc[:yesterday]
    ev_frame = walkforward_state_expectancy(
        history,
        history["state"],
        horizon=horizon,
        lookback=lookback,
        min_samples=min_samples,
        cost_bps=cost_bps,
    )
    if yesterday not in ev_frame.index:
        decisions.append({"action": "skip_entry", "reason": "no_ev_for_signal_date"})
        return new_tracker, decisions

    row = ev_frame.loc[yesterday]
    signal_ev = float(row["walkforward_ev"]) if pd.notna(row["walkforward_ev"]) else float("nan")
    if pd.isna(signal_ev) or signal_ev <= ev_threshold:
        decisions.append(
            {
                "action": "skip_entry",
                "reason": "ev_below_threshold",
                "signal_ev": signal_ev,
            }
        )
        return new_tracker, decisions

    target_exit_idx = today_idx + horizon
    if target_exit_idx >= len(frame.index):
        decisions.append({"action": "skip_entry", "reason": "horizon_past_data"})
        return new_tracker, decisions

    target_exit_date = frame.index[target_exit_idx]
    entry_price = float(frame.loc[today_ts, "Close"])
    new_tracker = new_tracker.open(
        symbol=symbol,
        signal_date=_iso(yesterday),
        entry_date=_iso(today_ts),
        entry_price=entry_price,
        horizon=horizon,
        state=int(row["state"]),
        signal_ev=signal_ev,
        target_exit_date=_iso(target_exit_date),
    )
    decisions.append(
        {
            "action": "open",
            "entry_date": _iso(today_ts),
            "entry_price": entry_price,
            "signal_ev": signal_ev,
            "state": int(row["state"]),
            "target_exit_date": _iso(target_exit_date),
        }
    )
    return new_tracker, decisions
