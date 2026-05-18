"""Tests for the paper-trading position tracker."""

import json

import pytest

from src.positions import Position, PositionTracker, SCHEMA_VERSION


def _open_tracker(symbol: str = "SPY") -> PositionTracker:
    return PositionTracker.empty().open(
        symbol=symbol,
        signal_date="2024-01-02",
        entry_date="2024-01-03",
        entry_price=100.0,
        horizon=5,
        state=7,
        signal_ev=0.012,
        target_exit_date="2024-01-10",
        position_id="abc123",
    )


def test_empty_tracker_starts_with_no_positions():
    tracker = PositionTracker.empty()
    assert tracker.positions == ()
    assert tracker.open_positions() == ()
    assert tracker.closed_positions() == ()


def test_open_appends_a_position_without_mutating_original():
    original = PositionTracker.empty()
    updated = original.open(
        symbol="SPY", signal_date="2024-01-02", entry_date="2024-01-03",
        entry_price=100.0, horizon=5, state=7, signal_ev=0.012,
        target_exit_date="2024-01-10",
    )
    assert original.positions == ()
    assert len(updated.positions) == 1
    assert updated.positions[0].symbol == "SPY"
    assert updated.positions[0].is_open is True


def test_close_records_net_return_and_marks_closed():
    tracker = _open_tracker()
    closed = tracker.close(
        position_id="abc123",
        exit_date="2024-01-10",
        exit_price=105.0,
        reason="horizon_reached",
        cost_bps=5.0,
    )
    pos = closed.find("abc123")
    assert pos is not None
    assert pos.is_open is False
    assert pos.closed_reason == "horizon_reached"
    assert pos.net_return == pytest.approx(0.05 - 5e-4)


def test_close_unknown_position_raises():
    tracker = _open_tracker()
    with pytest.raises(KeyError):
        tracker.close("missing", exit_date="2024-01-10", exit_price=100.0, reason="manual")


def test_close_already_closed_raises():
    tracker = _open_tracker().close("abc123", "2024-01-10", 105.0, "horizon_reached")
    with pytest.raises(ValueError):
        tracker.close("abc123", "2024-01-11", 106.0, "manual")


def test_has_open_position_filters_by_symbol():
    tracker = _open_tracker("SPY")
    assert tracker.has_open_position("SPY") is True
    assert tracker.has_open_position("QQQ") is False


def test_mark_to_market_returns_per_position_pnl():
    tracker = _open_tracker("SPY")
    pnl = tracker.mark_to_market({"SPY": 110.0, "QQQ": 999.0})
    assert pnl == {"abc123": pytest.approx(0.10)}


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "tracker.json"
    tracker = _open_tracker().close("abc123", "2024-01-10", 105.0, "horizon_reached", cost_bps=5.0)
    tracker.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert len(payload["positions"]) == 1

    reloaded = PositionTracker.load(path)
    assert reloaded.positions[0].net_return == pytest.approx(tracker.positions[0].net_return)


def test_load_missing_path_returns_empty_tracker(tmp_path):
    tracker = PositionTracker.load(tmp_path / "does_not_exist.json")
    assert tracker == PositionTracker.empty()


def test_load_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 999, "positions": []}))
    with pytest.raises(ValueError, match="Unsupported tracker schema version"):
        PositionTracker.load(path)
