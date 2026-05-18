"""Stateful paper-trading position tracker.

Research-only — there is NO order execution in this module. The tracker
records intent and mark-to-market state for trades the research pipeline
would have taken, persisted to a JSON file. Compliance and live broker
integration are explicitly out of scope.

Design choices
--------------
* :class:`Position` is a frozen dataclass, so a position is never mutated
  in place. Closing a position returns a new Position with the close fields
  populated.
* :class:`PositionTracker` is also immutable: ``open()`` / ``close()`` return
  new trackers. This makes it easy to reason about race conditions in the
  daily cron script.
* Persistence is JSON with atomic rename — no database, no schema migration.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Position:
    """One paper-trading position. Dates are ISO ``YYYY-MM-DD`` strings."""

    position_id: str
    symbol: str
    signal_date: str
    entry_date: str
    entry_price: float
    horizon: int
    state: int
    signal_ev: float
    target_exit_date: str
    exit_date: str | None = None
    exit_price: float | None = None
    net_return: float | None = None
    closed_reason: str | None = None  # 'horizon_reached', 'drift_alert', 'manual'

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PositionTracker:
    """Immutable container for paper-trading positions."""

    positions: tuple[Position, ...] = ()

    @classmethod
    def empty(cls) -> "PositionTracker":
        return cls(positions=())

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "PositionTracker":
        return cls(positions=tuple(Position(**record) for record in records))

    def to_records(self) -> list[dict[str, object]]:
        return [pos.to_dict() for pos in self.positions]

    def open_positions(self, symbol: str | None = None) -> tuple[Position, ...]:
        return tuple(
            pos for pos in self.positions
            if pos.is_open and (symbol is None or pos.symbol == symbol)
        )

    def closed_positions(self) -> tuple[Position, ...]:
        return tuple(pos for pos in self.positions if not pos.is_open)

    def has_open_position(self, symbol: str) -> bool:
        return any(pos.symbol == symbol and pos.is_open for pos in self.positions)

    def find(self, position_id: str) -> Position | None:
        for pos in self.positions:
            if pos.position_id == position_id:
                return pos
        return None

    def open(
        self,
        symbol: str,
        signal_date: str,
        entry_date: str,
        entry_price: float,
        horizon: int,
        state: int,
        signal_ev: float,
        target_exit_date: str,
        position_id: str | None = None,
    ) -> "PositionTracker":
        new_position = Position(
            position_id=position_id or uuid.uuid4().hex,
            symbol=symbol,
            signal_date=signal_date,
            entry_date=entry_date,
            entry_price=float(entry_price),
            horizon=int(horizon),
            state=int(state),
            signal_ev=float(signal_ev),
            target_exit_date=target_exit_date,
        )
        return PositionTracker(positions=self.positions + (new_position,))

    def close(
        self,
        position_id: str,
        exit_date: str,
        exit_price: float,
        reason: str,
        cost_bps: float = 0.0,
    ) -> "PositionTracker":
        target = self.find(position_id)
        if target is None:
            raise KeyError(f"Unknown position_id: {position_id}")
        if not target.is_open:
            raise ValueError(f"Position {position_id} is already closed")
        cost = cost_bps / 10_000.0
        gross_return = (float(exit_price) / target.entry_price) - 1.0
        net_return = gross_return - cost
        closed = replace(
            target,
            exit_date=exit_date,
            exit_price=float(exit_price),
            net_return=float(net_return),
            closed_reason=reason,
        )
        new_positions = tuple(closed if pos.position_id == position_id else pos for pos in self.positions)
        return PositionTracker(positions=new_positions)

    def mark_to_market(self, prices: dict[str, float]) -> dict[str, float]:
        """Return unrealised PnL per open position given current ``prices``."""
        pnl: dict[str, float] = {}
        for pos in self.open_positions():
            current = prices.get(pos.symbol)
            if current is None:
                continue
            pnl[pos.position_id] = float(current) / pos.entry_price - 1.0
        return pnl

    # ── persistence ────────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "positions": self.to_records(),
        }
        # Atomic write: tmp file + rename (POSIX/Windows safe).
        fd, tmp_name = tempfile.mkstemp(
            prefix=out_path.name + ".",
            suffix=".tmp",
            dir=str(out_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            os.replace(tmp_name, out_path)
        except Exception:
            if Path(tmp_name).exists():
                Path(tmp_name).unlink()
            raise
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "PositionTracker":
        in_path = Path(path)
        if not in_path.exists():
            return cls.empty()
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported tracker schema version: {version}")
        return cls.from_records(payload.get("positions", []))
