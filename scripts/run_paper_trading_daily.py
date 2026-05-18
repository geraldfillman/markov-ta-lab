"""Daily paper-trading step — loads tracker, runs one step per symbol, persists.

Pipeline per symbol:
1. Load the processed parquet (must already have today's bar appended).
2. Look up today's drift status from ``reports/tables/drift_status.csv``.
3. Call :func:`src.paper_trading.paper_trading_step` with the configured
   gates.
4. Append a row to ``reports/tables/paper_trading_decisions.csv``.

Persistence
-----------
The tracker lives at ``reports/paper_trading/tracker.json`` (atomic write).
This script is **research only** — it does not place orders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_COST_BPS, DEFAULT_SYMBOLS, FIRST_EXPERIMENT_SYMBOLS, REPORTS_DIR, TABLES_DIR
from src.data import load_processed
from src.paper_trading import paper_trading_step
from src.positions import PositionTracker


HORIZON = 5
LOOKBACK = 252
MIN_SAMPLES = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--full-universe", action="store_true")
    parser.add_argument(
        "--tracker-path",
        default=str(Path(REPORTS_DIR) / "paper_trading" / "tracker.json"),
    )
    parser.add_argument(
        "--decisions-output",
        default=str(Path(TABLES_DIR) / "paper_trading_decisions.csv"),
    )
    parser.add_argument(
        "--drift-status-path",
        default=str(Path(TABLES_DIR) / "drift_status.csv"),
        help="Where to read per-symbol drift alerts from (optional).",
    )
    parser.add_argument(
        "--today",
        default=None,
        help="Override today's date (ISO YYYY-MM-DD). Defaults to the last bar in each frame.",
    )
    return parser.parse_args()


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return args.symbols
    if args.full_universe:
        return DEFAULT_SYMBOLS
    return FIRST_EXPERIMENT_SYMBOLS


def _load_drift_map(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "symbol" not in frame.columns or "alert" not in frame.columns:
        return {}
    return {str(r.symbol): bool(r.alert) for r in frame.itertuples(index=False)}


def main() -> None:
    args = _parse_args()
    tracker = PositionTracker.load(args.tracker_path)
    drift_map = _load_drift_map(Path(args.drift_status_path))

    decision_rows: list[dict] = []
    for symbol in _resolve_symbols(args):
        frame = load_processed(symbol, provider=args.provider)
        if frame.empty:
            continue
        today = pd.Timestamp(args.today) if args.today else frame.index[-1]
        if today not in frame.index:
            continue
        tracker, decisions = paper_trading_step(
            symbol=symbol,
            frame=frame,
            tracker=tracker,
            today=today,
            horizon=HORIZON,
            lookback=LOOKBACK,
            min_samples=MIN_SAMPLES,
            cost_bps=DEFAULT_COST_BPS,
            drift_blocked=bool(drift_map.get(symbol, False)),
        )
        for decision in decisions:
            decision_rows.append({
                "run_date": str(today.date()),
                "symbol": symbol,
                **decision,
            })

    tracker_path = Path(args.tracker_path)
    tracker.save(tracker_path)

    decisions_path = Path(args.decisions_output)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame(decision_rows)
    if decisions_path.exists():
        existing = pd.read_csv(decisions_path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(decisions_path, index=False)
    print(
        f"Tracker now has {len(tracker.open_positions())} open / {len(tracker.closed_positions())} closed; "
        f"appended {len(new_rows)} decision rows to {decisions_path}"
    )


if __name__ == "__main__":
    main()
