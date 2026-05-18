"""Daily drift monitor — KL divergence between training and recent states.

For each symbol in the configured universe, loads processed bars (with a
``state`` column), splits into a training window (older bars) and a recent
window (most recent ``--recent`` bars), and runs :func:`src.drift.drift_alert`.
Writes ``reports/tables/drift_status.csv`` and prints a one-line summary.

Intended to be wired into the same cron that pulls today's bar via FMP and
re-runs the state labeller — but this script itself is pure file I/O.

Usage
-----
    python scripts/run_drift_monitor.py
    python scripts/run_drift_monitor.py --provider fmp --recent 60 --threshold 0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_SYMBOLS, FIRST_EXPERIMENT_SYMBOLS, N_STATES, TABLES_DIR
from src.data import load_processed
from src.drift import DEFAULT_KL_THRESHOLD, drift_alert


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--full-universe", action="store_true")
    parser.add_argument("--recent", type=int, default=60,
                        help="Number of most-recent bars to treat as the 'current' window.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_KL_THRESHOLD)
    parser.add_argument("--output", default="drift_status.csv")
    return parser.parse_args()


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return args.symbols
    if args.full_universe:
        return DEFAULT_SYMBOLS
    return FIRST_EXPERIMENT_SYMBOLS


def main() -> None:
    args = _parse_args()
    rows = []
    for symbol in _resolve_symbols(args):
        frame = load_processed(symbol, provider=args.provider)
        states = frame["state"].dropna()
        if states.size <= args.recent:
            rows.append({
                "symbol": symbol,
                "provider": args.provider or "default",
                "kl_divergence": float("nan"),
                "threshold": args.threshold,
                "alert": False,
                "n_training": 0,
                "n_current": int(states.size),
                "note": "insufficient_history",
            })
            continue
        training = states.iloc[: -args.recent]
        current = states.iloc[-args.recent :]
        report = drift_alert(training, current, n_states=N_STATES, threshold=args.threshold)
        rows.append({
            "symbol": symbol,
            "provider": args.provider or "default",
            "kl_divergence": report["kl_divergence"],
            "threshold": report["threshold"],
            "alert": report["alert"],
            "n_training": report["n_training"],
            "n_current": report["n_current"],
            "note": "ok",
        })

    out_path = Path(TABLES_DIR) / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    n_alerts = sum(1 for r in rows if r["alert"])
    print(f"Wrote drift status for {len(rows)} symbols ({n_alerts} alerts) to {out_path}")


if __name__ == "__main__":
    main()
