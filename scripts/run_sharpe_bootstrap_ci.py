"""Block-bootstrap Sharpe CI per symbol, wired into the reports pipeline.

For each symbol in the (configurable) universe, re-runs the walk-forward EV
backtest, collects the per-trade ``net_return`` series, and writes
``reports/tables/walkforward_sharpe_ci.csv`` with point Sharpe + bootstrap CI.

The block size defaults to ``horizon`` since trades are non-overlapping at
horizon spacing — using ``block_size == horizon`` preserves whatever residual
serial structure exists between adjacent (non-overlapping) trades.

Usage
-----
    python scripts/run_sharpe_bootstrap_ci.py
    python scripts/run_sharpe_bootstrap_ci.py --provider fmp --resamples 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtests import run_walkforward_ev_backtest
from src.config import DEFAULT_COST_BPS, FIRST_EXPERIMENT_SYMBOLS, TABLES_DIR
from src.data import load_processed
from src.metrics import bootstrap_sharpe_ci_from_trades


HORIZON = 5
LOOKBACK = 252
MIN_SAMPLES = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--resamples", type=int, default=2_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--output-suffix", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = args.symbols or FIRST_EXPERIMENT_SYMBOLS

    suffix = args.output_suffix
    if suffix is None and args.provider is not None:
        suffix = f"_{args.provider.lower()}"
    suffix = suffix or ""

    rows = []
    for symbol in symbols:
        frame = load_processed(symbol, provider=args.provider)
        result = run_walkforward_ev_backtest(
            frame,
            frame["state"],
            horizon=HORIZON,
            lookback=LOOKBACK,
            min_samples=MIN_SAMPLES,
            cost_bps=DEFAULT_COST_BPS,
        )
        trades = result["trades"]
        if trades.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "provider": args.provider or "default",
                    "n_trades": 0,
                    "sharpe_point": float("nan"),
                    "sharpe_ci_low": float("nan"),
                    "sharpe_ci_high": float("nan"),
                    "confidence": args.confidence,
                    "block_size": HORIZON,
                    "resamples": args.resamples,
                }
            )
            continue

        ci = bootstrap_sharpe_ci_from_trades(
            trades["net_return"].to_numpy(),
            confidence=args.confidence,
            n_resamples=args.resamples,
            block_size=HORIZON,
            avg_holding_period=HORIZON,
            random_state=args.seed,
        )
        rows.append(
            {
                "symbol": symbol,
                "provider": args.provider or "default",
                "n_trades": int(ci["n"]),
                "sharpe_point": ci["sharpe_point"],
                "sharpe_ci_low": ci["ci_low"],
                "sharpe_ci_high": ci["ci_high"],
                "confidence": args.confidence,
                "block_size": HORIZON,
                "resamples": args.resamples,
            }
        )

    summary = pd.DataFrame(rows)
    tables_dir = Path(TABLES_DIR)
    tables_dir.mkdir(parents=True, exist_ok=True)
    output = tables_dir / f"walkforward_sharpe_ci{suffix}.csv"
    summary.to_csv(output, index=False)
    print(f"Wrote {len(summary)} bootstrap Sharpe CIs to {output}")


if __name__ == "__main__":
    main()
