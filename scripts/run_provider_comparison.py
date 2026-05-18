"""Compare walk-forward backtest results across data providers.

Reads ``walkforward_backtest_summary_<provider>.csv`` for each requested
provider, joins by symbol, and writes ``provider_comparison.csv`` with the
per-symbol Sharpe / hit-rate / max-drawdown delta. Run after invoking
``scripts/run_walkforward_backtest.py --provider <name>`` for each provider.

Usage
-----
    python scripts/run_provider_comparison.py --providers fmp yfinance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TABLES_DIR


KEY_METRICS = ["sharpe", "total_return", "max_drawdown", "win_rate", "trade_count"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        nargs="+",
        required=True,
        help="Two or more provider names whose walkforward_backtest_summary_<name>.csv files exist.",
    )
    return parser.parse_args()


def _load_provider_summary(provider: str, tables_dir: Path) -> pd.DataFrame:
    path = tables_dir / f"walkforward_backtest_summary_{provider.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/run_walkforward_backtest.py --provider {provider}"
        )
    frame = pd.read_csv(path)
    frame["provider"] = provider.lower()
    return frame


def main() -> None:
    args = _parse_args()
    if len(args.providers) < 2:
        raise SystemExit("Need at least two providers to compare.")

    tables_dir = Path(TABLES_DIR)
    summaries = [_load_provider_summary(p, tables_dir) for p in args.providers]
    long_frame = pd.concat(summaries, ignore_index=True)
    columns = ["symbol", "provider", *KEY_METRICS]
    long_frame = long_frame[[c for c in columns if c in long_frame.columns]]

    base, *rest = args.providers
    pivot = long_frame.pivot_table(index="symbol", columns="provider", values=KEY_METRICS)
    pivot.columns = [f"{metric}__{provider}" for metric, provider in pivot.columns]
    pivot = pivot.reset_index()

    for other in rest:
        for metric in KEY_METRICS:
            base_col = f"{metric}__{base.lower()}"
            other_col = f"{metric}__{other.lower()}"
            if base_col in pivot.columns and other_col in pivot.columns:
                pivot[f"{metric}__delta_{other.lower()}_minus_{base.lower()}"] = (
                    pivot[other_col] - pivot[base_col]
                )

    output_path = tables_dir / "provider_comparison.csv"
    pivot.to_csv(output_path, index=False)
    print(f"Wrote provider comparison ({len(pivot)} symbols) to {output_path}")


if __name__ == "__main__":
    main()
