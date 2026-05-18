"""Run the walk-forward EV backtest across a configurable ETF universe.

Examples
--------
    python scripts/run_walkforward_backtest.py
    python scripts/run_walkforward_backtest.py --provider fmp
    python scripts/run_walkforward_backtest.py --symbols SPY QQQ --provider yfinance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtests import compare_backtest_to_baselines, run_walkforward_ev_backtest
from src.config import DEFAULT_COST_BPS, DEFAULT_SYMBOLS, FIRST_EXPERIMENT_SYMBOLS, RUNS_DIR, TABLES_DIR
from src.data import load_processed
from src.reports import generate_report


HORIZON = 5
LOOKBACK = 252
MIN_SAMPLES = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional data provider folder under data/processed (e.g. 'fmp', 'yfinance').",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Override the symbol universe (defaults to FIRST_EXPERIMENT_SYMBOLS).",
    )
    parser.add_argument(
        "--full-universe",
        action="store_true",
        help="Use DEFAULT_SYMBOLS (15 tickers) instead of FIRST_EXPERIMENT_SYMBOLS (9).",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Optional suffix appended to summary table filenames (e.g. '_fmp').",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.symbols:
        symbols = args.symbols
    elif args.full_universe:
        symbols = DEFAULT_SYMBOLS
    else:
        symbols = FIRST_EXPERIMENT_SYMBOLS
    suffix = args.output_suffix
    if suffix is None and args.provider is not None:
        suffix = f"_{args.provider.lower()}"
    suffix = suffix or ""

    rows = []
    comparison_tables = []

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
        stats = result["stats"]
        rows.append(
            {
                "symbol": symbol,
                "provider": args.provider or "default",
                "horizon": HORIZON,
                "lookback": LOOKBACK,
                "min_samples": MIN_SAMPLES,
                "total_return": stats["total_return"],
                "max_drawdown": stats["max_drawdown"],
                "sharpe": stats["sharpe"],
                "trade_count": stats["trade_count"],
                "win_rate": stats["win_rate"],
                "exposure_time": stats["exposure_time"],
                "benchmark_total_return": result["benchmark"]["total_return"],
            }
        )

        comparison = compare_backtest_to_baselines(
            frame,
            result,
            breakout_lookback=126,
            horizon=HORIZON,
            cost_bps=DEFAULT_COST_BPS,
        )
        comparison.insert(0, "symbol", symbol)
        comparison_tables.append(comparison)

    summary = pd.DataFrame(rows).sort_values("total_return", ascending=False)
    output_path = Path(TABLES_DIR) / f"walkforward_backtest_summary{suffix}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    comparison_output_path = Path(TABLES_DIR) / f"walkforward_baseline_comparison{suffix}.csv"
    pd.concat(comparison_tables, ignore_index=True).to_csv(comparison_output_path, index=False)

    generate_report(
        f"walkforward_backtest_prototype{suffix}",
        {
            "question": "Do state EV filters still work when estimated only from prior realized outcomes?",
            "hypothesis": "Walk-forward EV should reduce lookahead bias and reveal whether state payoff persistence exists.",
            "data_description": (
                f"Processed universe ({args.provider or 'default'} provider): {', '.join(symbols)}."
            ),
            "state_definitions": "Deterministic support/resistance state labels from src.states.",
            "model_setup": f"EV is re-estimated per bar using a {LOOKBACK}-bar lookback and at least {MIN_SAMPLES} prior samples.",
            "backtest_rules": f"Long-only, next-bar entry, fixed {HORIZON}-bar exit, non-overlapping trades, DEFAULT_COST_BPS included.",
            "results": f"Saved walk-forward summary table to {output_path}.",
            "benchmark_comparison": f"Saved walk-forward baseline comparison table to {comparison_output_path}.",
            "what_worked": "The prototype can now run without full-sample EV selection.",
            "what_failed": "This is still a simple current-state EV filter, not Markov-probability-weighted EV.",
            "bias_risk_checks": "EV estimates use only outcomes completed before the signal date.",
            "next_experiment": "Use behavior clusters to estimate family-level EV for sparse states.",
        },
        output_dir=RUNS_DIR,
    )

    print(f"Saved {len(summary)} symbol rows to {output_path}")


if __name__ == "__main__":
    main()
