"""Run the narrow long-only backtest across the first ETF universe.

Run from the repository root:

    python scripts/run_narrow_backtest.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtests import compare_backtest_to_baselines, run_backtest_readable
from src.config import DEFAULT_COST_BPS, FIRST_EXPERIMENT_SYMBOLS, RUNS_DIR, TABLES_DIR
from src.data import load_processed
from src.reports import generate_report


HORIZON = 5


def _load_expectancy(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    return table.set_index(["symbol", "state"]).sort_index()


def main() -> None:
    expectancy_path = Path(TABLES_DIR) / "state_expectancy.csv"
    expectancy = _load_expectancy(expectancy_path)
    rows = []
    comparison_tables = []

    for symbol in FIRST_EXPERIMENT_SYMBOLS:
        frame = load_processed(symbol)
        symbol_ev = expectancy.xs(symbol, level="symbol")
        result = run_backtest_readable(
            frame,
            frame["state"],
            symbol_ev,
            horizon=HORIZON,
            cost_bps=DEFAULT_COST_BPS,
        )
        stats = result["stats"]
        rows.append(
            {
                "symbol": symbol,
                "horizon": HORIZON,
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
    output_path = Path(TABLES_DIR) / "narrow_backtest_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    comparison_output_path = Path(TABLES_DIR) / "baseline_comparison.csv"
    pd.concat(comparison_tables, ignore_index=True).to_csv(comparison_output_path, index=False)

    generate_report(
        "narrow_backtest_prototype",
        {
            "question": "Do positive state expectancy filters produce useful long-only ETF trades?",
            "hypothesis": "States with positive historical 5-bar EV should produce better fixed-horizon trades after costs.",
            "data_description": f"Processed first experiment ETF universe: {', '.join(FIRST_EXPERIMENT_SYMBOLS)}.",
            "state_definitions": "Deterministic support/resistance state labels from src.states.",
            "model_setup": "Use state_expectancy.csv as the entry filter; enter when the current state's 5-bar EV is positive.",
            "backtest_rules": f"Long-only, next-bar entry, fixed {HORIZON}-bar exit, non-overlapping trades, DEFAULT_COST_BPS included.",
            "results": f"Saved summary table to {output_path}.",
            "benchmark_comparison": f"Saved baseline comparison table to {comparison_output_path}.",
            "what_worked": "The prototype now connects deterministic states, expected value tables, and readable backtest output.",
            "what_failed": "This is not yet walk-forward; the expectancy filter is estimated from the full sample.",
            "bias_risk_checks": "Known lookahead risk remains in full-sample expectancy selection. Next phase should make the EV filter walk-forward.",
            "next_experiment": "Replace full-sample EV lookup with walk-forward EV estimates and compare against simple breakout and MA baselines.",
        },
        output_dir=RUNS_DIR,
    )

    print(f"Saved {len(summary)} symbol rows to {output_path}")


if __name__ == "__main__":
    main()
