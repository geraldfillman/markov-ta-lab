"""Random-label permutation baseline (playbook robustness check).

Shuffles state labels per asset, re-runs the walk-forward EV backtest, and
appends the result to ``reports/tables/random_label_baseline.csv``. Also
merges the rows into ``baseline_comparison.csv`` so the dashboard's baseline
section can include the new row alongside ``state_ev_strategy``, ``buy_and_hold``,
``ma_crossover`` and ``breakout``.

Usage
-----
    python scripts/run_random_label_baseline.py
    python scripts/run_random_label_baseline.py --provider fmp --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtests import baseline_random_label_walkforward
from src.config import DEFAULT_COST_BPS, FIRST_EXPERIMENT_SYMBOLS, TABLES_DIR
from src.data import load_processed


HORIZON = 5
LOOKBACK = 252
MIN_SAMPLES = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=20260517)
    return parser.parse_args()


def _row_from_stats(symbol: str, stats: dict) -> dict:
    return {
        "symbol": symbol,
        "model": "random_label",
        "total_return": stats.get("total_return"),
        "max_drawdown": stats.get("max_drawdown"),
        "sharpe": stats.get("sharpe"),
        "trade_count": stats.get("trade_count"),
        "win_rate": stats.get("win_rate"),
        "avg_win": stats.get("avg_win"),
        "avg_loss": stats.get("avg_loss"),
        "exposure_time": stats.get("exposure_time"),
        "bars": stats.get("bars"),
    }


def _merge_into_baseline_comparison(new_rows: pd.DataFrame, tables_dir: Path) -> Path:
    baseline_path = tables_dir / "baseline_comparison.csv"
    if baseline_path.exists():
        existing = pd.read_csv(baseline_path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["symbol", "model"], keep="last")
    else:
        combined = new_rows.copy()

    # Recompute excess_vs_buy_hold for the new rows when buy_and_hold exists.
    if "excess_vs_buy_hold" not in combined.columns:
        combined["excess_vs_buy_hold"] = pd.NA
    bh_returns = (
        combined[combined["model"] == "buy_and_hold"].set_index("symbol")["total_return"].to_dict()
    )
    missing_excess = combined["excess_vs_buy_hold"].isna()
    combined.loc[missing_excess, "excess_vs_buy_hold"] = combined.loc[missing_excess].apply(
        lambda r: (r["total_return"] - bh_returns[r["symbol"]])
        if r["symbol"] in bh_returns and pd.notna(r["total_return"])
        else pd.NA,
        axis=1,
    )
    combined.to_csv(baseline_path, index=False)
    return baseline_path


def main() -> None:
    args = _parse_args()
    symbols = args.symbols or FIRST_EXPERIMENT_SYMBOLS

    rows = []
    for symbol in symbols:
        frame = load_processed(symbol, provider=args.provider)
        result = baseline_random_label_walkforward(
            frame,
            frame["state"],
            horizon=HORIZON,
            lookback=LOOKBACK,
            min_samples=MIN_SAMPLES,
            cost_bps=DEFAULT_COST_BPS,
            seed=args.seed,
        )
        rows.append(_row_from_stats(symbol, result["stats"]))

    tables_dir = Path(TABLES_DIR)
    tables_dir.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame(rows)
    new_rows.to_csv(tables_dir / "random_label_baseline.csv", index=False)
    merged_path = _merge_into_baseline_comparison(new_rows, tables_dir)
    print(f"Wrote {len(new_rows)} random-label rows; merged into {merged_path}")


if __name__ == "__main__":
    main()
