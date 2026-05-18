"""Run a compact walk-forward sensitivity grid across the ETF universe."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtests import run_walkforward_sensitivity
from src.config import FIRST_EXPERIMENT_SYMBOLS, TABLES_DIR
from src.data import load_processed


def main() -> None:
    rows = []
    for symbol in FIRST_EXPERIMENT_SYMBOLS:
        frame = load_processed(symbol)
        table = run_walkforward_sensitivity(
            frame,
            frame["state"],
            horizons=(5, 10),
            lookbacks=(126, 252),
            costs_bps=(5.0, 10.0),
            min_samples_values=(5, 10),
        )
        table.insert(0, "symbol", symbol)
        rows.append(table)

    result = pd.concat(rows, ignore_index=True)
    output_path = Path(TABLES_DIR) / "walkforward_sensitivity.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()
