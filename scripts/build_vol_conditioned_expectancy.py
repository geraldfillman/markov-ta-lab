"""Build state expectancy tables conditioned on volatility regime.

Run from the repository root:

    python scripts/build_vol_conditioned_expectancy.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_COST_BPS, FIRST_EXPERIMENT_SYMBOLS, FORECAST_HORIZONS, TABLES_DIR
from src.data import load_processed
from src.metrics import conditioned_state_expectancy_table
from src.volatility import classify_vol_state


def main() -> None:
    tables = []
    for symbol in FIRST_EXPERIMENT_SYMBOLS:
        frame = load_processed(symbol)
        vol_state = classify_vol_state(frame)
        table = conditioned_state_expectancy_table(
            frame,
            frame["state"],
            vol_state,
            horizons=tuple(FORECAST_HORIZONS),
            cost_bps=DEFAULT_COST_BPS,
        )
        if table.empty:
            continue
        table.insert(0, "symbol", symbol)
        table = table.set_index("symbol", append=True).reorder_levels(["symbol", "vol_state", "state"])
        tables.append(table)

    output_path = Path(TABLES_DIR) / "vol_conditioned_state_expectancy.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.concat(tables).sort_index() if tables else pd.DataFrame()
    result.to_csv(output_path)
    print(f"Saved {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()
