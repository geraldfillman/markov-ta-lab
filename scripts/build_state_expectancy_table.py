"""Build the first state expectancy CSV from processed market data.

Run from the repository root:

    python scripts/build_state_expectancy_table.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_COST_BPS, FIRST_EXPERIMENT_SYMBOLS, FORECAST_HORIZONS, TABLES_DIR
from src.data import load_processed
from src.metrics import save_state_expectancy_table, universe_state_expectancy_table


def main() -> None:
    data = {
        symbol: load_processed(symbol)
        for symbol in FIRST_EXPERIMENT_SYMBOLS
    }
    table = universe_state_expectancy_table(
        data,
        horizons=tuple(FORECAST_HORIZONS),
        cost_bps=DEFAULT_COST_BPS,
    )
    output_path = save_state_expectancy_table(table, Path(TABLES_DIR) / "state_expectancy.csv")
    print(f"Saved {len(table)} rows to {output_path}")


if __name__ == "__main__":
    main()
