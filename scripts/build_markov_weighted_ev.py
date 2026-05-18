"""Build walk-forward Markov probability weighted EV tables."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_COST_BPS, FIRST_EXPERIMENT_SYMBOLS, N_STATES, TABLES_DIR
from src.data import load_processed
from src.metrics import walkforward_markov_expected_value


HORIZON = 5
LOOKBACK = 252
MIN_SAMPLES = 10


def main() -> None:
    tables = []
    for symbol in FIRST_EXPERIMENT_SYMBOLS:
        frame = load_processed(symbol)
        table = walkforward_markov_expected_value(
            frame,
            frame["state"],
            n_states=N_STATES,
            horizon=HORIZON,
            lookback=LOOKBACK,
            min_samples=MIN_SAMPLES,
            cost_bps=DEFAULT_COST_BPS,
            alpha=1e-6,
        )
        table.insert(0, "symbol", symbol)
        tables.append(table.reset_index())

    result = pd.concat(tables, ignore_index=True)
    output_path = Path(TABLES_DIR) / "markov_weighted_ev.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()
