"""Build state expectancy pooled by asset behavior cluster."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering import cluster_assets
from src.config import DEFAULT_COST_BPS, FIRST_EXPERIMENT_SYMBOLS, FORECAST_HORIZONS, TABLES_DIR
from src.data import load_processed
from src.metrics import cluster_pooled_state_expectancy_table


def main() -> None:
    data = {symbol: load_processed(symbol) for symbol in FIRST_EXPERIMENT_SYMBOLS}
    clusters = cluster_assets(data, n_clusters=3)
    table = cluster_pooled_state_expectancy_table(
        data,
        clusters,
        horizons=tuple(FORECAST_HORIZONS),
        cost_bps=DEFAULT_COST_BPS,
    )
    output_path = Path(TABLES_DIR) / "cluster_pooled_state_expectancy.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path)
    print(f"Saved {len(table)} rows to {output_path}")


if __name__ == "__main__":
    main()
