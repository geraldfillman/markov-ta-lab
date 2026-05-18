"""Build structural behavior clusters for the first ETF universe.

Run from the repository root:

    python scripts/build_asset_clusters.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering import cluster_assets
from src.config import FIRST_EXPERIMENT_SYMBOLS, TABLES_DIR
from src.data import load_processed


def main() -> None:
    data = {
        symbol: load_processed(symbol)
        for symbol in FIRST_EXPERIMENT_SYMBOLS
    }
    clusters = cluster_assets(data, n_clusters=3)
    output_path = Path(TABLES_DIR) / "asset_behavior_clusters.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(output_path)
    print(f"Saved {len(clusters)} asset rows to {output_path}")


if __name__ == "__main__":
    main()
