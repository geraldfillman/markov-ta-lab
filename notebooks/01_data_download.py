"""Download and inspect OHLCV data for the first experiment universe.

Run from the repository root:

    python notebooks/01_data_download.py
    python notebooks/01_data_download.py --provider fmp
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_END, DEFAULT_START, FIRST_EXPERIMENT_SYMBOLS
from src.data import download_ohlcv, missing_data_report, save_processed, save_raw
from src.indicators import add_indicators
from src.levels import detect_levels
from src.states import label_states


def enrich_market_data(frame):
    enriched = detect_levels(add_indicators(frame))
    enriched["state"] = label_states(enriched)
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and enrich OHLCV market data.")
    parser.add_argument(
        "--provider",
        choices=["yfinance", "fmp"],
        default=os.getenv("MARKOV_DATA_PROVIDER", "yfinance"),
        help="Market data provider. Defaults to MARKOV_DATA_PROVIDER or yfinance.",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start date.")
    parser.add_argument("--end", default=DEFAULT_END, help="Exclusive/end date convention depends on provider.")
    parser.add_argument(
        "--symbols",
        default=",".join(FIRST_EXPERIMENT_SYMBOLS),
        help="Comma-separated symbols to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    data = download_ohlcv(symbols, args.start, args.end, provider=args.provider)
    enriched = {
        symbol: enrich_market_data(frame)
        for symbol, frame in data.items()
    }

    save_raw(data)
    save_processed(enriched)

    report = missing_data_report(data)
    print(report)


if __name__ == "__main__":
    main()
