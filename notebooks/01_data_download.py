"""Download and inspect OHLCV data for the first experiment universe.

Run from the repository root:

    python notebooks/01_data_download.py
"""

from src.config import DEFAULT_END, DEFAULT_START, FIRST_EXPERIMENT_SYMBOLS
from src.data import download_ohlcv, missing_data_report, save_processed, save_raw
from src.indicators import add_indicators
from src.levels import detect_levels


def main() -> None:
    data = download_ohlcv(FIRST_EXPERIMENT_SYMBOLS, DEFAULT_START, DEFAULT_END)
    enriched = {
        symbol: detect_levels(add_indicators(frame))
        for symbol, frame in data.items()
    }

    save_raw(data)
    save_processed(enriched)

    report = missing_data_report(data)
    print(report)


if __name__ == "__main__":
    main()
