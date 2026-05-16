"""Download and inspect OHLCV data for the first experiment universe.

Run from the repository root:

    python notebooks/01_data_download.py
"""

from src.config import DEFAULT_END, DEFAULT_START, FIRST_EXPERIMENT_SYMBOLS
from src.data import download_ohlcv, missing_data_report, save_processed, save_raw


def main() -> None:
    data = download_ohlcv(FIRST_EXPERIMENT_SYMBOLS, DEFAULT_START, DEFAULT_END)
    save_raw(data)
    save_processed(data)

    report = missing_data_report(data)
    print(report)


if __name__ == "__main__":
    main()
