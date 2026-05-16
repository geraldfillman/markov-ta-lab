"""Data Agent – clean OHLCV and macro data pipelines.

Responsibilities (from playbook §3.2):
- Pull daily OHLCV data (yfinance initially).
- Standardize columns: Open, High, Low, Close, Volume.
- Handle missing data.
- Save raw CSV and processed Parquet files.
- Support single ticker or list of tickers.
- Ensure timestamp alignment across assets.
"""

import pandas as pd


def download_ohlcv(
    symbols: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Download and return clean OHLCV data keyed by symbol.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to download.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of symbol → DataFrame with columns
        [Open, High, Low, Close, Volume] indexed by Date.
    """
    raise NotImplementedError("Implement in Data Agent phase")


def save_raw(data: dict[str, pd.DataFrame], output_dir: str = "data/raw") -> None:
    """Save raw OHLCV data as CSV files."""
    raise NotImplementedError


def save_processed(data: dict[str, pd.DataFrame], output_dir: str = "data/processed") -> None:
    """Save cleaned OHLCV data as Parquet files."""
    raise NotImplementedError


def load_processed(symbol: str, data_dir: str = "data/processed") -> pd.DataFrame:
    """Load a previously saved processed Parquet file."""
    raise NotImplementedError


def missing_data_report(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a summary of missing values per symbol."""
    raise NotImplementedError
