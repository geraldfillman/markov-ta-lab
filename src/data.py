"""Data Agent - clean OHLCV data pipeline."""

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import pandas as pd


REQUIRED_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
Downloader: TypeAlias = Callable[[str, str, str], pd.DataFrame]


def download_ohlcv(
    symbols: str | list[str],
    start: str,
    end: str,
    downloader: Downloader | None = None,
) -> dict[str, pd.DataFrame]:
    """Download and return clean OHLCV data keyed by symbol."""
    symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
    fetch = downloader or _download_with_yfinance

    data: dict[str, pd.DataFrame] = {}
    for symbol in symbol_list:
        clean_symbol = symbol.upper().strip()
        frame = fetch(clean_symbol, start, end)
        data[clean_symbol] = clean_ohlcv_frame(frame, clean_symbol)

    return data


def clean_ohlcv_frame(frame: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Normalize a vendor OHLCV frame to sorted Date-indexed OHLCV columns."""
    if frame.empty:
        name = f" for {symbol}" if symbol else ""
        raise ValueError(f"No OHLCV rows returned{name}")

    normalized = _flatten_yfinance_columns(frame, symbol)
    missing_columns = [column for column in REQUIRED_OHLCV_COLUMNS if column not in normalized.columns]
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

    result = normalized.loc[:, REQUIRED_OHLCV_COLUMNS].copy()
    result.index = pd.to_datetime(result.index)
    result.index.name = "Date"
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result = result.dropna(subset=REQUIRED_OHLCV_COLUMNS)

    return result


def save_raw(data: dict[str, pd.DataFrame], output_dir: str | Path = "data/raw") -> None:
    """Save raw OHLCV data as CSV files."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    for symbol, frame in data.items():
        frame.to_csv(directory / f"{symbol.upper()}.csv")


def save_processed(data: dict[str, pd.DataFrame], output_dir: str | Path = "data/processed") -> None:
    """Save cleaned OHLCV data as Parquet files."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    for symbol, frame in data.items():
        frame.to_parquet(directory / f"{symbol.upper()}.parquet")


def load_processed(symbol: str, data_dir: str | Path = "data/processed") -> pd.DataFrame:
    """Load a previously saved processed Parquet file."""
    path = Path(data_dir) / f"{symbol.upper()}.parquet"
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    return frame


def missing_data_report(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a summary of missing values per symbol."""
    rows = []
    for symbol, frame in data.items():
        missing_by_column = frame.isna().sum()
        record = {
            "symbol": symbol.upper(),
            "rows": len(frame),
            "columns": len(frame.columns),
            "total_missing": int(missing_by_column.sum()),
        }
        for column, count in missing_by_column.items():
            record[f"missing_{column}"] = int(count)
        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=["rows", "columns", "total_missing"]).rename_axis("symbol")

    return pd.DataFrame(rows).set_index("symbol").sort_index()


def _download_with_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )


def _flatten_yfinance_columns(frame: pd.DataFrame, symbol: str | None) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame.copy()

    columns = frame.columns
    if symbol and symbol in columns.get_level_values(-1):
        return frame.xs(symbol, level=-1, axis=1).copy()

    if symbol and symbol in columns.get_level_values(0):
        return frame.xs(symbol, level=0, axis=1).copy()

    if len(columns.levels[-1]) == 1:
        return frame.droplevel(-1, axis=1).copy()

    if len(columns.levels[0]) == 1:
        return frame.droplevel(0, axis=1).copy()

    raise ValueError("Cannot normalize multi-symbol OHLCV frame for a single symbol")
