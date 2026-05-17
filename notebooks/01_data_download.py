"""Download and inspect OHLCV data for the first experiment universe.

Run from the repository root:

    python notebooks/01_data_download.py
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    DEFAULT_END,
    DEFAULT_START,
    FIRST_EXPERIMENT_SYMBOLS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TABLES_DIR,
)
from src.data import download_ohlcv, missing_data_report, save_processed, save_raw
from src.indicators import add_indicators
from src.ingestion_status import IngestionRun, IngestionSymbolStatus, save_ingestion_status
from src.levels import detect_levels
from src.states import label_states


DEFAULT_STATUS_PATH = Path(TABLES_DIR) / "ingestion_status.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="yfinance")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--symbols", default=",".join(FIRST_EXPERIMENT_SYMBOLS))
    parser.add_argument("--raw-dir", default=RAW_DATA_DIR)
    parser.add_argument("--processed-dir", default=PROCESSED_DATA_DIR)
    parser.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH))
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_symbols(value: str) -> list[str]:
    return [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]


def _date_string(value) -> str:
    return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)


def _symbol_status(symbol: str, frame) -> IngestionSymbolStatus:
    if frame.empty:
        return IngestionSymbolStatus(
            symbol=symbol,
            status="error",
            rows=0,
            error="No OHLCV rows returned",
        )

    return IngestionSymbolStatus(
        symbol=symbol,
        status="success",
        rows=len(frame),
        first_date=_date_string(frame.index.min()),
        last_date=_date_string(frame.index.max()),
        total_missing=int(frame.isna().sum().sum()),
    )


def _missing_symbol_status(symbol: str) -> IngestionSymbolStatus:
    return IngestionSymbolStatus(
        symbol=symbol,
        status="error",
        rows=0,
        error=f"No data returned for requested symbol {symbol}",
    )


def _run_status(symbol_statuses: list[IngestionSymbolStatus]) -> str:
    if all(status.status == "error" for status in symbol_statuses):
        return "failed"
    return "partial" if any(status.status == "error" for status in symbol_statuses) else "success"


def _download_data(symbols: list[str], start: str, end: str, provider: str):
    if provider != "yfinance":
        raise ValueError(f"Unsupported provider: {provider}")

    data = {}
    symbol_statuses: list[IngestionSymbolStatus] = []
    for symbol in symbols:
        try:
            downloaded = download_ohlcv([symbol], start, end)
        except Exception as error:
            symbol_statuses.append(
                IngestionSymbolStatus(symbol=symbol, status="error", rows=0, error=str(error))
            )
            continue

        if symbol not in downloaded:
            symbol_statuses.append(_missing_symbol_status(symbol))
            continue

        frame = downloaded[symbol]
        symbol_status = _symbol_status(symbol, frame)
        symbol_statuses.append(symbol_status)
        if symbol_status.status == "success":
            data[symbol] = frame

    return data, symbol_statuses


def enrich_market_data(frame):
    enriched = detect_levels(add_indicators(frame))
    enriched["state"] = label_states(enriched)
    return enriched


def main() -> None:
    args = parse_args()
    symbols = _parse_symbols(args.symbols)
    status_path = Path(args.status_path)
    started_at = _now_iso()
    run_id = str(uuid4())

    def write_status(status: str, symbol_statuses=(), finished_at: str | None = None) -> None:
        save_ingestion_status(
            IngestionRun(
                run_id=run_id,
                provider=args.provider,
                start=args.start,
                end=args.end,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                symbols=tuple(symbol_statuses),
            ),
            status_path,
        )

    write_status(
        "running",
        [IngestionSymbolStatus(symbol=symbol, status="pending") for symbol in symbols],
    )

    symbol_statuses: list[IngestionSymbolStatus] = []
    try:
        data, symbol_statuses = _download_data(symbols, args.start, args.end, args.provider)
        if not data:
            raise RuntimeError("No usable OHLCV data returned")

        enriched = {
            symbol: enrich_market_data(frame)
            for symbol, frame in data.items()
        }

        save_raw(data, args.raw_dir)
        save_processed(enriched, args.processed_dir)

        report = missing_data_report(data)
        print(report)

        write_status(_run_status(symbol_statuses), symbol_statuses, finished_at=_now_iso())
    except Exception as error:
        if symbol_statuses and _run_status(symbol_statuses) == "failed":
            failed_statuses = symbol_statuses
        else:
            failed_statuses = [
                IngestionSymbolStatus(symbol=symbol, status="error", rows=0, error=str(error))
                for symbol in symbols
            ]
        write_status("failed", failed_statuses, finished_at=_now_iso())
        raise


if __name__ == "__main__":
    main()
