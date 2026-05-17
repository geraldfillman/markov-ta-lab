"""Tests for ingestion status persistence."""

from __future__ import annotations

from pathlib import Path

from src.ingestion_status import (
    IngestionRun,
    IngestionSymbolStatus,
    load_ingestion_status,
    save_ingestion_status,
)


def test_ingestion_status_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ingestion_status.json"
    run = IngestionRun(
        run_id="run-1",
        provider="yfinance",
        start="2024-01-01",
        end="2024-01-05",
        status="partial",
        started_at="2024-01-05T10:00:00Z",
        finished_at="2024-01-05T10:00:03Z",
        symbols=[
            IngestionSymbolStatus(
                symbol="SPY",
                status="success",
                rows=3,
                first_date="2024-01-02",
                last_date="2024-01-04",
                total_missing=0,
            ),
            IngestionSymbolStatus(
                symbol="BAD",
                status="error",
                rows=0,
                error="No OHLCV rows returned",
            ),
        ],
    )

    save_ingestion_status(run, path)

    loaded = load_ingestion_status(path)
    assert loaded is not None
    assert loaded["status"] == "partial"
    assert loaded["provider"] == "yfinance"
    assert loaded["symbols"][0]["symbol"] == "SPY"
    assert loaded["symbols"][1]["error"] == "No OHLCV rows returned"


def test_load_ingestion_status_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_ingestion_status(tmp_path / "missing.json") is None
