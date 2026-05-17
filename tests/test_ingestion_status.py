"""Tests for ingestion status persistence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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
    assert loaded == {
        "run_id": "run-1",
        "provider": "yfinance",
        "start": "2024-01-01",
        "end": "2024-01-05",
        "status": "partial",
        "started_at": "2024-01-05T10:00:00Z",
        "finished_at": "2024-01-05T10:00:03Z",
        "symbols": [
            {
                "symbol": "SPY",
                "status": "success",
                "rows": 3,
                "first_date": "2024-01-02",
                "last_date": "2024-01-04",
                "total_missing": 0,
                "error": None,
            },
            {
                "symbol": "BAD",
                "status": "error",
                "rows": 0,
                "first_date": None,
                "last_date": None,
                "total_missing": 0,
                "error": "No OHLCV rows returned",
            },
        ],
    }


def test_ingestion_run_normalizes_symbol_lists_to_tuple() -> None:
    run = IngestionRun(
        run_id="run-1",
        provider="yfinance",
        start="2024-01-01",
        end="2024-01-05",
        status="running",
        started_at="2024-01-05T10:00:00Z",
        symbols=[
            IngestionSymbolStatus(symbol="SPY", status="pending"),
        ],
    )

    assert run.symbols == (IngestionSymbolStatus(symbol="SPY", status="pending"),)


def test_save_ingestion_status_overwrite_cleans_temp_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ingestion_status.json"
    original = IngestionRun(
        run_id="run-1",
        provider="yfinance",
        start="2024-01-01",
        end="2024-01-05",
        status="success",
        started_at="2024-01-05T10:00:00Z",
    )
    replacement = IngestionRun(
        run_id="run-2",
        provider="yfinance",
        start="2024-01-08",
        end="2024-01-12",
        status="failed",
        started_at="2024-01-12T10:00:00Z",
    )
    save_ingestion_status(original, path)

    def fail_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        raise OSError(f"cannot replace {source} -> {destination}")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError):
        save_ingestion_status(replacement, path)

    loaded = load_ingestion_status(path)
    assert loaded is not None
    assert loaded["run_id"] == "run-1"
    assert list(tmp_path.glob(".ingestion_status.json.*.tmp")) == []
    assert list(tmp_path.glob(".ingestion_status.json.tmp")) == []


def test_load_ingestion_status_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_ingestion_status(tmp_path / "missing.json") is None
