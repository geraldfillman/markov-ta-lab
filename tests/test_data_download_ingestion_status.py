"""Tests for data-download ingestion status output."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "notebooks" / "01_data_download.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("data_download_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    ).rename_axis("Date")


def test_data_download_writes_ingestion_status(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    status_path = tmp_path / "tables" / "ingestion_status.json"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: module.argparse.Namespace(
            provider="yfinance",
            start="2024-01-01",
            end="2024-01-05",
            symbols="SPY,QQQ",
            status_path=str(status_path),
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
        ),
    )
    monkeypatch.setattr(
        module,
        "download_ohlcv",
        lambda symbols, start, end: {symbol: _sample_frame() for symbol in symbols},
    )
    monkeypatch.setattr(module, "enrich_market_data", lambda frame: frame.assign(state=0))

    module.main()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["provider"] == "yfinance"
    assert [row["symbol"] for row in payload["symbols"]] == ["SPY", "QQQ"]
    assert payload["symbols"][0]["rows"] == 3
    assert payload["symbols"][0]["first_date"] == "2024-01-02"


def test_data_download_writes_running_pending_status_before_download(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_script_module()
    status_path = tmp_path / "tables" / "ingestion_status.json"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    writes = []

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: module.argparse.Namespace(
            provider="yfinance",
            start="2024-01-01",
            end="2024-01-05",
            symbols="SPY,QQQ",
            status_path=str(status_path),
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
        ),
    )

    def record_status(run, path):
        writes.append(run.to_dict())
        return path

    def download_after_initial_status(symbols, start, end):
        assert writes
        assert writes[0]["status"] == "running"
        assert writes[0]["finished_at"] is None
        assert [row["symbol"] for row in writes[0]["symbols"]] == ["SPY", "QQQ"]
        assert all(row["status"] == "pending" for row in writes[0]["symbols"])
        return {symbol: _sample_frame() for symbol in symbols}

    monkeypatch.setattr(module, "save_ingestion_status", record_status)
    monkeypatch.setattr(module, "download_ohlcv", download_after_initial_status)
    monkeypatch.setattr(module, "enrich_market_data", lambda frame: frame.assign(state=0))

    module.main()

    assert len(writes) == 2
    assert writes[-1]["status"] == "success"


def test_data_download_marks_all_symbols_error_when_enrichment_fails(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_script_module()
    status_path = tmp_path / "tables" / "ingestion_status.json"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: module.argparse.Namespace(
            provider="yfinance",
            start="2024-01-01",
            end="2024-01-05",
            symbols="SPY,QQQ",
            status_path=str(status_path),
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
        ),
    )
    monkeypatch.setattr(
        module,
        "download_ohlcv",
        lambda symbols, start, end: {symbol: _sample_frame() for symbol in symbols},
    )

    def fail_enrichment(frame: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("enrichment failed")

    monkeypatch.setattr(module, "enrich_market_data", fail_enrichment)

    with pytest.raises(RuntimeError, match="enrichment failed"):
        module.main()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert [row["symbol"] for row in payload["symbols"]] == ["SPY", "QQQ"]
    assert all(row["status"] == "error" for row in payload["symbols"])
    assert all("enrichment failed" in row["error"] for row in payload["symbols"])


@pytest.mark.parametrize("failing_writer", ["save_raw", "save_processed"])
def test_data_download_marks_all_symbols_error_when_save_fails(
    monkeypatch, tmp_path: Path, failing_writer: str
) -> None:
    module = _load_script_module()
    status_path = tmp_path / "tables" / "ingestion_status.json"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: module.argparse.Namespace(
            provider="yfinance",
            start="2024-01-01",
            end="2024-01-05",
            symbols="SPY,QQQ",
            status_path=str(status_path),
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
        ),
    )
    monkeypatch.setattr(
        module,
        "download_ohlcv",
        lambda symbols, start, end: {symbol: _sample_frame() for symbol in symbols},
    )
    monkeypatch.setattr(module, "enrich_market_data", lambda frame: frame.assign(state=0))

    def fail_save(data, output_dir):
        raise RuntimeError(f"{failing_writer} failed")

    monkeypatch.setattr(module, failing_writer, fail_save)

    with pytest.raises(RuntimeError, match=f"{failing_writer} failed"):
        module.main()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert [row["symbol"] for row in payload["symbols"]] == ["SPY", "QQQ"]
    assert all(row["status"] == "error" for row in payload["symbols"])
    assert all(f"{failing_writer} failed" in row["error"] for row in payload["symbols"])


def test_data_download_marks_missing_requested_symbol_partial(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_script_module()
    status_path = tmp_path / "tables" / "ingestion_status.json"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: module.argparse.Namespace(
            provider="yfinance",
            start="2024-01-01",
            end="2024-01-05",
            symbols="SPY,QQQ",
            status_path=str(status_path),
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
        ),
    )
    calls: list[tuple[str, ...]] = []

    def download_one_symbol(symbols, start, end):
        calls.append(tuple(symbols))
        assert len(symbols) == 1
        symbol = symbols[0]
        if symbol == "QQQ":
            raise RuntimeError("No OHLCV rows returned for QQQ")
        return {symbol: _sample_frame()}

    monkeypatch.setattr(module, "download_ohlcv", download_one_symbol)
    monkeypatch.setattr(module, "enrich_market_data", lambda frame: frame.assign(state=0))

    module.main()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert calls == [("SPY",), ("QQQ",)]
    assert payload["status"] == "partial"
    assert [row["symbol"] for row in payload["symbols"]] == ["SPY", "QQQ"]
    assert payload["symbols"][0]["status"] == "success"
    assert payload["symbols"][0]["rows"] == 3
    assert payload["symbols"][0]["first_date"] == "2024-01-02"
    assert payload["symbols"][1]["status"] == "error"
    assert payload["symbols"][1]["rows"] == 0
    assert "No OHLCV rows returned for QQQ" == payload["symbols"][1]["error"]


def test_data_download_fails_without_saving_when_all_symbol_downloads_fail(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_script_module()
    status_path = tmp_path / "tables" / "ingestion_status.json"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: module.argparse.Namespace(
            provider="yfinance",
            start="2024-01-01",
            end="2024-01-05",
            symbols="SPY,QQQ",
            status_path=str(status_path),
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
        ),
    )

    def fail_download(symbols, start, end):
        raise RuntimeError(f"No OHLCV rows returned for {symbols[0]}")

    def fail_if_saving(data, output_dir):
        raise AssertionError("empty all-error data should not be saved")

    monkeypatch.setattr(module, "download_ohlcv", fail_download)
    monkeypatch.setattr(module, "save_raw", fail_if_saving)
    monkeypatch.setattr(module, "save_processed", fail_if_saving)
    monkeypatch.setattr(module, "enrich_market_data", lambda frame: frame.assign(state=0))

    with pytest.raises(RuntimeError, match="No usable OHLCV data returned"):
        module.main()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert [row["symbol"] for row in payload["symbols"]] == ["SPY", "QQQ"]
    assert all(row["status"] == "error" for row in payload["symbols"])
    assert payload["symbols"][0]["error"] == "No OHLCV rows returned for SPY"
    assert payload["symbols"][1]["error"] == "No OHLCV rows returned for QQQ"


def test_data_download_rejects_unsupported_provider_before_download(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_script_module()
    status_path = tmp_path / "tables" / "ingestion_status.json"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: module.argparse.Namespace(
            provider="fake",
            start="2024-01-01",
            end="2024-01-05",
            symbols="SPY,QQQ",
            status_path=str(status_path),
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
        ),
    )

    def fail_if_called(symbols, start, end):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(module, "download_ohlcv", fail_if_called)

    with pytest.raises(ValueError, match="Unsupported provider: fake"):
        module.main()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert all(row["status"] == "error" for row in payload["symbols"])
    assert all("Unsupported provider: fake" in row["error"] for row in payload["symbols"])
