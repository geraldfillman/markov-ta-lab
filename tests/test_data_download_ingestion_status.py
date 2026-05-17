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
            provider="fake",
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
        lambda symbols, start, end, provider: {symbol: _sample_frame() for symbol in symbols},
    )
    monkeypatch.setattr(module, "enrich_market_data", lambda frame: frame.assign(state=0))

    module.main()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["provider"] == "fake"
    assert [row["symbol"] for row in payload["symbols"]] == ["SPY", "QQQ"]
    assert payload["symbols"][0]["rows"] == 3
    assert payload["symbols"][0]["first_date"] == "2024-01-02"


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
            provider="fake",
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
        lambda symbols, start, end, provider: {symbol: _sample_frame() for symbol in symbols},
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
