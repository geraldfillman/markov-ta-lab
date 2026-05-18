"""Tests for Financial Modeling Prep market data integration."""

import os

import pandas as pd
import pytest

from src.fmp import FMPClient, FMPError, load_fmp_api_key, normalize_fmp_ohlcv


def test_load_fmp_api_key_prefers_environment_without_exposing_value(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_MODELING_PREP_API_KEY", "secret-from-env")
    dotenv = tmp_path / ".env"
    dotenv.write_text("FMP_API_KEY=secret-from-file\n", encoding="utf-8")

    assert load_fmp_api_key(dotenv_path=dotenv) == "secret-from-env"


def test_load_fmp_api_key_reads_supported_dotenv_names(tmp_path, monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("FINANCIAL_MODELING_PREP_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "ALPHA_VANTAGE_API_KEY=ignored\nFINANCIAL_MODELING_PREP_API_KEY=file-secret\n",
        encoding="utf-8",
    )

    assert load_fmp_api_key(dotenv_path=dotenv) == "file-secret"


def test_load_fmp_api_key_raises_without_leaking_known_values(tmp_path, monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("FINANCIAL_MODELING_PREP_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("ALPHA_VANTAGE_API_KEY=other-secret\n", encoding="utf-8")

    with pytest.raises(FMPError) as exc:
        load_fmp_api_key(dotenv_path=dotenv)

    message = str(exc.value)
    assert "FMP_API_KEY" in message
    assert "FINANCIAL_MODELING_PREP_API_KEY" in message
    assert "other-secret" not in message


def test_normalize_fmp_ohlcv_returns_existing_pipeline_schema():
    records = [
        {"date": "2024-01-03", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 1200},
        {"date": "2024-01-02", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1100},
    ]

    frame = normalize_fmp_ohlcv(records, "SPY")

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert frame.index.name == "Date"
    assert frame.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert frame.loc[pd.Timestamp("2024-01-03"), "Close"] == 102


def test_normalize_fmp_ohlcv_rejects_missing_required_fields():
    records = [{"date": "2024-01-02", "open": 100, "high": 102, "low": 99, "volume": 1100}]

    with pytest.raises(ValueError, match="Missing required FMP OHLCV fields"):
        normalize_fmp_ohlcv(records, "SPY")


def test_client_fetch_historical_eod_uses_stable_endpoint_and_masks_key():
    calls = []

    def transport(url: str):
        calls.append(url)
        return [
            {"date": "2024-01-02", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1100},
        ]

    client = FMPClient(api_key="super-secret", transport=transport)
    frame = client.fetch_historical_eod("SPY", "2024-01-01", "2024-01-05")

    assert len(frame) == 1
    assert "historical-price-eod/full" in calls[0]
    assert "symbol=SPY" in calls[0]
    assert "from=2024-01-01" in calls[0]
    assert "to=2024-01-05" in calls[0]
    assert "super-secret" in calls[0]
    assert "super-secret" not in repr(client)

