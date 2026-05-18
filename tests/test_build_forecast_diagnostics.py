"""Smoke tests for build_forecast_diagnostics runner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # bare imports used inside src/

from scripts.build_forecast_diagnostics import _load_real_history, run
from src.composite_states import StateRecord
from src.config import STATE_LABELS

EXPECTED_TABLES = {
    "fallback_usage.csv",
    "confidence_distribution.csv",
    "top_rare_states.csv",
    "coverage_by_symbol.csv",
    "forecast_warnings.csv",
}

EXPECTED_HEADERS = {
    "fallback_usage.csv": ["fallback_level", "count", "share"],
    "confidence_distribution.csv": ["confidence", "count", "share"],
    "top_rare_states.csv": ["composite_key", "sample_count"],
    "coverage_by_symbol.csv": [
        "symbol",
        "forecast_count",
        "mean_sample_count",
        "low_confidence_count",
        "low_confidence_share",
    ],
    "forecast_warnings.csv": ["warning", "count"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_parquet(tmp_path: Path, symbol: str, n_bars: int = 300) -> Path:
    """Write a minimal OHLCV parquet to tmp_path/<symbol>.parquet.

    Uses a simple random walk so the indicator pipeline has enough data to
    warm up (SMA-50 needs ≥50 rows, detect_levels lookback=126).  We use
    n_bars=300 to comfortably clear the 126-bar level-detection minimum.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", periods=n_bars, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bars)))
    high = close * (1 + rng.uniform(0.001, 0.02, n_bars))
    low = close * (1 - rng.uniform(0.001, 0.02, n_bars))
    open_ = close * (1 + rng.normal(0, 0.005, n_bars))
    volume = rng.integers(1_000_000, 10_000_000, n_bars).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    path = tmp_path / f"{symbol.upper()}.parquet"
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tables_dir(tmp_path: Path) -> Path:
    return tmp_path / "tables"


@pytest.fixture()
def fake_data_dir(tmp_path: Path) -> Path:
    """A directory with a single synthetic SPY parquet."""
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    _make_ohlcv_parquet(data_dir, "SPY")
    return data_dir


# ---------------------------------------------------------------------------
# Existing smoke tests — use synthetic_only=True to preserve original behaviour
# ---------------------------------------------------------------------------

def test_all_csvs_are_produced(tables_dir: Path) -> None:
    """All five CSVs must exist after run()."""
    run(
        symbols=["SPY", "QQQ"],
        history_bars=80,
        forecast_bars=10,
        seed=0,
        tables_dir=tables_dir,
        synthetic_only=True,
    )
    produced = {p.name for p in tables_dir.glob("*.csv")}
    assert EXPECTED_TABLES == produced


def test_csv_headers_are_correct(tables_dir: Path) -> None:
    """Each CSV must have the exact expected column headers."""
    import csv

    run(
        symbols=["SPY", "QQQ"],
        history_bars=80,
        forecast_bars=10,
        seed=0,
        tables_dir=tables_dir,
        synthetic_only=True,
    )
    for table_name, expected_cols in EXPECTED_HEADERS.items():
        path = tables_dir / table_name
        with path.open(newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
        assert header == expected_cols, (
            f"{table_name}: expected cols {expected_cols}, got {header}"
        )


def test_csv_are_non_empty(tables_dir: Path) -> None:
    """Each CSV must have at least one data row."""
    import csv

    run(
        symbols=["SPY", "QQQ"],
        history_bars=80,
        forecast_bars=10,
        seed=0,
        tables_dir=tables_dir,
        synthetic_only=True,
    )
    for table_name in EXPECTED_TABLES:
        path = tables_dir / table_name
        with path.open(newline="") as fh:
            reader = csv.reader(fh)
            _header = next(reader)
            rows = list(reader)
        assert rows, f"{table_name} has no data rows"


def test_coverage_by_symbol_has_both_symbols(tables_dir: Path) -> None:
    """coverage_by_symbol.csv must contain a row per input symbol."""
    import csv

    run(
        symbols=["SPY", "QQQ"],
        history_bars=80,
        forecast_bars=10,
        seed=0,
        tables_dir=tables_dir,
        synthetic_only=True,
    )
    path = tables_dir / "coverage_by_symbol.csv"
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        symbols_found = {row["symbol"] for row in reader}
    assert symbols_found == {"SPY", "QQQ"}


# ---------------------------------------------------------------------------
# New tests for _load_real_history
# ---------------------------------------------------------------------------

def test_load_real_history_returns_state_records(fake_data_dir: Path) -> None:
    """_load_real_history must return a non-empty list of StateRecord objects."""
    records = _load_real_history("SPY", fake_data_dir)
    assert records is not None, "_load_real_history returned None for a valid parquet"
    assert len(records) > 0, "Expected at least one StateRecord"
    assert all(isinstance(r, StateRecord) for r in records)


def test_load_real_history_valid_base_states(fake_data_dir: Path) -> None:
    """Every base_state in the returned records must be a known STATE_LABELS value."""
    valid_labels = set(STATE_LABELS.values())
    records = _load_real_history("SPY", fake_data_dir)
    assert records is not None
    for record in records:
        assert record.base_state in valid_labels, (
            f"Unknown base_state {record.base_state!r}"
        )


def test_load_real_history_returns_none_for_missing_symbol(fake_data_dir: Path) -> None:
    """_load_real_history must return None when the parquet does not exist."""
    result = _load_real_history("NONEXISTENT", fake_data_dir)
    assert result is None


def test_load_real_history_modifiers_are_none(fake_data_dir: Path) -> None:
    """Modifier slots must be None (only base_state is populated)."""
    records = _load_real_history("SPY", fake_data_dir)
    assert records is not None
    for record in records:
        assert record.volatility_state is None
        assert record.macro_state is None
        assert record.calendar_state is None
        assert record.liquidity_state is None


def test_run_with_real_data_dir_uses_discovered_symbols(
    tmp_path: Path, fake_data_dir: Path
) -> None:
    """run() with a data_dir containing SPY.parquet must list SPY in coverage CSV."""
    import csv

    tables = tmp_path / "tables"
    run(
        symbols=None,
        tables_dir=tables,
        data_dir=fake_data_dir,
        synthetic_only=False,
        forecast_bars=10,
    )
    path = tables / "coverage_by_symbol.csv"
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        symbols_found = {row["symbol"] for row in reader}
    assert "SPY" in symbols_found
