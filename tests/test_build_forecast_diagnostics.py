"""Smoke tests for build_forecast_diagnostics runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # bare imports used inside src/

from scripts.build_forecast_diagnostics import run

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


@pytest.fixture()
def tables_dir(tmp_path: Path) -> Path:
    return tmp_path / "tables"


def test_all_csvs_are_produced(tables_dir: Path) -> None:
    """All five CSVs must exist after run()."""
    run(
        symbols=["SPY", "QQQ"],
        history_bars=80,
        forecast_bars=10,
        seed=0,
        tables_dir=tables_dir,
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
    )
    path = tables_dir / "coverage_by_symbol.csv"
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        symbols_found = {row["symbol"] for row in reader}
    assert symbols_found == {"SPY", "QQQ"}
