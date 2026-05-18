"""Generate Phase F forecast-diagnostic CSV tables for the dashboard.

Produces five CSVs in reports/tables/:
  fallback_usage.csv
  confidence_distribution.csv
  top_rare_states.csv
  coverage_by_symbol.csv
  forecast_warnings.csv

By default the script uses real labeled state history derived from the
committed parquet files in data/processed/.  When a symbol's parquet is
missing or the indicator pipeline fails the script falls back to a
deterministic synthetic sequence and logs a warning.

Usage:
    python scripts/build_forecast_diagnostics.py
    python scripts/build_forecast_diagnostics.py --synthetic-only
    python scripts/build_forecast_diagnostics.py --data-dir path/to/processed
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # bare imports used inside src/

from src.composite_states import StateRecord
from src.conditional_markov import forecast_batch
from src.config import STATE_LABELS, TABLES_DIR
from src.coverage import state_counts as compute_state_counts
from src.dashboard_panels import (
    coverage_by_symbol_panel,
    fallback_usage_panel,
    low_confidence_panel,
    top_rare_states_panel,
    warnings_summary_panel,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = ROOT / "data" / "processed"
HISTORY_BARS_PER_SYMBOL = 500   # synthetic history bars (fallback only)
FORECAST_BARS_PER_SYMBOL = 60   # bars per symbol for which we produce forecasts
RANDOM_SEED = 42
CONFIDENCE_THRESHOLDS = (50, 30)

# Fallback synthetic symbols (used only when no parquets exist at all)
_FALLBACK_SYNTHETIC_SYMBOLS = ["SPY", "QQQ"]


def _make_synthetic_history(n_bars: int, rng: random.Random) -> list[StateRecord]:
    """Return a deterministic sequence of base-only StateRecords."""
    base_states = list(STATE_LABELS.values())
    return [StateRecord(base_state=rng.choice(base_states)) for _ in range(n_bars)]


def _load_real_history(symbol: str, data_dir: Path) -> list[StateRecord] | None:
    """Load parquet → indicators → levels → states → list[StateRecord].

    Returns None if the parquet is absent or any pipeline step fails so the
    caller can fall back to synthetic data without crashing.
    """
    try:
        from src.calendar_states import classify_calendar
        from src.data import load_processed
        from src.indicators import add_indicators
        from src.levels import detect_levels
        from src.states import label_states

        df = load_processed(symbol, data_dir=data_dir)
        df = add_indicators(df)
        df = detect_levels(df)
        state_ids = label_states(df)

        valid_ids = state_ids.dropna().astype(int)
        if valid_ids.empty:
            log.warning("%s: label_states returned no valid rows", symbol)
            return None

        records = [
            StateRecord(
                base_state=STATE_LABELS[sid],
                calendar_state=classify_calendar(date),
            )
            for date, sid in valid_ids.items()
            if sid in STATE_LABELS
        ]
        if not records:
            log.warning("%s: no state ids matched STATE_LABELS", symbol)
            return None

        return records

    except FileNotFoundError:
        log.warning("%s: parquet not found in %s", symbol, data_dir)
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: pipeline error — %s", symbol, exc)
        return None


def _discover_symbols(data_dir: Path) -> list[str]:
    """Return sorted list of symbols for which parquets exist in data_dir."""
    return sorted(p.stem for p in data_dir.glob("*.parquet"))


def run(
    symbols: list[str] | None = None,
    history_bars: int = HISTORY_BARS_PER_SYMBOL,
    forecast_bars: int = FORECAST_BARS_PER_SYMBOL,
    seed: int = RANDOM_SEED,
    tables_dir: str | Path = TABLES_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    synthetic_only: bool = False,
) -> dict[str, Path]:
    """Build all five panel CSVs and return a mapping {table_name: path}."""
    rng = random.Random(seed)
    out_dir = Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real_data_dir = Path(data_dir)

    # Resolve symbol list
    if symbols is None:
        if not synthetic_only:
            discovered = _discover_symbols(real_data_dir)
            resolved_symbols = discovered if discovered else _FALLBACK_SYNTHETIC_SYMBOLS
            if not discovered:
                log.warning(
                    "No parquets found in %s; using synthetic fallback symbols",
                    real_data_dir,
                )
        else:
            resolved_symbols = _FALLBACK_SYNTHETIC_SYMBOLS
    else:
        resolved_symbols = symbols

    all_forecasts: list = []
    all_symbols_label: list[str] = []  # parallel list to all_forecasts
    all_history: list[StateRecord] = []

    for symbol in resolved_symbols:
        # --- history for fitting the Markov model ---
        if not synthetic_only:
            history = _load_real_history(symbol, real_data_dir)
            if history is None:
                log.warning("%s: falling back to synthetic history", symbol)
                history = _make_synthetic_history(history_bars, rng)
        else:
            history = _make_synthetic_history(history_bars, rng)

        all_history.extend(history)

        # --- sequence to forecast ---
        # Use the tail of real history as the forecast window when available,
        # otherwise generate synthetic bars.
        if not synthetic_only:
            to_forecast_real = _load_real_history(symbol, real_data_dir)
            if to_forecast_real is not None:
                to_forecast: list[StateRecord] = to_forecast_real[-forecast_bars:]
            else:
                to_forecast = _make_synthetic_history(forecast_bars, rng)
        else:
            to_forecast = _make_synthetic_history(forecast_bars, rng)

        symbol_forecasts = forecast_batch(
            records_to_forecast=iter(to_forecast),
            all_history_records=iter(history),
            thresholds=CONFIDENCE_THRESHOLDS,
        )
        all_forecasts.extend(symbol_forecasts)
        all_symbols_label.extend([symbol] * len(symbol_forecasts))

    # Aggregate state counts across all history for the rare-states panel.
    sc = compute_state_counts(all_history, level=0)

    panels: dict[str, Path] = {}

    def _write(name: str, df) -> Path:
        path = out_dir / name
        df.to_csv(path, index=False)
        return path

    panels["fallback_usage.csv"] = _write(
        "fallback_usage.csv",
        fallback_usage_panel(all_forecasts),
    )
    panels["confidence_distribution.csv"] = _write(
        "confidence_distribution.csv",
        low_confidence_panel(all_forecasts),
    )
    panels["top_rare_states.csv"] = _write(
        "top_rare_states.csv",
        top_rare_states_panel(sc, top_n=20),
    )
    panels["coverage_by_symbol.csv"] = _write(
        "coverage_by_symbol.csv",
        coverage_by_symbol_panel(all_forecasts, all_symbols_label),
    )
    panels["forecast_warnings.csv"] = _write(
        "forecast_warnings.csv",
        warnings_summary_panel(all_forecasts),
    )

    for name, path in panels.items():
        print(f"  Wrote {path}")

    return panels


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        default=False,
        help="Force synthetic history for all symbols (skips parquet loading).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing processed *.parquet files (default: data/processed).",
    )
    args = parser.parse_args()

    print("Building Phase F forecast-diagnostic tables...")
    run(synthetic_only=args.synthetic_only, data_dir=args.data_dir)
    print("Done.")


if __name__ == "__main__":
    main()
