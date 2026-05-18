"""Generate Phase F forecast-diagnostic CSV tables for the dashboard.

Produces five CSVs in reports/tables/:
  fallback_usage.csv
  confidence_distribution.csv
  top_rare_states.csv
  coverage_by_symbol.csv
  forecast_warnings.csv

No external data fetches are needed.  A synthetic state history is built
from the repo's STATE_LABELS universe, which ensures this script always
runs offline (CI, local, no API key required).

Usage:
    python scripts/build_forecast_diagnostics.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # bare imports used inside src/

from src.composite_states import StateRecord
from src.conditional_markov import forecast_batch
from src.config import DEFAULT_SYMBOLS, STATE_LABELS, TABLES_DIR
from src.coverage import state_counts as compute_state_counts
from src.dashboard_panels import (
    coverage_by_symbol_panel,
    fallback_usage_panel,
    low_confidence_panel,
    top_rare_states_panel,
    warnings_summary_panel,
)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

SYMBOLS = DEFAULT_SYMBOLS
HISTORY_BARS_PER_SYMBOL = 500   # synthetic history bars to fit transitions on
FORECAST_BARS_PER_SYMBOL = 60   # bars per symbol for which we produce forecasts
RANDOM_SEED = 42
CONFIDENCE_THRESHOLDS = (50, 30)


def _make_synthetic_history(n_bars: int, rng: random.Random) -> list[StateRecord]:
    """Return a deterministic sequence of base-only StateRecords."""
    base_states = list(STATE_LABELS.values())
    return [StateRecord(base_state=rng.choice(base_states)) for _ in range(n_bars)]


def run(
    symbols: list[str] = SYMBOLS,
    history_bars: int = HISTORY_BARS_PER_SYMBOL,
    forecast_bars: int = FORECAST_BARS_PER_SYMBOL,
    seed: int = RANDOM_SEED,
    tables_dir: str | Path = TABLES_DIR,
) -> dict[str, Path]:
    """Build all five panel CSVs and return a mapping {table_name: path}."""
    rng = random.Random(seed)
    out_dir = Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_forecasts: list = []
    all_symbols_label: list[str] = []  # parallel list to all_forecasts

    for symbol in symbols:
        history = _make_synthetic_history(history_bars, rng)
        to_forecast = _make_synthetic_history(forecast_bars, rng)

        symbol_forecasts = forecast_batch(
            records_to_forecast=iter(to_forecast),
            all_history_records=iter(history),
            thresholds=CONFIDENCE_THRESHOLDS,
        )
        all_forecasts.extend(symbol_forecasts)
        all_symbols_label.extend([symbol] * len(symbol_forecasts))

    # Aggregate state counts across all history for the rare-states panel.
    full_history: list[StateRecord] = []
    rng2 = random.Random(seed)
    for _ in symbols:
        full_history.extend(_make_synthetic_history(history_bars, rng2))
    sc = compute_state_counts(full_history, level=0)

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
    print("Building Phase F forecast-diagnostic tables...")
    run()
    print("Done.")


if __name__ == "__main__":
    main()
