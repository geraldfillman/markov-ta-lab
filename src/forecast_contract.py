"""Forecast contract for the Markov TA Lab.

Defines the canonical 14-column forecast row schema and pure helper functions
to build contract-compliant rows from ConditionalForecast objects.

Only ``write_forecast_table`` performs I/O; everything else is pure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from conditional_markov import ConditionalForecast

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FORECAST_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "base_state",
    "conditions_requested",
    "conditions_used",
    "fallback_level",
    "sample_count",
    "probability",
    "expected_value_after_cost",
    "cost_model_used",
    "confidence",
    "historical_stability",
    "invalidation_trigger",
    "warnings",
)

assert len(FORECAST_COLUMNS) == 14, "FORECAST_COLUMNS must have exactly 14 entries"

# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def forecast_row(
    symbol: str,
    date: Any,
    conditional_forecast: ConditionalForecast,
    next_state: str | None = None,
) -> dict[str, Any]:
    """Build a single contract-compliant row dict from a ConditionalForecast.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"AAPL"``).
    date:
        Bar date — any type accepted by pandas (str, datetime, date, …).
    conditional_forecast:
        Result produced by :func:`conditional_markov.forecast`.
    next_state:
        When supplied, ``probability`` is set to
        ``conditional_forecast.next_state_probs.get(next_state)``.
        When ``None``, ``probability`` is left as ``None``.

    Returns
    -------
    dict with all 14 canonical columns populated (placeholders set to ``None``).
    """
    cf = conditional_forecast

    probability: float | None = None
    if next_state is not None:
        probability = cf.next_state_probs.get(next_state)

    return {
        "symbol": symbol,
        "date": date,
        "base_state": cf.base_state,
        "conditions_requested": cf.conditions_requested,
        "conditions_used": cf.conditions_used,
        "fallback_level": cf.fallback_level,
        "sample_count": cf.sample_count,
        "probability": probability,
        # Placeholders for later phases
        "expected_value_after_cost": None,
        "cost_model_used": None,
        "confidence": cf.confidence,
        "historical_stability": None,
        "invalidation_trigger": None,
        "warnings": cf.warnings,
    }


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def forecasts_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame with columns in canonical FORECAST_COLUMNS order.

    Missing columns in any row dict are filled with ``None``. Extra keys are
    silently dropped so callers are not penalised for adding diagnostic fields.

    Parameters
    ----------
    rows:
        List of dicts as returned by :func:`forecast_row`.  May be empty.

    Returns
    -------
    pd.DataFrame with exactly the 14 canonical columns in order.
    """
    if not rows:
        return pd.DataFrame(columns=list(FORECAST_COLUMNS))

    df = pd.DataFrame(rows)
    # Add any missing columns, drop extras, reorder to canonical order
    for col in FORECAST_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[list(FORECAST_COLUMNS)]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_forecast_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a forecast DataFrame to CSV, creating parent directories as needed.

    This is the ONLY function in this module that performs I/O.

    Parameters
    ----------
    df:
        DataFrame as returned by :func:`forecasts_to_dataframe`.
    path:
        Destination file path.

    Returns
    -------
    Resolved ``Path`` of the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out.resolve()
