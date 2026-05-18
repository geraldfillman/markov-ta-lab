"""Historical analogues: similarity lookup and metadata overlay. No prediction merging."""

from __future__ import annotations

import dataclasses
from typing import Optional

import pandas as pd

from composite_states import StateRecord, composite_key
from conditional_markov import ConditionalForecast


def build_state_path_index(
    records: list[StateRecord],
    window: int = 5,
) -> dict[tuple[str, ...], list[int]]:
    """Build a sliding-window index mapping state-key tuples to starting indices.

    Each key is a tuple of composite_key strings of length `window`.
    Values are lists of 0-based starting indices in `records`.
    """
    index: dict[tuple[str, ...], list[int]] = {}
    n = len(records)
    for i in range(n - window + 1):
        path = tuple(composite_key(records[j]) for j in range(i, i + window))
        index.setdefault(path, []).append(i)
    return index


def find_analogues(
    current_path: tuple[str, ...],
    index: dict[tuple[str, ...], list[int]],
    top_n: int = 5,
) -> list[int]:
    """Return starting indices of exact-match analogues (up to top_n).

    No fuzzy matching: if fewer than top_n exact matches exist, returns what is available.
    """
    matches = index.get(current_path, [])
    return matches[:top_n]


def analogue_outcomes(
    indices: list[int],
    future_returns: pd.Series,
    horizon: int = 5,
) -> pd.DataFrame:
    """Return a DataFrame of forward returns for each analogue start index.

    Columns: 'index', 't+1', ..., 't+{horizon}'.
    If an analogue index runs past the end of future_returns, those cells are NaN.
    """
    cols = ["index"] + [f"t+{h}" for h in range(1, horizon + 1)]
    rows: list[dict] = []
    for idx in indices:
        row: dict = {"index": idx}
        for h in range(1, horizon + 1):
            pos = idx + h
            if pos < len(future_returns):
                row[f"t+{h}"] = future_returns.iloc[pos]
            else:
                row[f"t+{h}"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def apply_analogue_metadata(
    forecast: ConditionalForecast,
    analogue_count: int,
) -> ConditionalForecast:
    """Return a new ConditionalForecast with 'few_analogues' warning if analogue_count < 3.

    Does NOT modify confidence or next_state_probs.
    """
    if analogue_count < 3:
        new_warnings = forecast.warnings + ("few_analogues",)
        return dataclasses.replace(forecast, warnings=new_warnings)
    return forecast
