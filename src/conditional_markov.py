"""Conditional Markov forecasting with composite state fallback.

Pure functions only: no I/O, no side effects, no mutation of inputs.
Builds on composite_states and coverage to forecast next-state probabilities
conditioned on the current modifier environment, backing off gracefully when
sample counts are thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from composite_states import MODIFIER_ORDER, StateRecord, composite_key, truncate_to_level
from coverage import select_fallback_level, state_counts, transition_counts


@dataclass(frozen=True)
class ConditionalForecast:
    """Result of a single conditional forecast."""

    base_state: str
    conditions_requested: dict[str, str | None]
    conditions_used: dict[str, str | None]
    fallback_level: int
    sample_count: int
    next_state_probs: dict[str, float]
    confidence: str
    warnings: tuple[str, ...]


def fit_transitions(
    records: Iterable[StateRecord],
    level_counts: dict[int, dict[str, int]] | None = None,
) -> dict[int, dict[tuple[str, str], int]]:
    """Pre-compute transition counts at every fallback level 0..N."""
    rec_list = list(records)
    max_level = len(MODIFIER_ORDER)
    return {
        level: transition_counts(rec_list, level=level)
        for level in range(max_level + 1)
    }


def fit_state_counts(records: Iterable[StateRecord]) -> dict[int, dict[str, int]]:
    """Pre-compute state counts at every fallback level 0..N."""
    rec_list = list(records)
    max_level = len(MODIFIER_ORDER)
    return {
        level: state_counts(rec_list, level=level)
        for level in range(max_level + 1)
    }


def _modifier_dict(record: StateRecord) -> dict[str, str | None]:
    return {field: getattr(record, field) for field in MODIFIER_ORDER}


def _confidence(sample_count: int, thresholds: tuple[int, int]) -> str:
    high, mid = thresholds
    if sample_count >= high:
        return "high"
    if sample_count >= mid:
        return "medium"
    return "low"


def _probs_at_level(
    from_key: str,
    level: int,
    transitions_by_level: dict[int, dict[tuple[str, str], int]],
) -> dict[str, float]:
    """Normalize outgoing transitions from from_key at the given level."""
    level_trans = transitions_by_level.get(level, {})
    raw = {tk: c for (fk, tk), c in level_trans.items() if fk == from_key}
    total = sum(raw.values())
    if total == 0:
        return {}
    return {tk: c / total for tk, c in raw.items()}


def forecast(
    record: StateRecord,
    state_counts_by_level: dict[int, dict[str, int]],
    transitions_by_level: dict[int, dict[tuple[str, str], int]],
    thresholds: tuple[int, int] = (50, 30),
) -> ConditionalForecast:
    """Forecast next-state probabilities for a single record.

    Walks the fallback chain to find a level with adequate samples, then
    normalises transition counts into probabilities.
    """
    max_level = len(MODIFIER_ORDER)
    conditions_requested = _modifier_dict(record)

    chosen_level = select_fallback_level(record, state_counts_by_level, thresholds)

    # Try the chosen level first; walk down further if no transitions found.
    search_levels = list(range(chosen_level, -1, -1))
    probs: dict[str, float] = {}
    used_level = chosen_level

    for lvl in search_levels:
        truncated = truncate_to_level(record, lvl)
        from_key = composite_key(truncated)
        probs = _probs_at_level(from_key, lvl, transitions_by_level)
        if probs:
            used_level = lvl
            break

    truncated_used = truncate_to_level(record, used_level)
    from_key_used = composite_key(truncated_used)
    sample_count = state_counts_by_level.get(used_level, {}).get(from_key_used, 0)
    conditions_used = _modifier_dict(truncated_used)
    conf = _confidence(sample_count, thresholds)

    warnings: list[str] = []
    if used_level < max_level:
        warnings.append("fallback_used")
    if conf == "low":
        warnings.append("low_sample")
    if not probs:
        warnings.append("no_transitions")

    return ConditionalForecast(
        base_state=record.base_state,
        conditions_requested=conditions_requested,
        conditions_used=conditions_used,
        fallback_level=used_level,
        sample_count=sample_count,
        next_state_probs=probs,
        confidence=conf,
        warnings=tuple(warnings),
    )


def forecast_batch(
    records_to_forecast: Iterable[StateRecord],
    all_history_records: Iterable[StateRecord],
    thresholds: tuple[int, int] = (50, 30),
) -> list[ConditionalForecast]:
    """Fit from history then forecast each input record."""
    history = list(all_history_records)
    sc_by_level = fit_state_counts(history)
    tr_by_level = fit_transitions(history)
    return [
        forecast(rec, sc_by_level, tr_by_level, thresholds)
        for rec in records_to_forecast
    ]
