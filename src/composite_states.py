"""Composite state records and derived condition keys.

Layered state representation: base price state + optional environmental
modifiers (volatility, calendar, macro, liquidity). The composite key is a
*derived artifact* used for grouping and lookup; it is never the primary
identifier and never mutates the base state enum.

Fallback hierarchy (most specific to least):

    base + volatility + macro + calendar + liquidity
    base + volatility + macro + calendar
    base + volatility + macro
    base + volatility
    base only

This module is pure: no I/O, no pandas dependency at module load. It exposes
the data contract every downstream consumer (coverage, conditional_markov,
forecast emitter, dashboard) must agree on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

MODIFIER_ORDER: tuple[str, ...] = (
    "volatility_state",
    "macro_state",
    "calendar_state",
    "liquidity_state",
)

WILDCARD = "*"
SEPARATOR = "|"


@dataclass(frozen=True)
class StateRecord:
    """Immutable layered state. Only `base_state` is required."""

    base_state: str
    volatility_state: str | None = None
    macro_state: str | None = None
    calendar_state: str | None = None
    liquidity_state: str | None = None

    def with_overrides(self, **changes: str | None) -> "StateRecord":
        """Return a new record with modifier fields replaced. Never mutates."""
        unknown = set(changes) - {"base_state", *MODIFIER_ORDER}
        if unknown:
            raise ValueError(f"Unknown StateRecord fields: {sorted(unknown)}")
        return replace(self, **changes)


def composite_key(record: StateRecord, fields: Sequence[str] | None = None) -> str:
    """Derive the canonical composite key for a state record.

    Modifier order is fixed (MODIFIER_ORDER) so that two records with the same
    semantic content always produce the same key. Missing modifiers collapse
    to the wildcard `*` rather than being dropped, which keeps key length
    constant across fallback levels and makes coverage tables tidy.
    """
    if not record.base_state:
        raise ValueError("StateRecord.base_state must be non-empty")

    selected = tuple(fields) if fields is not None else MODIFIER_ORDER
    _validate_fields(selected)

    parts: list[str] = [record.base_state]
    for field in selected:
        value = getattr(record, field)
        parts.append(value if value else WILDCARD)
    return SEPARATOR.join(parts)


def truncate_to_level(record: StateRecord, level: int) -> StateRecord:
    """Return a record retaining only the first `level` modifiers.

    `level=0` yields base-only. `level=len(MODIFIER_ORDER)` yields the full
    record unchanged. This is the primitive `conditional_markov.py` uses to
    back off when sample counts fall below thresholds.
    """
    if level < 0 or level > len(MODIFIER_ORDER):
        raise ValueError(
            f"level must be in [0, {len(MODIFIER_ORDER)}]; got {level}"
        )
    drop = MODIFIER_ORDER[level:]
    return record.with_overrides(**{field: None for field in drop})


def fallback_chain(record: StateRecord) -> list[StateRecord]:
    """Enumerate records from most-specific to base-only.

    Order matches the roadmap fallback hierarchy. Callers walk this list and
    stop at the first level whose sample count meets the threshold.
    """
    return [
        truncate_to_level(record, level)
        for level in range(len(MODIFIER_ORDER), -1, -1)
    ]


def _validate_fields(fields: Iterable[str]) -> None:
    unknown = [f for f in fields if f not in MODIFIER_ORDER]
    if unknown:
        raise ValueError(
            f"Unknown modifier fields: {unknown}; expected subset of {MODIFIER_ORDER}"
        )
