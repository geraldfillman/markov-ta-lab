"""Coverage and sample-count utilities for composite state records.

Pure functions only: no I/O, no side effects, no mutation of inputs.
DataFrames are returned; callers decide where to persist them.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import pandas as pd

from composite_states import (
    MODIFIER_ORDER,
    StateRecord,
    composite_key,
    fallback_chain,
    truncate_to_level,
)


def state_counts(
    records: Iterable[StateRecord],
    level: int | None = None,
) -> dict[str, int]:
    """Return composite-key → sample count for the given records.

    If `level` is provided each record is truncated before keying.
    """
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        if level is not None:
            rec = truncate_to_level(rec, level)
        counts[composite_key(rec)] += 1
    return dict(counts)


def transition_counts(
    records: Iterable[StateRecord],
    level: int | None = None,
    is_gap: list[bool] | None = None,
) -> dict[tuple[str, str], int]:
    """Return (from_key, to_key) → count for adjacent record pairs.

    Pairs that straddle a gap index (is_gap[i] True means the bar at index i
    starts a new segment) are skipped so cross-segment transitions are not
    counted.
    """
    rec_list = list(records)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for i in range(len(rec_list) - 1):
        # Skip if the next bar opens a new segment.
        if is_gap is not None and i + 1 < len(is_gap) and is_gap[i + 1]:
            continue
        a, b = rec_list[i], rec_list[i + 1]
        if level is not None:
            a = truncate_to_level(a, level)
            b = truncate_to_level(b, level)
        counts[(composite_key(a), composite_key(b))] += 1
    return dict(counts)


def coverage_table(
    records: Iterable[StateRecord],
    level: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with columns composite_key, sample_count, share.

    Rows are sorted by sample_count descending. share sums to 1.0 unless the
    input is empty, in which case an empty DataFrame is returned.
    """
    counts = state_counts(records, level=level)
    if not counts:
        return pd.DataFrame(columns=["composite_key", "sample_count", "share"])

    total = sum(counts.values())
    rows = [
        {"composite_key": k, "sample_count": v, "share": v / total}
        for k, v in counts.items()
    ]
    df = pd.DataFrame(rows).sort_values("sample_count", ascending=False)
    return df.reset_index(drop=True)


def transition_coverage_table(
    records: Iterable[StateRecord],
    level: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with columns from_key, to_key, count, row_share.

    row_share is the fraction of transitions *out of* each from_key.
    """
    counts = transition_counts(records, level=level)
    if not counts:
        return pd.DataFrame(columns=["from_key", "to_key", "count", "row_share"])

    # Aggregate row totals for row_share computation.
    from_totals: dict[str, int] = defaultdict(int)
    for (fk, _), c in counts.items():
        from_totals[fk] += c

    rows = [
        {
            "from_key": fk,
            "to_key": tk,
            "count": c,
            "row_share": c / from_totals[fk],
        }
        for (fk, tk), c in counts.items()
    ]
    df = pd.DataFrame(rows).sort_values(["from_key", "count"], ascending=[True, False])
    return df.reset_index(drop=True)


def rare_states(
    records: Iterable[StateRecord],
    min_count: int = 30,
    level: int | None = None,
) -> list[str]:
    """Return composite keys whose sample count is strictly below min_count."""
    counts = state_counts(records, level=level)
    return [k for k, v in counts.items() if v < min_count]


def select_fallback_level(
    record: StateRecord,
    level_counts: dict[int, dict[str, int]],
    thresholds: tuple[int, int] = (50, 30),
) -> int:
    """Return the highest level whose count meets the roadmap thresholds.

    Walks the fallback chain from most-specific to base-only and returns
    the first level whose sample count satisfies:
      - count >= high  → use that level
      - count >= mid   → use level - 1 (one step less specific)
      - count < mid    → continue falling back

    If no level meets the mid threshold, returns 0 (base-only).

    thresholds = (high, mid); default (50, 30).
    """
    high, mid = thresholds
    max_level = len(MODIFIER_ORDER)

    # fallback_chain returns max_level+1 entries: indices 0..max_level map to
    # levels max_level..0 respectively.
    chain = fallback_chain(record)  # most-specific first, base-only last
    for i, truncated in enumerate(chain):
        level = max_level - i
        key = composite_key(truncated)
        count = level_counts.get(level, {}).get(key, 0)

        if count >= high:
            return level
        if count >= mid:
            # One step less specific; never below 0.
            return max(0, level - 1)

    return 0
