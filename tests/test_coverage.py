"""Tests for src/coverage.py."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from composite_states import StateRecord
from coverage import (
    coverage_table,
    rare_states,
    select_fallback_level,
    state_counts,
    transition_counts,
    transition_coverage_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _r(base, vol=None, mac=None, cal=None, liq=None) -> StateRecord:
    return StateRecord(
        base_state=base,
        volatility_state=vol,
        macro_state=mac,
        calendar_state=cal,
        liquidity_state=liq,
    )


SIMPLE = [
    _r("bull", vol="HV"),
    _r("bull", vol="HV"),
    _r("bear", vol="LV"),
    _r("bull", vol="HV"),
]

FULL = [
    _r("bull", "HV", "risk_on", "week", "tight"),
    _r("bull", "HV", "risk_on", "week", "tight"),
    _r("bull", "HV", "risk_on", "week", "normal"),
    _r("bear", "LV", "risk_off", "eod", "loose"),
]


# ---------------------------------------------------------------------------
# state_counts
# ---------------------------------------------------------------------------

class TestStateCounts:
    def test_empty_returns_empty_dict(self):
        assert state_counts([]) == {}

    def test_aggregates_identical_records(self):
        counts = state_counts(SIMPLE)
        assert counts["bull|HV|*|*|*"] == 3
        assert counts["bear|LV|*|*|*"] == 1

    def test_level_0_truncates_to_base_only(self):
        # composite_key always emits wildcard slots; level=0 gives base|*|*|*|*
        counts = state_counts(SIMPLE, level=0)
        assert counts["bull|*|*|*|*"] == 3
        assert counts["bear|*|*|*|*"] == 1

    def test_level_1_keeps_one_modifier(self):
        counts = state_counts(FULL, level=1)
        assert counts["bull|HV|*|*|*"] == 3
        assert counts["bear|LV|*|*|*"] == 1

    def test_total_matches_input_length(self):
        counts = state_counts(FULL)
        assert sum(counts.values()) == len(FULL)


# ---------------------------------------------------------------------------
# transition_counts
# ---------------------------------------------------------------------------

class TestTransitionCounts:
    def test_empty_returns_empty(self):
        assert transition_counts([]) == {}

    def test_single_record_no_transitions(self):
        assert transition_counts([_r("bull")]) == {}

    def test_counts_adjacent_pairs(self):
        records = [_r("bull"), _r("bear"), _r("bull")]
        tc = transition_counts(records)
        assert tc[("bull|*|*|*|*", "bear|*|*|*|*")] == 1
        assert tc[("bear|*|*|*|*", "bull|*|*|*|*")] == 1

    def test_gap_skips_pair(self):
        records = [_r("bull"), _r("bear"), _r("bull")]
        # Mark index 1 as gap start → skip pair (bull→bear)
        tc = transition_counts(records, is_gap=[False, True, False])
        assert ("bull|*|*|*|*", "bear|*|*|*|*") not in tc
        assert tc.get(("bear|*|*|*|*", "bull|*|*|*|*"), 0) == 1

    def test_level_truncates_before_counting(self):
        records = [
            _r("bull", vol="HV"),
            _r("bull", vol="LV"),
        ]
        tc = transition_counts(records, level=0)
        assert tc[("bull|*|*|*|*", "bull|*|*|*|*")] == 1


# ---------------------------------------------------------------------------
# coverage_table
# ---------------------------------------------------------------------------

class TestCoverageTable:
    def test_empty_returns_empty_df(self):
        df = coverage_table([])
        assert list(df.columns) == ["composite_key", "sample_count", "share"]
        assert len(df) == 0

    def test_sorted_by_sample_count_desc(self):
        df = coverage_table(SIMPLE)
        assert df.iloc[0]["composite_key"] == "bull|HV|*|*|*"

    def test_shares_sum_to_one(self):
        df = coverage_table(FULL)
        assert abs(df["share"].sum() - 1.0) < 1e-9

    def test_columns_present(self):
        df = coverage_table(SIMPLE)
        assert set(df.columns) == {"composite_key", "sample_count", "share"}


# ---------------------------------------------------------------------------
# transition_coverage_table
# ---------------------------------------------------------------------------

class TestTransitionCoverageTable:
    def test_empty_returns_empty_df(self):
        df = transition_coverage_table([])
        assert list(df.columns) == ["from_key", "to_key", "count", "row_share"]

    def test_row_share_sums_to_one_per_from_key(self):
        records = [_r("bull"), _r("bear"), _r("bull"), _r("bear")]
        df = transition_coverage_table(records, level=0)
        for from_key, group in df.groupby("from_key"):
            assert abs(group["row_share"].sum() - 1.0) < 1e-9

    def test_columns_present(self):
        df = transition_coverage_table(SIMPLE)
        assert set(df.columns) == {"from_key", "to_key", "count", "row_share"}


# ---------------------------------------------------------------------------
# rare_states
# ---------------------------------------------------------------------------

class TestRareStates:
    def test_empty_returns_empty(self):
        assert rare_states([]) == []

    def test_below_threshold_flagged(self):
        result = rare_states(SIMPLE, min_count=2)
        assert "bear|LV|*|*|*" in result

    def test_at_threshold_not_flagged(self):
        # count=3 >= min_count=3 → not rare
        result = rare_states(SIMPLE, min_count=3)
        assert "bull|HV|*|*|*" not in result

    def test_above_threshold_not_flagged(self):
        result = rare_states(SIMPLE, min_count=2)
        assert "bull|HV|*|*|*" not in result

    def test_level_applied(self):
        result = rare_states(SIMPLE, min_count=10, level=0)
        # both "bull|*|*|*|*" (3) and "bear|*|*|*|*" (1) are below 10
        assert "bull|*|*|*|*" in result
        assert "bear|*|*|*|*" in result


# ---------------------------------------------------------------------------
# select_fallback_level
# ---------------------------------------------------------------------------

class TestSelectFallbackLevel:
    def _make_level_counts(self, level: int, key: str, n: int) -> dict[int, dict[str, int]]:
        return {level: {key: n}}

    def test_returns_highest_level_when_count_gte_high(self):
        rec = _r("bull", "HV", "risk_on", "week", "tight")
        # level 4 key has all 4 modifiers filled in, no wildcards
        level_counts = {4: {"bull|HV|risk_on|week|tight": 50}}
        assert select_fallback_level(rec, level_counts) == 4

    def test_boundary_exactly_50_returns_level_4(self):
        rec = _r("bull", "HV", "risk_on", "week", "tight")
        level_counts = {4: {"bull|HV|risk_on|week|tight": 50}}
        assert select_fallback_level(rec, level_counts) == 4

    def test_count_49_falls_to_mid_check(self):
        rec = _r("bull", "HV", "risk_on", "week", "tight")
        # level 4 count = 49: below high (50) but >= mid (30)
        # → mid fires at level 4, returns level 4-1 = 3
        level_counts = {
            4: {"bull|HV|risk_on|week|tight": 49},
        }
        result = select_fallback_level(rec, level_counts)
        assert result == 3

    def test_count_30_at_level_3_returns_2(self):
        rec = _r("bull", "HV", "risk_on", "week", "tight")
        level_counts = {
            4: {"bull|HV|risk_on|week|tight": 0},
            3: {"bull|HV|risk_on|week|*": 30},
        }
        assert select_fallback_level(rec, level_counts) == 2

    def test_count_29_falls_through_to_base(self):
        rec = _r("bull", "HV")
        # All levels have count < 30, so should return 0
        level_counts = {
            4: {"bull|HV|*|*|*": 29},
            0: {"bull|*|*|*|*": 29},
        }
        assert select_fallback_level(rec, level_counts) == 0

    def test_no_threshold_met_returns_0(self):
        rec = _r("bull", "HV")
        assert select_fallback_level(rec, {}) == 0

    def test_count_0_returns_0(self):
        rec = _r("bear", "LV", "risk_off")
        level_counts = {3: {"bear|LV|risk_off|*|*": 0}}
        assert select_fallback_level(rec, level_counts) == 0

    def test_high_threshold_exactly_50_full_level(self):
        rec = _r("bull", "HV")
        # For a record with 1 modifier, all levels 4→1 produce the same key
        level_counts = {4: {"bull|HV|*|*|*": 50}}
        assert select_fallback_level(rec, level_counts, thresholds=(50, 30)) == 4
