"""Tests for src/cost_models.py — Phase H Dynamic Cost Model."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.cost_models import (
    COST_BPS,
    ENTRY_BLOCK,
    LIQUIDITY_STATES,
    LIQUIDITY_THRESHOLDS,
    apply_costs,
    classify_liquidity,
    classify_liquidity_series,
    entry_blocked,
    estimate_cost_bps,
)
from src.composite_states import StateRecord, composite_key


# ---------------------------------------------------------------------------
# LIQUIDITY_STATES ordering
# ---------------------------------------------------------------------------

class TestLiquidityStates:
    def test_ordered_tuple(self):
        assert LIQUIDITY_STATES == (
            "NORMAL_LIQUIDITY",
            "THIN_LIQUIDITY",
            "STRESSED_LIQUIDITY",
            "PANIC_LIQUIDITY",
        )


# ---------------------------------------------------------------------------
# classify_liquidity — scalar
# ---------------------------------------------------------------------------

class TestClassifyLiquidity:
    def test_normal_all_moderate(self):
        result = classify_liquidity(
            volume_zscore_20=0.0,
            atr_14=0.01,
            realized_vol_20=0.01,
            gap_size=0.005,
        )
        assert result == "NORMAL_LIQUIDITY"

    def test_thin_low_volume(self):
        # volume_z < -1.0 triggers THIN
        result = classify_liquidity(
            volume_zscore_20=-1.5,
            atr_14=0.01,
            realized_vol_20=0.01,
            gap_size=0.001,
        )
        assert result == "THIN_LIQUIDITY"

    def test_thin_elevated_vol(self):
        # realized_vol > 0.025 but below STRESSED threshold triggers THIN
        result = classify_liquidity(
            volume_zscore_20=0.0,
            atr_14=0.01,
            realized_vol_20=0.03,
            gap_size=0.001,
        )
        assert result == "THIN_LIQUIDITY"

    def test_stressed_volume_z(self):
        # volume_z > 2.5 triggers STRESSED
        result = classify_liquidity(
            volume_zscore_20=3.0,
            atr_14=0.01,
            realized_vol_20=0.01,
            gap_size=0.001,
        )
        assert result == "STRESSED_LIQUIDITY"

    def test_stressed_realized_vol(self):
        result = classify_liquidity(
            volume_zscore_20=0.0,
            atr_14=0.01,
            realized_vol_20=0.045,
            gap_size=0.001,
        )
        assert result == "STRESSED_LIQUIDITY"

    def test_stressed_gap(self):
        result = classify_liquidity(
            volume_zscore_20=0.0,
            atr_14=0.01,
            realized_vol_20=0.01,
            gap_size=0.04,
        )
        assert result == "STRESSED_LIQUIDITY"

    def test_panic_volume_z(self):
        # volume_z > 4.0 triggers PANIC
        result = classify_liquidity(
            volume_zscore_20=5.0,
            atr_14=0.01,
            realized_vol_20=0.01,
            gap_size=0.001,
        )
        assert result == "PANIC_LIQUIDITY"

    def test_panic_realized_vol(self):
        result = classify_liquidity(
            volume_zscore_20=0.0,
            atr_14=0.01,
            realized_vol_20=0.07,
            gap_size=0.001,
        )
        assert result == "PANIC_LIQUIDITY"

    def test_panic_gap(self):
        result = classify_liquidity(
            volume_zscore_20=0.0,
            atr_14=0.01,
            realized_vol_20=0.01,
            gap_size=0.06,
        )
        assert result == "PANIC_LIQUIDITY"

    def test_panic_wins_over_stressed_same_bar(self):
        # Both PANIC and STRESSED conditions met — PANIC should win
        result = classify_liquidity(
            volume_zscore_20=5.0,   # above PANIC threshold 4.0
            atr_14=0.01,
            realized_vol_20=0.05,   # above STRESSED threshold 0.04
            gap_size=0.001,
        )
        assert result == "PANIC_LIQUIDITY"

    def test_nan_inputs_default_to_normal(self):
        # All NaN → NORMAL (no signals trigger)
        result = classify_liquidity(
            volume_zscore_20=float("nan"),
            atr_14=float("nan"),
            realized_vol_20=float("nan"),
            gap_size=float("nan"),
        )
        assert result == "NORMAL_LIQUIDITY"

    def test_none_inputs_default_to_normal(self):
        result = classify_liquidity(
            volume_zscore_20=None,
            atr_14=None,
            realized_vol_20=None,
            gap_size=None,
        )
        assert result == "NORMAL_LIQUIDITY"

    def test_nan_volume_z_other_signals_normal(self):
        # NaN volume_z + moderate everything else → NORMAL
        result = classify_liquidity(
            volume_zscore_20=float("nan"),
            atr_14=0.01,
            realized_vol_20=0.01,
            gap_size=0.001,
        )
        assert result == "NORMAL_LIQUIDITY"

    def test_nan_volume_z_stressed_rv(self):
        # NaN volume z-score, but realized_vol is STRESSED level → STRESSED
        result = classify_liquidity(
            volume_zscore_20=float("nan"),
            atr_14=0.01,
            realized_vol_20=0.05,
            gap_size=0.001,
        )
        assert result == "STRESSED_LIQUIDITY"

    def test_negative_gap_absolute_value_used(self):
        # Negative gap of -0.06 should trigger PANIC via abs()
        result = classify_liquidity(
            volume_zscore_20=0.0,
            atr_14=0.01,
            realized_vol_20=0.01,
            gap_size=-0.06,
        )
        assert result == "PANIC_LIQUIDITY"


# ---------------------------------------------------------------------------
# classify_liquidity_series — vectorized
# ---------------------------------------------------------------------------

class TestClassifyLiquiditySeries:
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_empty_df_returns_empty_series(self):
        df = pd.DataFrame(
            columns=["volume_zscore_20", "atr_14", "realized_vol_20", "gap_size"]
        )
        result = classify_liquidity_series(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_matches_scalar_classifier_elementwise(self):
        rows = [
            {"volume_zscore_20": 0.0,  "atr_14": 0.01, "realized_vol_20": 0.01,  "gap_size": 0.001},
            {"volume_zscore_20": -1.5, "atr_14": 0.01, "realized_vol_20": 0.01,  "gap_size": 0.001},
            {"volume_zscore_20": 3.0,  "atr_14": 0.01, "realized_vol_20": 0.01,  "gap_size": 0.001},
            {"volume_zscore_20": 5.0,  "atr_14": 0.01, "realized_vol_20": 0.01,  "gap_size": 0.001},
        ]
        df = self._make_df(rows)
        series_result = classify_liquidity_series(df)
        for i, row in enumerate(rows):
            scalar = classify_liquidity(
                row["volume_zscore_20"], row["atr_14"],
                row["realized_vol_20"], row["gap_size"],
            )
            assert series_result.iloc[i] == scalar

    def test_all_states_represented(self):
        rows = [
            {"volume_zscore_20": 0.0,  "atr_14": 0.01, "realized_vol_20": 0.01,  "gap_size": 0.001},
            {"volume_zscore_20": -1.5, "atr_14": 0.01, "realized_vol_20": 0.01,  "gap_size": 0.001},
            {"volume_zscore_20": 3.0,  "atr_14": 0.01, "realized_vol_20": 0.045, "gap_size": 0.001},
            {"volume_zscore_20": 5.0,  "atr_14": 0.01, "realized_vol_20": 0.07,  "gap_size": 0.06},
        ]
        df = self._make_df(rows)
        result = classify_liquidity_series(df)
        assert set(result.values) == set(LIQUIDITY_STATES)

    def test_index_preserved(self):
        df = pd.DataFrame(
            [{"volume_zscore_20": 0.0, "atr_14": 0.01, "realized_vol_20": 0.01, "gap_size": 0.001}],
            index=[42],
        )
        result = classify_liquidity_series(df)
        assert result.index.tolist() == [42]


# ---------------------------------------------------------------------------
# estimate_cost_bps
# ---------------------------------------------------------------------------

class TestEstimateCostBps:
    def test_normal_round_trip(self):
        assert estimate_cost_bps("NORMAL_LIQUIDITY") == pytest.approx(5.0)

    def test_thin_round_trip(self):
        assert estimate_cost_bps("THIN_LIQUIDITY") == pytest.approx(12.0)

    def test_stressed_round_trip(self):
        assert estimate_cost_bps("STRESSED_LIQUIDITY") == pytest.approx(30.0)

    def test_panic_round_trip(self):
        assert estimate_cost_bps("PANIC_LIQUIDITY") == pytest.approx(80.0)

    def test_entry_side_half_cost(self):
        assert estimate_cost_bps("NORMAL_LIQUIDITY", side="entry") == pytest.approx(2.5)

    def test_exit_side_half_cost(self):
        assert estimate_cost_bps("NORMAL_LIQUIDITY", side="exit") == pytest.approx(2.5)

    def test_overrides_honored(self):
        result = estimate_cost_bps("NORMAL_LIQUIDITY", overrides={"NORMAL_LIQUIDITY": 8.0})
        assert result == pytest.approx(8.0)

    def test_overrides_partial(self):
        # Only override PANIC; NORMAL should remain default
        result_panic = estimate_cost_bps("PANIC_LIQUIDITY", overrides={"PANIC_LIQUIDITY": 100.0})
        result_normal = estimate_cost_bps("NORMAL_LIQUIDITY", overrides={"PANIC_LIQUIDITY": 100.0})
        assert result_panic == pytest.approx(100.0)
        assert result_normal == pytest.approx(5.0)

    def test_unknown_state_falls_back_to_normal(self):
        # Unknown state should fall back to NORMAL cost
        result = estimate_cost_bps("UNKNOWN_STATE")
        assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# apply_costs
# ---------------------------------------------------------------------------

class TestApplyCosts:
    def test_empty_returns_empty(self):
        returns = pd.Series(dtype=float)
        states = pd.Series(dtype=str)
        result = apply_costs(returns, states)
        assert len(result) == 0

    def test_normal_round_trip_100bps_gross(self):
        # 100 bps gross return = 0.01 decimal; NORMAL 5 bps cost = 0.0005
        # net = 0.01 - 0.0005 = 0.0095 (95 bps)
        returns = pd.Series([0.01])
        states = pd.Series(["NORMAL_LIQUIDITY"])
        result = apply_costs(returns, states, sides="round_trip")
        assert result.iloc[0] == pytest.approx(0.0095)

    def test_panic_round_trip(self):
        # 100 bps gross; PANIC 80 bps cost = 0.008
        returns = pd.Series([0.01])
        states = pd.Series(["PANIC_LIQUIDITY"])
        result = apply_costs(returns, states)
        assert result.iloc[0] == pytest.approx(0.01 - 0.008)

    def test_entry_side_half_cost(self):
        # NORMAL entry = 2.5 bps = 0.00025 decimal
        returns = pd.Series([0.01])
        states = pd.Series(["NORMAL_LIQUIDITY"])
        result = apply_costs(returns, states, sides="entry")
        assert result.iloc[0] == pytest.approx(0.01 - 0.00025)

    def test_mixed_states_vectorized(self):
        returns = pd.Series([0.01, 0.02])
        states = pd.Series(["NORMAL_LIQUIDITY", "STRESSED_LIQUIDITY"])
        result = apply_costs(returns, states)
        assert result.iloc[0] == pytest.approx(0.01 - 5.0 / 10_000)
        assert result.iloc[1] == pytest.approx(0.02 - 30.0 / 10_000)

    def test_per_bar_sides_series(self):
        # entry = 2.5 bps = 0.00025 decimal; round_trip = 5 bps = 0.0005 decimal
        returns = pd.Series([0.01, 0.01])
        states = pd.Series(["NORMAL_LIQUIDITY", "NORMAL_LIQUIDITY"])
        sides = pd.Series(["entry", "round_trip"])
        result = apply_costs(returns, states, sides=sides)
        assert result.iloc[0] == pytest.approx(0.01 - 0.00025)
        assert result.iloc[1] == pytest.approx(0.01 - 0.0005)

    def test_overrides_applied(self):
        returns = pd.Series([0.01])
        states = pd.Series(["NORMAL_LIQUIDITY"])
        result = apply_costs(returns, states, overrides={"NORMAL_LIQUIDITY": 10.0})
        assert result.iloc[0] == pytest.approx(0.01 - 10.0 / 10_000)


# ---------------------------------------------------------------------------
# entry_blocked
# ---------------------------------------------------------------------------

class TestEntryBlocked:
    def test_panic_is_blocked(self):
        assert entry_blocked("PANIC_LIQUIDITY") is True

    def test_normal_not_blocked(self):
        assert entry_blocked("NORMAL_LIQUIDITY") is False

    def test_thin_not_blocked(self):
        assert entry_blocked("THIN_LIQUIDITY") is False

    def test_stressed_not_blocked(self):
        assert entry_blocked("STRESSED_LIQUIDITY") is False

    def test_unknown_state_not_blocked(self):
        assert entry_blocked("UNKNOWN_STATE") is False


# ---------------------------------------------------------------------------
# Smoke: StateRecord integration
# ---------------------------------------------------------------------------

class TestStateRecordIntegration:
    def test_build_state_record_with_liquidity(self):
        liq = classify_liquidity(
            volume_zscore_20=0.5,
            atr_14=0.01,
            realized_vol_20=0.015,
            gap_size=0.002,
        )
        record = StateRecord(base_state="BULL", liquidity_state=liq)
        assert record.liquidity_state == "NORMAL_LIQUIDITY"

    def test_composite_key_includes_liquidity(self):
        liq = classify_liquidity(
            volume_zscore_20=5.0,
            atr_14=0.01,
            realized_vol_20=0.07,
            gap_size=0.06,
        )
        record = StateRecord(base_state="BEAR", liquidity_state=liq)
        key = composite_key(record)
        assert "PANIC_LIQUIDITY" in key
        assert "BEAR" in key

    def test_composite_key_all_states(self):
        for state in LIQUIDITY_STATES:
            record = StateRecord(base_state="NEUTRAL", liquidity_state=state)
            key = composite_key(record)
            assert state in key
