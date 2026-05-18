"""Tests for src.composite_states."""

from __future__ import annotations

import pytest

from src.composite_states import (
    MODIFIER_ORDER,
    SEPARATOR,
    WILDCARD,
    StateRecord,
    composite_key,
    fallback_chain,
    truncate_to_level,
)


def _full_record() -> StateRecord:
    return StateRecord(
        base_state="FAILED_BREAKOUT",
        volatility_state="VOL_EXPANSION",
        macro_state="RATES_UP",
        calendar_state="PRE_OPEX",
        liquidity_state="NORMAL_LIQUIDITY",
    )


class TestStateRecord:
    def test_base_only_record_is_valid(self):
        rec = StateRecord(base_state="CONTINUATION")
        assert rec.base_state == "CONTINUATION"
        for field in MODIFIER_ORDER:
            assert getattr(rec, field) is None

    def test_record_is_immutable(self):
        rec = _full_record()
        with pytest.raises(Exception):
            rec.base_state = "X"  # type: ignore[misc]

    def test_record_is_hashable(self):
        rec = _full_record()
        assert {rec: 1}[rec] == 1

    def test_with_overrides_returns_new_instance(self):
        rec = _full_record()
        updated = rec.with_overrides(calendar_state=None)
        assert updated is not rec
        assert rec.calendar_state == "PRE_OPEX"
        assert updated.calendar_state is None

    def test_with_overrides_rejects_unknown_field(self):
        with pytest.raises(ValueError, match="Unknown StateRecord fields"):
            _full_record().with_overrides(bogus="x")  # type: ignore[arg-type]


class TestCompositeKey:
    def test_full_composite_uses_canonical_order(self):
        key = composite_key(_full_record())
        assert key == SEPARATOR.join([
            "FAILED_BREAKOUT",
            "VOL_EXPANSION",
            "RATES_UP",
            "PRE_OPEX",
            "NORMAL_LIQUIDITY",
        ])

    def test_missing_modifiers_become_wildcards(self):
        rec = StateRecord(base_state="CHOP_OR_NO_EDGE")
        key = composite_key(rec)
        assert key == SEPARATOR.join(["CHOP_OR_NO_EDGE"] + [WILDCARD] * len(MODIFIER_ORDER))

    def test_partial_modifiers_keep_position(self):
        rec = StateRecord(
            base_state="CONTINUATION",
            macro_state="RATES_UP",
        )
        key = composite_key(rec)
        parts = key.split(SEPARATOR)
        assert parts[0] == "CONTINUATION"
        assert parts[1 + MODIFIER_ORDER.index("macro_state")] == "RATES_UP"
        assert parts[1 + MODIFIER_ORDER.index("volatility_state")] == WILDCARD

    def test_field_subset_restricts_modifiers(self):
        rec = _full_record()
        key = composite_key(rec, fields=("volatility_state", "macro_state"))
        assert key == "FAILED_BREAKOUT|VOL_EXPANSION|RATES_UP"

    def test_empty_base_state_rejected(self):
        with pytest.raises(ValueError, match="base_state"):
            composite_key(StateRecord(base_state=""))

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="Unknown modifier fields"):
            composite_key(_full_record(), fields=("not_a_field",))

    def test_semantically_equal_records_produce_equal_keys(self):
        a = _full_record()
        b = StateRecord(
            base_state="FAILED_BREAKOUT",
            liquidity_state="NORMAL_LIQUIDITY",
            calendar_state="PRE_OPEX",
            macro_state="RATES_UP",
            volatility_state="VOL_EXPANSION",
        )
        assert composite_key(a) == composite_key(b)


class TestTruncateToLevel:
    def test_level_zero_yields_base_only(self):
        truncated = truncate_to_level(_full_record(), 0)
        assert truncated.base_state == "FAILED_BREAKOUT"
        for field in MODIFIER_ORDER:
            assert getattr(truncated, field) is None

    def test_full_level_is_identity(self):
        rec = _full_record()
        assert truncate_to_level(rec, len(MODIFIER_ORDER)) == rec

    def test_intermediate_level_keeps_prefix(self):
        truncated = truncate_to_level(_full_record(), 2)
        assert truncated.volatility_state == "VOL_EXPANSION"
        assert truncated.macro_state == "RATES_UP"
        assert truncated.calendar_state is None
        assert truncated.liquidity_state is None

    @pytest.mark.parametrize("level", [-1, len(MODIFIER_ORDER) + 1])
    def test_out_of_range_level_rejected(self, level):
        with pytest.raises(ValueError, match="level must be in"):
            truncate_to_level(_full_record(), level)


class TestFallbackChain:
    def test_chain_starts_most_specific_ends_base_only(self):
        chain = fallback_chain(_full_record())
        assert len(chain) == len(MODIFIER_ORDER) + 1
        assert chain[0] == _full_record()
        assert chain[-1] == StateRecord(base_state="FAILED_BREAKOUT")

    def test_chain_is_monotonically_dropping_modifiers(self):
        chain = fallback_chain(_full_record())
        for earlier, later in zip(chain, chain[1:]):
            earlier_mods = sum(getattr(earlier, f) is not None for f in MODIFIER_ORDER)
            later_mods = sum(getattr(later, f) is not None for f in MODIFIER_ORDER)
            assert later_mods == earlier_mods - 1
