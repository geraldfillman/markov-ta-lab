"""Tests for deterministic state labeling."""

import pandas as pd

from src.config import N_STATES, STATE_LABELS
from src.states import flag_rare_states, label_states, state_frequency_report


def _base_row() -> dict[str, float]:
    return {
        "Close": 100.0,
        "sma_20": 99.0,
        "sma_50": 98.0,
        "atr_14": 2.0,
        "bb_width_20": 0.10,
        "nearest_support": 90.0,
        "nearest_resistance": 110.0,
        "support_zone_low": 89.0,
        "support_zone_high": 91.0,
        "resistance_zone_low": 109.0,
        "resistance_zone_high": 111.0,
        "dist_to_support_atr": 5.0,
        "dist_to_resistance_atr": 5.0,
    }


def _frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    merged = []
    for overrides in rows:
        row = _base_row()
        row.update(overrides)
        merged.append(row)

    return pd.DataFrame(
        merged,
        index=pd.date_range("2024-01-02", periods=len(merged), freq="B", name="Date"),
    )


def test_every_bar_has_one_state_in_valid_range():
    states = label_states(
        _frame(
            [
                {"Close": 100.0},
                {"Close": 90.5, "dist_to_support_atr": 0.25},
                {"Close": 109.5, "dist_to_resistance_atr": 0.25},
                {"Close": 112.0},
            ]
        )
    )

    assert states.notna().all()
    assert states.index.name == "Date"
    assert states.between(0, N_STATES - 1).all()
    assert states.dtype == "int64"


def test_state_rules_are_deterministic_and_mutually_exclusive():
    df = _frame(
        [
            {"Close": 90.5, "dist_to_support_atr": 0.25},
            {"Close": 88.0},
            {"Close": 90.5, "dist_to_support_atr": 0.25},
            {"Close": 108.8, "dist_to_resistance_atr": 0.6, "bb_width_20": 0.03},
            {"Close": 109.5, "dist_to_resistance_atr": 0.25},
            {"Close": 112.5, "sma_20": 105.0, "sma_50": 100.0},
            {"Close": 109.5},
            {"Close": 108.0},
        ]
    )

    states = label_states(df, config={"compression_bb_width": 0.05})

    assert states.tolist() == [
        2,   # TOUCHING_SUPPORT
        4,   # SUPPORT_BREAKDOWN
        3,   # SUPPORT_RECLAIM
        6,   # COMPRESSION_BELOW_RESISTANCE
        5,   # APPROACHING_RESISTANCE
        7,   # RESISTANCE_BREAKOUT
        8,   # BREAKOUT_RETEST
        9,   # FAILED_BREAKOUT
    ]


def test_missing_features_fall_back_to_chop():
    df = _frame([{"Close": 100.0}])
    df.loc[df.index[0], "nearest_support"] = None

    states = label_states(df)

    assert states.iloc[0] == 11


def test_state_frequency_report_and_rare_flags():
    states = pd.Series([0, 0, 5, 5, 5, 11], name="state")

    report = state_frequency_report(states)
    rare = flag_rare_states(states, min_count=2)

    assert report.loc[5, "label"] == STATE_LABELS[5]
    assert report.loc[5, "count"] == 3
    assert report.loc[5, "percent"] == 50.0
    assert rare == [11]
