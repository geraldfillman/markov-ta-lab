"""Tests for support/resistance level features."""

import numpy as np
import pandas as pd

from src.levels import detect_levels


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=8, freq="B", name="Date")
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0, 12.5, 13.5, 14.0, 15.0],
            "High": [11.0, 12.0, 15.0, 14.0, 13.0, 16.0, 17.0, 18.0],
            "Low": [9.0, 10.0, 11.0, 12.0, 10.0, 12.0, 13.0, 14.0],
            "Close": [10.5, 11.5, 14.0, 12.5, 12.0, 15.0, 16.0, 17.0],
            "Volume": [100, 110, 120, 130, 140, 150, 160, 170],
            "atr_14": [2.0] * 8,
        },
        index=index,
    )


def test_levels_use_prior_data_only():
    result = detect_levels(_sample_frame(), lookback=3, atr_mult=0.5)

    # At index 3, prior highs are 11, 12, 15 and prior lows are 9, 10, 11.
    row = result.iloc[3]
    assert row["nearest_resistance"] == 15.0
    assert row["nearest_support"] == 9.0

    changed_current_bar = _sample_frame()
    changed_current_bar.iloc[3, changed_current_bar.columns.get_loc("High")] = 100.0
    changed = detect_levels(changed_current_bar, lookback=3, atr_mult=0.5)

    assert changed.iloc[3]["nearest_resistance"] == row["nearest_resistance"]


def test_zone_width_is_atr_based():
    result = detect_levels(_sample_frame(), lookback=3, atr_mult=0.5)
    row = result.iloc[3]

    assert row["resistance_zone_low"] == 14.0
    assert row["resistance_zone_high"] == 16.0
    assert row["support_zone_low"] == 8.0
    assert row["support_zone_high"] == 10.0


def test_nearest_level_and_distance_populated_after_warmup():
    result = detect_levels(_sample_frame(), lookback=3, atr_mult=0.5)
    warmed = result.iloc[3:]

    assert warmed["nearest_support"].notna().all()
    assert warmed["nearest_resistance"].notna().all()
    assert np.isclose(result.iloc[3]["dist_to_resistance_atr"], (15.0 - 12.5) / 2.0)
    assert np.isclose(result.iloc[3]["dist_to_support_atr"], (12.5 - 9.0) / 2.0)
