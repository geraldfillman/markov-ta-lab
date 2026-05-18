"""Tests for volatility regime classification."""

import numpy as np
import pandas as pd

from src.config import VOL_STATE_LABELS
from src.volatility import classify_vol_state, vol_adjusted_position_size


def test_classify_vol_state_uses_rolling_percentiles():
    index = pd.date_range("2024-01-02", periods=6, freq="B", name="Date")
    df = pd.DataFrame(
        {"realized_vol_20": [0.10, 0.11, 0.12, 0.13, 0.30, 0.08]},
        index=index,
    )

    states = classify_vol_state(df, lookback=3)

    assert states.index.equals(index)
    assert states.iloc[4] == 2
    assert states.iloc[5] == 0
    assert states.map(VOL_STATE_LABELS).iloc[4] == "HIGH_VOL"


def test_vol_adjusted_position_size_is_inverse_to_atr():
    low_atr = vol_adjusted_position_size(100_000.0, 0.01, atr=2.0, atr_multiple=2.0)
    high_atr = vol_adjusted_position_size(100_000.0, 0.01, atr=4.0, atr_multiple=2.0)

    assert low_atr == 250.0
    assert high_atr == 125.0


def test_vol_adjusted_position_size_rejects_non_positive_risk_inputs():
    with np.testing.assert_raises(ValueError):
        vol_adjusted_position_size(100_000.0, 0.01, atr=0.0)
