"""Tests for the HMM Regime Agent."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")

from src.hmm_models import (
    DEFAULT_EMISSION_COLUMNS,
    HMMFitResult,
    fit_hmm,
    label_regimes_by_behavior,
    regime_filter_signal,
    select_emissions,
)


def _two_regime_emissions(n_per_regime: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    calm = rng.normal(loc=[0.001, 0.005], scale=[0.002, 0.001], size=(n_per_regime, 2))
    storm = rng.normal(loc=[-0.002, 0.020], scale=[0.010, 0.005], size=(n_per_regime, 2))
    data = np.vstack([calm, storm, calm, storm])
    index = pd.date_range("2024-01-01", periods=len(data), freq="B")
    return pd.DataFrame(data, columns=["return_1d", "realized_vol_20"], index=index)


def test_fit_hmm_returns_regimes_aligned_to_input_index():
    emissions = _two_regime_emissions()
    result = fit_hmm(emissions, n_regimes=2, n_starts=3, random_state=42)

    assert isinstance(result, HMMFitResult)
    assert result.regimes.index.equals(emissions.index)
    assert result.regimes.dropna().astype(int).between(0, 1).all()
    assert result.transition_matrix.shape == (2, 2)
    np.testing.assert_allclose(result.transition_matrix.sum(axis=1), 1.0, atol=1e-6)


def test_fit_hmm_separates_high_and_low_vol_regimes():
    emissions = _two_regime_emissions()
    result = fit_hmm(emissions, n_regimes=2, n_starts=5, random_state=7)
    labels = label_regimes_by_behavior(
        result.regimes,
        emissions["return_1d"],
        emissions["realized_vol_20"],
    )
    assert set(labels.values()) <= {
        "RISK_ON_TREND",
        "RISK_ON_VOLATILE",
        "RISK_OFF_VOLATILE",
        "RISK_OFF_QUIET",
    }
    assert len(labels) == 2


def test_fit_hmm_rejects_invalid_args():
    emissions = _two_regime_emissions()
    with pytest.raises(ValueError):
        fit_hmm(emissions, n_regimes=0)
    with pytest.raises(ValueError):
        fit_hmm(emissions, n_starts=0)
    with pytest.raises(ValueError):
        fit_hmm(pd.DataFrame(columns=["return_1d"]))


def test_fit_hmm_needs_enough_clean_rows():
    emissions = pd.DataFrame(
        {"return_1d": [0.01, np.nan, 0.02]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    with pytest.raises(ValueError):
        fit_hmm(emissions, n_regimes=3)


def test_select_emissions_picks_available_columns():
    df = pd.DataFrame(
        {
            "return_1d": [0.0, 0.01],
            "realized_vol_20": [0.1, 0.2],
            "noise": [1, 2],
        }
    )
    result = select_emissions(df)
    assert set(result.columns) <= set(DEFAULT_EMISSION_COLUMNS)
    assert list(result.columns) == ["return_1d", "realized_vol_20"]


def test_select_emissions_raises_when_no_columns_present():
    with pytest.raises(ValueError):
        select_emissions(pd.DataFrame({"unrelated": [1, 2]}))


def test_regime_filter_signal_masks_unallowed_regimes():
    regimes = pd.Series([0, 1, 0, np.nan, 1], dtype="float64")
    labels_map = {0: "RISK_ON_TREND", 1: "RISK_OFF_QUIET"}
    mask = regime_filter_signal(regimes, ["RISK_ON_TREND"], labels_map)
    assert mask.tolist() == [True, False, True, False, False]


def test_regime_filter_signal_empty_allowed_raises():
    regimes = pd.Series([0, 1])
    with pytest.raises(ValueError):
        regime_filter_signal(regimes, [], {0: "A"})
