"""Tests for structural asset clustering."""

import pandas as pd

from src.clustering import asset_behavior_features, cluster_assets


def _asset(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Close": closes,
            "Volume": volumes or [100.0] * len(closes),
        },
        index=pd.date_range("2024-01-02", periods=len(closes), freq="B", name="Date"),
    )


def test_asset_behavior_features_summarize_volatility_and_trend():
    data = {
        "LOW": _asset([100.0, 101.0, 102.0, 103.0, 104.0]),
        "HIGH": _asset([100.0, 120.0, 90.0, 125.0, 80.0]),
    }

    features = asset_behavior_features(data)

    assert features.index.tolist() == ["HIGH", "LOW"]
    assert features.loc["HIGH", "realized_volatility"] > features.loc["LOW", "realized_volatility"]
    assert features.loc["LOW", "trend_persistence"] > features.loc["HIGH", "trend_persistence"]
    assert {"reversal_frequency", "volume_stability"}.issubset(features.columns)


def test_cluster_assets_groups_low_vol_assets_together():
    data = {
        "LOW_A": _asset([100.0, 101.0, 102.0, 103.0, 104.0]),
        "LOW_B": _asset([50.0, 50.5, 51.0, 51.5, 52.0]),
        "HIGH": _asset([100.0, 120.0, 90.0, 125.0, 80.0]),
    }

    clusters = cluster_assets(data, n_clusters=2)

    assert clusters.loc["LOW_A", "cluster_id"] == clusters.loc["LOW_B", "cluster_id"]
    assert clusters.loc["HIGH", "cluster_id"] != clusters.loc["LOW_A", "cluster_id"]
    assert "cluster_label" in clusters.columns
