"""Clustering Agent - structural behavior grouping for assets."""

import numpy as np
import pandas as pd


CLUSTER_LABELS = {
    0: "low_vol_defensive",
    1: "medium_vol_cyclical",
    2: "high_vol_speculative",
}


def asset_behavior_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize each asset into clustering features."""
    rows = []
    for symbol, frame in data.items():
        if "Close" not in frame.columns:
            raise ValueError(f"{symbol} frame must include a Close column")

        close = frame["Close"].astype(float).dropna()
        returns = close.pct_change().dropna()
        volume = frame["Volume"].astype(float).reindex(close.index) if "Volume" in frame.columns else pd.Series(dtype=float)

        rows.append(
            {
                "symbol": symbol.upper(),
                "realized_volatility": _safe_float(returns.std(ddof=0) * np.sqrt(252)),
                "trend_persistence": _safe_float(returns.autocorr(lag=1)),
                "reversal_frequency": _reversal_frequency(returns),
                "volume_stability": _volume_stability(volume),
                "average_return": _safe_float(returns.mean()),
            }
        )

    if not rows:
        return pd.DataFrame().rename_axis("symbol")

    return pd.DataFrame(rows).set_index("symbol").sort_index()


def cluster_assets(data: dict[str, pd.DataFrame], n_clusters: int = 3) -> pd.DataFrame:
    """Assign assets to interpretable behavior clusters sorted by volatility."""
    features = asset_behavior_features(data)
    if features.empty:
        return features
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if n_clusters > len(features):
        raise ValueError("n_clusters cannot exceed number of assets")

    ranked = features.sort_values(["realized_volatility", "reversal_frequency"], ascending=[True, True]).copy()
    cluster_ids = []
    for rank in range(len(ranked)):
        cluster_ids.append(min(int(np.floor(rank * n_clusters / len(ranked))), n_clusters - 1))

    ranked["cluster_id"] = cluster_ids
    ranked["cluster_label"] = ranked["cluster_id"].map(lambda value: CLUSTER_LABELS.get(value, f"cluster_{value}"))
    return ranked.sort_index()


def _reversal_frequency(returns: pd.Series) -> float:
    signs = np.sign(returns[returns != 0.0])
    if len(signs) < 2:
        return np.nan
    return float((signs != signs.shift(1)).iloc[1:].mean())


def _volume_stability(volume: pd.Series) -> float:
    clean = volume.dropna()
    if len(clean) < 2 or np.isclose(clean.mean(), 0.0):
        return np.nan
    coefficient_of_variation = clean.std(ddof=0) / clean.mean()
    return float(1.0 / (1.0 + coefficient_of_variation))


def _safe_float(value: float) -> float:
    return float(value) if pd.notna(value) and np.isfinite(value) else np.nan
