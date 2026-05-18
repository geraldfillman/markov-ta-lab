"""HMM Regime Agent — hidden regime detection.

Responsibilities (from playbook §3.9):
- Fit Gaussian HMM to emissions (returns, vol, ATR pct, volume z, S/R dist).
- Multiple random starts; pick the best log-likelihood.
- Label hidden states by realised behavior (not component number).
- Compare HMM regimes to visible technical states.
- Use HMM as a filter, not a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_EMISSION_COLUMNS = (
    "return_1d",
    "realized_vol_20",
    "atr_14",
    "volume_zscore_20",
    "dist_to_support_atr",
    "dist_to_resistance_atr",
)


@dataclass
class HMMFitResult:
    """Container for a fitted Gaussian HMM."""

    model: object
    regimes: pd.Series
    log_likelihood: float
    means: np.ndarray
    covariances: np.ndarray
    transition_matrix: np.ndarray
    columns: tuple[str, ...]


def fit_hmm(
    emissions: pd.DataFrame,
    n_regimes: int = 3,
    n_starts: int = 10,
    covariance_type: str = "diag",
    n_iter: int = 200,
    random_state: int = 0,
) -> HMMFitResult:
    """Fit a Gaussian HMM with multiple random starts; return the best by log-likelihood.

    Each start uses `random_state + i` as seed. Rows with any NaN are dropped
    before fitting, and the returned `regimes` Series is reindexed to the
    original emissions index (NaN for dropped rows).
    """
    if n_regimes < 1:
        raise ValueError("n_regimes must be at least 1")
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
    if emissions.empty:
        raise ValueError("emissions must contain at least one row")

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("hmmlearn is required to fit HMM models") from exc

    clean = emissions.dropna()
    if len(clean) < n_regimes * 2:
        raise ValueError(
            f"need at least {n_regimes * 2} clean rows to fit a {n_regimes}-state HMM"
        )

    features = clean.to_numpy(dtype=float)
    columns = tuple(clean.columns)

    best_ll = -np.inf
    best_model = None
    best_regimes: np.ndarray | None = None

    for start_idx in range(n_starts):
        seed = random_state + start_idx
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=seed,
        )
        try:
            model.fit(features)
            ll = float(model.score(features))
        except Exception:  # pragma: no cover - hmmlearn convergence failures
            continue
        if ll > best_ll:
            best_ll = ll
            best_model = model
            best_regimes = model.predict(features)

    if best_model is None or best_regimes is None:
        raise RuntimeError("All HMM fits failed to converge")

    regimes = pd.Series(best_regimes, index=clean.index, name="regime", dtype="int64")
    regimes = regimes.reindex(emissions.index)

    return HMMFitResult(
        model=best_model,
        regimes=regimes,
        log_likelihood=best_ll,
        means=np.asarray(best_model.means_, dtype=float),
        covariances=_covariances_from_model(best_model, n_regimes),
        transition_matrix=np.asarray(best_model.transmat_, dtype=float),
        columns=columns,
    )


def label_regimes_by_behavior(
    regimes: pd.Series,
    returns: pd.Series,
    volatility: pd.Series,
) -> dict[int, str]:
    """Assign human-readable names to hidden regimes by realised return and vol."""
    aligned = pd.DataFrame(
        {"regime": regimes, "returns": returns, "vol": volatility}
    ).dropna()
    if aligned.empty:
        return {}

    aligned["regime"] = aligned["regime"].astype(int)
    stats = aligned.groupby("regime").agg(
        avg_return=("returns", "mean"),
        avg_vol=("vol", "mean"),
    )
    if stats.empty:
        return {}

    return_median = float(stats["avg_return"].median())
    vol_median = float(stats["avg_vol"].median())

    labels: dict[int, str] = {}
    for regime_id, row in stats.iterrows():
        bullish = row["avg_return"] >= return_median
        high_vol = row["avg_vol"] >= vol_median
        if bullish and not high_vol:
            labels[int(regime_id)] = "RISK_ON_TREND"
        elif bullish and high_vol:
            labels[int(regime_id)] = "RISK_ON_VOLATILE"
        elif not bullish and high_vol:
            labels[int(regime_id)] = "RISK_OFF_VOLATILE"
        else:
            labels[int(regime_id)] = "RISK_OFF_QUIET"

    return labels


def select_emissions(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the subset of emission columns that exist in df."""
    candidates = tuple(columns) if columns is not None else DEFAULT_EMISSION_COLUMNS
    available = [column for column in candidates if column in df.columns]
    if not available:
        raise ValueError(
            "df does not contain any of the requested emission columns: "
            f"{list(candidates)}"
        )
    return df[available].copy()


def regime_filter_signal(
    regimes: pd.Series,
    allowed_labels: Sequence[str],
    labels_map: dict[int, str],
) -> pd.Series:
    """Boolean series — True when the bar's regime label is in allowed_labels."""
    allowed = set(allowed_labels)
    if not allowed:
        raise ValueError("allowed_labels must not be empty")

    def _to_label(value: object) -> str | None:
        if pd.isna(value):
            return None
        return labels_map.get(int(value))

    return regimes.map(_to_label).isin(allowed)


def _covariances_from_model(model: object, n_regimes: int) -> np.ndarray:
    """Normalise hmmlearn covariance output to (n_components, n_features, n_features)."""
    covars = getattr(model, "covars_", None)
    if covars is None:
        return np.zeros((n_regimes, 1, 1), dtype=float)
    array = np.asarray(covars, dtype=float)
    if array.ndim == 2:
        return np.stack([np.diag(row) for row in array])
    return array
