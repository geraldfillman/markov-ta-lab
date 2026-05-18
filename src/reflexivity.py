"""Reflexivity diagnostics: detects cascade/reflexivity regimes. Diagnostic-only, no prediction change."""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

from conditional_markov import ConditionalForecast

# Weights sum to 1.0:
#   price_change_pct (abs, clipped): 0.25 — magnitude of price dislocation
#   volume_zscore_20 (clipped abs):  0.35 — volume surge (key reflexivity signal)
#   correlation_to_market (abs):     0.20 — co-movement amplifying cascade
#   rolling_kurtosis (normalised):   0.20 — fat-tail clustering
_WEIGHTS = (0.25, 0.35, 0.20, 0.20)


def _safe(v: Optional[float]) -> float:
    """Return 0.0 if v is None or NaN."""
    if v is None:
        return 0.0
    try:
        return 0.0 if math.isnan(v) else float(v)
    except (TypeError, ValueError):
        return 0.0


def compute_reflexivity_score(
    price_change_pct: Optional[float],
    volume_zscore_20: Optional[float],
    correlation_to_market: Optional[float],
    rolling_kurtosis: Optional[float],
) -> float:
    """Return a reflexivity score in [0, 1].

    Higher score = more reflexive (price moves driven by volume + correlated cascade).
    Weights: price_change=0.25, volume_z=0.35, correlation=0.20, kurtosis=0.20.
    """
    pc = min(max(abs(_safe(price_change_pct)) / 0.05, 0.0), 1.0)   # 5% move as ceiling
    vz = min(max(abs(_safe(volume_zscore_20)) / 4.0, 0.0), 1.0)    # z=4 as ceiling
    cm = min(max(abs(_safe(correlation_to_market)), 0.0), 1.0)      # already [0,1]
    rk = min(max((_safe(rolling_kurtosis) - 3.0) / 10.0, 0.0), 1.0) # excess kurtosis, ceiling=10

    score = _WEIGHTS[0] * pc + _WEIGHTS[1] * vz + _WEIGHTS[2] * cm + _WEIGHTS[3] * rk
    return round(min(max(score, 0.0), 1.0), 6)


def apply_reflexivity_warning(
    forecast: ConditionalForecast,
    score: float,
    threshold: float = 0.7,
) -> ConditionalForecast:
    """Return a new ConditionalForecast with a reflexivity warning if score >= threshold.

    Does NOT modify confidence or next_state_probs — purely diagnostic.
    """
    if score < threshold:
        return forecast

    new_warnings = forecast.warnings + ("reflexivity_elevated",)
    return dataclasses.replace(forecast, warnings=new_warnings)
