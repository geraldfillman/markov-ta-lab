"""Tail-risk override layer: downgrades confidence or zeros next_state_probs in extreme regimes."""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

from conditional_markov import ConditionalForecast

# Weights sum to 1.0:
#   realized_vol_20 (normalised): 0.35  — primary driver of tail events
#   atr_zscore (clipped abs):     0.30  — range expansion signal
#   gap_size (abs, clipped 0-1):  0.20  — overnight jump risk
#   drawdown_pct (clipped 0-1):   0.15  — magnitude of ongoing loss
_WEIGHTS = (0.35, 0.30, 0.20, 0.15)


def _safe(v: Optional[float]) -> float:
    """Return 0.0 if v is None or NaN."""
    if v is None:
        return 0.0
    try:
        return 0.0 if math.isnan(v) else float(v)
    except (TypeError, ValueError):
        return 0.0


def compute_tail_risk_score(
    realized_vol_20: Optional[float],
    atr_zscore: Optional[float],
    gap_size: Optional[float],
    drawdown_pct: Optional[float],
) -> float:
    """Return a tail-risk score in [0, 1].

    Weights: realized_vol_20=0.35, atr_zscore=0.30, gap_size=0.20, drawdown_pct=0.15.
    Each input is normalised/clipped to [0, 1] before weighting.
    """
    rv = min(max(_safe(realized_vol_20) / 0.60, 0.0), 1.0)   # 60% ann-vol as ceiling
    az = min(max(abs(_safe(atr_zscore)) / 4.0, 0.0), 1.0)    # z=4 as ceiling
    gs = min(max(abs(_safe(gap_size)) / 0.05, 0.0), 1.0)     # 5% gap as ceiling
    dd = min(max(abs(_safe(drawdown_pct)) / 0.30, 0.0), 1.0) # 30% drawdown as ceiling

    score = _WEIGHTS[0] * rv + _WEIGHTS[1] * az + _WEIGHTS[2] * gs + _WEIGHTS[3] * dd
    return round(min(max(score, 0.0), 1.0), 6)


def apply_tail_risk_override(
    forecast: ConditionalForecast,
    score: float,
    *,
    threshold_high: float = 0.75,
    threshold_extreme: float = 0.90,
) -> ConditionalForecast:
    """Return a new ConditionalForecast adjusted for tail risk.

    Below threshold_high: unchanged.
    threshold_high <= score < threshold_extreme: confidence downgraded one notch, warning added.
    score >= threshold_extreme: confidence set to 'low', next_state_probs zeroed, warning added.
    """
    _CONFIDENCE_ORDER = ("low", "medium", "high")

    if score < threshold_high:
        return forecast

    if score < threshold_extreme:
        # Downgrade one notch
        idx = _CONFIDENCE_ORDER.index(forecast.confidence) if forecast.confidence in _CONFIDENCE_ORDER else 1
        new_confidence = _CONFIDENCE_ORDER[max(idx - 1, 0)]
        new_warnings = forecast.warnings + ("tail_risk_elevated",)
        return dataclasses.replace(forecast, confidence=new_confidence, warnings=new_warnings)

    # Extreme: zero out probs, force low confidence
    zeroed_probs = {k: 0.0 for k in forecast.next_state_probs}
    new_warnings = forecast.warnings + ("tail_risk_extreme",)
    return dataclasses.replace(
        forecast,
        confidence="low",
        next_state_probs=zeroed_probs,
        warnings=new_warnings,
    )
