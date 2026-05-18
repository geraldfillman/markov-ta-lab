"""Cross-asset stress overlay: classifies market stress and downgrades ConditionalForecast confidence."""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

from conditional_markov import ConditionalForecast

STRESS_LEVELS: tuple[str, ...] = ("CALM", "ELEVATED", "STRESS", "PANIC")

_CONFIDENCE_ORDER = ("low", "medium", "high")


def classify_cross_asset_stress(
    vix: Optional[float],
    credit_spread: Optional[float],
    dollar_zscore: Optional[float],
    thresholds: Optional[dict] = None,
) -> str:
    """Return the highest stress level triggered by any input signal (NaN-safe)."""
    if thresholds is None:
        thresholds = {}

    panic_vix = thresholds.get("panic_vix", 40.0)
    panic_cs = thresholds.get("panic_cs", 8.0)
    panic_dz = thresholds.get("panic_dz", 3.0)
    stress_vix = thresholds.get("stress_vix", 28.0)
    stress_cs = thresholds.get("stress_cs", 5.5)
    stress_dz = thresholds.get("stress_dz", 2.0)
    elevated_vix = thresholds.get("elevated_vix", 20.0)
    elevated_cs = thresholds.get("elevated_cs", 4.0)
    elevated_dz = thresholds.get("elevated_dz", 1.5)

    def _safe(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        try:
            return None if math.isnan(v) else v
        except (TypeError, ValueError):
            return None

    v = _safe(vix)
    cs = _safe(credit_spread)
    dz = _safe(dollar_zscore)
    abs_dz = abs(dz) if dz is not None else None

    # PANIC
    if (v is not None and v > panic_vix) or (cs is not None and cs > panic_cs) or (abs_dz is not None and abs_dz > panic_dz):
        return "PANIC"

    # STRESS
    if (v is not None and v > stress_vix) or (cs is not None and cs > stress_cs) or (abs_dz is not None and abs_dz > stress_dz):
        return "STRESS"

    # ELEVATED
    if (v is not None and v > elevated_vix) or (cs is not None and cs > elevated_cs) or (abs_dz is not None and abs_dz > elevated_dz):
        return "ELEVATED"

    return "CALM"


def apply_stress_overlay(forecast: ConditionalForecast, stress: str) -> ConditionalForecast:
    """Return a new ConditionalForecast with confidence downgraded and stress warning appended."""
    warning = f"cross_asset_{stress.lower()}"
    new_warnings = forecast.warnings + (warning,)

    current = forecast.confidence
    new_confidence = current

    if stress == "STRESS":
        # high -> medium; medium and low unchanged
        if current == "high":
            new_confidence = "medium"
    elif stress == "PANIC":
        # high/medium -> low
        if current in ("high", "medium"):
            new_confidence = "low"

    return dataclasses.replace(forecast, confidence=new_confidence, warnings=new_warnings)
