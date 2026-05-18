"""Dynamic cost model: liquidity-state classification and trade cost estimation.

Classifies each bar into a liquidity regime using volume, volatility, and gap
signals, then provides cost-in-bps and entry-block logic for backtesting.
Pure: no I/O, no side effects.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Liquidity state constants — ordered most-liquid to least
# ---------------------------------------------------------------------------

LIQUIDITY_STATES: tuple[str, ...] = (
    "NORMAL_LIQUIDITY",
    "THIN_LIQUIDITY",
    "STRESSED_LIQUIDITY",
    "PANIC_LIQUIDITY",
)

# ---------------------------------------------------------------------------
# Classification thresholds (module-level defaults, fully overridable)
# ---------------------------------------------------------------------------

LIQUIDITY_THRESHOLDS: dict[str, dict[str, float]] = {
    # PANIC: extreme dislocation — very high volume spike OR high realized vol
    # OR large overnight gap.  Entry is blocked in this regime.
    "PANIC": {
        "volume_z_above": 4.0,      # volume z-score threshold (spike up)
        "realized_vol_above": 0.06, # 20-bar realized vol as decimal (6 %)
        "gap_abs_above": 0.05,      # abs overnight gap as decimal (5 %)
    },
    # STRESSED: significant but tradeable stress — elevated vol or moderate gap.
    "STRESSED": {
        "volume_z_above": 2.5,
        "realized_vol_above": 0.04,
        "gap_abs_above": 0.03,
    },
    # THIN: low-volume session or mildly elevated vol; wider spreads expected.
    "THIN": {
        "volume_z_below": -1.0,     # volume z-score threshold (drying up)
        "realized_vol_above": 0.025,
    },
    # NORMAL: all other conditions — baseline cost applies.
}

# ---------------------------------------------------------------------------
# Cost table (bps) and entry-block set
# ---------------------------------------------------------------------------

COST_BPS: dict[str, float] = {
    "NORMAL_LIQUIDITY":   5.0,   # 5 bps base round-trip
    "THIN_LIQUIDITY":    12.0,
    "STRESSED_LIQUIDITY": 30.0,
    "PANIC_LIQUIDITY":   80.0,   # defensive / entry-block regime
}

ENTRY_BLOCK: frozenset[str] = frozenset({"PANIC_LIQUIDITY"})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(value: Optional[float]) -> Optional[float]:
    """Return None if value is None or NaN, else the value itself."""
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        return None
    return value


def _is_panic(
    volume_z: Optional[float],
    realized_vol: Optional[float],
    gap_abs: Optional[float],
    t: dict,
) -> bool:
    """True if any PANIC threshold is breached."""
    if volume_z is not None and volume_z > t["volume_z_above"]:
        return True
    if realized_vol is not None and realized_vol > t["realized_vol_above"]:
        return True
    if gap_abs is not None and gap_abs > t["gap_abs_above"]:
        return True
    return False


def _is_stressed(
    volume_z: Optional[float],
    realized_vol: Optional[float],
    gap_abs: Optional[float],
    t: dict,
) -> bool:
    """True if any STRESSED threshold is breached."""
    if volume_z is not None and volume_z > t["volume_z_above"]:
        return True
    if realized_vol is not None and realized_vol > t["realized_vol_above"]:
        return True
    if gap_abs is not None and gap_abs > t["gap_abs_above"]:
        return True
    return False


def _is_thin(
    volume_z: Optional[float],
    realized_vol: Optional[float],
    t: dict,
) -> bool:
    """True if any THIN threshold is breached."""
    if volume_z is not None and volume_z < t["volume_z_below"]:
        return True
    if realized_vol is not None and realized_vol > t["realized_vol_above"]:
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_liquidity(
    volume_zscore_20: Optional[float],
    atr_14: Optional[float],
    realized_vol_20: Optional[float],
    gap_size: Optional[float],
    thresholds: Optional[dict] = None,
) -> str:
    """Classify a single bar into a liquidity state string.

    Highest-severity wins (PANIC > STRESSED > THIN > NORMAL).  Any input that
    is None or NaN contributes nothing; missing data defaults to NORMAL for
    that signal.  atr_14 is accepted for interface completeness but is not used
    in the default threshold logic (reserved for future extension).
    """
    t = thresholds if thresholds is not None else LIQUIDITY_THRESHOLDS

    vz = _safe(volume_zscore_20)
    rv = _safe(realized_vol_20)
    gap = None if gap_size is None else _safe(abs(gap_size))

    if _is_panic(vz, rv, gap, t["PANIC"]):
        return "PANIC_LIQUIDITY"
    if _is_stressed(vz, rv, gap, t["STRESSED"]):
        return "STRESSED_LIQUIDITY"
    if _is_thin(vz, rv, t["THIN"]):
        return "THIN_LIQUIDITY"
    return "NORMAL_LIQUIDITY"


def classify_liquidity_series(
    df: pd.DataFrame,
    thresholds: Optional[dict] = None,
) -> pd.Series:
    """Vectorized liquidity classification over a DataFrame.

    Expects columns: volume_zscore_20, atr_14, realized_vol_20, gap_size.
    Returns a Series of state strings aligned to df.index.
    Empty input returns an empty Series.
    """
    if df.empty:
        return pd.Series(dtype=str)

    def _row_classify(row: pd.Series) -> str:
        return classify_liquidity(
            volume_zscore_20=row.get("volume_zscore_20"),
            atr_14=row.get("atr_14"),
            realized_vol_20=row.get("realized_vol_20"),
            gap_size=row.get("gap_size"),
            thresholds=thresholds,
        )

    return df.apply(_row_classify, axis=1)


def estimate_cost_bps(
    liquidity_state: str,
    side: str = "round_trip",
    overrides: Optional[dict[str, float]] = None,
) -> float:
    """Return estimated transaction cost in basis points for a given state.

    side='round_trip' (default) returns the full cost.
    side='entry' or 'exit' returns half the round-trip cost.
    overrides replaces per-state bps values before lookup.
    """
    cost_table = dict(COST_BPS)
    if overrides:
        cost_table.update(overrides)

    bps = cost_table.get(liquidity_state, cost_table["NORMAL_LIQUIDITY"])

    if side in ("entry", "exit"):
        return bps / 2.0
    return bps


def apply_costs(
    returns: pd.Series,
    liquidity_states: pd.Series,
    sides: "pd.Series | str" = "round_trip",
    overrides: Optional[dict[str, float]] = None,
) -> pd.Series:
    """Subtract transaction costs from gross returns.

    returns            — gross returns as decimals (e.g. 0.01 = 1 %)
    liquidity_states   — aligned Series of state strings
    sides              — scalar str or aligned Series of side strings
    overrides          — optional per-state bps override dict

    Returns net returns (gross - cost_decimal).  Empty input returns empty.
    """
    if returns.empty:
        return returns.copy()

    if isinstance(sides, str):
        sides_series = pd.Series(sides, index=returns.index)
    else:
        sides_series = sides

    def _cost_decimal(state: str, side: str) -> float:
        return estimate_cost_bps(state, side=side, overrides=overrides) / 10_000.0

    costs = pd.Series(
        [
            _cost_decimal(state, side)
            for state, side in zip(liquidity_states, sides_series)
        ],
        index=returns.index,
        dtype=float,
    )
    return returns - costs


def entry_blocked(liquidity_state: str) -> bool:
    """Return True if new entries should be blocked in this liquidity state."""
    return liquidity_state in ENTRY_BLOCK
