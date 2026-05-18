"""Memory features derived from base-state series. Pure pandas/numpy, no I/O."""

import numpy as np
import pandas as pd

FAILED_BREAKOUT_ID = 9


def state_age(states: pd.Series) -> pd.Series:
    """Consecutive bar count (including current) for which the state has not changed."""
    if states.empty:
        return states.copy()
    groups = (states != states.shift(1)).cumsum()
    return groups.groupby(groups).cumcount() + 1


def prior_state(states: pd.Series, lookback: int = 1) -> pd.Series:
    """State value at t-lookback bars; NaN for warmup."""
    if states.empty:
        return states.copy().astype(float)
    return states.shift(lookback).where(lambda s: s.notna(), other=np.nan)


def prior_n_state_path(states: pd.Series, n: int = 3) -> pd.Series:
    """Prior n states joined by '>'; NaN during warmup."""
    if states.empty:
        return pd.Series(dtype=object, index=states.index)

    result = pd.array([None] * len(states), dtype=object)
    values = states.to_numpy()
    for i in range(len(values)):
        if i < n:
            result[i] = None
        else:
            path = ">".join(str(int(v)) for v in values[i - n : i])
            result[i] = path

    return pd.Series(result, index=states.index)


def entry_velocity(
    states: pd.Series,
    prices: pd.Series,
    window: int = 5,
) -> pd.Series:
    """Abs pct-change of price over `window` bars leading into the current state-entry bar.

    Defined on entry bars (state change), forward-filled within each run.
    NaN where price history is insufficient.
    """
    if states.empty:
        return pd.Series(dtype=float, index=states.index)

    is_entry = (states != states.shift(1)).to_numpy()
    price_arr = prices.reindex(states.index).to_numpy(dtype=float)
    out = np.full(len(states), np.nan)

    for i in range(len(states)):
        if is_entry[i]:
            start = i - window
            if start >= 0 and not np.isnan(price_arr[start]) and not np.isnan(price_arr[i]):
                pct = abs(price_arr[i] - price_arr[start]) / price_arr[start]
                out[i] = pct

    # Forward-fill within each run
    result = pd.Series(out, index=states.index)
    result = result.ffill()
    return result


def failed_breakout_memory(states: pd.Series, lookback: int = 20) -> pd.Series:
    """Count of FAILED_BREAKOUT occurrences in the trailing `lookback` bars (exclusive of current)."""
    if states.empty:
        return pd.Series(dtype=int, index=states.index)

    is_fb = (states == FAILED_BREAKOUT_ID).astype(int)
    rolled = is_fb.shift(1).fillna(0).rolling(window=lookback, min_periods=1).sum()
    return rolled.astype(int)


def build_path_features(
    states: pd.Series,
    prices: pd.Series,
    *,
    state_age_col: bool = True,
    prior_state_col: bool = True,
    path_n: int = 3,
    velocity_window: int = 5,
    failed_lookback: int = 20,
) -> pd.DataFrame:
    """Assemble all path features into a single DataFrame aligned to `states`."""
    cols: dict[str, pd.Series] = {}

    if state_age_col:
        cols["state_age"] = state_age(states)
    if prior_state_col:
        cols["prior_state"] = prior_state(states)

    cols["prior_path"] = prior_n_state_path(states, n=path_n)
    cols["entry_velocity"] = entry_velocity(states, prices, window=velocity_window)
    cols["failed_breakout_memory"] = failed_breakout_memory(states, lookback=failed_lookback)

    if not cols:
        return pd.DataFrame(index=states.index)

    return pd.DataFrame(cols, index=states.index)
