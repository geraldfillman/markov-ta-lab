"""Expected Value Agent – trade quality metrics.

Responsibilities (from playbook §3.7):
- Average forward returns by state and horizon.
- Average adverse/favourable excursion.
- Win rate, average win, average loss.
- Expected value after costs.
- Compare Markov forecast to baseline state behavior.

EV formula:
    EV = P(win) × Avg(win) − P(loss) × Avg(loss) − costs
    EV_state,horizon = mean(forward_return | current_state, horizon) − costs

Known unimplemented surface:
    adverse_excursion and favorable_excursion are reserved trade-quality
    metrics and are not part of the current pipeline contract yet.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import STATE_LABELS, TABLES_DIR
from src.markov import estimate_transition_matrix, forecast_state_probs


def _forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Return close-to-future-close returns for a positive bar horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    return close.shift(-horizon) / close - 1.0


def _state_stats(returns: pd.Series) -> dict[str, float | int]:
    """Summarize forward returns for one state and one horizon."""
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    count = int(returns.count())
    mean = float(returns.mean())
    stderr = float(returns.std(ddof=1) / np.sqrt(count)) if count > 1 else np.nan
    margin = 1.96 * stderr if pd.notna(stderr) else np.nan

    return {
        "count": count,
        "avg_forward_return": mean,
        "win_rate": float((returns > 0.0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "ci_low": mean - margin if pd.notna(margin) else np.nan,
        "ci_high": mean + margin if pd.notna(margin) else np.nan,
    }


def state_expectancy_table(
    df: pd.DataFrame,
    states: pd.Series,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Compute expected value per state per horizon, after costs.

    Returns
    -------
    pd.DataFrame
        Index: state, with one set of expectancy columns per horizon.
    """
    if "Close" not in df.columns:
        raise ValueError("df must include a Close column")

    aligned = pd.DataFrame(
        {
            "Close": df["Close"],
            "state": states.reindex(df.index),
        },
        index=df.index,
    )
    cost = cost_bps / 10_000.0
    rows: dict[int, dict[str, object]] = {}

    for horizon in horizons:
        forward = _forward_returns(aligned["Close"], horizon)
        horizon_frame = pd.DataFrame(
            {
                "state": aligned["state"],
                "forward_return": forward,
            },
            index=aligned.index,
        ).dropna(subset=["state", "forward_return"])

        if horizon_frame.empty:
            continue

        horizon_frame["state"] = horizon_frame["state"].astype(int)
        for state, group in horizon_frame.groupby("state", sort=True):
            state = int(state)
            rows.setdefault(state, {"label": STATE_LABELS.get(state, str(state))})
            stats = _state_stats(group["forward_return"])
            rows[state][f"count_{horizon}"] = stats["count"]
            rows[state][f"avg_forward_return_{horizon}"] = stats["avg_forward_return"]
            rows[state][f"win_rate_{horizon}"] = stats["win_rate"]
            rows[state][f"avg_win_{horizon}"] = stats["avg_win"]
            rows[state][f"avg_loss_{horizon}"] = stats["avg_loss"]
            rows[state][f"ci_low_{horizon}"] = stats["ci_low"]
            rows[state][f"ci_high_{horizon}"] = stats["ci_high"]
            rows[state][f"ev_after_cost_{horizon}"] = stats["avg_forward_return"] - cost

    if not rows:
        return pd.DataFrame(index=pd.Index([], name="state"))

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "state"
    return result.sort_index()


def universe_state_expectancy_table(
    data: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    cost_bps: float = 10.0,
    state_col: str = "state",
) -> pd.DataFrame:
    """Compute state expectancy tables for a symbol-keyed processed universe."""
    tables = []
    for symbol, frame in data.items():
        if state_col not in frame.columns:
            raise ValueError(f"{symbol} frame must include a {state_col} column")

        table = state_expectancy_table(
            frame,
            frame[state_col],
            horizons=horizons,
            cost_bps=cost_bps,
        )
        if table.empty:
            continue

        table.insert(0, "symbol", symbol.upper())
        table = table.set_index("symbol", append=True).reorder_levels(["symbol", "state"])
        tables.append(table)

    if not tables:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "state"]))

    return pd.concat(tables).sort_index()


def conditioned_state_expectancy_table(
    df: pd.DataFrame,
    states: pd.Series,
    condition: pd.Series,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    cost_bps: float = 10.0,
    condition_name: str = "vol_state",
) -> pd.DataFrame:
    """Compute state expectancy grouped by an external regime condition."""
    if "Close" not in df.columns:
        raise ValueError("df must include a Close column")

    aligned = pd.DataFrame(
        {
            "Close": df["Close"],
            "state": states.reindex(df.index),
            condition_name: condition.reindex(df.index),
        },
        index=df.index,
    )
    cost = cost_bps / 10_000.0
    rows: dict[tuple[int, int], dict[str, object]] = {}

    for horizon in horizons:
        horizon_frame = pd.DataFrame(
            {
                "state": aligned["state"],
                condition_name: aligned[condition_name],
                "forward_return": _forward_returns(aligned["Close"], horizon),
            },
            index=aligned.index,
        ).dropna(subset=["state", condition_name, "forward_return"])

        if horizon_frame.empty:
            continue

        horizon_frame["state"] = horizon_frame["state"].astype(int)
        horizon_frame[condition_name] = horizon_frame[condition_name].astype(int)
        for (condition_value, state), group in horizon_frame.groupby([condition_name, "state"], sort=True):
            key = (int(condition_value), int(state))
            rows.setdefault(key, {"label": STATE_LABELS.get(int(state), str(state))})
            stats = _state_stats(group["forward_return"])
            rows[key][f"count_{horizon}"] = stats["count"]
            rows[key][f"avg_forward_return_{horizon}"] = stats["avg_forward_return"]
            rows[key][f"win_rate_{horizon}"] = stats["win_rate"]
            rows[key][f"avg_win_{horizon}"] = stats["avg_win"]
            rows[key][f"avg_loss_{horizon}"] = stats["avg_loss"]
            rows[key][f"ci_low_{horizon}"] = stats["ci_low"]
            rows[key][f"ci_high_{horizon}"] = stats["ci_high"]
            rows[key][f"ev_after_cost_{horizon}"] = stats["avg_forward_return"] - cost

    if not rows:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=[condition_name, "state"]))

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index = pd.MultiIndex.from_tuples(result.index, names=[condition_name, "state"])
    return result.sort_index()


def cluster_pooled_state_expectancy_table(
    data: dict[str, pd.DataFrame],
    clusters: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    cost_bps: float = 10.0,
    state_col: str = "state",
    cluster_col: str = "cluster_label",
) -> pd.DataFrame:
    """Pool state expectancy by behavior cluster to reduce sparse-symbol noise."""
    if cluster_col not in clusters.columns:
        raise ValueError(f"clusters must include {cluster_col}")

    cluster_lookup = clusters[cluster_col]
    if "symbol" in clusters.columns:
        cluster_lookup = clusters.set_index("symbol")[cluster_col]
    cluster_lookup.index = cluster_lookup.index.astype(str).str.upper()

    tables = []
    for cluster_label, symbols in cluster_lookup.groupby(cluster_lookup):
        observations = []
        for symbol in symbols.index:
            if symbol not in data:
                continue
            frame = data[symbol]
            if state_col not in frame.columns:
                raise ValueError(f"{symbol} frame must include a {state_col} column")
            for horizon in horizons:
                symbol_obs = pd.DataFrame(
                    {
                        "state": frame[state_col],
                        "horizon": horizon,
                        "forward_return": _forward_returns(frame["Close"], horizon),
                    },
                    index=frame.index,
                ).dropna(subset=["state", "forward_return"])
                observations.append(symbol_obs)

        if not observations:
            continue

        pooled = pd.concat(observations, axis=0)
        cost = cost_bps / 10_000.0
        rows: dict[int, dict[str, object]] = {}
        for (state_key, horizon_key), group in pooled.groupby(["state", "horizon"], sort=True):
            state_id = int(state_key)
            horizon_val = int(horizon_key)
            rows.setdefault(state_id, {"label": STATE_LABELS.get(state_id, str(state_id))})
            stats = _state_stats(group["forward_return"])
            rows[state_id][f"count_{horizon_val}"] = stats["count"]
            rows[state_id][f"avg_forward_return_{horizon_val}"] = stats["avg_forward_return"]
            rows[state_id][f"win_rate_{horizon_val}"] = stats["win_rate"]
            rows[state_id][f"avg_win_{horizon_val}"] = stats["avg_win"]
            rows[state_id][f"avg_loss_{horizon_val}"] = stats["avg_loss"]
            rows[state_id][f"ci_low_{horizon_val}"] = stats["ci_low"]
            rows[state_id][f"ci_high_{horizon_val}"] = stats["ci_high"]
            rows[state_id][f"ev_after_cost_{horizon_val}"] = stats["avg_forward_return"] - cost

        if not rows:
            continue

        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "state"
        table.insert(0, "cluster_label", cluster_label)
        table = table.set_index("cluster_label", append=True).reorder_levels(["cluster_label", "state"])
        tables.append(table)

    if not tables:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["cluster_label", "state"]))

    return pd.concat(tables).sort_index()


def walkforward_state_expectancy(
    df: pd.DataFrame,
    states: pd.Series,
    horizon: int = 5,
    lookback: int = 252,
    min_samples: int = 5,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Estimate current-state EV using only prior realized forward outcomes."""
    if "Close" not in df.columns:
        raise ValueError("df must include a Close column")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")

    aligned = pd.DataFrame({"Close": df["Close"], "state": states.reindex(df.index)}, index=df.index)
    aligned["forward_return"] = _forward_returns(aligned["Close"], horizon)
    records = []
    cost = cost_bps / 10_000.0

    for pos in range(horizon + min_samples, len(aligned)):
        current_state = aligned.iloc[pos]["state"]
        if pd.isna(current_state):
            continue

        train_signal_end = pos - horizon - 1
        train_signal_start = max(0, train_signal_end - lookback + 1)
        train_rows = aligned.iloc[train_signal_start:train_signal_end + 1]

        current_state = int(current_state)
        matching_returns = train_rows.loc[train_rows["state"] == current_state, "forward_return"].dropna()
        train_count = int(matching_returns.count())
        walkforward_ev = np.nan
        if train_count >= min_samples:
            walkforward_ev = float(matching_returns.mean() - cost)

        records.append(
            {
                "date": aligned.index[pos],
                "state": current_state,
                "train_count": train_count,
                "walkforward_ev": walkforward_ev,
            }
        )

    if not records:
        return pd.DataFrame(columns=["state", "train_count", "walkforward_ev"]).rename_axis(aligned.index.name or "date")

    return pd.DataFrame(records).set_index("date")


def walkforward_markov_expected_value(
    df: pd.DataFrame,
    states: pd.Series,
    n_states: int,
    horizon: int = 5,
    lookback: int = 252,
    min_samples: int = 5,
    cost_bps: float = 10.0,
    alpha: float = 0.0,
) -> pd.DataFrame:
    """Estimate Markov probability weighted EV using prior transitions and payoffs."""
    if "Close" not in df.columns:
        raise ValueError("df must include a Close column")
    if n_states < 1:
        raise ValueError("n_states must be at least 1")

    aligned = pd.DataFrame({"Close": df["Close"], "state": states.reindex(df.index)}, index=df.index)
    records = []
    ev_col = f"ev_after_cost_{horizon}"
    count_col = f"count_{horizon}"

    for pos in range(horizon + min_samples, len(aligned)):
        current_state = aligned.iloc[pos]["state"]
        if pd.isna(current_state):
            continue

        train_signal_end = pos - horizon - 1
        train_signal_start = max(0, train_signal_end - lookback + 1)
        train_close_end = train_signal_end + horizon
        train_frame = aligned.iloc[train_signal_start:train_close_end + 1]
        transition_states = aligned["state"].iloc[train_signal_start:train_signal_end + 1].dropna()

        current_state = int(current_state)
        markov_ev = np.nan
        coverage = 0.0
        weighted_samples = 0.0
        if len(transition_states) >= 2:
            payoff_table = state_expectancy_table(
                train_frame[["Close"]],
                train_frame["state"],
                horizons=(horizon,),
                cost_bps=cost_bps,
            )
            matrix = estimate_transition_matrix(transition_states, n_states=n_states, alpha=alpha)
            probabilities = forecast_state_probs(matrix, current_state=current_state, horizon=horizon)
            result = forecast_expected_value(probabilities, payoff_table, horizon=horizon)
            contributions = result["contributions"]
            valid = contributions[contributions["sample_count"].fillna(0) >= min_samples] if not contributions.empty else contributions
            if not valid.empty:
                markov_ev = float(valid["weighted_ev"].sum())
                coverage = float(valid["probability"].sum())
                weighted_samples = float((valid["probability"] * valid["sample_count"]).sum())

        records.append(
            {
                "date": aligned.index[pos],
                "current_state": current_state,
                "markov_weighted_ev": markov_ev,
                "coverage": coverage,
                "weighted_samples": weighted_samples,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["current_state", "markov_weighted_ev", "coverage", "weighted_samples"]
        ).rename_axis(aligned.index.name or "date")

    return pd.DataFrame(records).set_index("date")


def save_state_expectancy_table(
    table: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    """Save a state expectancy table as CSV and return the output path."""
    if output_path is None:
        output_path = Path(TABLES_DIR) / "state_expectancy.csv"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return path


def forecast_expected_value(
    probabilities: np.ndarray,
    expectancy_table: pd.DataFrame,
    horizon: int,
) -> dict[str, object]:
    """Combine Markov destination probabilities with historical state payoff estimates."""
    ev_col = f"ev_after_cost_{horizon}"
    count_col = f"count_{horizon}"
    if ev_col not in expectancy_table.columns:
        raise ValueError(f"expectancy_table must include {ev_col}")

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional vector")

    rows = []
    for state, probability in enumerate(probs):
        if state not in expectancy_table.index:
            continue

        state_ev = expectancy_table.loc[state, ev_col]
        if pd.isna(state_ev):
            continue

        sample_count = expectancy_table.loc[state, count_col] if count_col in expectancy_table.columns else np.nan
        rows.append(
            {
                "state": state,
                "label": expectancy_table.loc[state, "label"] if "label" in expectancy_table.columns else STATE_LABELS.get(state, str(state)),
                "probability": float(probability),
                "state_ev": float(state_ev),
                "weighted_ev": float(probability) * float(state_ev),
                "sample_count": sample_count,
            }
        )

    contributions = pd.DataFrame(rows).set_index("state") if rows else pd.DataFrame(
        columns=["label", "probability", "state_ev", "weighted_ev", "sample_count"]
    ).rename_axis("state")

    return {
        "horizon": horizon,
        "expected_value": float(contributions["weighted_ev"].sum()) if not contributions.empty else np.nan,
        "coverage": float(contributions["probability"].sum()) if not contributions.empty else 0.0,
        "contributions": contributions,
    }


def bootstrap_confidence_interval(
    values: pd.Series | np.ndarray,
    statistic: str = "mean",
    n_resamples: int = 2_000,
    block_size: int | None = None,
    confidence: float = 0.95,
    random_state: int = 0,
) -> dict[str, float]:
    """Return a bootstrap confidence interval for a statistic over `values`.

    For serially correlated samples (returns, equity curves) pass `block_size`
    to enable moving block bootstrap and preserve autocorrelation. With
    `block_size=None` this is plain IID bootstrap.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1 (exclusive)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}

    func = _STATISTIC_FUNCS.get(statistic)
    if func is None:
        raise ValueError(f"unsupported statistic: {statistic}")

    rng = np.random.default_rng(random_state)
    if block_size is None or block_size <= 1:
        samples = rng.integers(0, array.size, size=(n_resamples, array.size))
        resampled = array[samples]
        stats = func(resampled, axis=1)
    else:
        block_size = min(block_size, array.size)
        n_blocks = int(np.ceil(array.size / block_size))
        max_start = array.size - block_size + 1
        starts = rng.integers(0, max_start, size=(n_resamples, n_blocks))
        offsets = np.arange(block_size)
        indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)
        indices = indices[:, : array.size]
        resampled = array[indices]
        stats = func(resampled, axis=1)

    lower = (1.0 - confidence) / 2.0
    upper = 1.0 - lower
    return {
        "point": float(func(array, axis=0)),
        "ci_low": float(np.quantile(stats, lower)),
        "ci_high": float(np.quantile(stats, upper)),
        "n": int(array.size),
    }


_STATISTIC_FUNCS: dict[str, "callable"] = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
}


def adverse_excursion(
    df: pd.DataFrame,
    entry_indices: pd.Index,
    horizon: int,
) -> pd.Series:
    """Max adverse excursion (MAE) for each entry within horizon bars."""
    raise NotImplementedError


def favorable_excursion(
    df: pd.DataFrame,
    entry_indices: pd.Index,
    horizon: int,
) -> pd.Series:
    """Max favorable excursion (MFE) for each entry within horizon bars."""
    raise NotImplementedError


def sensitivity_stability_summary(
    sensitivity: pd.DataFrame,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Per-symbol stability summary across a parameter sensitivity grid.

    Produces a small table (one row per symbol) with the median, std, IQR,
    and min/max of ``metric`` across the (lookback × min_samples × cost_bps
    × horizon) grid. Higher std / wider IQR = more parameter-sensitive ⇒
    less robust strategy on that symbol.
    """
    if "symbol" not in sensitivity.columns:
        raise ValueError("sensitivity must include a 'symbol' column")
    if metric not in sensitivity.columns:
        raise ValueError(f"sensitivity must include a {metric!r} column")

    rows: list[dict[str, float | int | str]] = []
    grouped = sensitivity.groupby("symbol", dropna=True)
    for symbol, group in grouped:
        values = group[metric].dropna().astype(float)
        if values.empty:
            continue
        q25, q75 = float(values.quantile(0.25)), float(values.quantile(0.75))
        rows.append(
            {
                "symbol": str(symbol),
                "n_configs": int(values.size),
                f"{metric}_median": float(values.median()),
                f"{metric}_std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                f"{metric}_iqr": q75 - q25,
                f"{metric}_min": float(values.min()),
                f"{metric}_max": float(values.max()),
                f"{metric}_share_negative": float((values < 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(f"{metric}_median", ascending=False).reset_index(drop=True)


def bootstrap_sharpe_ci_from_trades(
    trade_returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    block_size: int | None = None,
    annualization: float = 252.0,
    avg_holding_period: float = 1.0,
    random_state: int = 0,
) -> dict[str, float]:
    """Bootstrap a Sharpe-like CI for a series of trade-level net returns.

    Sharpe is computed per-resample as ``mean / std`` (no risk-free rate
    subtraction — fine for relative comparisons), then optionally
    annualised by ``sqrt(annualization / avg_holding_period)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1 (exclusive)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if annualization <= 0 or avg_holding_period <= 0:
        raise ValueError("annualization and avg_holding_period must be positive")

    arr = np.asarray(trade_returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return {"sharpe_point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": int(n)}

    rng = np.random.default_rng(random_state)
    if block_size is None or block_size <= 1:
        indices = rng.integers(0, n, size=(n_resamples, n))
        resampled = arr[indices]
    else:
        block_size = min(block_size, n)
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, n - block_size + 1, size=(n_resamples, n_blocks))
        offsets = np.arange(block_size)
        indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)[:, :n]
        resampled = arr[indices]

    means = resampled.mean(axis=1)
    stds = resampled.std(axis=1, ddof=1)
    sharpes = np.divide(means, stds, out=np.zeros_like(means), where=stds > 0.0)
    scale = float(np.sqrt(annualization / avg_holding_period))

    point_std = arr.std(ddof=1)
    point_sharpe = float(arr.mean() / point_std) if point_std > 0 else float("nan")

    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    return {
        "sharpe_point": point_sharpe * scale,
        "ci_low": float(np.quantile(sharpes, lower_q)) * scale,
        "ci_high": float(np.quantile(sharpes, upper_q)) * scale,
        "n": int(n),
    }
