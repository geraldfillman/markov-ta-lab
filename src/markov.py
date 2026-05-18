"""Markov Agent - visible Markov chain model."""

import numpy as np
import pandas as pd


def estimate_transition_matrix(
    states: pd.Series,
    n_states: int,
    alpha: float = 0.0,
) -> np.ndarray:
    """Estimate transition probabilities with optional Laplace smoothing."""
    _validate_n_states(n_states)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    values = states.dropna().astype(int).to_numpy()
    counts = np.full((n_states, n_states), alpha, dtype=float)

    if values.size >= 2:
        src = values[:-1]
        dst = values[1:]
        valid = (src >= 0) & (src < n_states) & (dst >= 0) & (dst < n_states)
        if valid.any():
            np.add.at(counts, (src[valid], dst[valid]), 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums != 0,
    )

    empty_rows = np.where(row_sums.ravel() == 0)[0]
    for state_id in empty_rows:
        matrix[state_id, state_id] = 1.0

    return matrix


def forecast_state_probs(
    P: np.ndarray,
    current_state: int,
    horizon: int,
) -> np.ndarray:
    """Return horizon-step state probabilities from current_state."""
    _validate_transition_matrix(P)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not 0 <= current_state < P.shape[0]:
        raise ValueError("current_state is outside the transition matrix")

    powered = np.linalg.matrix_power(P, horizon)
    probs = powered[current_state].astype(float)
    return _normalize_probability_vector(probs)


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a transition matrix."""
    _validate_transition_matrix(P)

    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    vector = np.real(eigenvectors[:, index])

    if vector.sum() < 0:
        vector = -vector

    vector = np.maximum(vector, 0.0)
    if np.isclose(vector.sum(), 0.0):
        vector = np.ones(P.shape[0], dtype=float)

    return _normalize_probability_vector(vector)


def walkforward_forecasts(
    states: pd.Series,
    n_states: int,
    lookback: int,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Estimate transition matrices from prior windows and forecast each bar."""
    _validate_n_states(n_states)
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must be positive")

    clean_states = states.dropna().astype(int)
    records = []

    for i in range(lookback, len(clean_states)):
        historical_states = clean_states.iloc[i - lookback:i]
        current_state = int(clean_states.iloc[i])
        matrix = estimate_transition_matrix(historical_states, n_states=n_states, alpha=0.0)

        record: dict[str, float | int | pd.Timestamp] = {
            "date": clean_states.index[i],
            "current_state": current_state,
        }

        for horizon in horizons:
            probs = forecast_state_probs(matrix, current_state, horizon)
            for state_id, probability in enumerate(probs):
                record[f"prob_h{horizon}_state{state_id}"] = float(probability)

        records.append(record)

    if not records:
        return pd.DataFrame().rename_axis(states.index.name or "date")

    return pd.DataFrame(records).set_index("date")


def _normalize_probability_vector(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(values.astype(float), 0.0)
    total = clipped.sum()
    if np.isclose(total, 0.0):
        return np.full(len(clipped), 1.0 / len(clipped))
    return clipped / total


def _validate_n_states(n_states: int) -> None:
    if n_states < 1:
        raise ValueError("n_states must be at least 1")


def _validate_transition_matrix(P: np.ndarray) -> None:
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("transition matrix must be square")
    if not np.isfinite(P).all():
        raise ValueError("transition matrix contains non-finite values")
