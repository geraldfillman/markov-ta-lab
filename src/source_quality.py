"""Market data provider quality comparison utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import REQUIRED_OHLCV_COLUMNS


def summarize_source_frames(data: dict[str, pd.DataFrame], provider: str) -> pd.DataFrame:
    """Return per-symbol coverage and missing-value summary for one provider."""
    rows = []
    for symbol, frame in data.items():
        if frame.empty:
            first_date = None
            last_date = None
        else:
            first_date = str(pd.Timestamp(frame.index.min()).date())
            last_date = str(pd.Timestamp(frame.index.max()).date())

        record = {
            "symbol": symbol.upper(),
            "provider": provider,
            "rows": int(len(frame)),
            "first_date": first_date,
            "last_date": last_date,
            "total_missing": int(frame.reindex(columns=REQUIRED_OHLCV_COLUMNS).isna().sum().sum()),
        }
        for column in REQUIRED_OHLCV_COLUMNS:
            record[f"missing_{column}"] = int(frame[column].isna().sum()) if column in frame else int(len(frame))
        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=["provider", "rows", "first_date", "last_date", "total_missing"]).rename_axis("symbol")
    return pd.DataFrame(rows).set_index("symbol").sort_index()


def compare_ohlcv_sources(
    left: dict[str, pd.DataFrame],
    right: dict[str, pd.DataFrame],
    left_name: str,
    right_name: str,
    close_tolerance: float = 0.01,
    volume_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Compare coverage and overlapping OHLCV values between two providers."""
    symbols = sorted(set(left) | set(right))
    rows = []
    for symbol in symbols:
        left_frame = left.get(symbol, _empty_ohlcv_frame())
        right_frame = right.get(symbol, _empty_ohlcv_frame())
        overlap_index = left_frame.index.intersection(right_frame.index)
        left_only_rows = len(left_frame.index.difference(right_frame.index))
        right_only_rows = len(right_frame.index.difference(left_frame.index))

        close_diffs = pd.Series(dtype=float)
        volume_diffs = pd.Series(dtype=float)
        if len(overlap_index):
            left_overlap = left_frame.reindex(overlap_index)
            right_overlap = right_frame.reindex(overlap_index)
            close_diffs = (left_overlap["Close"].astype(float) - right_overlap["Close"].astype(float)).abs()
            volume_diffs = (left_overlap["Volume"].astype(float) - right_overlap["Volume"].astype(float)).abs()

        rows.append(
            {
                "symbol": symbol,
                f"{left_name}_rows": int(len(left_frame)),
                f"{right_name}_rows": int(len(right_frame)),
                "overlap_rows": int(len(overlap_index)),
                f"{left_name}_only_rows": int(left_only_rows),
                f"{right_name}_only_rows": int(right_only_rows),
                "close_mismatch_rows": int((close_diffs > close_tolerance).sum()) if len(close_diffs) else 0,
                "max_abs_close_diff": _safe_max(close_diffs),
                "mean_abs_close_diff": _safe_mean(close_diffs),
                "volume_mismatch_rows": int((volume_diffs > volume_tolerance).sum()) if len(volume_diffs) else 0,
                "max_abs_volume_diff": _safe_max(volume_diffs),
                "mean_abs_volume_diff": _safe_mean(volume_diffs),
            }
        )

    if not rows:
        return pd.DataFrame().rename_axis("symbol")
    return pd.DataFrame(rows).set_index("symbol").sort_index()


def _empty_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS, index=pd.DatetimeIndex([], name="Date"))


def _safe_max(values: pd.Series) -> float:
    return float(values.max()) if len(values) else np.nan


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else np.nan

