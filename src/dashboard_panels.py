"""Pure DataFrame-building functions for coverage/fallback dashboard panels."""

from __future__ import annotations

import pandas as pd

from conditional_markov import ConditionalForecast

# ---------------------------------------------------------------------------
# Panel 1 – Fallback usage
# ---------------------------------------------------------------------------

def fallback_usage_panel(forecasts: list[ConditionalForecast]) -> pd.DataFrame:
    """Aggregate how often each fallback level was used across forecasts."""
    columns = ["fallback_level", "count", "share"]
    if not forecasts:
        return pd.DataFrame(columns=columns)

    level_counts: dict[int, int] = {}
    for f in forecasts:
        level_counts[f.fallback_level] = level_counts.get(f.fallback_level, 0) + 1

    total = len(forecasts)
    rows = [
        {"fallback_level": lvl, "count": cnt, "share": cnt / total}
        for lvl, cnt in sorted(level_counts.items())
    ]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Panel 2 – Low-confidence breakdown
# ---------------------------------------------------------------------------

def low_confidence_panel(forecasts: list[ConditionalForecast]) -> pd.DataFrame:
    """Break down forecast counts by confidence tier (high/medium/low)."""
    columns = ["confidence", "count", "share"]
    if not forecasts:
        return pd.DataFrame(columns=columns)

    conf_counts: dict[str, int] = {}
    for f in forecasts:
        conf_counts[f.confidence] = conf_counts.get(f.confidence, 0) + 1

    total = len(forecasts)
    # Fixed order: high → medium → low
    order = ["high", "medium", "low"]
    rows = []
    for tier in order:
        if tier in conf_counts:
            cnt = conf_counts[tier]
            rows.append({"confidence": tier, "count": cnt, "share": cnt / total})
    # Any unexpected tiers appended after
    for tier, cnt in conf_counts.items():
        if tier not in order:
            rows.append({"confidence": tier, "count": cnt, "share": cnt / total})
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Panel 3 – Top rare states
# ---------------------------------------------------------------------------

def top_rare_states_panel(
    state_counts: dict[str, int],
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the rarest composite states sorted ascending by sample_count."""
    columns = ["composite_key", "sample_count"]
    if not state_counts:
        return pd.DataFrame(columns=columns)

    sorted_items = sorted(state_counts.items(), key=lambda kv: kv[1])[:top_n]
    rows = [{"composite_key": k, "sample_count": v} for k, v in sorted_items]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Panel 4 – Coverage by symbol
# ---------------------------------------------------------------------------

def coverage_by_symbol_panel(
    forecasts: list[ConditionalForecast],
    symbols: list[str],
) -> pd.DataFrame:
    """Summarise forecast quality per symbol using parallel lists."""
    columns = [
        "symbol",
        "forecast_count",
        "mean_sample_count",
        "low_confidence_count",
        "low_confidence_share",
    ]
    if not forecasts or not symbols:
        return pd.DataFrame(columns=columns)

    # Group forecasts by position index → symbol
    sym_groups: dict[str, list[ConditionalForecast]] = {}
    for sym, fc in zip(symbols, forecasts):
        sym_groups.setdefault(sym, []).append(fc)

    rows = []
    for sym, fcs in sym_groups.items():
        n = len(fcs)
        mean_sc = sum(f.sample_count for f in fcs) / n
        low_cnt = sum(1 for f in fcs if f.confidence == "low")
        rows.append(
            {
                "symbol": sym,
                "forecast_count": n,
                "mean_sample_count": mean_sc,
                "low_confidence_count": low_cnt,
                "low_confidence_share": low_cnt / n,
            }
        )
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Panel 5 – Warnings summary
# ---------------------------------------------------------------------------

def warnings_summary_panel(forecasts: list[ConditionalForecast]) -> pd.DataFrame:
    """Count each warning type across all forecasts, sorted descending."""
    columns = ["warning", "count"]
    if not forecasts:
        return pd.DataFrame(columns=columns)

    warn_counts: dict[str, int] = {}
    for f in forecasts:
        for w in f.warnings:
            warn_counts[w] = warn_counts.get(w, 0) + 1

    if not warn_counts:
        return pd.DataFrame(columns=columns)

    rows = sorted(
        [{"warning": w, "count": c} for w, c in warn_counts.items()],
        key=lambda r: r["count"],
        reverse=True,
    )
    return pd.DataFrame(rows, columns=columns)
