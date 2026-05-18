"""Tests for static HTML dashboard generation."""

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.dashboard import generate_dashboard


def _scratch_dir() -> Path:
    path = Path(".test_output") / f"dashboard_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_generate_dashboard_writes_self_contained_html():
    root = _scratch_dir()
    tables_dir = root / "tables"
    dashboard_dir = root / "dashboard"
    tables_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"symbol": "SPY", "total_return": 0.50, "max_drawdown": -0.10, "sharpe": 0.40, "trade_count": 12, "win_rate": 0.58, "exposure_time": 0.40, "benchmark_total_return": 0.80},
            {"symbol": "QQQ", "total_return": 0.75, "max_drawdown": -0.12, "sharpe": 0.60, "trade_count": 10, "win_rate": 0.60, "exposure_time": 0.35, "benchmark_total_return": 1.20},
        ]
    ).to_csv(tables_dir / "walkforward_backtest_summary.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": "SPY", "model": "state_ev_strategy", "total_return": 0.50, "excess_vs_buy_hold": -0.30, "bars": None},
            {"symbol": "SPY", "model": "buy_and_hold", "total_return": 0.80, "excess_vs_buy_hold": 0.0},
        ]
    ).to_csv(tables_dir / "walkforward_baseline_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": "SPY", "vol_state": 0, "state": 1, "label": "APPROACHING_SUPPORT", "count_5": 20, "ev_after_cost_5": 0.01},
            {"symbol": "QQQ", "vol_state": 2, "state": 7, "label": "RESISTANCE_BREAKOUT", "count_5": 15, "ev_after_cost_5": -0.02},
        ]
    ).to_csv(tables_dir / "vol_conditioned_state_expectancy.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": "SPY", "realized_volatility": 0.17, "trend_persistence": -0.1, "cluster_id": 0, "cluster_label": "low_vol_defensive"},
            {"symbol": "QQQ", "realized_volatility": 0.21, "trend_persistence": -0.08, "cluster_id": 1, "cluster_label": "medium_vol_cyclical"},
        ]
    ).to_csv(tables_dir / "asset_behavior_clusters.csv", index=False)
    pd.DataFrame(
        [{"cluster_label": "low_vol_defensive", "state": 1, "label": "APPROACHING_SUPPORT", "count_5": 40, "ev_after_cost_5": 0.02, "ci_low_5": 0.0, "ci_high_5": 0.04}]
    ).to_csv(tables_dir / "cluster_pooled_state_expectancy.csv", index=False)
    pd.DataFrame(
        [{"symbol": "SPY", "date": "2024-01-05", "current_state": 1, "markov_weighted_ev": 0.01, "coverage": 0.9, "weighted_samples": 22}]
    ).to_csv(tables_dir / "markov_weighted_ev.csv", index=False)
    pd.DataFrame(
        [{"symbol": "SPY", "horizon": 5, "lookback": 126, "cost_bps": 10, "min_samples": 5, "total_return": 0.2, "sharpe": 0.3, "trade_count": 8}]
    ).to_csv(tables_dir / "walkforward_sensitivity.csv", index=False)

    output_path = generate_dashboard(tables_dir=tables_dir, output_dir=dashboard_dir)

    html = output_path.read_text(encoding="utf-8")
    assert output_path.name == "index.html"
    assert "Institutional Flow Markov Lab" in html
    assert "Walk-Forward Performance" in html
    assert "Plain-English Summary" in html
    assert "What needs more research" in html
    assert "Baseline Comparison" in html
    assert "Volatility-Conditioned EV" in html
    assert "Asset Behavior Clusters" in html
    assert "Cluster-Pooled EV" in html
    assert "Markov-Weighted EV" in html
    assert "Sensitivity Tests" in html
    assert "dashboardData" in html
    assert "SPY" in html
