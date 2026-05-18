# Agent Roles & Responsibilities

This file documents the modular agent responsibilities for the Markov TA Research Lab.
See the full playbook at `..\markov_ta_agent_playbook.md` for detailed specifications.

## Agent Summary

| # | Agent | Primary Deliverables |
|---|-------|---------------------|
| 1 | Orchestrator | `README.md`, `AGENTS.md`, experiment run logs |
| 2 | Data | `src/data.py`, `src/fmp.py`, raw/processed datasets, `01_data_download.py` |
| 3 | Indicator | `src/indicators.py`, indicator tests |
| 4 | Support/Resistance | `src/levels.py`, `02_support_resistance_states.ipynb` |
| 5 | State Labeling | `src/states.py`, state frequency reports |
| 6 | Markov | `src/markov.py`, transition matrices, multi-step forecasts |
| 7 | Expected Value | `src/metrics.py`, expectancy tables, bootstrap Sharpe CI, stability summary |
| 8 | Backtest | `src/backtests.py`, walk-forward EV, baselines, random-label permutation |
| 9 | HMM Regime | `src/hmm_models.py`, behaviour-labelled regimes |
| 10 | Change-Point | `src/changepoints.py`, ruptures Pelt + pause signal |
| 11 | Volatility | `src/volatility.py`, vol-conditioned EV |
| 12 | Macro Filter | `src/macro.py`, conditional matrices, **Sharpe-lift acceptance gate (≥ 0.20)** |
| 13 | Reporting | `src/reports.py`, `src/dashboard.py`, experiment reports |
| 14 | Drift Monitor | `src/drift.py`, `scripts/run_drift_monitor.py`, `drift_status.csv` |
| 15 | Position Tracker | `src/positions.py` — immutable `Position` + `PositionTracker` with atomic JSON persistence |
| 16 | Paper Trading | `src/paper_trading.py`, `scripts/run_paper_trading_daily.py`, `paper_trading_decisions.csv` |

## Build Order

1. **Week 1 – Foundation:** repo, env, data, indicators, support/resistance — ✅ shipped
2. **Week 2 – Visible Markov:** states, transition matrix, forecasts, expected values — ✅ shipped
3. **Week 3 – Backtesting:** readable strategy, parameter sweeps, baselines, first report — ✅ shipped
4. **Week 4 – Filters:** volatility, HMM, change-point — ✅ shipped
5. **Week 5 – Robustness:** expand assets, regime tests, sensitivity, bootstrap CIs, random-label baseline — ✅ shipped
6. **Phase C – Macro filter:** conditional transition matrices, Sharpe-lift gate — ✅ shipped
7. **Phase D – Paper-trading readiness:** drift monitor + stateful tracker + daily step — ✅ shipped
8. **Phase E – Distribution:** hatchling-built wheel (`pyproject.toml`), CI pytest matrix, GitHub Pages dashboard publish — ✅ shipped

## Key Constraints

- No lookahead bias — walk-forward EV uses only completed prior outcomes.
- Research before production — no broker execution code in this repo.
- Must beat simple baselines *and* the random-label permutation baseline.
- 8–12 states max — currently 12.
- One Python environment.
- Macro filter must lift Sharpe by ≥ 0.20 net-of-cost or be dropped.

## Current Phase

All phases A–E are complete. The test suite reports `140 passed, 3 skipped` (the skips require the optional `ruptures`, `hmmlearn`, and `hypothesis` libraries — installable via `pip install ".[hmm,changepoint,dev]"`).

Packaging is hatchling-based; `python -m build --wheel` produces `markov_ta_lab-0.1.0-py3-none-any.whl`. CI runs the test matrix on Python 3.11 and 3.12 (`.github/workflows/test.yml`) and deploys the static dashboard to GitHub Pages on every push to `main` (`.github/workflows/dashboard.yml`).
