# Agent Roles & Responsibilities

This file documents the modular agent responsibilities for the Markov TA Research Lab.
See the full playbook at `..\markov_ta_agent_playbook.md` for detailed specifications.

## Agent Summary

| # | Agent | Primary Deliverables |
|---|-------|---------------------|
| 1 | Orchestrator | `README.md`, `AGENTS.md`, experiment run logs |
| 2 | Data | `src/data.py`, raw/processed datasets, `01_data_download.ipynb` |
| 3 | Indicator | `src/indicators.py`, indicator tests |
| 4 | Support/Resistance | `src/levels.py`, `02_support_resistance_states.ipynb` |
| 5 | State Labeling | `src/states.py`, state frequency reports |
| 6 | Markov | `src/markov.py`, `03_markov_transition_matrix.ipynb` |
| 7 | Expected Value | `src/metrics.py`, expectancy tables |
| 8 | Backtest | `src/backtests.py`, `04_vectorbt_backtest.ipynb`, `05_backtesting_py_readable_test.ipynb` |
| 9 | HMM Regime | `src/hmm_models.py`, `06_hmm_regime_model.ipynb` |
| 10 | Change-Point | `src/changepoints.py`, `07_changepoint_detection.ipynb` |
| 11 | Volatility | `src/volatility.py`, `08_volatility_filter_garch.ipynb` |
| 12 | Macro Filter | Macro filter report, conditional transition matrices |
| 13 | Reporting | `src/reports.py`, `09_combined_signal_dashboard.ipynb`, experiment reports |

## Build Order

1. **Week 1 - Foundation:** repo, env, data, indicators, support/resistance
2. **Week 2 - Visible Markov:** states, transition matrix, forecasts, expected values
3. **Week 3 - Backtesting:** readable strategy, parameter sweeps, baselines, first report
4. **Week 4 - Filters:** volatility, HMM, change-point
5. **Week 5 - Robustness:** expand assets, regime tests, sensitivity, bootstrap CIs

## Key Constraints

- No lookahead bias
- Research before production
- Must beat simple baselines
- 8-12 states max initially
- One Python environment

## Current Phase

The repository is in the setup phase. Keep modules importable and tests runnable, but do not implement strategy logic until each phase has its own test-first implementation slice.
