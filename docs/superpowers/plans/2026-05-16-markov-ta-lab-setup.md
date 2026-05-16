# Markov TA Lab Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the Markov TA research lab into a reproducible Python project skeleton that follows the playbook without implementing strategy logic early.

**Architecture:** Keep the first version as one Python repo with docs, environment files, importable modules, placeholder research notebooks, data/report directories, and smoke tests. The setup phase makes future Data, Indicator, Level, State, Markov, and Backtest agents work in bounded phases instead of building the whole trading research system at once.

**Tech Stack:** Python 3.11, pandas, NumPy, scipy, scikit-learn, yfinance, ta, vectorbt, backtesting.py, hmmlearn, ruptures, arch, pykalman, filterpy, statsmodels, pytest, Jupyter.

---

### Task 1: Project Contract

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Create: `docs/superpowers/plans/2026-05-16-markov-ta-lab-setup.md`
- Create: `pyproject.toml`
- Test: `tests/test_project_setup.py`

- [x] **Step 1: Write the failing setup test**

```python
def test_setup_plan_exists_and_names_bootstrap_scope():
    plan = ROOT / "docs" / "superpowers" / "plans" / "2026-05-16-markov-ta-lab-setup.md"
    assert plan.exists()
    text = plan.read_text(encoding="utf-8")
    assert "Bootstrap the Markov TA research lab" in text
    assert "Do not implement strategy logic" in text
```

- [x] **Step 2: Run the setup test and confirm it fails**

Run: `python -m pytest tests\test_project_setup.py -q`

Expected: fails because the setup plan and `pyproject.toml` do not exist.

- [x] **Step 3: Add pytest configuration**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-ra"
```

- [x] **Step 4: Save this implementation plan**

This file defines the bootstrap scope, marks strategy logic as out of scope, and records the verification commands.

### Task 2: Bootstrap Scaffold

**Files:**
- Existing: `src/__init__.py`
- Existing: `src/config.py`
- Existing: `src/data.py`
- Existing: `src/indicators.py`
- Existing: `src/levels.py`
- Existing: `src/states.py`
- Existing: `src/markov.py`
- Existing: `src/hmm_models.py`
- Existing: `src/changepoints.py`
- Existing: `src/volatility.py`
- Existing: `src/backtests.py`
- Existing: `src/metrics.py`
- Existing: `src/plotting.py`
- Existing: `src/reports.py`
- Existing: `tests/test_smoke.py`

- [x] **Step 1: Keep modules importable**

The bootstrap phase allows placeholder functions for later agents, but every module must import cleanly so notebooks and future tests have stable targets.

- [x] **Step 2: Keep future implementation tests skipped**

The existing level, state, Markov, and backtest tests document acceptance criteria while staying skipped until those phases are implemented with TDD.

- [x] **Step 3: Verify smoke imports**

Run: `python -m pytest tests\test_smoke.py -q`

Expected: all import smoke tests pass.

### Task 3: Environment Setup

**Files:**
- Existing: `requirements.txt`
- Existing: `environment.yml`

- [x] **Step 1: Keep one environment**

The repo uses the playbook's `markov-lab` Python 3.11 conda environment and installs Python packages from `requirements.txt`.

- [x] **Step 2: Verify test runner dependency**

Run: `python -m pytest -q`

Expected: smoke and setup tests pass; future implementation tests may be skipped until those phases are active.

### Task 4: Phase Gate

**Files:**
- Existing: `AGENTS.md`
- Existing: `README.md`

- [x] **Step 1: Lock setup scope**

Do not implement strategy logic, backtesting logic, HMM filters, change-point filters, or live trading integrations in the setup phase.

- [ ] **Step 2: Next implementation slice**

After setup verification, the next slice should be the Data Agent: implement `src/data.py` using yfinance, write data-shape tests first, and save processed parquet outputs under `data/processed/`.
