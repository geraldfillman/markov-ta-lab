# Markov + Technical Analysis Research Lab

> When price interacts with support/resistance or trend structure, what is the probability of each next market state, and does that probability create positive expected value after costs under the current macro/volatility regime?

## Purpose

Research, simulation, and paper-trading preparation **only**. This repo is not financial advice and should not be used to deploy live capital without independent validation, risk controls, and compliance review.

## Core Research Loop

1. Pull clean market data (yfinance or Financial Modeling Prep)
2. Engineer technical features
3. Detect support/resistance zones
4. Label each bar into a market state
5. Estimate Markov transition probabilities using only past data
6. Forecast 1-step and multi-step future state probabilities
7. Convert probabilities into expected value
8. Backtest with realistic costs and walk-forward estimation
9. Layer HMM, change-point, volatility, and macro filters on top
10. Permutation, sensitivity, and bootstrap robustness checks
11. Drift-monitored paper-trading tracker (no broker execution)
12. Produce repeatable reports and a static HTML dashboard

## Quick Start

The repo is packaged with [hatchling](https://hatch.pypa.io/) and PEP 621 metadata. Choose either the editable-install path (development) or the conda environment path (notebooks).

### Supported Python versions

CI is the compatibility contract: Python 3.11 and 3.12. Python 3.14 may work locally, but it is not part of the supported matrix until it is added to `.github/workflows/test.yml`.

### Editable install (recommended for development)

```bash
python -m venv .venv
. .venv/bin/activate    # PowerShell: & ".\.venv\Scripts\Activate.ps1"
pip install -e ".[dev]"
```

Optional research extras: `hmm`, `changepoint`, `garch`, `kalman`, `ta`, `backtest`, `viz`, `notebook`. Combine as needed:

```bash
pip install -e ".[dev,hmm,changepoint]"
```

### Conda environment (for notebook kernels)

```bash
conda create -n markov-lab python=3.11 -y
conda activate markov-lab
pip install -e ".[dev,notebook]"
python -m ipykernel install --user --name markov-lab --display-name "Python (markov-lab)"
```

### GitHub Codespaces

The repo includes a `.devcontainer/devcontainer.json` for GitHub Codespaces. It uses Python 3.11 and runs `pip install -e ".[dev,notebook]"` after the container is created so tests, notebooks, and editable package imports are ready in the browser workspace.

### Build a wheel

```bash
python -m build --wheel
# -> dist/markov_ta_lab-0.1.0-py3-none-any.whl
```

## Verify Setup

```bash
python -m pytest -q
```

The full suite (`140 passed, 3 skipped`) runs in ~3 s. Skips are only for the optional `ruptures`, `hmmlearn`, and `hypothesis` packages — install them if you want the HMM / change-point / property-based property tests to execute too.

## Project Structure

```text
markov-ta-lab/
  AGENTS.md              # Agent roles and responsibilities
  README.md              # This file
  pyproject.toml         # PEP 621 metadata, hatchling build, pytest config
  requirements.txt       # Pinned runtime dependencies
  environment.yml        # Conda environment spec (notebook kernels)

  .github/workflows/     # CI: pytest matrix + GitHub Pages dashboard publish
  docs/                  # Onboarding, data source, FMP, and project-state guides
  notebooks/             # Research notebooks/scripts numbered by phase
  src/                   # Reusable Python modules (shipped as the wheel)
  scripts/               # CLI entry points for each pipeline stage
  tests/                 # Unit, integration, and property tests
  data/raw/              # Raw downloaded market data (gitignored)
  data/processed/        # Cleaned parquet files (gitignored; per-provider subdirs)
  reports/charts/        # Visualization outputs
  reports/tables/        # CSV metric tables
  reports/runs/          # Experiment run logs
  reports/dashboard/     # Generated static HTML dashboard
  reports/paper_trading/ # Persisted PositionTracker JSON
```

## Modules

| File | Purpose |
|---|---|
| `src/data.py` | OHLCV download/clean/load; multi-provider via `provider=` kwarg |
| `src/fmp.py` | Financial Modeling Prep REST client + dotenv key loader |
| `src/indicators.py` | Technical indicators (SMA, EMA, ATR, RSI, BB width, returns, vol) |
| `src/levels.py` | Prior-bar support/resistance zones |
| `src/states.py` | Deterministic state labels (12 states, see `STATE_LABELS`) |
| `src/markov.py` | Transition matrix estimation + multi-step forecasts |
| `src/metrics.py` | Expected value tables, walk-forward EV, bootstrap Sharpe CI, sensitivity stability summary |
| `src/backtests.py` | Readable + walk-forward backtests, baselines, random-label permutation, signal mask |
| `src/volatility.py` | Volatility regime classifier + position sizing |
| `src/clustering.py` | Asset-behavior K-style clustering |
| `src/hmm_models.py` | HMM regimes (`hmmlearn`) with behaviour-based labelling |
| `src/changepoints.py` | Change-point detection (`ruptures`) + pause signal |
| `src/macro.py` | Macro regime classifier + conditional transition matrices + Sharpe-lift gate |
| `src/drift.py` | KL-divergence drift monitor on state-frequency distributions |
| `src/positions.py` | Immutable `Position` + `PositionTracker` with atomic JSON persistence |
| `src/paper_trading.py` | One-bar paper-trading step honouring drift + macro gates |
| `src/dashboard.py` | Self-contained static HTML dashboard generator |
| `src/reports.py` | Markdown experiment-report writer |
| `src/source_quality.py` | Cross-provider data-quality comparison |

## Scripts

```text
scripts/build_state_expectancy_table.py      # State EV table
scripts/run_narrow_backtest.py               # Full-sample prototype (lookahead-prone)
scripts/build_vol_conditioned_expectancy.py  # EV conditioned on volatility regime
scripts/run_walkforward_backtest.py          # Walk-forward EV backtest (--provider, --full-universe)
scripts/build_asset_clusters.py              # Asset-behavior clusters
scripts/build_cluster_pooled_expectancy.py   # Cluster-pooled state EV
scripts/build_markov_weighted_ev.py          # Markov probability-weighted EV
scripts/run_sensitivity_tests.py             # Walk-forward sensitivity grid
scripts/run_sensitivity_stability.py         # Per-symbol stability summary on the grid
scripts/run_sharpe_bootstrap_ci.py           # Block-bootstrap Sharpe CIs per symbol
scripts/run_random_label_baseline.py         # Permutation-baseline check
scripts/run_provider_comparison.py           # Diff walk-forward results across providers
scripts/run_drift_monitor.py                 # Daily KL-divergence state-drift alert
scripts/run_paper_trading_daily.py           # Daily paper-trading step (no execution)
scripts/build_dashboard.py                   # Render reports/dashboard/index.html
scripts/compare_data_sources.py              # OHLCV vendor-quality comparison
scripts/render_markdown_pdf.py               # Convert run reports to PDF
```

## Robustness Stack (Phase B)

The pipeline includes these guardrails so a single positive number doesn't get over-interpreted:

- **Random-label permutation baseline** — shuffles state labels, re-runs walk-forward EV, surfaces `random_label` rows in `baseline_comparison.csv`. If performance survives the shuffle, the model is exploiting an artefact.
- **Block-bootstrap Sharpe CI** — `walkforward_sharpe_ci.csv` reports point Sharpe + CI per symbol, with block size = trade horizon to preserve serial structure.
- **Sensitivity stability summary** — `sensitivity_stability_summary.csv` collapses the parameter grid into one row per symbol (median, std, IQR, share-of-negative-Sharpe).
- **Macro acceptance gate** — `src/macro.py::evaluate_macro_filter_sharpe_lift` enforces `MIN_REQUIRED_SHARPE_LIFT = 0.20` (playbook §3.12). The macro filter must clear that hurdle or be dropped.
- **Drift monitor** — `src/drift.py::drift_alert` flags when the recent state-frequency distribution diverges from the training window's via KL.

## Paper-Trading Pipeline (Phase D)

Research only — no broker, no orders. Two daily scripts:

```bash
python scripts/run_drift_monitor.py --provider fmp
python scripts/run_paper_trading_daily.py --provider fmp
```

The tracker lives at `reports/paper_trading/tracker.json` (atomic write, schema-versioned). Decisions append to `reports/tables/paper_trading_decisions.csv`. Drift alerts from the first script are consumed as an entry gate by the second.

## Non-Negotiable Rules

- **No lookahead bias** — every signal uses only prior data; walk-forward EV uses only outcomes whose exits were known before the signal date.
- **Research before production** — prove value before building a bot.
- **Beat baselines first** — buy-and-hold, MA filter, breakout, **and random-label permutation**.
- **States must be useful** — 8-12 states max, each explainable and tradable.
- **Macro filter has a hurdle** — Sharpe lift ≥ 0.20 net-of-cost or it gets dropped.
- **No execution code** — `src/paper_trading.py` records intent and exits at horizon; it does not place orders.

## Continuous Integration

Two GitHub Actions workflows live under `.github/workflows/`:

- **`test.yml`** — runs `pytest` on Python 3.11 and 3.12 for every push/PR to `main`, and sanity-builds the wheel on 3.11.
- **`dashboard.yml`** — on push to `main`, runs `scripts/build_dashboard.py` against the committed `reports/tables/*.csv`, then publishes `reports/dashboard/index.html` to GitHub Pages.

Enable Pages once in the repo settings (Settings → Pages → Source: GitHub Actions).

## See Also

- [AGENTS.md](AGENTS.md) — agent responsibilities and build order.
- [docs/roadmap.md](docs/roadmap.md) — architecture priorities and loop-closing roadmap.
- [docs/learning/current-project-guide-2026-05-17.md](docs/learning/current-project-guide-2026-05-17.md) — full state-of-the-project guide.
- `..\markov_ta_agent_playbook.md` — source playbook.
