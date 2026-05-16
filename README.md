# Markov + Technical Analysis Research Lab

> When price interacts with support/resistance or trend structure, what is the probability of each next market state, and does that probability create positive expected value after costs under the current macro/volatility regime?

## Purpose

Research, simulation, and paper-trading preparation **only**. This repo is not financial advice and should not be used to deploy live capital without independent validation, risk controls, and compliance review.

## Core Research Loop

1. Pull clean market data
2. Engineer technical features
3. Detect support/resistance zones
4. Label each bar into a market state
5. Estimate Markov transition probabilities using only past data
6. Forecast 1-step and multi-step future state probabilities
7. Convert probabilities into expected value
8. Backtest with realistic costs and walk-forward estimation
9. Add HMM, change-point, volatility, and macro filters only after the simple system works
10. Produce repeatable reports

## Quick Start

```bash
conda create -n markov-lab python=3.11 -y
conda activate markov-lab
pip install -r requirements.txt
python -m ipykernel install --user --name markov-lab --display-name "Python (markov-lab)"
```

## Verify Setup

```bash
python -m pytest -q
```

The initial setup is expected to pass smoke/setup tests while skipping future phase tests for levels, states, Markov modeling, and backtesting until those agents are implemented.

## Project Structure

```text
markov-ta-lab/
  AGENTS.md              # Agent roles and responsibilities
  README.md              # This file
  environment.yml        # Conda environment spec
  requirements.txt       # pip dependencies
  pyproject.toml         # pytest configuration

  docs/superpowers/plans/ # Saved setup and implementation plans
  notebooks/              # Research notebooks/scripts numbered by phase
  src/                    # Reusable Python modules
  data/raw/               # Raw downloaded market data
  data/processed/         # Cleaned parquet files
  reports/charts/         # Visualization outputs
  reports/tables/         # CSV metric tables
  reports/runs/           # Experiment run logs
  tests/                  # Unit and setup tests
```

## Non-Negotiable Rules

- **No lookahead bias** - every signal uses only prior data
- **Research before production** - prove value before building a bot
- **Beat baselines first** - buy-and-hold, MA filter, simple breakout, mean-reversion, randomized labels
- **States must be useful** - 8-12 states max, each explainable and tradable

See `AGENTS.md` for agent responsibilities and `..\markov_ta_agent_playbook.md` for the full source playbook.
