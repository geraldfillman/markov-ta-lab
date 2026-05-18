# Experiment Report: narrow_backtest_prototype

**Date:** 2026-05-17

## Question

Do positive state expectancy filters produce useful long-only ETF trades?

## Hypothesis

States with positive historical 5-bar EV should produce better fixed-horizon trades after costs.

## Data

Processed first experiment ETF universe: SPY, QQQ, IWM, XLK, SMH, XLE, XLF, GLD, TLT.

## State Definitions

Deterministic support/resistance state labels from src.states.

## Model Setup

Use state_expectancy.csv as the entry filter; enter when the current state's 5-bar EV is positive.

## Backtest Rules

Long-only, next-bar entry, fixed 5-bar exit, non-overlapping trades, DEFAULT_COST_BPS included.

## Results

Saved summary table to reports\tables\narrow_backtest_summary.csv.

## Benchmark Comparison

Saved baseline comparison table to reports\tables\baseline_comparison.csv.

## What Worked

The prototype now connects deterministic states, expected value tables, and readable backtest output.

## What Failed

This is not yet walk-forward; the expectancy filter is estimated from the full sample.

## Bias / Risk Checks

Known lookahead risk remains in full-sample expectancy selection. Next phase should make the EV filter walk-forward.

## Next Experiment

Replace full-sample EV lookup with walk-forward EV estimates and compare against simple breakout and MA baselines.
