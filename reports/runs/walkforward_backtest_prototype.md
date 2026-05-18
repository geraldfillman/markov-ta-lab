# Experiment Report: walkforward_backtest_prototype

**Date:** 2026-05-17

## Question

Do state EV filters still work when estimated only from prior realized outcomes?

## Hypothesis

Walk-forward EV should reduce lookahead bias and reveal whether state payoff persistence exists.

## Data

Processed universe (default provider): SPY, QQQ, IWM, XLK, SMH, XLE, XLF, GLD, TLT.

## State Definitions

Deterministic support/resistance state labels from src.states.

## Model Setup

EV is re-estimated per bar using a 252-bar lookback and at least 10 prior samples.

## Backtest Rules

Long-only, next-bar entry, fixed 5-bar exit, non-overlapping trades, DEFAULT_COST_BPS included.

## Results

Saved walk-forward summary table to reports\tables\walkforward_backtest_summary.csv.

## Benchmark Comparison

Saved walk-forward baseline comparison table to reports\tables\walkforward_baseline_comparison.csv.

## What Worked

The prototype can now run without full-sample EV selection.

## What Failed

This is still a simple current-state EV filter, not Markov-probability-weighted EV.

## Bias / Risk Checks

EV estimates use only outcomes completed before the signal date.

## Next Experiment

Use behavior clusters to estimate family-level EV for sparse states.
