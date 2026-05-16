# Computing Technical Markers From yfinance Data

This note explains how the Markov TA Lab can turn basic market data into technical markers, support/resistance states, transition probabilities, and expected-value estimates.

## What yfinance Provides

yfinance gives the raw daily candle data we need for a first research lab:

```text
Date
Open
High
Low
Close
Volume
```

It can also provide adjusted close, dividends, splits, and some metadata, but the first project phase should stay focused on clean OHLCV candles.

## What We Compute

Most technical markers are derived from OHLCV data:

```text
Returns
ATR
RSI
Moving averages
Bollinger Band width
Realized volatility
Volume z-score
Support/resistance zones
Distance to support/resistance
State labels
Transition probabilities
Expected value
Backtest metrics
```

The key rule is that every marker at date `t` must use only data available at date `t` or earlier. For simulated trading, the signal should generally be acted on after the close that produced it, not during the same bar.

## Basic Derived Fields

From OHLCV:

```text
daily_return = Close.pct_change()
log_return = log(Close / Close.shift(1))
range = High - Low
body = Close - Open
upper_wick = High - max(Open, Close)
lower_wick = min(Open, Close) - Low
gap = Open - Close.shift(1)
```

These describe candle behavior and become useful later for state labeling.

## Trend Markers

Computed from `Close`:

```text
SMA_20 = Close.rolling(20).mean()
SMA_50 = Close.rolling(50).mean()
SMA_200 = Close.rolling(200).mean()

EMA_20 = Close.ewm(span=20).mean()
EMA_50 = Close.ewm(span=50).mean()

above_200dma = Close > SMA_200
trend_up = SMA_50 > SMA_200
distance_to_50dma = (Close - SMA_50) / Close
distance_to_200dma = (Close - SMA_200) / Close
```

These answer whether the asset is in an uptrend, downtrend, or neutral trend regime.

## Volatility Markers

Computed from `High`, `Low`, and `Close`:

```text
true_range = max(
  High - Low,
  abs(High - Close.shift(1)),
  abs(Low - Close.shift(1))
)

ATR_14 = true_range.rolling(14).mean()
ATR_percent = ATR_14 / Close
realized_vol_20 = daily_return.rolling(20).std() * sqrt(252)
```

Volatility states can be derived from rolling percentiles:

```text
ATR_percentile = ATR_percent.rolling(252).rank(pct=True)

LOW_VOL = ATR_percentile < 0.20
HIGH_VOL = ATR_percentile > 0.80
```

The first version can use simple volatility states:

```text
LOW_VOL
NORMAL_VOL
HIGH_VOL
VOL_COMPRESSION
VOL_EXPANSION
```

## Momentum Markers

Computed from `Close`:

```text
RSI_14
rolling_return_5 = Close / Close.shift(5) - 1
rolling_return_10 = Close / Close.shift(10) - 1
rolling_return_20 = Close / Close.shift(20) - 1
rate_of_change_20 = Close / Close.shift(20) - 1
```

These help separate continuation, exhaustion, and chop.

## Volume Markers

Computed from `Volume`:

```text
volume_sma_20 = Volume.rolling(20).mean()
volume_zscore = (Volume - Volume.rolling(20).mean()) / Volume.rolling(20).std()
```

Example breakout confirmation:

```text
breakout_volume_confirmed = volume_zscore > 1.0
```

## Bollinger And Compression Markers

Computed from `Close`:

```text
bb_mid = Close.rolling(20).mean()
bb_std = Close.rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
bb_width = (bb_upper - bb_lower) / bb_mid
```

Compression can be estimated with a rolling percentile:

```text
bb_width_percentile = bb_width.rolling(252).rank(pct=True)
compression = bb_width_percentile < 0.25
```

This is useful for identifying states such as `COMPRESSION_BELOW_RESISTANCE`.

## Support And Resistance Markers

Support/resistance should be modeled as zones, not exact lines. A simple first version can use prior rolling highs and lows:

```text
prior_high = High.shift(1).rolling(126).max()
prior_low = Low.shift(1).rolling(126).min()
```

The `.shift(1)` is important. It prevents today's high or low from defining today's support/resistance.

ATR-normalized zones:

```text
resistance = prior_high
support = prior_low

resistance_zone_upper = resistance + 0.5 * ATR
resistance_zone_lower = resistance - 0.5 * ATR

support_zone_upper = support + 0.5 * ATR
support_zone_lower = support - 0.5 * ATR
```

Distance markers:

```text
distance_to_resistance_atr = (resistance - Close) / ATR
distance_to_support_atr = (Close - support) / ATR
```

Example interpretations:

```text
near_resistance = distance_to_resistance_atr between 0 and 1
touching_resistance = Close inside resistance zone
breakout = Close > resistance_zone_upper
near_support = distance_to_support_atr between 0 and 1
support_breakdown = Close < support_zone_lower
```

## State Labels

Once indicators and levels exist, each bar can be labeled into exactly one state:

```text
0  FAR_FROM_LEVEL
1  APPROACHING_RESISTANCE
2  COMPRESSION_BELOW_RESISTANCE
3  RESISTANCE_BREAKOUT
4  BREAKOUT_RETEST
5  CONTINUATION
6  FAILED_BREAKOUT
7  APPROACHING_SUPPORT
8  SUPPORT_TOUCH
9  SUPPORT_RECLAIM
10 SUPPORT_BREAKDOWN
11 CHOP_OR_NO_EDGE
```

Example deterministic rules:

```text
APPROACHING_RESISTANCE:
  Close is below resistance zone
  Distance to resistance <= 1 ATR

COMPRESSION_BELOW_RESISTANCE:
  Close is below resistance zone
  Distance to resistance <= 1 ATR
  Bollinger width is contracting
  Recent lows are rising

RESISTANCE_BREAKOUT:
  Close > resistance_zone_upper

BREAKOUT_RETEST:
  A breakout happened recently
  Price is testing old resistance from above
  Close remains above or inside the retest zone

FAILED_BREAKOUT:
  A breakout happened recently
  Close falls back below the old resistance zone

CONTINUATION:
  Price remains above the breakout level
  Trend remains positive
```

These labels should be deterministic before adding hidden regimes or machine learning.

## Transition Matrix

After every date has a state, transition probabilities are estimated by counting state changes:

```text
APPROACHING_RESISTANCE -> COMPRESSION_BELOW_RESISTANCE
COMPRESSION_BELOW_RESISTANCE -> RESISTANCE_BREAKOUT
RESISTANCE_BREAKOUT -> BREAKOUT_RETEST
BREAKOUT_RETEST -> CONTINUATION
```

For each state:

```text
P(next_state | current_state)
```

Example:

```text
When current state is BREAKOUT_RETEST:

CONTINUATION       42%
CHOP_OR_NO_EDGE    25%
FAILED_BREAKOUT    18%
FAR_FROM_LEVEL     15%
```

## Multi-Step Forecasts

A one-step forecast uses the transition matrix directly. A five-day forecast uses the matrix raised to the fifth power:

```text
P^5
```

A ten-day forecast uses:

```text
P^10
```

This lets the system ask:

```text
Given today's state is BREAKOUT_RETEST,
what is the probability of being in CONTINUATION within 5 or 10 bars?
```

## Expected Value

Forward returns are computed from `Close` for research evaluation:

```text
forward_return_5 = Close.shift(-5) / Close - 1
forward_return_10 = Close.shift(-10) / Close - 1
```

These future values can be used to evaluate historical outcomes, but they must not be used to generate historical signals.

State-level expected value:

```text
EV_state,horizon = mean(forward_return | current_state, horizon) - costs
```

Example:

```text
BREAKOUT_RETEST, 5-day horizon:
  sample count: 184
  avg forward return: +0.38%
  win rate: 57%
  avg win: +1.25%
  avg loss: -0.78%
  cost assumption: 0.10%
  EV after costs: +0.28%
```

## Baselines

The Markov model should be compared against simple baselines:

```text
Buy and hold
Close > 200DMA
Simple breakout above 126-day high
Simple mean reversion after support touch
Randomized state labels
```

If the Markov state model cannot beat simple baselines, adding HMMs, change-point filters, or macro filters is premature.

## First Practical Pipeline

The first full research run should be:

```text
1. Download OHLCV from yfinance
2. Clean and align symbols
3. Add indicators
4. Add support/resistance zones
5. Add state labels
6. Estimate rolling transition matrices
7. Compute 5-day and 10-day state probabilities
8. Compute state-level expected value
9. Backtest using only past-estimated probabilities
10. Compare against breakout baselines
```

The edge, if any, comes from whether the state definitions and transition probabilities create positive expected value out of sample.

## Potential FMP Extension

Financial Modeling Prep can be useful after the yfinance-first pipeline is stable. Likely uses:

```text
Alternative OHLCV source
ETF and stock metadata
Financial statement context
Sector or industry classification
Calendar/event data
Fundamental filters
```

API keys should stay in `.env` and should never be committed or printed in logs.
