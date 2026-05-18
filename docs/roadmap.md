# Markov TA Lab Roadmap

Last updated: 2026-05-17

## Direction

Markov TA Lab should evolve into a probabilistic market-structure and regime-intelligence platform.

It should not become a black-box predictor, an indicator zoo, or an overfit strategy optimizer.

The core separation stays:

```text
Base states describe what price is doing.
Modifier states describe the environment price is operating inside.
The transition engine estimates what usually happens next.
The diagnostic layer decides whether that probability is trustworthy.
```

## Architectural Principles

- Keep base price states small, explainable, and stable. The current 8-12 state discipline remains the guardrail.
- Do not create giant flat labels such as `FAILED_BREAKOUT_PRE_OPEX_HIGH_VOL_RATES_UP`.
- Prefer layered state records over string-only composites.
- Every probability or EV estimate must expose its evidence quality: sample count, fallback level, stability, and confidence.
- Add new data and modifiers only when they improve the forecast contract or diagnostics.
- Baseline superiority and robustness gates remain mandatory before treating any result as evidence.

## Target State Record

Future conditional engines should pass structured state records rather than mutating the base state enum:

```json
{
  "base_state": "FAILED_BREAKOUT",
  "volatility_state": "VOL_EXPANSION",
  "calendar_state": "PRE_OPEX",
  "macro_state": "RATES_UP",
  "liquidity_state": "NORMAL_LIQUIDITY"
}
```

Composite strings are acceptable for grouping and reporting, but they should be derived artifacts:

```text
FAILED_BREAKOUT|VOL_EXPANSION|PRE_OPEX|RATES_UP
```

## Forecast Contract

The next architecture spine is a single forecast/diagnostic contract. Every forecast row should eventually include:

```text
symbol
date
base_state
conditions_requested
conditions_used
fallback_level
sample_count
probability
expected_value_after_cost
cost_model_used
confidence
historical_stability
invalidation_trigger
warnings
```

This contract prevents probability misuse. The system should never say only "60% probability"; it should also say how many samples supported that estimate, whether it used a fallback, and whether the estimate has been stable.

## Integration Priorities

### 1. Confidence, Coverage, and Fallback Spine

Build this before adding more regime engines.

Primary goals:

- Measure coverage for base and conditioned states.
- Detect sparse states and rare transitions.
- Select a fallback level when a composite state is under-sampled.
- Emit confidence buckets and warnings with every forecast.
- Report fallback usage so the dashboard can show where the model is relying on weak evidence.

Candidate modules:

```text
src/composite_states.py
src/conditional_markov.py
src/coverage.py
```

Fallback hierarchy:

```text
base + volatility + macro + calendar
base + volatility + macro
base + volatility
base only
```

Initial thresholds:

```text
n >= 50  -> use full composite
n >= 30  -> use partial composite
n < 30   -> fallback to base state
```

Outputs:

```text
reports/tables/state_coverage.csv
reports/tables/transition_coverage.csv
reports/tables/fallback_usage.csv
reports/tables/low_confidence_forecasts.csv
```

### 2. Calendar State Engine

Calendar states are the best first new modifier because they are deterministic, cheap, and do not require a new market-data vendor.

Start with:

```text
OPEX_WEEK
MONTH_END
HOLIDAY_LIQUIDITY
CPI_WINDOW
FOMC_WINDOW
```

Defer `TREASURY_AUCTION_WINDOW` until the project has a reliable calendar source.

Candidate module:

```text
src/calendar_states.py
```

### 3. Dynamic Cost Model

Execution reality should be integrated before higher-level tail-risk narratives. A state can have positive gross EV and still fail after realistic liquidity costs.

Start with a simple deterministic model:

```text
NORMAL_LIQUIDITY   -> base cost
THIN_LIQUIDITY     -> elevated cost
STRESSED_LIQUIDITY -> high cost
PANIC_LIQUIDITY    -> defensive cost / entry block
```

Inputs should come from fields the repo already computes or can compute cheaply:

```text
volume_zscore_20
atr_14
realized_vol_20
gap size
symbol / cluster defaults
```

Candidate module:

```text
src/cost_models.py
```

### 4. Path Memory Features

Add memory without changing the base Markov state enum.

Candidate module:

```text
src/path_features.py
```

Initial features:

```text
state_age
prior_state
prior_3_state_path
entry_velocity
failed_breakout_memory
```

These should feed diagnostics and conditional grouping, not create permanent giant state labels.

### 5. Environmental Overlays

Only after the confidence/fallback spine is in place, expand environmental modifiers.

Near-term overlays:

```text
volatility states
macro states
liquidity states
cross-asset stress states
```

The repo already has `src/volatility.py` and `src/macro.py`; prefer extending those contracts before adding parallel modules.

## Deferred Modules

The following ideas are valuable but should wait until the forecast contract and fallback diagnostics are stable:

```text
src/options_states.py
src/reflexivity.py
src/tail_risk.py
src/correlation_states.py
src/portfolio_states.py
src/analogues.py
src/regime_dashboard.py
```

Reason: these modules depend on trustworthy confidence, fallback, and coverage metadata. Without that spine, they risk becoming extra labels instead of decision-quality diagnostics.

## Loop-Closing Design

Every limitation should map to a diagnostic and corrective module:

```text
Limitation -> Diagnostic -> Corrective Module -> Validation Layer
```

Examples:

```text
Sparse composites -> coverage report -> fallback hierarchy -> out-of-sample comparison
Non-stationarity -> stability score -> rolling matrices -> walk-forward validation
Execution reality -> cost diagnostics -> dynamic cost model -> baseline comparison after costs
Probability misuse -> confidence report -> forecast contract -> dashboard warnings
```

## Phase Roadmap

### Phase F: Forecast Trust Spine

Deliver:

- Composite condition-key builder.
- Fallback selection.
- Coverage diagnostics.
- Confidence buckets.
- Dashboard coverage/fallback summary.

Success criteria:

- Forecast rows say which conditioning level was used.
- Low-sample states are flagged, not silently trusted.
- Existing walk-forward outputs remain reproducible.

### Phase G: Deterministic Modifiers

Deliver:

- Calendar state engine.
- Extended volatility-state contract if needed.
- Conditional EV/transition reports using the fallback spine.

Success criteria:

- Calendar conditioning improves interpretability without creating sparse-state overfit.
- Reports show whether calendar effects survive baselines and stability checks.

### Phase H: Cost Reality

Deliver:

- Liquidity/cost state classifier.
- Dynamic cost model.
- Walk-forward backtests that can consume state-dependent costs.

Success criteria:

- Strategy EV and Sharpe are reported after realistic cost assumptions.
- Stress-liquidity periods can reduce confidence or block entries.

### Phase I: Memory and Non-Stationarity

Deliver:

- Path-memory features.
- Rolling transition matrices.
- Probability stability score.

Success criteria:

- Forecasts can distinguish fresh states from stale states.
- Transition estimates show whether probabilities are stable enough to trust.

### Phase J: Systemic and Tail Overlays

Deliver only after Phases F-I:

- Cross-asset stress overlay.
- Tail-risk override layer.
- Reflexivity/cascade-risk diagnostics.
- Historical analogues.

Success criteria:

- Tail-risk modules reduce confidence or adjust costs rather than inventing high-precision predictions.
- Systemic overlays improve risk diagnostics without bypassing baseline gates.

## Near-Term Implementation Slice

Recommended next build:

```text
Forecast Confidence + Fallback Engine
```

Minimal scope:

```text
src/composite_states.py
src/coverage.py
src/conditional_markov.py
tests/test_composite_states.py
tests/test_coverage.py
tests/test_conditional_markov.py
```

Do not add new market data in this slice. Use existing base states, volatility states, and macro regimes.

First dashboard addition:

```text
Fallback usage
Low-confidence forecast count
Top rare states
Coverage by symbol
```

## Final Principle

The Markov model should answer:

```text
What usually happens next?
```

The diagnostic systems should answer:

```text
Can we trust that probability right now?
```

The roadmap should prioritize that distinction before adding more market overlays.
