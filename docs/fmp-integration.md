# Financial Modeling Prep Integration

The project can use Financial Modeling Prep as an optional REST market data provider, side-by-side with yfinance, behind the same `Downloader` protocol.

## Secret Handling

Put the key in the repo-local `.env` file or in your shell environment.

Supported names:

```text
FMP_API_KEY
FINANCIAL_MODELING_PREP_API_KEY
```

The existing `.env` file currently uses:

```text
FINANCIAL_MODELING_PREP_API_KEY
```

Do not commit `.env`, print API keys, or embed keys in reports. The project client (`src/fmp.py::FMPClient`) redacts the key in its object representation and raises missing-key errors without including any secret values. Behaviour is asserted by `tests/test_fmp.py`.

## Run With FMP

The provider can be selected per-script. Either pass `--provider fmp` or set `MARKOV_DATA_PROVIDER=fmp` for the session.

### Data ingestion

```powershell
python notebooks\01_data_download.py --provider fmp
```

The output shape is identical to the yfinance pipeline:

```text
data/raw/*.csv
data/processed/*.parquet
```

Processed frames still use:

```text
Date index
Open
High
Low
Close
Volume
```

### Walk-forward backtest with FMP data

```powershell
python scripts\run_walkforward_backtest.py --provider fmp
python scripts\run_walkforward_backtest.py --provider fmp --full-universe   # 15-ticker DEFAULT_SYMBOLS
```

`--provider fmp` does two things:

1. `load_processed(symbol, provider="fmp")` looks first under `data/processed/fmp/SYMBOL.parquet`, then falls back to the legacy `data/processed/SYMBOL.parquet` for backward compatibility.
2. Summary tables are written with a provider suffix: `walkforward_backtest_summary_fmp.csv` and `walkforward_baseline_comparison_fmp.csv`.

The same `--provider fmp` flag is wired into:

```text
scripts\run_random_label_baseline.py
scripts\run_drift_monitor.py
scripts\run_paper_trading_daily.py
scripts\run_sharpe_bootstrap_ci.py
```

### Cross-provider comparison

After running the walk-forward backtest with both providers, diff the results:

```powershell
python scripts\run_walkforward_backtest.py --provider yfinance
python scripts\run_walkforward_backtest.py --provider fmp
python scripts\run_provider_comparison.py --providers yfinance fmp
```

That writes `reports/tables/provider_comparison.csv` with one row per symbol and `<metric>__delta_<other>_minus_<base>` columns for Sharpe, total return, max drawdown, win rate, and trade count.

This is a real research artefact: any non-trivial Sharpe delta between vendors means the conclusion is provider-sensitive and should not be reported without naming the vendor.

## Source-Quality Check

Before changing research assumptions, run the source-quality comparison:

```powershell
python scripts\compare_data_sources.py --left-provider fmp --right-provider yfinance --start 2010-01-01 --end 2026-05-16
```

Outputs:

```text
reports/tables/fmp_vs_yfinance_source_quality.csv
reports/tables/fmp_vs_yfinance_source_summary.csv
```

The comparison checks:

```text
row coverage
overlapping dates
provider-only dates
close differences
volume differences
missing OHLCV values
```

Close differences are the main price-integrity check. Volume differences are often vendor-definition differences and only matter when extreme or clustered around specific dates.

## Current Scope

Implemented:

```text
FMP daily EOD OHLCV via the stable historical-price-eod/full endpoint
FMP provider switch in download_ohlcv
local .env key loading with masking
mocked tests with no live API dependency
FMP-vs-yfinance source-quality comparison
--provider flag on walk-forward, random-label, sharpe-bootstrap, drift, and paper-trading scripts
Per-provider load_processed() with legacy fallback
Cross-provider walkforward delta table
```

Not implemented yet:

```text
bulk EOD refresh job
batch quote snapshots
commodity symbol discovery
intraday data
fundamentals
calendar/events
MCP-driven workflows
```

## MCP Note

FMP also documents an MCP server URL pattern:

```text
https://financialmodelingprep.com/mcp?apikey=YOUR_FMP_API_KEY
```

Use MCP later for ad hoc analyst-style queries. For repeatable backtests, REST ingestion into parquet remains the system of record.
