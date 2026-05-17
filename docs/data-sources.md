# Data Sources

This project starts with programmatic sources that are practical for repeatable research. Free web pages can still be useful for sanity checks, but they should not be scraped unless the source clearly permits automated extraction.

## Active Programmatic Sources

| Source | Use | Status | Notes |
|---|---|---|---|
| yfinance / Yahoo Finance | Daily OHLCV candles for ETFs | Active | Current first-pass source for raw market data. Good for research prototypes, not a guaranteed production data feed. |

## Available Keyed Sources

| Source | Use | Status | Notes |
|---|---|---|---|
| Financial Modeling Prep | OHLCV, fundamentals, calendars, profiles, sector/industry metadata | Candidate | API keys are available locally in the user's `.env`. Do not commit or print keys. Add only when the project needs a second provider or fundamentals. |

## Manual Reference / Sanity-Check Sources

| Source | Use | Status | Notes |
|---|---|---|---|
| StatMuse | Quick natural-language market quote checks | Manual reference only | Free pages may be useful for quick checks, but StatMuse terms prohibit scraping, mining, or extraction without permission or an API license. Do not wire into automated ingestion. |
| MacroTrends | Long-horizon charts and historical context | Manual reference only | Useful as a browser-based cross-check where pages are publicly viewable. Treat as manual reference unless an approved API/license is available. |

## Source Selection Rules

1. Prefer sources with stable APIs for repeatable research.
2. Keep API keys in environment variables or `.env` files outside git.
3. Do not scrape sites that prohibit automated extraction.
4. Record the source used in every experiment report.
5. Use manual web sources only to sanity-check figures, not as the system of record.
