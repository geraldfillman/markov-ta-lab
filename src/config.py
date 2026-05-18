"""Global configuration and constants for the research lab."""

# ─── Default ticker universe ───────────────────────────────────────────
DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "SMH", "XLE", "XLF", "XLI", "XLU", "XLV",
    "GLD", "SLV", "TLT", "UUP",
]

FIRST_EXPERIMENT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "XLK", "SMH", "XLE", "XLF", "GLD", "TLT",
]

# ─── Data parameters ───────────────────────────────────────────────────
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2026-05-16"

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# ─── State definitions ─────────────────────────────────────────────────
STATE_LABELS = {
    0: "FAR_FROM_LEVEL",
    1: "APPROACHING_SUPPORT",
    2: "TOUCHING_SUPPORT",
    3: "SUPPORT_RECLAIM",
    4: "SUPPORT_BREAKDOWN",
    5: "APPROACHING_RESISTANCE",
    6: "COMPRESSION_BELOW_RESISTANCE",
    7: "RESISTANCE_BREAKOUT",
    8: "BREAKOUT_RETEST",
    9: "FAILED_BREAKOUT",
    10: "CONTINUATION",
    11: "CHOP_OR_NO_EDGE",
}

N_STATES = len(STATE_LABELS)

# ─── Volatility state labels ───────────────────────────────────────────
VOL_STATE_LABELS = {
    0: "LOW_VOL",
    1: "NORMAL_VOL",
    2: "HIGH_VOL",
    3: "VOL_COMPRESSION",
    4: "VOL_EXPANSION",
}

# ─── Macro regime labels ───────────────────────────────────────────────
MACRO_LABELS = {
    0: "RISK_ON",
    1: "RISK_OFF",
    2: "NEUTRAL",
}

# ─── Backtest defaults ─────────────────────────────────────────────────
DEFAULT_COST_BPS = 5          # 5 basis points round-trip
DEFAULT_SLIPPAGE_BPS = 5      # 5 basis points estimated slippage
DEFAULT_RISK_PER_TRADE = 0.005  # 0.5% of portfolio

# ─── Forecast horizons ─────────────────────────────────────────────────
FORECAST_HORIZONS = [1, 5, 10, 20]

# ─── Markov defaults ───────────────────────────────────────────────────
DEFAULT_MARKOV_LOOKBACK = 252   # ~1 year of trading days
DEFAULT_LAPLACE_ALPHA = 1e-6

# ─── Support/Resistance defaults ───────────────────────────────────────
DEFAULT_SR_LOOKBACK = 126       # ~6 months
DEFAULT_ATR_ZONE_MULT = 0.5

# ─── Report paths ──────────────────────────────────────────────────────
REPORTS_DIR = "reports"
CHARTS_DIR = "reports/charts"
TABLES_DIR = "reports/tables"
RUNS_DIR = "reports/runs"
