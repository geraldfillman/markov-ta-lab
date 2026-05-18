"""Compare OHLCV coverage and value alignment between two market data providers."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_END, DEFAULT_START, FIRST_EXPERIMENT_SYMBOLS, TABLES_DIR
from src.data import download_ohlcv
from src.source_quality import compare_ohlcv_sources, summarize_source_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OHLCV data providers.")
    parser.add_argument("--left-provider", default="fmp", choices=["fmp", "yfinance"])
    parser.add_argument("--right-provider", default="yfinance", choices=["fmp", "yfinance"])
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--symbols", default=",".join(FIRST_EXPERIMENT_SYMBOLS))
    parser.add_argument("--output-dir", default=TABLES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    left = download_ohlcv(symbols, args.start, args.end, provider=args.left_provider)
    right = download_ohlcv(symbols, args.start, args.end, provider=args.right_provider)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = compare_ohlcv_sources(left, right, args.left_provider, args.right_provider)
    left_summary = summarize_source_frames(left, args.left_provider)
    right_summary = summarize_source_frames(right, args.right_provider)
    source_summary = left_summary.reset_index().merge(
        right_summary.reset_index(),
        on="symbol",
        how="outer",
        suffixes=(f"_{args.left_provider}", f"_{args.right_provider}"),
    ).set_index("symbol")

    comparison_path = output_dir / f"{args.left_provider}_vs_{args.right_provider}_source_quality.csv"
    summary_path = output_dir / f"{args.left_provider}_vs_{args.right_provider}_source_summary.csv"
    comparison.to_csv(comparison_path)
    source_summary.to_csv(summary_path)

    print(f"Saved {len(comparison)} comparison rows to {comparison_path}")
    print(f"Saved {len(source_summary)} source summary rows to {summary_path}")
    print(comparison)


if __name__ == "__main__":
    main()

