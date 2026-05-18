"""Summarise the walk-forward sensitivity grid into a per-symbol stability table.

Reads ``reports/tables/walkforward_sensitivity.csv`` and writes
``reports/tables/sensitivity_stability_summary.csv`` (median Sharpe, std,
IQR, share of negative-Sharpe configs). Powers the Research QA dashboard
card so we can see which symbols hold up across the parameter grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TABLES_DIR
from src.metrics import sensitivity_stability_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="sharpe")
    parser.add_argument("--input", default="walkforward_sensitivity.csv")
    parser.add_argument("--output", default="sensitivity_stability_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tables_dir = Path(TABLES_DIR)
    src = tables_dir / args.input
    if not src.exists():
        raise SystemExit(f"Missing input: {src}. Run scripts/run_sensitivity_tests.py first.")
    sens = pd.read_csv(src)
    summary = sensitivity_stability_summary(sens, metric=args.metric)
    out_path = tables_dir / args.output
    summary.to_csv(out_path, index=False)
    print(f"Wrote stability summary ({len(summary)} symbols) to {out_path}")


if __name__ == "__main__":
    main()
