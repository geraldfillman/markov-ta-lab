"""Build the static HTML results dashboard.

Run from the repository root:

    python scripts/build_dashboard.py

Phase F forecast-diagnostic CSVs are generated automatically when any are
missing.  Pass --skip-diagnostics to suppress that step.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TABLES_DIR
from src.dashboard import generate_dashboard

_PHASE_F_TABLES = [
    "fallback_usage.csv",
    "confidence_distribution.csv",
    "top_rare_states.csv",
    "coverage_by_symbol.csv",
    "forecast_warnings.csv",
]


def _phase_f_csvs_present(tables_dir: Path) -> bool:
    return all((tables_dir / name).exists() for name in _PHASE_F_TABLES)


def _ensure_phase_f_csvs(tables_dir: Path) -> None:
    if _phase_f_csvs_present(tables_dir):
        return
    print("Phase F diagnostic CSVs missing — generating now...")
    from scripts.build_forecast_diagnostics import run as build_diagnostics  # noqa: PLC0415
    build_diagnostics(tables_dir=tables_dir)


def main() -> None:
    skip = "--skip-diagnostics" in sys.argv
    tables_dir = Path(TABLES_DIR)
    if not skip:
        _ensure_phase_f_csvs(tables_dir)
    output_path = generate_dashboard()
    print(f"Saved dashboard to {output_path}")


if __name__ == "__main__":
    main()
