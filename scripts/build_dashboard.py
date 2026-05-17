"""Build the static HTML results dashboard.

Run from the repository root:

    python scripts/build_dashboard.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard import generate_dashboard


def main() -> None:
    output_path = generate_dashboard()
    print(f"Saved dashboard to {output_path}")


if __name__ == "__main__":
    main()
