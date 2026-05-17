"""Pytest bootstrap for local development."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEST_OUTPUT_DIR = ROOT / ".test_output"


def ensure_test_output_dir() -> Path:
    """Create the project-local pytest output directory if needed."""
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    return TEST_OUTPUT_DIR


ensure_test_output_dir()


def pytest_sessionstart(session) -> None:  # type: ignore[no-untyped-def]
    ensure_test_output_dir()


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    configured = getattr(config.option, "basetemp", None)
    if configured is None:
        return

    base = Path(configured)
    if base.name == "pytest-tmp":
        config.option.basetemp = str(TEST_OUTPUT_DIR / f"pytest-tmp-{os.getpid()}")
