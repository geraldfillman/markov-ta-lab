"""Tests for backtesting (src/backtests.py).

Key checks (from playbook §7.5):
- Transaction costs included.
- Positions are shifted correctly (no same-bar future info).
- Benchmark comparison included.
"""

import pytest


def test_costs_included():
    """Backtest results should differ from zero-cost results."""
    pytest.skip("Implement after backtests.py is built")


def test_no_same_bar_signal():
    """Signals should not use same-bar close for entry."""
    pytest.skip("Implement after backtests.py is built")


def test_benchmark_exists():
    """Results dict should include benchmark comparison."""
    pytest.skip("Implement after backtests.py is built")
