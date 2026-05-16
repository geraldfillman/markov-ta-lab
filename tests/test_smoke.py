"""Smoke test – verify all src modules import cleanly."""

import importlib
import pytest

MODULES = [
    "src.config",
    "src.data",
    "src.indicators",
    "src.levels",
    "src.states",
    "src.markov",
    "src.hmm_models",
    "src.changepoints",
    "src.volatility",
    "src.backtests",
    "src.metrics",
    "src.plotting",
    "src.reports",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name):
    """Every src module should import without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None
