"""Smoke test – verify all src modules import cleanly."""

import importlib
import pytest

MODULES = [
    "src.config",
    "src.data",
    "src.fmp",
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
    "src.macro",
    "src.source_quality",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name):
    """Every src module should import without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None
