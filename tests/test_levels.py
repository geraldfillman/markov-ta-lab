"""Tests for support/resistance levels (src/levels.py).

Key checks (from playbook §7.2):
- Levels use prior data only (no lookahead).
- Zone width is ATR-normalized.
- Nearest support/resistance is inspectable.
"""

import pytest


def test_levels_use_prior_data_only():
    """Levels at bar i should only depend on bars < i."""
    pytest.skip("Implement after levels.py is built")


def test_zone_width_is_atr_based():
    """Zone boundaries should respect the ATR multiplier."""
    pytest.skip("Implement after levels.py is built")


def test_nearest_level_populated():
    """After warm-up, nearest_support and nearest_resistance should not be NaN."""
    pytest.skip("Implement after levels.py is built")
