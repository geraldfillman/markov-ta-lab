"""Tests for state labeling (src/states.py).

Key checks (from playbook §7.3):
- Every bar has exactly one state.
- State definitions produce valid integer labels.
- State frequencies are reasonable.
"""

import pytest


def test_every_bar_has_one_state():
    """Every bar should have exactly one state label, no NaNs."""
    pytest.skip("Implement after states.py is built")


def test_state_labels_in_valid_range():
    """State labels should be integers in [0, N_STATES)."""
    pytest.skip("Implement after states.py is built")


def test_state_frequency_minimum():
    """No state should have fewer than min_count unless intentionally merged."""
    pytest.skip("Implement after states.py is built")
