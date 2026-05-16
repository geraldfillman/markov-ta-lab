"""Tests for Markov model (src/markov.py).

Key checks (from playbook §7.4):
- Transition matrix rows sum to 1.
- Sparse rows are handled.
- Matrix shape matches n_states.
- Multi-step forecasts produce valid probability vectors.
"""

import pytest


def test_rows_sum_to_one():
    """Each row of the transition matrix should sum to 1.0."""
    pytest.skip("Implement after markov.py is built")


def test_matrix_shape():
    """Transition matrix shape should be (n_states, n_states)."""
    pytest.skip("Implement after markov.py is built")


def test_sparse_row_handling():
    """States with zero observations should produce a valid row (not NaN)."""
    pytest.skip("Implement after markov.py is built")


def test_multistep_forecast_is_probability():
    """Multi-step forecast should sum to ~1.0 and have no negatives."""
    pytest.skip("Implement after markov.py is built")
