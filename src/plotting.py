"""Plotting utilities for the research lab.

Provides chart helpers for:
- Transition matrix heatmaps
- State timeline overlays
- Equity curves
- Level visualisation
- Regime overlays
- Parameter sensitivity surfaces
"""

import pandas as pd
import numpy as np


def plot_transition_heatmap(P: np.ndarray, state_labels: dict | None = None) -> None:
    """Display a heatmap of the transition probability matrix."""
    raise NotImplementedError


def plot_state_timeline(df: pd.DataFrame, states: pd.Series, symbol: str = "") -> None:
    """Overlay state labels on a price chart."""
    raise NotImplementedError


def plot_equity_curve(equity: pd.Series, benchmark: pd.Series | None = None) -> None:
    """Plot strategy equity vs benchmark."""
    raise NotImplementedError


def plot_regime_overlay(df: pd.DataFrame, regimes: pd.Series, symbol: str = "") -> None:
    """Shade price chart by HMM regime."""
    raise NotImplementedError


def plot_parameter_sensitivity(results: pd.DataFrame, x: str, y: str, metric: str) -> None:
    """2D heatmap of a backtest metric across two parameter axes."""
    raise NotImplementedError
