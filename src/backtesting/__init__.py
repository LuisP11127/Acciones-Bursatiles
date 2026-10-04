"""Backtesting Engine (walk-forward, sin información futura)."""

from .engine import Rules, check_exit, run_strategy
from .metrics import full_metrics

__all__ = ["Rules", "check_exit", "run_strategy", "full_metrics"]
