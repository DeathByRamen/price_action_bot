"""
Portfolio-level risk controls.

Enforces maximum exposure, correlated position limits, and per-symbol caps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskConfig:
    """Configuration for portfolio risk controls."""
    max_total_exposure_pct: float = 0.50
    max_correlated_positions: int = 3
    correlation_threshold: float = 0.70
    max_single_position_pct: float = 0.10
    max_sector_exposure_pct: float = 0.25


class PortfolioRiskManager:
    """
    Enforces portfolio-level risk constraints.

    Checks new positions against existing portfolio to prevent
    excessive concentration and correlated risk.
    """

    def __init__(self, config: Optional[PortfolioRiskConfig] = None):
        self.config = config or PortfolioRiskConfig()
        self._correlation_cache: Dict[str, Dict[str, float]] = {}

    def can_open_position(
        self,
        symbol: str,
        notional: float,
        capital: float,
        current_positions: Dict[str, float],
        returns_history: Optional[Dict[str, np.ndarray]] = None,
    ) -> tuple[bool, str]:
        """
        Check if opening a new position violates risk constraints.

        Parameters
        ----------
        symbol : str
            Symbol to open.
        notional : float
            Proposed position notional.
        capital : float
            Current total capital.
        current_positions : dict[str, float]
            Existing open positions {symbol: notional}.
        returns_history : dict[str, ndarray] | None
            Recent returns per symbol for correlation checking.

        Returns
        -------
        (allowed, reason) tuple.
        """
        if notional / capital > self.config.max_single_position_pct:
            return False, (
                f"Position size {notional/capital:.1%} exceeds "
                f"max {self.config.max_single_position_pct:.0%}"
            )

        total_exposure = sum(current_positions.values()) + notional
        if total_exposure / capital > self.config.max_total_exposure_pct:
            return False, (
                f"Total exposure {total_exposure/capital:.1%} would exceed "
                f"max {self.config.max_total_exposure_pct:.0%}"
            )

        if returns_history and len(current_positions) > 0:
            correlated_count = self._count_correlated(
                symbol, set(current_positions.keys()), returns_history
            )
            if correlated_count >= self.config.max_correlated_positions:
                return False, (
                    f"{correlated_count} correlated positions already open "
                    f"(max {self.config.max_correlated_positions})"
                )

        return True, "ok"

    def _count_correlated(
        self,
        new_symbol: str,
        existing_symbols: Set[str],
        returns_history: Dict[str, np.ndarray],
    ) -> int:
        """Count how many existing positions are highly correlated with the new symbol."""
        if new_symbol not in returns_history:
            return 0

        new_returns = returns_history[new_symbol]
        correlated = 0

        for sym in existing_symbols:
            if sym not in returns_history:
                continue
            existing_returns = returns_history[sym]
            min_len = min(len(new_returns), len(existing_returns))
            if min_len < 10:
                continue

            corr = np.corrcoef(new_returns[-min_len:], existing_returns[-min_len:])[0, 1]
            if abs(corr) >= self.config.correlation_threshold:
                correlated += 1

        return correlated

    def compute_correlation_matrix(
        self,
        returns_history: Dict[str, np.ndarray],
        min_periods: int = 24,
    ) -> tuple[list[str], np.ndarray]:
        """
        Compute pairwise correlation matrix for all symbols.

        Returns (symbol_list, correlation_matrix).
        """
        symbols = [s for s, r in returns_history.items() if len(r) >= min_periods]
        n = len(symbols)
        if n < 2:
            return symbols, np.eye(n)

        min_len = min(len(returns_history[s]) for s in symbols)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                r_i = returns_history[symbols[i]][-min_len:]
                r_j = returns_history[symbols[j]][-min_len:]
                corr = np.corrcoef(r_i, r_j)[0, 1]
                matrix[i, j] = corr
                matrix[j, i] = corr

        return symbols, matrix
