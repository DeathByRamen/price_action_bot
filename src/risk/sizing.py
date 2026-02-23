"""
Position sizing strategies.

Implements Kelly criterion, volatility targeting, and fixed-fraction sizing.
All sizers return a notional amount (in quote currency) for a given trade.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""

    @abstractmethod
    def compute_size(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_atr_pct: float = 0.0,
        conviction: float = 0.5,
    ) -> float:
        """
        Compute position notional.

        Parameters
        ----------
        capital : float
            Available capital.
        win_rate : float
            Historical win rate (0 to 1).
        avg_win : float
            Average winning trade return (fraction, e.g. 0.02 = 2%).
        avg_loss : float
            Average losing trade return (fraction, positive, e.g. 0.01 = 1%).
        current_atr_pct : float
            Current ATR as fraction of price (for volatility sizing).
        conviction : float
            Model conviction for this signal (0 to 1).

        Returns
        -------
        float : position notional in quote currency.
        """
        ...


class KellySizer(PositionSizer):
    """
    Kelly criterion position sizing.

    f* = (p * b - q) / b

    where p = win probability, q = 1-p, b = avg_win / avg_loss.

    Uses fractional Kelly (default half-Kelly) for safety.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        max_position_pct: float = 0.10,
        min_position_pct: float = 0.01,
    ):
        self.fraction = fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct

    def compute_size(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_atr_pct: float = 0.0,
        conviction: float = 0.5,
    ) -> float:
        if avg_loss <= 0 or win_rate <= 0:
            return capital * self.min_position_pct

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_fraction = (p * b - q) / b

        if kelly_fraction <= 0:
            return capital * self.min_position_pct

        position_pct = kelly_fraction * self.fraction
        position_pct = max(self.min_position_pct, min(self.max_position_pct, position_pct))

        position_pct *= conviction

        return capital * position_pct


class VolatilitySizer(PositionSizer):
    """
    Volatility-targeted position sizing.

    Sizes positions inversely proportional to current ATR, targeting
    a fixed dollar risk per trade.
    """

    def __init__(
        self,
        target_risk_pct: float = 0.01,
        max_position_pct: float = 0.10,
        atr_multiplier: float = 2.0,
    ):
        self.target_risk_pct = target_risk_pct
        self.max_position_pct = max_position_pct
        self.atr_multiplier = atr_multiplier

    def compute_size(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_atr_pct: float = 0.0,
        conviction: float = 0.5,
    ) -> float:
        if current_atr_pct <= 0:
            return capital * 0.02

        risk_per_unit = current_atr_pct * self.atr_multiplier
        dollar_risk = capital * self.target_risk_pct
        notional = dollar_risk / risk_per_unit

        max_notional = capital * self.max_position_pct
        return min(notional, max_notional)


class FixedFractionSizer(PositionSizer):
    """Simple fixed-fraction position sizing."""

    def __init__(self, fraction: float = 0.05):
        self.fraction = fraction

    def compute_size(
        self,
        capital: float,
        win_rate: float = 0.0,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
        current_atr_pct: float = 0.0,
        conviction: float = 0.5,
    ) -> float:
        return capital * self.fraction
