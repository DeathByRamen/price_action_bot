"""
Entry and exit rules for trade management.

Provides configurable stop-loss, take-profit, time-stop, signal reversal,
and liquidity filtering logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntryExitConfig:
    """Configuration for entry and exit rules."""
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    time_stop_candles: int = 24
    min_volume_24h: float = 100_000.0
    max_spread_bps: float = 50.0
    reversal_min_conviction: float = 0.5


class EntryExitRules:
    """
    Evaluates entry and exit conditions for trades.

    Provides methods to check whether a trade should be entered
    or exited based on configurable rules.
    """

    def __init__(self, config: EntryExitConfig | None = None):
        self.config = config or EntryExitConfig()

    def should_enter(
        self,
        volume_24h: float,
        spread_bps: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Check if a symbol passes entry filters.

        Returns (allowed, reason).
        """
        if volume_24h < self.config.min_volume_24h:
            return False, f"Volume ${volume_24h:,.0f} below min ${self.config.min_volume_24h:,.0f}"

        if spread_bps > self.config.max_spread_bps:
            return False, f"Spread {spread_bps:.1f}bps exceeds max {self.config.max_spread_bps:.0f}bps"

        return True, "ok"

    def compute_stop_loss(
        self,
        entry_price: float,
        atr: float,
        side: str,
    ) -> float:
        """
        Compute ATR-based stop-loss price.

        Parameters
        ----------
        entry_price : float
        atr : float
            Current ATR value (in price units).
        side : str
            "LONG" or "SHORT".

        Returns
        -------
        Stop-loss price.
        """
        distance = atr * self.config.stop_loss_atr_mult
        if side == "LONG":
            return entry_price - distance
        else:
            return entry_price + distance

    def compute_take_profit(
        self,
        entry_price: float,
        atr: float,
        side: str,
        predicted_magnitude: float = 0.0,
    ) -> float:
        """
        Compute take-profit price.

        Uses the larger of ATR-based target and model-predicted magnitude.
        """
        atr_target = atr * self.config.take_profit_atr_mult
        mag_target = entry_price * abs(predicted_magnitude) if predicted_magnitude else 0

        distance = max(atr_target, mag_target)

        if side == "LONG":
            return entry_price + distance
        else:
            return entry_price - distance

    def should_exit_on_reversal(
        self,
        position_side: str,
        new_direction: str,
        new_conviction: float,
    ) -> bool:
        """Check if a position should be closed due to signal reversal."""
        if new_conviction < self.config.reversal_min_conviction:
            return False

        if position_side == "LONG" and new_direction == "DOWN":
            return True
        if position_side == "SHORT" and new_direction == "UP":
            return True

        return False

    def should_time_stop(self, candles_held: int) -> bool:
        """Check if a position should be closed due to time limit."""
        return candles_held >= self.config.time_stop_candles
