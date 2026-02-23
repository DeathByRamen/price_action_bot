"""
Drawdown management and circuit breakers.

Monitors equity drawdown and reduces/halts trading when thresholds are breached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DrawdownState(Enum):
    NORMAL = "normal"
    REDUCED = "reduced"
    HALTED = "halted"
    RECOVERING = "recovering"


@dataclass
class DrawdownConfig:
    """Configuration for drawdown management."""
    reduce_threshold_pct: float = 5.0
    halt_threshold_pct: float = 10.0
    reduce_factor: float = 0.5
    recovery_pct: float = 50.0
    cooldown_periods: int = 24


class DrawdownManager:
    """
    Monitors drawdown and adjusts trading behavior.

    States:
    - NORMAL: trade at full size
    - REDUCED: reduce position sizes by reduce_factor after reduce_threshold hit
    - HALTED: no new trades after halt_threshold hit
    - RECOVERING: gradually increasing size as equity recovers
    """

    def __init__(self, config: DrawdownConfig | None = None):
        self.config = config or DrawdownConfig()
        self.peak_equity: float = 0.0
        self.state = DrawdownState.NORMAL
        self.periods_in_state: int = 0
        self._halted_at_equity: float = 0.0

    def update(self, current_equity: float) -> DrawdownState:
        """Update drawdown state based on current equity. Call each period."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity <= 0:
            return self.state

        drawdown_pct = ((self.peak_equity - current_equity) / self.peak_equity) * 100
        self.periods_in_state += 1

        if self.state == DrawdownState.HALTED:
            if self._halted_at_equity > 0:
                recovery = (current_equity - self._halted_at_equity) / self._halted_at_equity * 100
                if recovery >= self.config.recovery_pct * (self.config.halt_threshold_pct / 100):
                    self.state = DrawdownState.RECOVERING
                    self.periods_in_state = 0
                    logger.info(
                        "Drawdown recovery started: equity=$%.2f (recovered %.1f%%)",
                        current_equity, recovery,
                    )
            return self.state

        if self.state == DrawdownState.RECOVERING:
            if drawdown_pct < self.config.reduce_threshold_pct:
                self.state = DrawdownState.NORMAL
                self.periods_in_state = 0
                logger.info("Drawdown fully recovered — resuming normal trading")
            elif drawdown_pct >= self.config.halt_threshold_pct:
                self.state = DrawdownState.HALTED
                self._halted_at_equity = current_equity
                self.periods_in_state = 0
                logger.warning(
                    "Drawdown exceeded halt threshold (%.1f%%) during recovery — halting",
                    drawdown_pct,
                )
            return self.state

        if drawdown_pct >= self.config.halt_threshold_pct:
            self.state = DrawdownState.HALTED
            self._halted_at_equity = current_equity
            self.periods_in_state = 0
            logger.warning(
                "CIRCUIT BREAKER: Drawdown %.1f%% exceeds halt threshold %.1f%% — "
                "all new trades halted",
                drawdown_pct, self.config.halt_threshold_pct,
            )
        elif drawdown_pct >= self.config.reduce_threshold_pct:
            if self.state != DrawdownState.REDUCED:
                self.state = DrawdownState.REDUCED
                self.periods_in_state = 0
                logger.warning(
                    "Drawdown %.1f%% exceeds reduce threshold %.1f%% — "
                    "position sizes reduced to %.0f%%",
                    drawdown_pct, self.config.reduce_threshold_pct,
                    self.config.reduce_factor * 100,
                )
        else:
            if self.state != DrawdownState.NORMAL:
                self.state = DrawdownState.NORMAL
                self.periods_in_state = 0

        return self.state

    def get_size_multiplier(self) -> float:
        """
        Returns a multiplier for position sizing based on current state.

        NORMAL = 1.0, REDUCED = reduce_factor, HALTED = 0.0,
        RECOVERING = gradually increasing from reduce_factor to 1.0.
        """
        if self.state == DrawdownState.NORMAL:
            return 1.0
        elif self.state == DrawdownState.REDUCED:
            return self.config.reduce_factor
        elif self.state == DrawdownState.HALTED:
            return 0.0
        elif self.state == DrawdownState.RECOVERING:
            progress = min(
                self.periods_in_state / max(self.config.cooldown_periods, 1), 1.0
            )
            return self.config.reduce_factor + (1.0 - self.config.reduce_factor) * progress
        return 1.0

    @property
    def can_trade(self) -> bool:
        return self.state != DrawdownState.HALTED
