"""
Portfolio and position tracking for backtesting.

Tracks open/closed positions, P&L, equity curve, and exposure over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .costs import TransactionCosts

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single trading position."""
    symbol: str
    side: str              # "LONG" or "SHORT"
    entry_price: float
    size: float            # position size in base currency units
    entry_time: str
    notional: float        # entry_price * size
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    def close(
        self,
        exit_price: float,
        exit_time: str,
        tx_costs: TransactionCosts,
        reason: str = "signal",
    ) -> float:
        """Close the position and compute P&L. Returns net P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason

        if self.side == "LONG":
            self.pnl = (exit_price - self.entry_price) / self.entry_price * self.notional
        else:
            self.pnl = (self.entry_price - exit_price) / self.entry_price * self.notional

        entry_cost = tx_costs.entry_cost(self.notional)
        exit_cost = tx_costs.exit_cost(self.notional)
        self.costs = entry_cost + exit_cost
        self.net_pnl = self.pnl - self.costs

        return self.net_pnl

    def unrealized_pnl(self, current_price: float) -> float:
        """Compute unrealized P&L at current price."""
        if self.side == "LONG":
            return (current_price - self.entry_price) / self.entry_price * self.notional
        else:
            return (self.entry_price - current_price) / self.entry_price * self.notional


class Portfolio:
    """
    Tracks all positions and equity over time.

    Parameters
    ----------
    initial_capital : float
        Starting capital in quote currency (e.g., USDT).
    max_positions : int
        Maximum number of concurrent open positions.
    max_position_pct : float
        Maximum fraction of capital for a single position.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        max_positions: int = 10,
        max_position_pct: float = 0.10,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct

        self.open_positions: dict[str, Position] = {}
        self.closed_positions: list[Position] = []
        self.equity_curve: list[tuple[str, float]] = []

    @property
    def num_open(self) -> int:
        return len(self.open_positions)

    @property
    def total_exposure(self) -> float:
        return sum(p.notional for p in self.open_positions.values())

    def equity(self, prices: dict[str, float]) -> float:
        """Compute total equity (cash + unrealized P&L)."""
        unrealized = sum(
            p.unrealized_pnl(prices.get(p.symbol, p.entry_price))
            for p in self.open_positions.values()
        )
        return self.cash + unrealized

    def can_open(self) -> bool:
        """Check if we can open a new position."""
        return self.num_open < self.max_positions

    def position_size(self, price: float) -> float:
        """Compute position notional based on available capital and limits."""
        max_notional = self.cash * self.max_position_pct
        return min(max_notional, self.cash * 0.95)

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        timestamp: str,
        tx_costs: TransactionCosts,
        size_override: Optional[float] = None,
    ) -> Optional[Position]:
        """Open a new position. Returns the Position if opened, None if rejected."""
        if symbol in self.open_positions:
            return None
        if not self.can_open():
            return None

        notional = size_override or self.position_size(price)
        if notional <= 0 or notional > self.cash:
            return None

        entry_cost = tx_costs.entry_cost(notional)
        self.cash -= entry_cost

        size = notional / price
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            size=size,
            entry_time=timestamp,
            notional=notional,
        )
        self.open_positions[symbol] = pos
        return pos

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: str,
        tx_costs: TransactionCosts,
        reason: str = "signal",
    ) -> Optional[float]:
        """Close an open position. Returns net P&L or None if no position."""
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return None

        net_pnl = pos.close(price, timestamp, tx_costs, reason)
        self.cash += pos.notional + net_pnl
        self.closed_positions.append(pos)
        return net_pnl

    def close_all(
        self,
        prices: dict[str, float],
        timestamp: str,
        tx_costs: TransactionCosts,
        reason: str = "end_of_backtest",
    ) -> float:
        """Close all open positions. Returns total net P&L."""
        total_pnl = 0.0
        symbols = list(self.open_positions.keys())
        for sym in symbols:
            price = prices.get(sym, self.open_positions[sym].entry_price)
            pnl = self.close_position(sym, price, timestamp, tx_costs, reason)
            if pnl is not None:
                total_pnl += pnl
        return total_pnl

    def record_equity(self, timestamp: str, prices: dict[str, float]) -> None:
        """Record equity at a point in time."""
        self.equity_curve.append((timestamp, self.equity(prices)))

    def get_trade_log(self) -> list[dict]:
        """Return closed trades as a list of dicts."""
        return [
            {
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": p.entry_price,
                "exit_price": p.exit_price,
                "entry_time": p.entry_time,
                "exit_time": p.exit_time,
                "notional": p.notional,
                "pnl": p.pnl,
                "costs": p.costs,
                "net_pnl": p.net_pnl,
                "exit_reason": p.exit_reason,
            }
            for p in self.closed_positions
        ]
