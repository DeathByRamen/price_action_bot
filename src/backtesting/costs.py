"""
Transaction cost modeling for backtesting.

Models maker/taker fees, funding rate costs, and slippage estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransactionCosts:
    """
    Models all costs associated with executing a trade.

    Parameters
    ----------
    maker_fee : float
        Maker fee as a fraction (e.g., 0.0002 = 0.02%).
    taker_fee : float
        Taker fee as a fraction (e.g., 0.0006 = 0.06%).
    slippage_bps : float
        Estimated slippage in basis points (e.g., 5.0 = 0.05%).
    funding_rate_per_8h : float
        Average funding rate per 8-hour period (e.g., 0.0001 = 0.01%).
    use_maker : bool
        Whether to use maker fees (limit orders) or taker fees (market orders).
    """
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006
    slippage_bps: float = 5.0
    funding_rate_per_8h: float = 0.0001
    use_maker: bool = False

    @property
    def trade_fee(self) -> float:
        return self.maker_fee if self.use_maker else self.taker_fee

    def entry_cost(self, notional: float) -> float:
        """Total cost to open a position (fee + slippage)."""
        fee = notional * self.trade_fee
        slippage = notional * (self.slippage_bps / 10_000)
        return fee + slippage

    def exit_cost(self, notional: float) -> float:
        """Total cost to close a position (fee + slippage)."""
        return self.entry_cost(notional)

    def round_trip_cost(self, notional: float) -> float:
        """Total cost for a complete trade (entry + exit)."""
        return self.entry_cost(notional) + self.exit_cost(notional)

    def funding_cost(self, notional: float, hours_held: float, is_long: bool = True) -> float:
        """
        Estimated funding cost over the holding period.

        Longs pay when funding is positive, shorts pay when negative.
        We assume the average funding rate applies.
        """
        periods = hours_held / 8.0
        cost = notional * abs(self.funding_rate_per_8h) * periods
        return cost
