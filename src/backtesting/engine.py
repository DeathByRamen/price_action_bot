"""
Core backtesting engine.

Walks forward through historical data, generates signals via a pluggable
SignalGenerator, and simulates trades with realistic transaction costs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .costs import TransactionCosts
from .metrics import PerformanceReport, compute_metrics
from .portfolio import Portfolio
from .signals import SignalGenerator, TradeSignal

logger = logging.getLogger(__name__)


class Backtester:
    """
    Event-driven backtesting engine.

    Parameters
    ----------
    signal_generator : SignalGenerator
        Generates trade signals from market data.
    costs : TransactionCosts
        Transaction cost model.
    initial_capital : float
        Starting capital in quote currency.
    max_positions : int
        Maximum concurrent open positions.
    max_position_pct : float
        Maximum fraction of capital per position.
    max_hold_candles : int
        Force-close positions after this many candles (time stop).
    stop_loss_pct : float
        Stop-loss as fraction of entry price (e.g., 0.02 = 2%).
        Set to 0 to disable.
    take_profit_pct : float
        Take-profit as fraction of entry price (e.g., 0.03 = 3%).
        Set to 0 to disable.
    """

    def __init__(
        self,
        signal_generator: SignalGenerator,
        costs: Optional[TransactionCosts] = None,
        initial_capital: float = 10_000.0,
        max_positions: int = 10,
        max_position_pct: float = 0.10,
        max_hold_candles: int = 24,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
    ):
        self.signal_generator = signal_generator
        self.costs = costs or TransactionCosts()
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.max_hold_candles = max_hold_candles
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def run(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> PerformanceReport:
        """
        Run the backtest on historical data.

        Parameters
        ----------
        symbol_data : dict[str, pd.DataFrame]
            Historical OHLCV data per symbol.  Each DataFrame must have
            columns: open, high, low, close, volume, ts.
            Must be sorted chronologically.
        start_idx : int
            Row index to start backtesting from (allows warmup period).
        end_idx : int | None
            Row index to stop at (exclusive). None = end of data.

        Returns
        -------
        PerformanceReport with all performance metrics.
        """
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            max_positions=self.max_positions,
            max_position_pct=self.max_position_pct,
        )

        ref_symbol = next(iter(symbol_data))
        ref_df = symbol_data[ref_symbol]
        max_idx = len(ref_df)
        end_idx = min(end_idx or max_idx, max_idx)

        position_entry_idx: dict[str, int] = {}
        benchmark_prices: list[float] = []

        btc_df = symbol_data.get("BTCUSDT")

        logger.info(
            "Starting backtest: %d symbols, indices [%d, %d), capital=$%.0f",
            len(symbol_data), start_idx, end_idx, self.initial_capital,
        )

        for idx in range(start_idx, end_idx):
            current_prices: dict[str, float] = {}
            current_highs: dict[str, float] = {}
            current_lows: dict[str, float] = {}
            timestamp = ""

            for sym, df in symbol_data.items():
                if idx < len(df):
                    current_prices[sym] = float(df["close"].iloc[idx])
                    current_highs[sym] = float(df["high"].iloc[idx])
                    current_lows[sym] = float(df["low"].iloc[idx])
                    if not timestamp:
                        timestamp = str(df["ts"].iloc[idx]) if "ts" in df.columns else str(idx)

            if btc_df is not None and idx < len(btc_df):
                benchmark_prices.append(float(btc_df["close"].iloc[idx]))

            self._check_stops(
                portfolio, current_prices, current_highs, current_lows,
                timestamp, position_entry_idx, idx,
            )

            window_data = {}
            for sym, df in symbol_data.items():
                if idx < len(df):
                    window_data[sym] = df.iloc[:idx + 1]

            signals = self.signal_generator.generate_signals(window_data, timestamp)

            self._process_signals(
                portfolio, signals, current_prices, timestamp, position_entry_idx, idx,
            )

            portfolio.record_equity(timestamp, current_prices)

        final_prices = {}
        for sym, df in symbol_data.items():
            if end_idx - 1 < len(df):
                final_prices[sym] = float(df["close"].iloc[end_idx - 1])
        final_ts = str(ref_df["ts"].iloc[end_idx - 1]) if "ts" in ref_df.columns else str(end_idx - 1)
        portfolio.close_all(final_prices, final_ts, self.costs, "end_of_backtest")

        benchmark_returns = None
        if len(benchmark_prices) > 1:
            bp = np.array(benchmark_prices)
            benchmark_returns = np.diff(bp) / bp[:-1]

        report = compute_metrics(
            equity_curve=portfolio.equity_curve,
            trade_log=portfolio.get_trade_log(),
            initial_capital=self.initial_capital,
            benchmark_returns=benchmark_returns,
        )

        logger.info("Backtest complete:\n%s", report.summary())
        return report

    def _check_stops(
        self,
        portfolio: Portfolio,
        prices: dict[str, float],
        highs: dict[str, float],
        lows: dict[str, float],
        timestamp: str,
        entry_indices: dict[str, int],
        current_idx: int,
    ) -> None:
        """Check stop-loss, take-profit, and time-stop for all open positions."""
        to_close: list[tuple[str, float, str]] = []

        for sym, pos in list(portfolio.open_positions.items()):
            price = prices.get(sym, pos.entry_price)
            high = highs.get(sym, price)
            low = lows.get(sym, price)
            entry_idx = entry_indices.get(sym, current_idx)
            candles_held = current_idx - entry_idx

            if self.stop_loss_pct > 0:
                if pos.side == "LONG" and low <= pos.entry_price * (1 - self.stop_loss_pct):
                    stop_price = pos.entry_price * (1 - self.stop_loss_pct)
                    to_close.append((sym, stop_price, "stop_loss"))
                    continue
                elif pos.side == "SHORT" and high >= pos.entry_price * (1 + self.stop_loss_pct):
                    stop_price = pos.entry_price * (1 + self.stop_loss_pct)
                    to_close.append((sym, stop_price, "stop_loss"))
                    continue

            if self.take_profit_pct > 0:
                if pos.side == "LONG" and high >= pos.entry_price * (1 + self.take_profit_pct):
                    tp_price = pos.entry_price * (1 + self.take_profit_pct)
                    to_close.append((sym, tp_price, "take_profit"))
                    continue
                elif pos.side == "SHORT" and low <= pos.entry_price * (1 - self.take_profit_pct):
                    tp_price = pos.entry_price * (1 - self.take_profit_pct)
                    to_close.append((sym, tp_price, "take_profit"))
                    continue

            if self.max_hold_candles > 0 and candles_held >= self.max_hold_candles:
                to_close.append((sym, price, "time_stop"))

        for sym, exit_price, reason in to_close:
            portfolio.close_position(sym, exit_price, timestamp, self.costs, reason)
            entry_indices.pop(sym, None)

    def _process_signals(
        self,
        portfolio: Portfolio,
        signals: list[TradeSignal],
        prices: dict[str, float],
        timestamp: str,
        entry_indices: dict[str, int],
        current_idx: int,
    ) -> None:
        """Process signals: close reversed positions and open new ones."""
        for signal in signals:
            sym = signal.symbol
            price = prices.get(sym)
            if price is None:
                continue

            if sym in portfolio.open_positions:
                pos = portfolio.open_positions[sym]
                should_close = (
                    (pos.side == "LONG" and signal.action == "SHORT") or
                    (pos.side == "SHORT" and signal.action == "LONG")
                )
                if should_close:
                    portfolio.close_position(sym, price, timestamp, self.costs, "reversal")
                    entry_indices.pop(sym, None)
                else:
                    continue

            if signal.action in ("LONG", "SHORT") and portfolio.can_open():
                pos = portfolio.open_position(
                    sym, signal.action, price, timestamp, self.costs,
                )
                if pos is not None:
                    entry_indices[sym] = current_idx
