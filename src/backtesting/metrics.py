"""
Financial performance metrics for backtesting.

Computes Sharpe, Sortino, Calmar, drawdown, win rate, profit factor,
and other standard quant metrics from an equity curve and trade log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PerformanceReport:
    """Container for backtest performance metrics."""
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_hours: float = 0.0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_duration_hours: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    initial_capital: float = 0.0
    final_equity: float = 0.0
    benchmark_return_pct: float = 0.0
    information_ratio: float = 0.0

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 55,
            "  BACKTEST PERFORMANCE REPORT",
            "=" * 55,
            f"  Initial Capital:      ${self.initial_capital:>12,.2f}",
            f"  Final Equity:         ${self.final_equity:>12,.2f}",
            f"  Net P&L:              ${self.net_pnl:>12,.2f}",
            f"  Total Return:         {self.total_return_pct:>11.2f}%",
            f"  Annualized Return:    {self.annualized_return_pct:>11.2f}%",
            "",
            f"  Sharpe Ratio:         {self.sharpe_ratio:>11.3f}",
            f"  Sortino Ratio:        {self.sortino_ratio:>11.3f}",
            f"  Calmar Ratio:         {self.calmar_ratio:>11.3f}",
            f"  Max Drawdown:         {self.max_drawdown_pct:>11.2f}%",
            f"  DD Duration:          {self.max_drawdown_duration_hours:>8.0f}  hrs",
            "",
            f"  Total Trades:         {self.total_trades:>11d}",
            f"  Win Rate:             {self.win_rate:>11.1f}%",
            f"  Avg Win:              {self.avg_win_pct:>11.3f}%",
            f"  Avg Loss:             {self.avg_loss_pct:>11.3f}%",
            f"  Profit Factor:        {self.profit_factor:>11.3f}",
            f"  Avg Duration:         {self.avg_trade_duration_hours:>8.1f}  hrs",
            f"  Total Fees:           ${self.total_fees:>12,.2f}",
            "",
            f"  Benchmark Return:     {self.benchmark_return_pct:>11.2f}%",
            f"  Information Ratio:    {self.information_ratio:>11.3f}",
            "=" * 55,
        ]
        return "\n".join(lines)


def compute_metrics(
    equity_curve: list[tuple[str, float]],
    trade_log: list[dict],
    initial_capital: float,
    benchmark_returns: Optional[np.ndarray] = None,
    hours_per_period: float = 1.0,
) -> PerformanceReport:
    """
    Compute comprehensive performance metrics from backtest results.

    Parameters
    ----------
    equity_curve : list of (timestamp, equity) tuples
    trade_log : list of trade dicts from Portfolio.get_trade_log()
    initial_capital : starting capital
    benchmark_returns : array of benchmark period returns (for information ratio)
    hours_per_period : hours per equity curve observation (1.0 for hourly)
    """
    report = PerformanceReport(initial_capital=initial_capital)

    if not equity_curve:
        return report

    equities = np.array([e[1] for e in equity_curve], dtype=np.float64)
    report.final_equity = float(equities[-1])
    report.net_pnl = report.final_equity - initial_capital
    report.total_return_pct = (report.net_pnl / initial_capital) * 100

    periods_per_year = (365.25 * 24) / hours_per_period
    n_periods = len(equities)
    if n_periods > 1:
        total_return = equities[-1] / equities[0]
        years = n_periods / periods_per_year
        if years > 0 and total_return > 0:
            report.annualized_return_pct = (total_return ** (1 / years) - 1) * 100

    returns = np.diff(equities) / equities[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) > 1:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret > 0:
            report.sharpe_ratio = float(
                (mean_ret / std_ret) * math.sqrt(periods_per_year)
            )

        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside, ddof=1)
            if downside_std > 0:
                report.sortino_ratio = float(
                    (mean_ret / downside_std) * math.sqrt(periods_per_year)
                )

    peak = np.maximum.accumulate(equities)
    drawdowns = (equities - peak) / peak
    report.max_drawdown_pct = float(abs(np.min(drawdowns)) * 100)

    if report.max_drawdown_pct > 0:
        report.calmar_ratio = report.annualized_return_pct / report.max_drawdown_pct

    in_drawdown = drawdowns < 0
    if np.any(in_drawdown):
        dd_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
        dd_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]
        if len(dd_starts) > 0:
            if len(dd_ends) == 0 or (len(dd_ends) > 0 and dd_ends[-1] < dd_starts[-1]):
                dd_ends = np.append(dd_ends, len(equities) - 1)
            max_dur = 0
            for s, e in zip(dd_starts, dd_ends):
                dur = (e - s) * hours_per_period
                max_dur = max(max_dur, dur)
            report.max_drawdown_duration_hours = float(max_dur)

    if trade_log:
        report.total_trades = len(trade_log)
        report.total_fees = sum(t.get("costs", 0) for t in trade_log)

        net_pnls = [t.get("net_pnl", 0) for t in trade_log]
        winners = [p for p in net_pnls if p > 0]
        losers = [p for p in net_pnls if p <= 0]

        report.winning_trades = len(winners)
        report.losing_trades = len(losers)
        report.win_rate = (len(winners) / len(trade_log)) * 100 if trade_log else 0.0

        if winners:
            win_notionals = [
                t.get("notional", 1) for t in trade_log if t.get("net_pnl", 0) > 0
            ]
            report.avg_win_pct = float(
                np.mean([w / n * 100 for w, n in zip(winners, win_notionals)])
            )
        if losers:
            loss_notionals = [
                t.get("notional", 1) for t in trade_log if t.get("net_pnl", 0) <= 0
            ]
            report.avg_loss_pct = float(
                np.mean([loss / n * 100 for loss, n in zip(losers, loss_notionals)])
            )

        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        report.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        durations = []
        for t in trade_log:
            et = t.get("entry_time", "")
            xt = t.get("exit_time", "")
            if et and xt:
                try:
                    from datetime import datetime
                    entry_dt = datetime.fromisoformat(et.replace("Z", "+00:00"))
                    exit_dt = datetime.fromisoformat(xt.replace("Z", "+00:00"))
                    dur_h = (exit_dt - entry_dt).total_seconds() / 3600
                    durations.append(dur_h)
                except Exception:
                    pass
        if durations:
            report.avg_trade_duration_hours = float(np.mean(durations))

    if benchmark_returns is not None and len(benchmark_returns) > 0 and len(returns) > 0:
        min_len = min(len(returns), len(benchmark_returns))
        active = returns[:min_len] - benchmark_returns[:min_len]
        if np.std(active) > 0:
            report.information_ratio = float(
                np.mean(active) / np.std(active) * math.sqrt(periods_per_year)
            )
        report.benchmark_return_pct = float(
            (np.prod(1 + benchmark_returns) - 1) * 100
        )

    return report
