#!/usr/bin/env python3
"""
Backtest CLI for PA Bot.

Runs the backtesting engine on historical data from the database.

Usage:
    python scripts/run_backtest.py --interval 60 --days 90 --capital 10000
    python scripts/run_backtest.py --walk-forward --folds 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd

from src.backtesting.costs import TransactionCosts
from src.backtesting.engine import Backtester
from src.backtesting.metrics import PerformanceReport
from src.backtesting.signals import PredictorSignalGenerator
from src.data.storage import Storage
from src.features.indicators import MAX_WARMUP_PERIODS
from src.model.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def load_data(
    storage: Storage,
    interval: str = "60",
    days: int = 90,
    min_candles: int = 500,
) -> dict[str, pd.DataFrame]:
    """Load historical OHLCV data from the database."""
    symbols_df = await storage.db.execute_fetchall(
        "SELECT DISTINCT symbol FROM ohlcv WHERE interval = ?",
        (interval,),
    )
    symbols = [row[0] for row in symbols_df]

    symbol_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = await storage.get_ohlcv(sym, interval=interval, limit=days * 24 * 4)
        if df is not None and len(df) >= min_candles:
            df = df.sort_values("ts").reset_index(drop=True)
            symbol_data[sym] = df

    logger.info(
        "Loaded %d symbols with >= %d candles (interval=%s, days=%d)",
        len(symbol_data), min_candles, interval, days,
    )
    return symbol_data


def run_single_backtest(
    symbol_data: dict[str, pd.DataFrame],
    model_path: str,
    interval: str,
    capital: float,
    max_positions: int,
    stop_loss: float,
    take_profit: float,
    max_hold: int,
) -> PerformanceReport:
    """Run a single backtest pass."""
    from src.features.indicators import get_feature_columns

    predictor = Predictor(
        model_path=model_path,
        num_features=len(get_feature_columns()),
        window_size=168,
        device="cpu",
    )

    signal_gen = PredictorSignalGenerator(
        predictor=predictor,
        min_conviction=0.3,
        min_prob=0.45,
        min_magnitude=0.002,
    )

    costs = TransactionCosts(
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_bps=5.0,
    )

    backtester = Backtester(
        signal_generator=signal_gen,
        costs=costs,
        initial_capital=capital,
        max_positions=max_positions,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        max_hold_candles=max_hold,
    )

    start_idx = MAX_WARMUP_PERIODS + 168
    return backtester.run(symbol_data, start_idx=start_idx)


def run_walk_forward(
    symbol_data: dict[str, pd.DataFrame],
    model_path: str,
    interval: str,
    capital: float,
    n_folds: int = 5,
) -> None:
    """Run walk-forward backtest with multiple folds."""
    from src.features.indicators import get_feature_columns

    ref_sym = next(iter(symbol_data))
    total_rows = len(symbol_data[ref_sym])
    warmup = MAX_WARMUP_PERIODS + 168
    usable = total_rows - warmup
    fold_size = usable // (n_folds + 1)

    logger.info(
        "Walk-forward: %d total rows, %d usable, %d per fold, %d folds",
        total_rows, usable, fold_size, n_folds,
    )

    fold_reports: list[PerformanceReport] = []

    for fold in range(n_folds):
        train_end = warmup + fold_size * (fold + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, total_rows)

        if test_end <= test_start:
            break

        logger.info(
            "Fold %d/%d: test window [%d, %d) (%d candles)",
            fold + 1, n_folds, test_start, test_end, test_end - test_start,
        )

        predictor = Predictor(
            model_path=model_path,
            num_features=len(get_feature_columns()),
            window_size=168,
            device="cpu",
        )
        signal_gen = PredictorSignalGenerator(predictor=predictor)
        costs = TransactionCosts()

        bt = Backtester(
            signal_generator=signal_gen,
            costs=costs,
            initial_capital=capital,
        )
        report = bt.run(symbol_data, start_idx=test_start, end_idx=test_end)
        fold_reports.append(report)

        logger.info(
            "Fold %d: Sharpe=%.3f, Return=%.2f%%, DD=%.2f%%",
            fold + 1, report.sharpe_ratio,
            report.total_return_pct, report.max_drawdown_pct,
        )

    if fold_reports:
        avg_sharpe = np.mean([r.sharpe_ratio for r in fold_reports])
        avg_return = np.mean([r.total_return_pct for r in fold_reports])
        avg_dd = np.mean([r.max_drawdown_pct for r in fold_reports])
        avg_wr = np.mean([r.win_rate for r in fold_reports])

        logger.info(
            "\n=== Walk-Forward Summary ===\n"
            "  Avg Sharpe:     %.3f\n"
            "  Avg Return:     %.2f%%\n"
            "  Avg Max DD:     %.2f%%\n"
            "  Avg Win Rate:   %.1f%%\n"
            "  Folds:          %d",
            avg_sharpe, avg_return, avg_dd, avg_wr, len(fold_reports),
        )


async def main():
    parser = argparse.ArgumentParser(description="PA Bot Backtester")
    parser.add_argument("--interval", default="60", help="Candle interval")
    parser.add_argument("--days", type=int, default=90, help="Days of data")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--stop-loss", type=float, default=0.02, help="Stop-loss pct")
    parser.add_argument("--take-profit", type=float, default=0.03, help="Take-profit pct")
    parser.add_argument("--max-hold", type=int, default=24, help="Max hold candles")
    parser.add_argument("--model-path", default=None, help="Model checkpoint path")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward mode")
    parser.add_argument("--folds", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--db", default=None, help="Override DB path")
    args = parser.parse_args()

    model_path = args.model_path or f"data/models/model_final_{args.interval}.pt"

    async with Storage(args.db) as storage:
        symbol_data = await load_data(storage, args.interval, args.days)

    if not symbol_data:
        logger.error("No data loaded — cannot backtest")
        return

    if args.walk_forward:
        run_walk_forward(symbol_data, model_path, args.interval, args.capital, args.folds)
    else:
        report = run_single_backtest(
            symbol_data, model_path, args.interval, args.capital,
            args.max_positions, args.stop_loss, args.take_profit, args.max_hold,
        )
        print(report.summary())


if __name__ == "__main__":
    asyncio.run(main())
