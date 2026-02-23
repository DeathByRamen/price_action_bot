#!/usr/bin/env python3
"""
Prediction entrypoint -- the cron target.

Single-timeframe (default):
    0 * * * * cd /path/to/pa_bot && python scripts/run_prediction.py

Multi-timeframe (1h + 15m ensemble):
    */15 * * * * cd /path/to/pa_bot && python scripts/run_prediction.py --multi-timeframe

    The --multi-timeframe flag runs both the primary (1h) and secondary (15m)
    models and combines their signals via the ensemble module.
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_multi_timeframe_pipeline, run_pipeline


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run crypto prediction pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml"),
        help="Path to settings.yaml config file",
    )
    parser.add_argument("--db", type=str, default=None, help="Override SQLite DB path")
    parser.add_argument("--model", type=str, default=None, help="Override model checkpoint path")
    parser.add_argument(
        "--interval",
        type=str,
        default=None,
        help="Candle interval override for single-timeframe mode (e.g. 15, 60)",
    )
    parser.add_argument(
        "--multi-timeframe",
        action="store_true",
        help="Run multi-timeframe ensemble (primary + secondary from config)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    start = datetime.now(timezone.utc)
    logging.info("=" * 60)
    mode_label = "multi-timeframe" if args.multi_timeframe else "single-timeframe"
    logging.info("PA Bot prediction run (%s) starting at %s", mode_label, start.isoformat())
    logging.info("=" * 60)

    config = load_config(args.config)

    if args.multi_timeframe:
        combined = asyncio.run(
            run_multi_timeframe_pipeline(config, db_path=args.db)
        )
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logging.info("Run complete: %d combined predictions in %.1f seconds", len(combined), elapsed)

        for i, p in enumerate(combined[:5], 1):
            logging.info(
                "  #%d %s %s (1h=%s, 15m=%s, score=%.4f)",
                i,
                p.symbol,
                p.agreement_label,
                p.primary.direction,
                p.secondary.direction,
                p.combined_score,
            )
    else:
        predictions = asyncio.run(
            run_pipeline(
                config,
                db_path=args.db,
                model_path=args.model,
                interval=args.interval,
            )
        )
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logging.info("Run complete: %d predictions in %.1f seconds", len(predictions), elapsed)

        for i, p in enumerate(predictions[:5], 1):
            logging.info(
                "  #%d %s %s (%.1f%% conf, %+.2f%% mag, score=%.4f)",
                i,
                p.symbol,
                p.direction,
                max(p.prob_up, p.prob_flat, p.prob_down) * 100,
                p.magnitude * 100,
                p.signal_score,
            )


if __name__ == "__main__":
    main()
