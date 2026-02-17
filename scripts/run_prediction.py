#!/usr/bin/env python3
"""
Hourly prediction entrypoint -- the cron target.

Usage:
    python scripts/run_prediction.py [--config config/settings.yaml]

Designed to be called from crontab:
    0 * * * * cd /path/to/pa_bot && python scripts/run_prediction.py >> logs/cron.log 2>&1
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_pipeline


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hourly crypto prediction pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml"),
        help="Path to settings.yaml config file",
    )
    parser.add_argument("--db", type=str, default=None, help="Override SQLite DB path")
    parser.add_argument("--model", type=str, default=None, help="Override model checkpoint path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    start = datetime.now(timezone.utc)
    logging.info("=" * 60)
    logging.info("PA Bot prediction run starting at %s", start.isoformat())
    logging.info("=" * 60)

    config = load_config(args.config)

    predictions = asyncio.run(
        run_pipeline(config, db_path=args.db, model_path=args.model)
    )

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logging.info(
        "Run complete: %d predictions in %.1f seconds", len(predictions), elapsed
    )

    # Print top 5 to stdout for quick cron log review
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
