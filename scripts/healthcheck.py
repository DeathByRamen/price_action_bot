#!/usr/bin/env python3
"""
Health check script for PA Bot production deployment.

Checks:
  - Last prediction timestamp < 2 hours ago
  - Last retrain < 26 hours ago
  - Last order book snapshot < 20 minutes ago
  - Database file exists and is < threshold size
  - Disk space available

Exits with code 0 on healthy, 1 on any failure.
Designed to run every 30 minutes via cron:
  */30 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/healthcheck.py 2>&1 | mail -s "[PA Bot] Health Alert" mcgillfinance@gmail.com

Or use the --notify flag to send email alerts on failure.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.data.storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_PREDICTION_AGE_HOURS = 2
MAX_RETRAIN_AGE_HOURS = 26
MAX_ORDERBOOK_AGE_MINUTES = 20
MAX_DB_SIZE_GB = 5.0
MIN_DISK_FREE_GB = 2.0


async def run_checks(db_path: str | None = None) -> list[str]:
    """Run all health checks and return list of failure messages."""
    failures: list[str] = []
    now = datetime.now(timezone.utc)

    async with Storage(db_path) as storage:
        db = storage._db_path

        if not os.path.exists(db):
            failures.append(f"Database not found at {db}")
            return failures

        db_size_gb = os.path.getsize(db) / (1024 ** 3)
        if db_size_gb > MAX_DB_SIZE_GB:
            failures.append(f"Database size {db_size_gb:.2f} GB exceeds {MAX_DB_SIZE_GB} GB limit")

        try:
            latest_pred = await storage.db.execute_fetchall(
                "SELECT MAX(created_at) FROM predictions"
            )
            if latest_pred and latest_pred[0][0]:
                pred_ts = datetime.fromisoformat(latest_pred[0][0].replace("Z", "+00:00"))
                age = now - pred_ts
                if age > timedelta(hours=MAX_PREDICTION_AGE_HOURS):
                    failures.append(
                        f"Last prediction is {age.total_seconds()/3600:.1f}h old "
                        f"(max {MAX_PREDICTION_AGE_HOURS}h)"
                    )
            else:
                failures.append("No predictions found in database")
        except Exception as e:
            failures.append(f"Error checking predictions: {e}")

        try:
            latest_retrain = await storage.db.execute_fetchall(
                "SELECT MAX(run_date) FROM accuracy_log"
            )
            if latest_retrain and latest_retrain[0][0]:
                retrain_date = datetime.strptime(
                    latest_retrain[0][0], "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc)
                age = now - retrain_date
                if age > timedelta(hours=MAX_RETRAIN_AGE_HOURS):
                    failures.append(
                        f"Last retrain is {age.total_seconds()/3600:.1f}h old "
                        f"(max {MAX_RETRAIN_AGE_HOURS}h)"
                    )
        except Exception as e:
            failures.append(f"Error checking retrain: {e}")

        try:
            latest_ob = await storage.db.execute_fetchall(
                "SELECT MAX(ts) FROM order_book_snapshots"
            )
            if latest_ob and latest_ob[0][0]:
                ob_ts = datetime.fromisoformat(latest_ob[0][0].replace("Z", "+00:00"))
                age = now - ob_ts
                if age > timedelta(minutes=MAX_ORDERBOOK_AGE_MINUTES):
                    failures.append(
                        f"Last order book snapshot is {age.total_seconds()/60:.0f}min old "
                        f"(max {MAX_ORDERBOOK_AGE_MINUTES}min)"
                    )
        except Exception:
            pass  # Table may not exist yet

    disk_usage = shutil.disk_usage("/")
    free_gb = disk_usage.free / (1024 ** 3)
    if free_gb < MIN_DISK_FREE_GB:
        failures.append(f"Disk space low: {free_gb:.1f} GB free (min {MIN_DISK_FREE_GB} GB)")

    return failures


def main():
    parser = argparse.ArgumentParser(description="PA Bot health check")
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--notify", action="store_true", help="Send email on failure")
    args = parser.parse_args()

    failures = asyncio.run(run_checks(args.db))

    if not failures:
        logger.info("All health checks passed")
        sys.exit(0)

    for f in failures:
        logger.error("HEALTH CHECK FAILED: %s", f)

    if args.notify:
        try:
            _send_alert(failures)
        except Exception as e:
            logger.error("Failed to send alert email: %s", e)

    sys.exit(1)


def _send_alert(failures: list[str]) -> None:
    """Send email alert for health check failures."""
    import smtplib
    from email.mime.text import MIMEText

    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "")
    password = os.getenv("SMTP_PASSWORD", "")
    to_addr = os.getenv("EMAIL_TO", "mcgillfinance@gmail.com")

    if not password:
        logger.warning("SMTP_PASSWORD not set, cannot send alert")
        return

    body = "PA Bot Health Check Failures:\n\n"
    for f in failures:
        body += f"  - {f}\n"
    body += f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}"

    msg = MIMEText(body)
    msg["Subject"] = f"[PA Bot] Health Alert - {len(failures)} issue(s)"
    msg["From"] = username
    msg["To"] = to_addr

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)

    logger.info("Alert email sent to %s", to_addr)


if __name__ == "__main__":
    main()
