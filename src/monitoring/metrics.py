"""
Prometheus metrics exporter for PA Bot monitoring.

Exposes prediction performance, API health, data freshness,
and system resource metrics via an HTTP endpoint.

Run as a background thread alongside the main prediction loop.
"""

from __future__ import annotations

import logging
import time
from threading import Thread
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        start_http_server,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.info("prometheus_client not installed — metrics disabled")


if HAS_PROMETHEUS:
    PREDICTIONS_TOTAL = Counter(
        "pabot_predictions_total",
        "Total predictions generated",
        ["direction", "interval"],
    )
    PREDICTION_LATENCY = Histogram(
        "pabot_prediction_latency_seconds",
        "Time to generate predictions",
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
    )
    PREDICTION_ACCURACY = Gauge(
        "pabot_prediction_accuracy",
        "Rolling prediction accuracy",
        ["interval"],
    )
    API_REQUESTS = Counter(
        "pabot_api_requests_total",
        "Total API requests",
        ["api", "status"],
    )
    DATA_FRESHNESS = Gauge(
        "pabot_data_freshness_seconds",
        "Age of latest data in seconds",
        ["source"],
    )
    MODEL_SHARPE = Gauge(
        "pabot_model_sharpe_ratio",
        "Model Sharpe ratio from backtest",
        ["model_name"],
    )
    ACTIVE_POSITIONS = Gauge(
        "pabot_active_positions",
        "Number of active positions",
    )
    PORTFOLIO_EQUITY = Gauge(
        "pabot_portfolio_equity",
        "Current portfolio equity",
    )
    DB_SIZE_BYTES = Gauge(
        "pabot_database_size_bytes",
        "Database file size in bytes",
    )


class MetricsServer:
    """
    Starts a Prometheus metrics HTTP server on a background thread.

    Usage:
        server = MetricsServer(port=8000)
        server.start()
        # ... record metrics ...
        server.stop()
    """

    def __init__(self, port: int = 8000):
        self.port = port
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        """Start the metrics server in a background thread."""
        if not HAS_PROMETHEUS:
            logger.warning("prometheus_client not installed — skipping metrics server")
            return

        self._thread = Thread(
            target=self._run,
            daemon=True,
            name="metrics-server",
        )
        self._thread.start()
        logger.info("Prometheus metrics server started on port %d", self.port)

    def _run(self) -> None:
        start_http_server(self.port)
        while True:
            time.sleep(3600)

    def stop(self) -> None:
        """Stop is a no-op since the thread is daemonic."""
        pass


def record_prediction(direction: str, interval: str = "60") -> None:
    """Record a prediction event."""
    if HAS_PROMETHEUS:
        PREDICTIONS_TOTAL.labels(direction=direction, interval=interval).inc()


def record_prediction_latency(seconds: float) -> None:
    """Record prediction generation latency."""
    if HAS_PROMETHEUS:
        PREDICTION_LATENCY.observe(seconds)


def record_accuracy(accuracy: float, interval: str = "60") -> None:
    """Update rolling accuracy gauge."""
    if HAS_PROMETHEUS:
        PREDICTION_ACCURACY.labels(interval=interval).set(accuracy)


def record_api_request(api: str, status: str = "success") -> None:
    """Record an API request."""
    if HAS_PROMETHEUS:
        API_REQUESTS.labels(api=api, status=status).inc()


def record_data_freshness(source: str, age_seconds: float) -> None:
    """Update data freshness gauge."""
    if HAS_PROMETHEUS:
        DATA_FRESHNESS.labels(source=source).set(age_seconds)


def record_model_sharpe(model_name: str, sharpe: float) -> None:
    """Update model Sharpe ratio gauge."""
    if HAS_PROMETHEUS:
        MODEL_SHARPE.labels(model_name=model_name).set(sharpe)


def record_db_size(size_bytes: int) -> None:
    """Update database size gauge."""
    if HAS_PROMETHEUS:
        DB_SIZE_BYTES.set(size_bytes)
