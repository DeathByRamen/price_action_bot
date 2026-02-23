"""
Pluggable notification dispatcher.

Reads enabled channels from config and fans out messages to each.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional

from src.model.predictor import Prediction

logger = logging.getLogger(__name__)


class Notifier(ABC):
    """Base class for all notification channels."""

    @abstractmethod
    async def send(self, message: str, subject: Optional[str] = None) -> bool:
        """Send a message. Returns True on success."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class Dispatcher:
    """Routes formatted prediction alerts to all registered notifiers."""

    def __init__(self) -> None:
        self._channels: List[Notifier] = []

    def register(self, notifier: Notifier) -> None:
        self._channels.append(notifier)
        logger.info("Registered notifier: %s", notifier.name)

    async def dispatch(
        self,
        predictions: List[Prediction],
        top_n: int = 10,
        interval: str = "60",
    ) -> None:
        """Format predictions into an alert message and send to all channels."""
        if not predictions:
            logger.info("No predictions to dispatch")
            return

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        message = self._format_message(predictions[:top_n], interval=interval, timestamp=now_utc)
        subject = f"[PA Bot] {interval}m Predictions — {now_utc}"

        for channel in self._channels:
            try:
                ok = await channel.send(message, subject=subject)
                if ok:
                    logger.info("Sent alert via %s", channel.name)
                else:
                    logger.warning("Failed to send via %s", channel.name)
            except Exception as exc:
                logger.error("Error sending via %s: %s", channel.name, exc)

    async def dispatch_raw(self, message: str, subject: Optional[str] = None) -> None:
        """Send a pre-formatted message to all channels."""
        for channel in self._channels:
            try:
                ok = await channel.send(message, subject=subject)
                if ok:
                    logger.info("Sent message via %s", channel.name)
                else:
                    logger.warning("Failed to send via %s", channel.name)
            except Exception as exc:
                logger.error("Error sending via %s: %s", channel.name, exc)

    @staticmethod
    def _format_message(
        predictions: List[Prediction],
        interval: str = "60",
        timestamp: str = "",
    ) -> str:
        """Build a plain-text alert from ranked predictions."""
        header = f"PA Bot — {interval}m Predictions"
        if timestamp:
            header += f"  |  {timestamp}"

        up_count = sum(1 for p in predictions if p.direction == "UP")
        down_count = sum(1 for p in predictions if p.direction == "DOWN")
        flat_count = sum(1 for p in predictions if p.direction == "FLAT")

        lines = [
            header,
            "=" * len(header),
            f"Top {len(predictions)} signals by conviction",
            f"Market bias: {up_count} UP / {down_count} DOWN / {flat_count} FLAT",
            "",
            f"{'#':>3}  {'Symbol':<14} {'Dir':>5}  {'Prob':>6}  {'Mag%':>7}  {'Price':>12}  {'Score':>7}",
            f"{'─'*3}  {'─'*14} {'─'*5}  {'─'*6}  {'─'*7}  {'─'*12}  {'─'*7}",
        ]

        for i, p in enumerate(predictions, 1):
            max_prob = max(p.prob_up, p.prob_flat, p.prob_down)
            dir_icon = "▲" if p.direction == "UP" else "▼" if p.direction == "DOWN" else "─"
            lines.append(
                f"{i:>3}  {p.symbol:<14} {dir_icon} {p.direction:>3}  "
                f"{max_prob:>5.1%}  {p.magnitude:>+6.2%}  "
                f"{p.current_price:>12.4f}  {p.signal_score:>7.4f}"
            )

        lines.append("")
        lines.append("Legend: Dir = predicted direction | Prob = confidence")
        lines.append("        Mag% = predicted move size | Score = signal strength")

        return "\n".join(lines)
