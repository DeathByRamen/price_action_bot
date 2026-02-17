"""
Pluggable notification dispatcher.

Reads enabled channels from config and fans out messages to each.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.model.predictor import Prediction

logger = logging.getLogger(__name__)


class Notifier(ABC):
    """Base class for all notification channels."""

    @abstractmethod
    async def send(self, message: str) -> bool:
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

    async def dispatch(self, predictions: List[Prediction], top_n: int = 10) -> None:
        """Format predictions into an alert message and send to all channels."""
        if not predictions:
            logger.info("No predictions to dispatch")
            return

        message = self._format_message(predictions[:top_n])

        for channel in self._channels:
            try:
                ok = await channel.send(message)
                if ok:
                    logger.info("Sent alert via %s", channel.name)
                else:
                    logger.warning("Failed to send via %s", channel.name)
            except Exception as exc:
                logger.error("Error sending via %s: %s", channel.name, exc)

    async def dispatch_raw(self, message: str) -> None:
        """Send a pre-formatted message to all channels."""
        for channel in self._channels:
            try:
                ok = await channel.send(message)
                if ok:
                    logger.info("Sent message via %s", channel.name)
                else:
                    logger.warning("Failed to send via %s", channel.name)
            except Exception as exc:
                logger.error("Error sending via %s: %s", channel.name, exc)

    @staticmethod
    def _format_message(predictions: List[Prediction]) -> str:
        """Build a plain-text / markdown alert from ranked predictions."""
        lines = [
            "**PA Bot Crypto Predictions**",
            f"Top {len(predictions)} signals by conviction:\n",
            "```",
            f"{'#':>3} {'Symbol':<12} {'Dir':>5} {'Prob':>7} {'Mag%':>8} {'Price':>12} {'Score':>7}",
            f"{'---':>3} {'------':<12} {'---':>5} {'----':>7} {'----':>8} {'-----':>12} {'-----':>7}",
        ]

        for i, p in enumerate(predictions, 1):
            max_prob = max(p.prob_up, p.prob_flat, p.prob_down)
            lines.append(
                f"{i:>3} {p.symbol:<12} {p.direction:>5} {max_prob:>6.1%} "
                f"{p.magnitude:>+7.2%} {p.current_price:>12.4f} {p.signal_score:>7.4f}"
            )

        lines.append("```")

        # Add quick legend
        lines.append("\nDir: predicted direction | Prob: confidence | Mag%: predicted move | Score: signal strength")

        return "\n".join(lines)
