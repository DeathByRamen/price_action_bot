"""
Telegram bot notifier.

Uses the Telegram Bot API sendMessage endpoint to push alerts.
"""

from __future__ import annotations

import logging

import aiohttp

from .dispatcher import Notifier

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"
MAX_TELEGRAM_MSG_LEN = 4096


class TelegramNotifier(Notifier):
    """Send messages to a Telegram chat via Bot API."""

    def __init__(self, bot_token: str, chat_id: str):
        self._bot_token = bot_token
        self._chat_id = chat_id

    @property
    def name(self) -> str:
        return "Telegram"

    async def send(self, message: str) -> bool:
        if len(message) > MAX_TELEGRAM_MSG_LEN:
            message = message[: MAX_TELEGRAM_MSG_LEN - 20] + "\n... (truncated)"

        url = f"{TELEGRAM_API}/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()
                    if data.get("ok"):
                        return True
                    logger.warning("Telegram API error: %s", data.get("description"))
                    return False
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False
