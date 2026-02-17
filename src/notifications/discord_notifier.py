"""
Discord webhook notifier.

Sends prediction alerts to a Discord channel via webhook URL.
"""

from __future__ import annotations

import logging

import aiohttp

from .dispatcher import Notifier

logger = logging.getLogger(__name__)

MAX_DISCORD_MSG_LEN = 2000


class DiscordNotifier(Notifier):
    """Send messages to a Discord channel via webhook."""

    def __init__(self, webhook_url: str):
        self._webhook_url = webhook_url

    @property
    def name(self) -> str:
        return "Discord"

    async def send(self, message: str) -> bool:
        # Discord has a 2000 char limit; truncate if needed
        if len(message) > MAX_DISCORD_MSG_LEN:
            message = message[: MAX_DISCORD_MSG_LEN - 20] + "\n... (truncated)"

        payload = {"content": message}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status in (200, 204):
                        return True
                    body = await resp.text()
                    logger.warning(
                        "Discord webhook returned %d: %s", resp.status, body[:200]
                    )
                    return False
        except Exception as exc:
            logger.error("Discord send failed: %s", exc)
            return False
