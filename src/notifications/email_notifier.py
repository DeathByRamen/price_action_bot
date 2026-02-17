"""
Email (SMTP) notifier.

Sends prediction alerts via email using aiosmtplib.
Falls back to synchronous smtplib if aiosmtplib is not installed.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from .dispatcher import Notifier

logger = logging.getLogger(__name__)


class EmailNotifier(Notifier):
    """Send prediction alerts via SMTP email."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: list[str],
        use_tls: bool = True,
        subject_prefix: str = "[PA Bot]",
    ):
        self._host = smtp_host
        self._port = smtp_port
        self._username = username
        self._password = password
        self._from = from_addr
        self._to = to_addrs
        self._use_tls = use_tls
        self._subject_prefix = subject_prefix

    @property
    def name(self) -> str:
        return "Email"

    async def send(self, message: str) -> bool:
        """Send email synchronously (SMTP is blocking; wrapped for Notifier API)."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{self._subject_prefix} Crypto Prediction Alert"
            msg["From"] = self._from
            msg["To"] = ", ".join(self._to)

            # Plain text body
            msg.attach(MIMEText(message, "plain"))

            # HTML body (wrap in <pre> for monospace table)
            html_body = f"<html><body><pre>{message}</pre></body></html>"
            msg.attach(MIMEText(html_body, "html"))

            if self._use_tls:
                server = smtplib.SMTP(self._host, self._port)
                server.ehlo()
                server.starttls()
            else:
                server = smtplib.SMTP(self._host, self._port)

            server.ehlo()
            server.login(self._username, self._password)
            server.sendmail(self._from, self._to, msg.as_string())
            server.quit()
            return True

        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return False
