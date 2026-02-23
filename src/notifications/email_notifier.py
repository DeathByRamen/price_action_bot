"""
Email (SMTP) notifier.

Sends prediction alerts via email using smtplib with styled HTML.
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

    async def send(self, message: str, subject: Optional[str] = None) -> bool:
        """Send email with styled HTML body."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject or f"{self._subject_prefix} Crypto Prediction Alert"
            msg["From"] = self._from
            msg["To"] = ", ".join(self._to)

            msg.attach(MIMEText(message, "plain"))

            html_body = self._build_html(message)
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

    @staticmethod
    def _build_html(plain_text: str) -> str:
        """Convert the plain-text alert into styled HTML."""
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    margin: 0;
    padding: 20px;
    color: #333;
  }}
  .container {{
    max-width: 700px;
    margin: 0 auto;
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    overflow: hidden;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    color: #ffffff;
    padding: 20px 24px;
  }}
  .header h1 {{
    margin: 0 0 4px 0;
    font-size: 18px;
    font-weight: 600;
  }}
  .header .subtitle {{
    font-size: 13px;
    color: #a0aec0;
  }}
  .content {{
    padding: 20px 24px;
  }}
  pre {{
    background: #f8f9fa;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 16px;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 12.5px;
    line-height: 1.6;
    overflow-x: auto;
    white-space: pre;
    color: #2d3748;
  }}
  .legend {{
    font-size: 12px;
    color: #718096;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #e2e8f0;
  }}
  .footer {{
    text-align: center;
    font-size: 11px;
    color: #a0aec0;
    padding: 12px 24px;
    border-top: 1px solid #f0f0f0;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="content">
    <pre>{plain_text}</pre>
  </div>
  <div class="footer">
    PA Bot &mdash; Automated Crypto Prediction System
  </div>
</div>
</body>
</html>"""
