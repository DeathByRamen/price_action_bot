"""
Main hourly pipeline orchestrator.

Each run:
  1. Discover all futures symbols via tickers endpoint
  2. Fetch latest candles for each symbol (incremental gap-fill)
  3. For each symbol with enough data, compute indicators + run inference
  4. Rank predictions by signal strength
  5. Dispatch top signals to notification channels
  6. Log all predictions to the database
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import yaml

from src.api.bitunix_client import BitunixClient
from src.data.collector import DataCollector
from src.data.storage import Storage
from src.model.predictor import Prediction, Predictor
from src.notifications.dispatcher import Dispatcher, Notifier
from src.notifications.discord_notifier import DiscordNotifier
from src.notifications.telegram_notifier import TelegramNotifier
from src.notifications.email_notifier import EmailNotifier

logger = logging.getLogger(__name__)


def build_dispatcher(config: Dict) -> Dispatcher:
    """Create a Dispatcher with notifiers enabled in the config."""
    dispatcher = Dispatcher()
    notif_cfg = config.get("notifications", {})

    # Discord
    discord_cfg = notif_cfg.get("discord", {})
    if discord_cfg.get("enabled") and discord_cfg.get("webhook_url"):
        dispatcher.register(DiscordNotifier(webhook_url=discord_cfg["webhook_url"]))

    # Telegram
    telegram_cfg = notif_cfg.get("telegram", {})
    if telegram_cfg.get("enabled") and telegram_cfg.get("bot_token") and telegram_cfg.get("chat_id"):
        dispatcher.register(
            TelegramNotifier(
                bot_token=telegram_cfg["bot_token"],
                chat_id=telegram_cfg["chat_id"],
            )
        )

    # Email
    email_cfg = notif_cfg.get("email", {})
    if email_cfg.get("enabled") and email_cfg.get("smtp_host"):
        dispatcher.register(
            EmailNotifier(
                smtp_host=email_cfg["smtp_host"],
                smtp_port=email_cfg.get("smtp_port", 587),
                username=email_cfg["username"],
                password=email_cfg["password"],
                from_addr=email_cfg["from_addr"],
                to_addrs=email_cfg["to_addrs"],
                use_tls=email_cfg.get("use_tls", True),
            )
        )

    return dispatcher


async def run_pipeline(
    config: Dict,
    db_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> List[Prediction]:
    """
    Execute the full hourly prediction pipeline.

    Parameters
    ----------
    config : dict
        Loaded settings.yaml contents.
    db_path : str | None
        Override for SQLite database path.
    model_path : str | None
        Override for model checkpoint path.

    Returns
    -------
    List of ranked Prediction objects.
    """
    model_cfg = config.get("model", {})
    pipeline_cfg = config.get("pipeline", {})
    window_size = model_cfg.get("window_size", 168)
    hidden_dim = model_cfg.get("hidden_dim", 128)
    top_n = pipeline_cfg.get("top_n_signals", 10)
    lookback_candles = pipeline_cfg.get("lookback_candles", 10)
    candle_limit = window_size + 100  # enough for indicators + window

    # 1. Set up components
    dispatcher = build_dispatcher(config)
    predictor = Predictor(
        model_path=model_path,
        hidden_dim=hidden_dim,
        window_size=window_size,
    )

    all_predictions: List[Prediction] = []

    async with BitunixClient() as client, Storage(db_path) as storage:
        collector = DataCollector(client, storage)

        # 2. Discover symbols
        symbols = await collector.discover_futures_symbols()
        logger.info("Pipeline: processing %d symbols", len(symbols))

        # 3. Fetch latest candles (gap-fill)
        new_candles = await collector.fetch_latest_candles(
            symbols, interval="60", lookback=lookback_candles
        )
        logger.info("Fetched %d new candles across all symbols", new_candles)

        # 4. For each symbol: load candles -> indicators -> predict
        symbol_latest_ts: dict[str, str] = {}
        for symbol in symbols:
            try:
                df = await storage.get_candles(symbol, limit=candle_limit)
                if df.empty or len(df) < window_size + 50:
                    continue

                # Track the latest candle timestamp for this symbol
                symbol_latest_ts[symbol] = str(df["ts"].iloc[-1])

                pred = predictor.predict_symbol(df, symbol)
                if pred is not None:
                    all_predictions.append(pred)
            except Exception as exc:
                logger.warning("Prediction failed for %s: %s", symbol, exc)

        logger.info("Generated %d predictions", len(all_predictions))

        # 5. Rank by signal strength
        ranked = predictor.rank_predictions(all_predictions)

        # 6. Dispatch notifications
        await dispatcher.dispatch(ranked, top_n=top_n)

        # 7. Log predictions to database with actual candle timestamps
        if ranked:
            pred_rows = [
                (
                    p.symbol,
                    symbol_latest_ts.get(p.symbol, ""),
                    p.direction,
                    p.prob_up,
                    p.prob_flat,
                    p.prob_down,
                    p.magnitude,
                    p.signal_score,
                )
                for p in ranked
            ]
            await storage.insert_predictions(pred_rows)
            logger.info("Logged %d predictions to database", len(pred_rows))

    return ranked
