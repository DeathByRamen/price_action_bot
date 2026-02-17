"""
Main pipeline orchestrator — supports single and multi-timeframe modes.

Single-timeframe (default):
  Runs predictions at one interval (e.g. 1h).

Multi-timeframe:
  Runs predictions at two intervals (e.g. 1h + 15m), combines them
  via the ensemble module, and dispatches the combined signals.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.api.bitunix_client import BitunixClient
from src.data.collector import DataCollector
from src.data.storage import Storage
from src.features.indicators import MAX_WARMUP_PERIODS
from src.model.predictor import Prediction, Predictor
from src.model.ensemble import (
    MultiTimeframePrediction,
    combine_timeframes,
    compute_adaptive_weights,
    format_multi_timeframe_message,
)
from src.notifications.dispatcher import Dispatcher, Notifier
from src.notifications.discord_notifier import DiscordNotifier
from src.notifications.telegram_notifier import TelegramNotifier
from src.notifications.email_notifier import EmailNotifier

logger = logging.getLogger(__name__)


def build_dispatcher(config: Dict) -> Dispatcher:
    """Create a Dispatcher with notifiers enabled in the config."""
    dispatcher = Dispatcher()
    notif_cfg = config.get("notifications", {})

    # Discord — config or env var
    discord_cfg = notif_cfg.get("discord", {})
    discord_url = discord_cfg.get("webhook_url") or os.getenv("DISCORD_WEBHOOK_URL", "")
    if discord_cfg.get("enabled") and discord_url:
        dispatcher.register(DiscordNotifier(webhook_url=discord_url))

    # Telegram — config or env vars
    telegram_cfg = notif_cfg.get("telegram", {})
    tg_token = telegram_cfg.get("bot_token") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat = telegram_cfg.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID", "")
    if telegram_cfg.get("enabled") and tg_token and tg_chat:
        dispatcher.register(TelegramNotifier(bot_token=tg_token, chat_id=tg_chat))

    # Email — config or env vars
    email_cfg = notif_cfg.get("email", {})
    smtp_password = email_cfg.get("password") or os.getenv("SMTP_PASSWORD", "")
    smtp_username = email_cfg.get("username") or os.getenv("SMTP_USERNAME", "")
    smtp_host = email_cfg.get("smtp_host") or os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = email_cfg.get("smtp_port") or int(os.getenv("SMTP_PORT", "587"))
    from_addr = email_cfg.get("from_addr") or os.getenv("EMAIL_FROM", "")
    to_addrs = email_cfg.get("to_addrs") or os.getenv("EMAIL_TO", "").split(",")

    if email_cfg.get("enabled") and smtp_host and smtp_password:
        dispatcher.register(
            EmailNotifier(
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                username=smtp_username,
                password=smtp_password,
                from_addr=from_addr,
                to_addrs=[a.strip() for a in to_addrs if a.strip()],
                use_tls=email_cfg.get("use_tls", True),
            )
        )

    return dispatcher


async def run_pipeline(
    config: Dict,
    db_path: Optional[str] = None,
    model_path: Optional[str] = None,
    interval: Optional[str] = None,
) -> List[Prediction]:
    """
    Execute the prediction pipeline for a single timeframe.

    Parameters
    ----------
    config : dict
        Loaded settings.yaml contents.
    db_path : str | None
        Override for SQLite database path.
    model_path : str | None
        Override for model checkpoint path.
    interval : str | None
        Candle interval override (e.g. "15", "60").  Defaults to config value.

    Returns
    -------
    List of ranked Prediction objects.
    """
    model_cfg = config.get("model", {})
    pipeline_cfg = config.get("pipeline", {})
    interval = interval or pipeline_cfg.get("interval", "60")
    window_size = model_cfg.get("window_size", 168)
    hidden_dim = model_cfg.get("hidden_dim", 128)
    top_n = pipeline_cfg.get("top_n_signals", 10)
    lookback_candles = pipeline_cfg.get("lookback_candles", 10)
    candle_limit = window_size + MAX_WARMUP_PERIODS + 10

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
        symbols = await collector.discover_tradeable_symbols()
        logger.info("Pipeline [%sm]: processing %d symbols", interval, len(symbols))

        # 3. Fetch latest candles (gap-fill)
        new_candles = await collector.fetch_latest_candles(
            symbols, interval=interval, lookback=lookback_candles
        )
        logger.info("Fetched %d new %sm candles", new_candles, interval)

        # 4. Load candles for all symbols, then batch-predict
        symbol_latest_ts: dict[str, str] = {}
        symbol_dfs: dict[str, "pd.DataFrame"] = {}
        for symbol in symbols:
            try:
                df = await storage.get_candles(
                    symbol, limit=candle_limit, interval=interval
                )
                if df.empty or len(df) < window_size + MAX_WARMUP_PERIODS:
                    continue
                symbol_latest_ts[symbol] = str(df["ts"].iloc[-1])
                symbol_dfs[symbol] = df
            except Exception as exc:
                logger.warning("Failed to load candles for %s: %s", symbol, exc)

        # Batch inference: single forward pass for all symbols
        try:
            all_predictions = predictor.predict_batch(symbol_dfs)
        except Exception as exc:
            logger.error("Batch prediction failed: %s — falling back to per-symbol", exc)
            for symbol, df in symbol_dfs.items():
                try:
                    pred = predictor.predict_symbol(df, symbol)
                    if pred is not None:
                        all_predictions.append(pred)
                except Exception as inner_exc:
                    logger.warning("Prediction failed for %s: %s", symbol, inner_exc)

        logger.info("Generated %d predictions [%sm]", len(all_predictions), interval)

        # 5. Rank by signal strength
        ranked = predictor.rank_predictions(all_predictions)

        # 6. Dispatch notifications (only in single-timeframe mode)
        if not model_path or "multi" not in str(model_path):
            await dispatcher.dispatch(ranked, top_n=top_n)

        # 7. Log predictions to database
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
            await storage.insert_predictions(pred_rows, interval=interval)
            logger.info("Logged %d predictions to database [%sm]", len(pred_rows), interval)

    return ranked


async def run_multi_timeframe_pipeline(
    config: Dict,
    db_path: Optional[str] = None,
) -> List[MultiTimeframePrediction]:
    """
    Execute the multi-timeframe prediction pipeline.

    Runs predictions at two intervals (primary + secondary), combines
    them via the ensemble module, and dispatches the combined signals.

    Returns
    -------
    List of MultiTimeframePrediction, sorted by combined score.
    """
    tf_cfg = config.get("timeframes", {})
    primary_cfg = tf_cfg.get("primary", {})
    secondary_cfg = tf_cfg.get("secondary", {})
    top_n = config.get("pipeline", {}).get("top_n_signals", 10)

    primary_interval = primary_cfg.get("interval", "60")
    secondary_interval = secondary_cfg.get("interval", "15")
    default_primary_weight = primary_cfg.get("weight", 0.6)
    default_secondary_weight = secondary_cfg.get("weight", 0.4)
    primary_window = primary_cfg.get("window_size", 168)
    secondary_window = secondary_cfg.get("window_size", 672)
    primary_checkpoint = primary_cfg.get("checkpoint", "data/models/model_final_60.pt")
    secondary_checkpoint = secondary_cfg.get("checkpoint", "data/models/model_final_15.pt")

    # Build config overrides for each timeframe
    primary_config = {**config, "model": {**config.get("model", {}), "window_size": primary_window}}
    secondary_config = {**config, "model": {**config.get("model", {}), "window_size": secondary_window}}

    # Compute adaptive weights from recent per-interval accuracy
    async with Storage(db_path) as storage:
        pri_acc = await storage.get_recent_interval_accuracy(primary_interval, days=7)
        sec_acc = await storage.get_recent_interval_accuracy(secondary_interval, days=7)

    primary_weight, secondary_weight = compute_adaptive_weights(
        primary_accuracy=pri_acc,
        secondary_accuracy=sec_acc,
        default_primary=default_primary_weight,
        default_secondary=default_secondary_weight,
    )

    logger.info(
        "Multi-timeframe: %sm (weight=%.3f) + %sm (weight=%.3f)",
        primary_interval, primary_weight, secondary_interval, secondary_weight,
    )

    # Run primary timeframe
    logger.info("Running primary timeframe (%sm)...", primary_interval)
    primary_preds = await run_pipeline(
        primary_config,
        db_path=db_path,
        model_path=primary_checkpoint,
        interval=primary_interval,
    )

    # Run secondary timeframe
    logger.info("Running secondary timeframe (%sm)...", secondary_interval)
    secondary_preds = await run_pipeline(
        secondary_config,
        db_path=db_path,
        model_path=secondary_checkpoint,
        interval=secondary_interval,
    )

    # Combine via ensemble
    combined = combine_timeframes(
        primary_preds, secondary_preds,
        primary_weight=primary_weight,
        secondary_weight=secondary_weight,
    )

    # Dispatch combined notifications
    dispatcher = build_dispatcher(config)
    if combined and dispatcher._channels:
        message = format_multi_timeframe_message(
            combined,
            top_n=top_n,
            primary_label=f"{primary_interval}m",
            secondary_label=f"{secondary_interval}m",
        )
        for channel in dispatcher._channels:
            try:
                await channel.send(message)
            except Exception as exc:
                logger.error("Failed to send multi-TF alert via %s: %s", channel.name, exc)

    return combined
