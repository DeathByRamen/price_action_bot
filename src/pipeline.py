"""
Main pipeline orchestrator — supports single and multi-timeframe modes.

Single-timeframe (default):
  Runs predictions at one interval (e.g. 1h).

Multi-timeframe:
  Runs predictions at two intervals (e.g. 1h + 15m), combines them
  via the ensemble module, and dispatches the combined signals.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from src.api.bitunix_client import BitunixClient
from src.data.collector import DataCollector
from src.data.storage import Storage
from src.features.derivatives import (
    compute_coinalyze_features,
    compute_cross_asset_features,
    compute_funding_rate_features,
)
from src.features.indicators import MAX_WARMUP_PERIODS
from src.features.orderbook import compute_orderbook_features
from src.model.drift import DriftMonitor
from src.model.ensemble import (
    MultiTimeframePrediction,
    combine_timeframes,
    compute_adaptive_weights,
    format_multi_timeframe_message,
)
from src.model.predictor import Prediction, Predictor
from src.model.regime import REGIME_NAMES, RegimeDetector
from src.notifications.discord_notifier import DiscordNotifier
from src.notifications.dispatcher import Dispatcher
from src.notifications.email_notifier import EmailNotifier
from src.notifications.telegram_notifier import TelegramNotifier
from src.risk.drawdown import DrawdownConfig, DrawdownManager, DrawdownState
from src.risk.rules import EntryExitConfig, EntryExitRules

logger = logging.getLogger(__name__)

DRIFT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "drift_state.json"
)


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

        # 4. Load candles for all symbols, then enrich with OB/derivatives
        symbol_latest_ts: dict[str, str] = {}
        symbol_dfs: dict[str, object] = {}

        btc_df = await storage.get_candles("BTCUSDT", limit=candle_limit, interval=interval)

        for symbol in symbols:
            try:
                df = await storage.get_candles(
                    symbol, limit=candle_limit, interval=interval
                )
                if df.empty or len(df) < window_size + MAX_WARMUP_PERIODS:
                    continue
                symbol_latest_ts[symbol] = str(df["ts"].iloc[-1])

                start_ts = str(df["ts"].iloc[0]) if "ts" in df.columns else None

                try:
                    ob_df = await storage.get_order_book_snapshots(
                        symbol, start_ts=start_ts, limit=candle_limit * 2
                    )
                    if not ob_df.empty:
                        df = compute_orderbook_features(ob_df, df)
                except Exception:
                    pass

                try:
                    oi_df = await storage.get_coinalyze_oi(symbol, start_ts=start_ts, limit=candle_limit)
                    liq_df = await storage.get_coinalyze_liquidations(symbol, start_ts=start_ts, limit=candle_limit)
                    ls_df = await storage.get_coinalyze_long_short(symbol, start_ts=start_ts, limit=candle_limit)
                    if not oi_df.empty or not liq_df.empty or not ls_df.empty:
                        df = compute_coinalyze_features(oi_df, liq_df, ls_df, df)
                except Exception:
                    pass

                try:
                    fr_df = await storage.get_funding_rate_snapshots(
                        symbol, start_ts=start_ts, limit=candle_limit
                    )
                    if not fr_df.empty:
                        df = compute_funding_rate_features(fr_df, df)
                except Exception:
                    pass

                try:
                    if not btc_df.empty and symbol != "BTCUSDT":
                        df = compute_cross_asset_features(btc_df, df)
                except Exception:
                    pass

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

        # 5. Regime detection — adjust signals per market state
        regime_label = "Unknown"
        regime_adjustments: Dict[str, float] = {}
        try:
            detector = RegimeDetector()
            detector.fit(btc_df)
            regime = detector.predict(btc_df)
            regime_label = REGIME_NAMES.get(regime, "Unknown")
            regime_adjustments = detector.get_regime_adjustments(regime)
            logger.info("Detected market regime: %s", regime_label)

            size_mult = regime_adjustments.get("size_multiplier", 1.0)
            conf_boost = regime_adjustments.get("confidence_boost", 0.0)
            for p in all_predictions:
                p.signal_score *= size_mult
                p.conviction = min(1.0, max(0.0, p.conviction + conf_boost))
                p.regime = regime_label
        except Exception as exc:
            logger.warning("Regime detection failed: %s", exc)

        # 6. Entry rules — filter by volume/spread
        risk_cfg = config.get("risk", {})
        entry_cfg = EntryExitConfig(
            min_volume_24h=risk_cfg.get("min_volume_24h", 100_000.0),
            max_spread_bps=risk_cfg.get("max_spread_bps", 50.0),
        )
        entry_rules = EntryExitRules(entry_cfg)
        filtered: List[Prediction] = []
        for p in all_predictions:
            sym_df = symbol_dfs.get(p.symbol)
            if sym_df is None:
                filtered.append(p)
                continue
            vol_24h = float(sym_df["volume"].tail(24).sum()) if "volume" in sym_df.columns else 0.0
            spread = getattr(p, "ob_spread_bps", 0.0)
            allowed, reason = entry_rules.should_enter(vol_24h, spread)
            if allowed:
                filtered.append(p)
            else:
                logger.debug("Entry filter rejected %s: %s", p.symbol, reason)
        all_predictions = filtered

        # 7. Drawdown management — suppress/reduce signals on drawdown
        drawdown_label = ""
        try:
            dd_cfg = DrawdownConfig(
                reduce_threshold_pct=risk_cfg.get("drawdown_reduce_pct", 5.0),
                halt_threshold_pct=risk_cfg.get("drawdown_halt_pct", 10.0),
            )
            dd_manager = DrawdownManager(dd_cfg)
            simulated_equity = await _load_simulated_equity(storage)
            dd_state = dd_manager.update(simulated_equity)
            drawdown_label = dd_state.value

            if dd_state == DrawdownState.HALTED:
                logger.warning("Drawdown HALT — suppressing all signals")
                all_predictions = []
            elif dd_state == DrawdownState.REDUCED:
                top_n = max(1, top_n // 2)
                logger.warning("Drawdown REDUCED — cutting top_n to %d", top_n)
        except Exception as exc:
            logger.warning("Drawdown manager failed: %s", exc)

        # 8. Rank by signal strength
        ranked = predictor.rank_predictions(all_predictions)

        # 9. Drift monitoring — continuous alerting
        drift_warning = ""
        try:
            drift_mon = _load_drift_monitor()
            for p in ranked:
                drift_mon.record_prediction(p.direction, p.prob_up, p.prob_flat, p.prob_down)
            if len(drift_mon._prediction_history) >= drift_mon.config.min_samples:
                if drift_mon._baseline_distribution is None:
                    drift_mon.set_baseline()
                drift_report = drift_mon.compute_drift()
                if drift_report.distribution_shift:
                    drift_warning = f"Distribution shift detected (KL={drift_report.kl_divergence:.4f})"
                if drift_report.calibration_degraded:
                    drift_warning += f" | Calibration degraded (ECE={drift_report.calibration_error:.4f})"
            _save_drift_monitor(drift_mon)
        except Exception as exc:
            logger.warning("Drift monitoring failed: %s", exc)

        # 10. Dispatch notifications with enriched context
        if not model_path or "multi" not in str(model_path):
            await dispatcher.dispatch(
                ranked,
                top_n=top_n,
                interval=interval,
                regime=regime_label,
                drift_warning=drift_warning,
                drawdown_state=drawdown_label,
            )

        # 11. Log predictions to database
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


async def _load_simulated_equity(storage: "Storage") -> float:
    """Load simulated equity from recent prediction accuracy in the DB."""
    try:
        rows = await storage.db.execute_fetchall(
            "SELECT actual_direction, predicted_direction, predicted_magnitude "
            "FROM prediction_scores ORDER BY scored_at DESC LIMIT 200"
        )
        if not rows:
            return 10000.0
        equity = 10000.0
        for row in reversed(rows):
            actual, predicted, mag = row
            mag = float(mag) if mag else 0.0
            if actual == predicted and predicted != "FLAT":
                equity += equity * abs(mag)
            elif actual != predicted and predicted != "FLAT":
                equity -= equity * abs(mag) * 0.5
        return equity
    except Exception:
        return 10000.0


def _load_drift_monitor() -> DriftMonitor:
    """Load drift monitor state from disk."""
    monitor = DriftMonitor()
    try:
        if os.path.exists(DRIFT_STATE_PATH):
            with open(DRIFT_STATE_PATH) as f:
                state = json.load(f)
            monitor._prediction_history = state.get("history", [])
            baseline = state.get("baseline")
            if baseline is not None:
                import numpy as np
                monitor._baseline_distribution = np.array(baseline)
    except Exception:
        pass
    return monitor


def _save_drift_monitor(monitor: DriftMonitor) -> None:
    """Persist drift monitor state to disk."""
    try:
        state = {
            "history": monitor._prediction_history[-500:],
            "baseline": (
                monitor._baseline_distribution.tolist()
                if monitor._baseline_distribution is not None
                else None
            ),
        }
        os.makedirs(os.path.dirname(DRIFT_STATE_PATH), exist_ok=True)
        with open(DRIFT_STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


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
        from datetime import datetime
        from datetime import timezone as tz
        now_utc = datetime.now(tz.utc).strftime("%Y-%m-%d %H:%M UTC")
        subject = (
            f"[PA Bot] Ensemble ({primary_interval}m + {secondary_interval}m) "
            f"— {now_utc}"
        )
        message = format_multi_timeframe_message(
            combined,
            top_n=top_n,
            primary_label=f"{primary_interval}m",
            secondary_label=f"{secondary_interval}m",
        )
        for channel in dispatcher._channels:
            try:
                await channel.send(message, subject=subject)
            except Exception as exc:
                logger.error("Failed to send multi-TF alert via %s: %s", channel.name, exc)

    return combined
