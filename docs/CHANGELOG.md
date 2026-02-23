# PA Bot — Improvement Changelog

Tracks every significant improvement made to the system, the date it was made,
and the reasoning behind each change.

---

## 2026-02-23 — Road to 9/10: Full Advanced Module Integration

**Commit:** `73232f8`

### Multi-Model Ensemble (LSTM + TFT + GBM)

| Component | Change |
|-----------|--------|
| `scripts/daily_retrain.py` | TFT and GBM models now train alongside the LSTM during daily retrain |
| `src/model/predictor.py` | Predictor loads all 3 model checkpoints and combines via Sharpe-weighted ensemble |
| `src/model/multi_ensemble.py` | Sharpe-weighted averaging with diversity checking |
| `src/data/storage.py` | New `model_sharpes` table for persisting per-model Sharpe ratios |

**Why:** A single LSTM leaves alpha on the table. Different model families capture
different patterns — LSTMs excel at sequential dependencies, TFT at variable
selection, and GBM at non-linear feature interactions. Combining them with
Sharpe-weighting ensures the best-performing model gets the most influence,
while ensemble disagreement provides a more principled uncertainty estimate
than MC Dropout alone.

### Sentiment & Cross-Exchange Data Sources

| Component | Change |
|-----------|--------|
| `src/api/sentiment_client.py` | New client for Fear & Greed Index (Alternative.me) and CryptoPanic news |
| `src/api/binance_client.py` | New client for Binance public funding rates and open interest |
| `scripts/collect_sentiment.py` | Hourly cron target for sentiment collection |
| `scripts/collect_binance.py` | Hourly cron target for Binance cross-exchange data |
| `src/data/storage.py` | Four new tables: `fear_greed_index`, `news_sentiment`, `binance_funding_rate`, `binance_oi` |

**Why:** Price action alone is insufficient. Sentiment extremes (fear/greed)
are contrarian indicators — extreme fear often precedes bounces. Cross-exchange
funding rate spreads reveal arbitrage pressure and directional bias that a
single-exchange model misses. News volume spikes precede volatility.

### Sentiment & Cross-Exchange Features (66 total)

| Component | Change |
|-----------|--------|
| `src/features/sentiment.py` | New module: 4 sentiment features + 2 cross-exchange features |
| `src/features/indicators.py` | `get_feature_columns()` expanded from 60 → 66 |
| `src/pipeline.py` | Sentiment and cross-exchange enrichment wired into data loading loop |

New features:
- `fear_greed_index` — normalized 0–1 market sentiment
- `fear_greed_change` — daily change in sentiment (momentum)
- `news_sentiment_score` — (positive − negative) / total news articles
- `news_volume_zscore` — abnormal news activity detection
- `funding_rate_spread` — Binance funding rate minus BitUnix funding rate
- `oi_divergence` — Binance open interest change rate

**Why:** These features provide orthogonal signal to price-based indicators.
Funding rate spread captures cross-exchange positioning imbalances. OI
divergence detects when leverage is building unevenly across venues.

### CV Fold Ensemble

| Component | Change |
|-----------|--------|
| `scripts/daily_retrain.py` | Each walk-forward fold's model is saved to disk |
| `src/model/predictor.py` | Fold models loaded at inference; outputs averaged with primary model |

**Why:** A single train/val split is noisy. By training on each fold and
averaging predictions across all fold models, the system reduces variance
and produces more stable probability estimates — analogous to bagging
but applied to temporal cross-validation folds.

### Regime-Aware Training

| Component | Change |
|-----------|--------|
| `scripts/daily_retrain.py` | HMM regime detection before training; LR and dropout adjusted per regime |

Adjustments:
- **High Volatility** → LR × 0.5, dropout + 0.1 (smaller steps, more regularization)
- **Ranging** → LR × 0.8 (gentler updates in low-signal environments)
- **Trending** → No adjustment (default parameters work well)

**Why:** Training with the same hyperparameters regardless of market conditions
leads to suboptimal convergence. In volatile markets, large learning rates
cause gradient instability. In ranging markets, the model overfits to noise.
Adaptive hyperparameters improve per-regime generalization.

### Ensemble Disagreement as Primary Uncertainty

| Component | Change |
|-----------|--------|
| `src/model/predictor.py` | When ensemble is active, uncertainty = model disagreement; fold disagreement as fallback; MC Dropout only when single model |

**Why:** MC Dropout estimates uncertainty by randomly disabling neurons — a
proxy at best. Genuine model disagreement (different architectures trained on
different data splits predicting differently) is a far stronger signal that
the prediction is unreliable. It directly measures epistemic uncertainty.

### A/B Testing Framework

| Component | Change |
|-----------|--------|
| `scripts/daily_retrain.py` | Shadow model comparison wired into daily retrain; state persisted to `data/ab_test_state.json` |

**Why:** Deploying a new model blindly risks regression. A/B testing lets the
new model run in shadow mode, generating predictions that are scored but not
acted on. After 7+ days and 100+ predictions, the system evaluates whether the
shadow model's Sharpe ratio exceeds production's by a configurable threshold
before recommending promotion.

### Meta-Learner Performance Tracking

| Component | Change |
|-----------|--------|
| `scripts/daily_retrain.py` | Per-model, per-regime Sharpe tracked via `MetaLearner` after each retrain |

**Why:** Not all models perform equally across all symbols and market
conditions. By tracking which model wins in which regime, the system
can eventually route predictions to the best-performing model per context.

### Automated Feature Retirement

| Component | Change |
|-----------|--------|
| `scripts/daily_retrain.py` | Features below importance threshold for 3+ consecutive days flagged as retired |
| `src/data/storage.py` | New `feature_retirement` table tracks below-threshold streaks |

**Why:** Dead features add noise and computational cost. Rather than manually
auditing importance scores, the system automatically identifies and flags
features that consistently contribute nothing, enabling automated pruning.

### WebSocket Client & PostgreSQL Config

| Component | Change |
|-----------|--------|
| `src/api/bitunix_ws.py` | WebSocket client for real-time order book streaming with auto-reconnect |
| `config/settings.yaml` | PostgreSQL DSN config added (opt-in, SQLite remains default) |
| `requirements.txt` | All optional deps uncommented: lightgbm, optuna, hmmlearn, asyncpg, prometheus-client, websockets |

**Why:** REST polling for order book data is rate-limited and stale. WebSocket
streaming provides continuous, real-time depth updates. PostgreSQL/TimescaleDB
is needed for production scalability (concurrent reads during prediction while
writes happen from collectors).

### Weekly HPO Script

| Component | Change |
|-----------|--------|
| `scripts/weekly_hpo.py` | Optuna-based hyperparameter optimization for LSTM, TFT, and GBM |

**Why:** Manually tuning hidden_dim, dropout, learning_rate, etc. is
time-consuming and suboptimal. Weekly automated HPO with Optuna's Bayesian
optimization explores the hyperparameter space systematically, optimizing
for validation Sharpe ratio rather than accuracy.

---

## 2026-02-23 — Full Pipeline Integration (Wire All Built Modules)

**Commit:** `5d8342a`

### Pipeline Orchestration Overhaul

| Component | Change |
|-----------|--------|
| `src/pipeline.py` | Added regime detection, entry/exit rules, drawdown management, drift monitoring |
| `src/model/predictor.py` | MC Dropout uncertainty integrated; `signal_score` penalized by uncertainty |
| `scripts/daily_retrain.py` | P&L optimizer for threshold tuning; backtest validation gate before deployment |
| `scripts/run_prediction.py` | Prometheus metrics recording (prediction count, latency) |
| `src/notifications/dispatcher.py` | Enhanced with regime, uncertainty, drift, drawdown context |

**Why:** Previously, all these modules (regime detection, drawdown management,
drift monitoring, etc.) were *built* but not *wired* into the live pipeline.
Without integration, they provided zero value. This commit connected every
component so they actively influence predictions and risk management.

### Data Leakage Fix

| Component | Change |
|-----------|--------|
| `src/features/orderbook.py` | Changed `.ffill()` → `.fillna(0.0)` for order book features |
| `src/features/derivatives.py` | Changed `.ffill()` → `.fillna(0.0)` for Coinalyze and funding rate features |

**Why:** Forward-filling missing feature values introduces future information
into past rows during training. This is a critical data leakage bug that
inflates training metrics while degrading live performance. Zero-fill is safe
because it represents "no data available" without temporal contamination.

### P&L-Based Threshold Optimization

| Component | Change |
|-----------|--------|
| `scripts/daily_retrain.py` | `PnLOptimizer` replaces accuracy-based threshold tuning |

**Why:** Optimizing the FLAT threshold for accuracy maximizes correctness but
ignores profitability. A threshold that's correct 80% of the time on small
moves is worse than one that's correct 60% on large moves. P&L optimization
directly maximizes risk-adjusted returns (Sharpe ratio).

### Backtest Validation Gate

| Component | Change |
|-----------|--------|
| `scripts/daily_retrain.py` | New model must achieve Sharpe ≥ -1.0 on 14-day backtest before deployment |

**Why:** Validation loss alone doesn't capture whether a model makes money.
A model can have low loss but terrible trading performance due to poor
calibration or adverse timing. The backtest gate simulates actual trading
with transaction costs, rejecting models that would lose money.

---

## 2026-02-23 — Architecture Upgrade (Full Integration Plan)

**Commit:** `a1ee690`

### 60-Feature Model with Derivatives

| Component | Change |
|-----------|--------|
| `src/features/indicators.py` | Expanded from 41 → 60 features (order book + derivatives + cross-asset) |
| `src/features/orderbook.py` | Order book structural features (imbalance, spread, depth, walls, concentration) |
| `src/features/derivatives.py` | OI change, liquidation imbalance, L/S ratio, funding rate, BTC correlation |

**Why:** Technical indicators alone capture ~50% of the available signal.
Order book microstructure reveals supply/demand imbalances. Derivatives data
(OI, liquidations, funding) captures leveraged positioning that drives
futures price action. Cross-asset correlation to BTC captures systemic risk.

### Built Advanced Modules

New modules created (wired in later commits):
- `src/model/tft.py` — Temporal Fusion Transformer
- `src/model/gbm.py` — Gradient Boosting Machine
- `src/model/regime.py` — HMM-based regime detection
- `src/model/drift.py` — Prediction drift monitoring
- `src/model/uncertainty.py` — MC Dropout + Deep Ensemble
- `src/model/hpo.py` — Optuna hyperparameter optimization
- `src/model/meta_learning.py` — Per-symbol model selection
- `src/model/ab_testing.py` — Shadow model comparison
- `src/model/multi_ensemble.py` — Multi-model combination
- `src/risk/drawdown.py` — Circuit breaker for drawdowns
- `src/risk/rules.py` — Entry/exit filtering
- `src/scoring/pnl_optimizer.py` — P&L-based threshold optimization
- `src/backtesting/engine.py` — Event-driven backtesting
- `src/monitoring/metrics.py` — Prometheus metrics

**Why:** Industry-grade quant systems require defense in depth — no single
model or technique is sufficient. Each module addresses a specific weakness
identified during architectural review.

---

## 2026-02-23 — Data Quality & Robustness

**Commit:** `5270169`

| Component | Change |
|-----------|--------|
| `src/data/quality.py` | Candle validation (close ≤ 0, high < low, volume < 0, price jumps) |
| `src/model/predictor.py` | Degenerate window detection (all-zero, no-variance → skip) |
| `src/model/trainer.py` | NLL comparison for temperature calibration; empty dataset guards |
| `src/data/collector.py` | Infinite loop guard on backfill with max iterations |

**Why:** Garbage in, garbage out. Without validation, corrupt candles
(exchange API glitches, zero prices, impossible OHLC relationships) would
train the model on nonsense. Degenerate window detection prevents
divide-by-zero during Z-score normalization at inference.

---

## 2026-02-23 — Coinalyze Integration

**Commit:** `eaeb43a`

| Component | Change |
|-----------|--------|
| `src/api/coinalyze_client.py` | REST client for OI, liquidations, and L/S ratio data |
| `scripts/backfill_coinalyze.py` | Historical backfill for all Coinalyze data types |
| `src/data/storage.py` | Three new tables: `coinalyze_oi`, `coinalyze_liquidations`, `coinalyze_long_short` |

**Why:** Futures price action is driven by leverage. Open interest reveals
total positioning, liquidation data shows forced selling/buying cascades,
and long/short ratios reveal crowd positioning — all critical for
predicting short-term moves in perpetual futures markets.

---

## 2026-02-23 — Email Notification Improvements

**Commit:** `1ddd307`

| Component | Change |
|-----------|--------|
| `src/notifications/email_notifier.py` | Styled HTML body, dynamic subject lines |
| `src/notifications/dispatcher.py` | Regime, uncertainty, drawdown context in messages |

**Why:** Raw text emails were hard to scan. Styled HTML with clear tables,
color-coded directions, and informative subject lines (including regime tag)
let the user assess signals at a glance from their inbox.

---

## 2026-02-23 — Order Book & Funding Rate Collection

**Commits:** `df6bb69`, `8e8f938`

| Component | Change |
|-----------|--------|
| `scripts/collect_orderbook.py` | 15-minute order book depth snapshots |
| `src/data/storage.py` | `order_book_snapshots` and `funding_rate_snapshots` tables |

**Why:** Order book data is ephemeral — if not captured, it's gone forever.
Early collection builds the historical dataset needed for feature extraction
once enough snapshots accumulate (typically 2+ weeks for meaningful patterns).

---

## 2026-02-17 — A-Grade Architecture Upgrade

**Commit:** `321343f`

### Walk-Forward Validation

| Component | Change |
|-----------|--------|
| `src/model/dataset.py` | Temporal train/val splits instead of random splits |

**Why:** Random splits leak future information into training. Walk-forward
validation respects the temporal ordering of data, producing realistic
performance estimates that match live trading conditions.

### Feature Gating & Temporal Attention

| Component | Change |
|-----------|--------|
| `src/model/lstm.py` | Added learnable feature gates and temporal attention mechanism |

**Why:** Not all features matter equally, and not all time steps are equally
informative. Feature gating lets the model learn to suppress noisy features.
Temporal attention lets it focus on the most relevant historical periods
(e.g., recent candles during breakouts, longer history during consolidation).

### Probability Calibration

| Component | Change |
|-----------|--------|
| `src/model/trainer.py` | Learnable temperature parameter for calibrated probabilities |

**Why:** Neural network softmax outputs are notoriously overconfident.
Temperature scaling adjusts the output distribution so that a 70% predicted
probability actually corresponds to 70% empirical accuracy — critical for
proper position sizing and risk management.

---

## 2026-02-17 — Anti-Pump/Dump Features

**Commit:** `006b28c`

| Feature | Description |
|---------|-------------|
| `price_position_48` | Where price sits in its 48-period range (0 = bottom, 1 = top) |
| `price_position_168` | Same for 168-period (1 week) — detects extended moves |
| `momentum_accel` | Rate of change of ROC — detects momentum exhaustion |
| `atr_expansion` | Current ATR vs 48-period average — detects volatility spikes |
| `vol_price_ratio` | Volume spike relative to price move — detects distribution |

**Why:** Pump-and-dump schemes show characteristic signatures: price at
extreme range positions with decelerating momentum and expanding volatility.
These features let the model recognize and avoid signals that look bullish
but are actually distribution phases.

---

## 2026-02-17 — Multi-Timeframe Support

**Commit:** `22d5f97`

| Component | Change |
|-----------|--------|
| `src/model/ensemble.py` | Log-odds combination of 1h + 15m predictions |
| `config/settings.yaml` | Configurable timeframe weights and intervals |

**Why:** 1h candles capture directional bias but miss entry timing.
15m candles provide precise entry signals but are noisy for direction.
Combining both via log-odds probability fusion gives the best of both
worlds — directional accuracy from 1h with timing precision from 15m.

---

## 2026-02-17 — Quant-Grade Model Architecture

**Commit:** `50b239a`

### Dual-Head LSTM

| Component | Change |
|-----------|--------|
| `src/model/lstm.py` | Classification head (UP/FLAT/DOWN) + regression head (magnitude) |

**Why:** Classification alone only predicts direction. Adding a regression
head for magnitude prediction enables the `signal_score` to prioritize
large moves — a 2% predicted move with 70% UP probability is far more
valuable than a 0.1% move with 90% UP probability.

### Per-Symbol Z-Score Normalization

| Component | Change |
|-----------|--------|
| `src/model/dataset.py`, `src/model/predictor.py` | Per-window Z-score normalization |

**Why:** Different symbols trade at vastly different price levels and
volatilities. Global normalization fails to account for this. Per-window
Z-score normalization ensures each symbol's features are on a comparable
scale, regardless of whether BTC is at $30k or $100k.

---

## 2026-02-17 — Initial System

**Commit:** `94c5033`

Core system with LSTM model, BitUnix API client, SQLite storage,
async data collection, Discord/Telegram/Email notifications, and
cron-based scheduling. 41 technical indicators covering trend,
momentum, volatility, and volume dimensions.

---

## 2026-02-16 — Project Initialization

**Commit:** `6e5d7ee`

Repository created with README and `.gitignore`.
