# PA Bot -- Crypto Price Action Predictor

An automated crypto analysis system that pulls futures market data from BitUnix, computes technical indicators, runs predictions through deep-learning models, and broadcasts ranked signals via Discord, Telegram, and/or email. Supports **multi-timeframe analysis** (1h + 15m ensemble), **backtesting**, **risk management**, and **regime-aware prediction**.

## Architecture

```
Prediction Pipeline:
  Cron (every 15m) -- run_prediction.py --multi-timeframe
    -> Fetch all BitUnix futures tickers (concurrent)
    -> Pull latest OHLCV candles (1h + 15m intervals)
    -> Merge order book, funding rate, and Coinalyze features
    -> Compute 41+ scale-invariant technical indicators
    -> Feed sliding window into model ensemble (LSTM + TFT + GBM)
    -> MC Dropout uncertainty estimation
    -> Regime-aware signal filtering
    -> Risk management checks (position sizing, drawdown, correlation)
    -> Rank coins by signal strength
    -> Dispatch alerts to Discord / Telegram / Email
    -> Log predictions to database

Daily Cycle:
  Cron (daily @ 00:05 UTC) -- daily_retrain.py
    -> Gap-fill recent candles for all intervals
    -> Score predictions against actual outcomes
    -> P&L-based threshold optimization (Sharpe, not accuracy)
    -> Regime detection update (HMM on BTC/ETH)
    -> Retrain all model families (LSTM, TFT, GBM)
    -> Meta-learner model selection update
    -> Drift monitoring and calibration check
    -> Send accuracy digest via notifications

Data Collection:
  Cron (every 15m) -- collect_orderbook.py  (order book + funding rates)
  Cron (hourly)    -- collect_coinalyze.py  (OI, liquidations, L/S ratios)
```

## Project Structure

```
pa_bot/
├── config/
│   ├── settings.yaml              # Configuration (model params, notifications, timeframes)
│   └── prometheus.yml             # Prometheus scrape config (Phase 7)
├── src/
│   ├── api/
│   │   ├── bitunix_client.py      # Async REST client for BitUnix futures + spot
│   │   ├── coinalyze_client.py    # Async Coinalyze API client (OI, liquidations, L/S)
│   │   └── rate_limiter.py        # Token-bucket rate limiter
│   ├── data/
│   │   ├── collector.py           # Data fetch orchestration
│   │   ├── storage.py             # SQLite backend (OHLCV, predictions, OB, etc.)
│   │   ├── postgres_storage.py    # PostgreSQL/TimescaleDB backend (Phase 7)
│   │   └── quality.py             # Candle validation, gap detection, data health
│   ├── features/
│   │   ├── indicators.py          # 41 scale-invariant technical indicators
│   │   ├── orderbook.py           # Order book structural/liquidity features
│   │   └── derivatives.py         # OI, liquidation, funding rate, cross-asset features
│   ├── model/
│   │   ├── architecture.py        # LSTM with feature gating + temporal attention
│   │   ├── tft.py                 # Temporal Fusion Transformer
│   │   ├── gbm.py                 # LightGBM / sklearn gradient boosting
│   │   ├── dataset.py             # Time-series windowing + walk-forward splits
│   │   ├── trainer.py             # Training loop with early stopping + calibration
│   │   ├── predictor.py           # Inference engine
│   │   ├── ensemble.py            # Multi-timeframe prediction combiner
│   │   ├── multi_ensemble.py      # Multi-model ensemble (LSTM + TFT + GBM)
│   │   ├── uncertainty.py         # MC Dropout + deep ensemble uncertainty
│   │   ├── importance.py          # Permutation importance for feature auditing
│   │   ├── regime.py              # HMM-based market regime detection
│   │   ├── drift.py               # Prediction drift + calibration monitoring
│   │   ├── meta_learning.py       # Per-symbol/regime model selection
│   │   ├── hpo.py                 # Optuna hyperparameter optimization
│   │   └── ab_testing.py          # Shadow model A/B testing framework
│   ├── scoring/
│   │   ├── accuracy.py            # Prediction accuracy evaluation
│   │   ├── adaptive.py            # Adaptive threshold + sample weight tuning
│   │   └── pnl_optimizer.py       # Sharpe-based threshold + weight optimization
│   ├── risk/
│   │   ├── sizing.py              # Kelly criterion, volatility, fixed-fraction sizing
│   │   ├── portfolio_risk.py      # Max exposure, correlation limits
│   │   ├── drawdown.py            # Circuit breaker (reduce/halt/recover)
│   │   └── rules.py               # ATR stop-loss, take-profit, time stops
│   ├── backtesting/
│   │   ├── engine.py              # Event-driven backtester
│   │   ├── costs.py               # Fee, slippage, funding cost modeling
│   │   ├── metrics.py             # Sharpe, Sortino, Calmar, drawdown, profit factor
│   │   ├── portfolio.py           # Position tracking, equity curve, trade log
│   │   └── signals.py             # Signal generator interface + predictor wrapper
│   ├── monitoring/
│   │   └── metrics.py             # Prometheus metrics exporter
│   ├── notifications/
│   │   ├── dispatcher.py          # Pluggable notification router
│   │   ├── discord_notifier.py
│   │   ├── telegram_notifier.py
│   │   └── email_notifier.py
│   └── pipeline.py                # Main prediction pipeline orchestrator
├── scripts/
│   ├── backfill_data.py           # One-time historical OHLCV download
│   ├── backfill_coinalyze.py      # One-time Coinalyze historical download
│   ├── train_model.py             # Model training entrypoint
│   ├── daily_retrain.py           # Daily retrain cron target
│   ├── run_prediction.py          # Prediction cron target
│   ├── run_backtest.py            # Backtesting CLI (single + walk-forward)
│   ├── collect_orderbook.py       # OB + funding rate snapshot cron target
│   ├── collect_coinalyze.py       # Coinalyze data collection cron target
│   ├── healthcheck.py             # Production health monitoring
│   └── migrate_to_postgres.py     # SQLite -> PostgreSQL migration
├── tests/
│   ├── conftest.py                # Shared test fixtures
│   ├── test_quality.py            # Data quality validation tests
│   ├── test_indicators.py         # Technical indicator tests
│   ├── test_dataset.py            # Dataset windowing + label tests
│   ├── test_predictor.py          # Inference + normalization tests
│   ├── test_scoring.py            # Accuracy computation tests
│   └── test_ensemble.py           # Multi-timeframe combination tests
├── .github/workflows/
│   ├── ci.yml                     # Lint + test on push/PR
│   └── deploy.yml                 # Auto-deploy to Hetzner on merge
├── Dockerfile                     # Multi-stage Docker build
├── docker-compose.yml             # Full stack (app + TimescaleDB + Prometheus + Grafana)
├── pyproject.toml                 # ruff, mypy, pytest configuration
├── requirements.txt
├── .env.example
└── .gitignore
```

## Prerequisites

- Python 3.10+
- pip
- (Optional) NVIDIA GPU with CUDA for faster training

## Quick Start

### 1. Install Dependencies

```bash
cd pa_bot

# CPU (works everywhere, recommended for most setups)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# GPU (only if you have NVIDIA + CUDA 12.1+)
# pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2. Configure

Copy and edit the config file:

```bash
cp .env.example .env
# Edit .env with your notification credentials (Discord webhook, Telegram bot, etc.)
# Edit config/settings.yaml to tune model and pipeline parameters
```

No BitUnix API keys are needed for market data -- the public endpoints are unauthenticated.

### 3. Backfill Historical Data

Download historical candles for all BitUnix futures pairs:

```bash
# Multi-timeframe (1h + 15m at once)
python scripts/backfill_data.py --candles 2000 --all-timeframes
```

### 4. Train the Model (Initial)

```bash
# Train 1h model
python scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 60

# Train 15m model (for multi-timeframe)
python scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 15
```

### 5. Run a Prediction

```bash
python scripts/run_prediction.py --multi-timeframe
```

### 6. Run a Backtest

```bash
# Single pass backtest
python scripts/run_backtest.py --interval 60 --days 90 --capital 10000

# Walk-forward backtest (realistic out-of-sample evaluation)
python scripts/run_backtest.py --walk-forward --folds 5
```

### 7. Set Up Cron

```bash
# Ensemble predictions every hour at :05
5 * * * * cd /path/to/pa_bot && python scripts/run_prediction.py --multi-timeframe >> logs/prediction.log 2>&1

# Daily retrain at 00:05 UTC
5 0 * * * cd /path/to/pa_bot && python scripts/daily_retrain.py >> logs/retrain.log 2>&1

# Order book + funding rate snapshots every 15 minutes
*/15 * * * * cd /path/to/pa_bot && python scripts/collect_orderbook.py >> logs/orderbook.log 2>&1

# Coinalyze data hourly at :10
10 * * * * cd /path/to/pa_bot && python scripts/collect_coinalyze.py >> logs/coinalyze.log 2>&1

# Health check every 30 minutes
*/30 * * * * cd /path/to/pa_bot && python scripts/healthcheck.py --notify >> logs/health.log 2>&1
```

## Model Architecture

The system supports three model families that can be ensembled:

### LSTM (Primary)
- **Input:** Sliding window of per-window Z-score normalized indicator vectors
- **Feature gating:** Learned sigmoid gate dynamically weights features per timestep
- **Encoder:** 2-layer unidirectional LSTM (128 hidden units) — causal processing only
- **Temporal attention:** Bahdanau-style attention over all LSTM hidden states
- **Dual heads:** Classification (UP/FLAT/DOWN) + regression (magnitude)
- **Temperature:** Post-training calibration for accurate probability estimates

### Temporal Fusion Transformer (TFT)
- Variable Selection Network for interpretable feature importance
- Multi-head causal attention across time dimension
- Gated Residual Networks for non-linear processing

### Gradient Boosting (GBM)
- LightGBM (or sklearn fallback) on flattened feature windows
- Fast to train, strong non-neural baseline
- Feature importance via split counts

### Ensemble
- Multi-model combination weighted by recent Sharpe ratio
- Diversity checking (correlation between model predictions)
- Multi-timeframe combination via log-odds probability fusion

## Risk Management

- **Position Sizing:** Kelly criterion (half-Kelly default), volatility targeting, fixed fraction
- **Portfolio Controls:** Maximum total exposure (50%), correlated position limits (3), per-symbol caps (10%)
- **Drawdown Management:** Circuit breaker (5% = reduce, 10% = halt), gradual recovery
- **Entry/Exit Rules:** ATR-based stop-loss/take-profit, time stops, signal reversal exits, liquidity filters

## Backtesting

The backtesting engine provides realistic historical simulation:

- Event-driven architecture with configurable signal generators
- Transaction cost modeling (maker/taker fees, slippage, funding costs)
- Comprehensive metrics: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor
- Walk-forward mode for unbiased out-of-sample evaluation
- BTC buy-and-hold benchmark comparison

## Technical Indicators

The feature engineering module computes 41+ scale-invariant indicators:

| Category   | Indicators                                                    |
|------------|---------------------------------------------------------------|
| Trend      | EMA(9, 21, 50), MACD(12,26,9), ADX(14)                       |
| Momentum   | RSI(14), Stochastic RSI, Williams %R, ROC(12)                 |
| Volatility | Bollinger Bands(20,2), ATR(14), Keltner Channels              |
| Volume     | OBV, Acc/Dist, Volume SMA Ratio, Volume Z-Score, VWAP         |
| Custom     | % changes (1h/4h/24h/3d/7d), candle ratios, EMA crosses, BB signals |
| Anti-pump  | Price position (2d/7d range), momentum acceleration, ATR expansion, volume-price divergence |
| Order Book | Imbalance, spread (bps), depth ratio, concentration           |
| Derivatives| OI change/Z-score, liquidation imbalance, L/S ratio, funding rate |
| Cross-Asset| BTC return, correlation to BTC, BTC dominance proxy           |

## Notifications

Enable any combination in `config/settings.yaml`:

- **Discord:** Set `webhook_url` from your server's channel settings
- **Telegram:** Create a bot via @BotFather, get `bot_token` and `chat_id`
- **Email:** Configure SMTP credentials (Gmail app passwords recommended)

## Production Deployment

See `docs/DEPLOYMENT.md` for a step-by-step Hetzner VPS guide.

For Docker deployment:

```bash
# Full stack with TimescaleDB + Prometheus + Grafana
docker-compose up -d
```

## Monitoring

- **Health checks:** `scripts/healthcheck.py` monitors predictions, retraining, data freshness, DB size, disk space
- **Prometheus metrics:** Prediction latency, accuracy, API health, portfolio equity
- **Grafana dashboards:** Real-time performance visualization

## Testing

```bash
python -m pytest tests/ -v
```

## License

MIT
