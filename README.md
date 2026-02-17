# PA Bot -- Crypto Price Action Predictor

An automated crypto analysis system that pulls futures market data from BitUnix every hour, computes technical indicators, runs predictions through an LSTM deep-learning model, and broadcasts ranked signals via Discord, Telegram, and/or email.

## Architecture

```
Cron (hourly) -- run_prediction.py
  -> Fetch all BitUnix futures tickers (concurrent)
  -> Pull latest OHLCV candles (1h interval)
  -> Store in SQLite
  -> Compute 30+ technical indicators (RSI, MACD, BB, ATR, OBV, VWAP, ...)
  -> Feed sliding window into bidirectional LSTM model (inference only, ~1s total)
  -> Dual output: P(UP/FLAT/DOWN) + predicted % magnitude
  -> Rank coins by signal strength
  -> Dispatch alerts to Discord / Telegram / Email
  -> Log predictions to database

Cron (daily @ 00:05 UTC) -- daily_retrain.py
  -> Gap-fill last 48h of candles
  -> Back up existing model checkpoint
  -> Retrain LSTM on rolling 60-day window (~5-15 min on CPU)
  -> Save new model_final.pt (picked up automatically by next hourly run)
```

## Project Structure

```
pa_bot/
├── config/
│   └── settings.yaml          # Configuration (API keys, model params, notifications)
├── src/
│   ├── api/
│   │   ├── bitunix_client.py  # Async REST client for BitUnix futures + spot
│   │   └── rate_limiter.py    # Token-bucket rate limiter (10 req/sec)
│   ├── data/
│   │   ├── collector.py       # Data fetch orchestration
│   │   ├── storage.py         # SQLite backend for OHLCV + predictions
│   │   └── backfill.py        # Historical data download
│   ├── features/
│   │   └── indicators.py      # 30+ technical indicators via pandas + ta
│   ├── model/
│   │   ├── architecture.py    # Bidirectional LSTM with dual heads
│   │   ├── dataset.py         # Time-series windowing + walk-forward splits
│   │   ├── trainer.py         # Training loop with early stopping
│   │   └── predictor.py       # Inference engine
│   ├── notifications/
│   │   ├── dispatcher.py      # Pluggable notification router
│   │   ├── discord_notifier.py
│   │   ├── telegram_notifier.py
│   │   └── email_notifier.py
│   └── pipeline.py            # Hourly pipeline orchestrator
├── scripts/
│   ├── backfill_data.py       # One-time historical download
│   ├── train_model.py         # Model training entrypoint
│   ├── daily_retrain.py       # Daily retrain cron target
│   └── run_prediction.py      # Hourly prediction cron target
├── data/                      # Local data (gitignored)
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
pip install -r requirements.txt
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
python scripts/backfill_data.py --candles 2000
```

This fetches ~83 days of hourly data per symbol. Takes a few minutes depending on how many pairs are listed.

Options:
- `--candles N` -- number of hourly candles per symbol (default: 2000)
- `--concurrency N` -- parallel API requests (default: 5)
- `--symbols BTCUSDT,ETHUSDT` -- limit to specific pairs

### 4. Train the Model (Initial)

```bash
python scripts/train_model.py --epochs 100 --window 168 --patience 10
```

Options:
- `--window N` -- input window size in hours (default: 168 = 7 days)
- `--horizon N` -- prediction horizon in hours (default: 1)
- `--folds N` -- walk-forward cross-validation folds (default: 3)
- `--hidden N` -- LSTM hidden dimension (default: 128)
- `--batch-size N` -- training batch size (default: 64)
- `--rolling-days N` -- only use the most recent N days of data per symbol (omit for all data)
- `--backup` -- create a timestamped backup of the existing checkpoint before overwriting

The best model checkpoint is saved to `data/models/model_final.pt`.

### 5. Run a Prediction

```bash
python scripts/run_prediction.py --config config/settings.yaml
```

### 6. Set Up Cron (Hourly Predictions + Daily Retrain)

The system uses two cron jobs: hourly predictions and a daily model retrain that keeps the LSTM adapted to current market regimes.

Add to your crontab (`crontab -e`):

```bash
# Hourly predictions at the top of every hour
0 * * * * cd /path/to/pa_bot && /path/to/python scripts/run_prediction.py >> logs/cron.log 2>&1

# Daily model retrain at 00:05 UTC (after midnight candle closes)
5 0 * * * cd /path/to/pa_bot && /path/to/python scripts/daily_retrain.py >> logs/retrain.log 2>&1
```

On Windows, use Task Scheduler to create two tasks:
- **Hourly prediction:** run `python scripts/run_prediction.py` every hour
- **Daily retrain:** run `python scripts/daily_retrain.py` once per day at 00:05 UTC

The daily retrain:
1. Gap-fills the last 48h of candle data (catches any missed hours)
2. Backs up the existing model checkpoint with a timestamp
3. Retrains on a rolling 60-day window (configurable in `config/settings.yaml`)
4. Saves the new `model_final.pt` -- the next hourly prediction run picks it up automatically

## Model Details

- **Input:** Sliding window of 168 hours (7 days) of normalized technical indicator vectors
- **Encoder:** 2-layer bidirectional LSTM (128 hidden units per direction)
- **Classification head:** Softmax over [UP, FLAT, DOWN] -- probability distribution
- **Regression head:** Predicted % price change magnitude
- **Loss:** Weighted cross-entropy + Huber loss with tunable lambda
- **Validation:** Walk-forward splitting to prevent lookahead bias
- **Training:** All symbols jointly, so the model learns general crypto patterns

## Technical Indicators

The feature engineering module computes 35+ indicators:

| Category   | Indicators                                                    |
|------------|---------------------------------------------------------------|
| Trend      | EMA(9, 21, 50), MACD(12,26,9), ADX(14)                       |
| Momentum   | RSI(14), Stochastic RSI, Williams %R, ROC(12)                 |
| Volatility | Bollinger Bands(20,2), ATR(14), Keltner Channels              |
| Volume     | OBV, Acc/Dist, Volume SMA Ratio, Volume Z-Score, VWAP         |
| Custom     | % changes (1h/4h/24h), candle ratios, EMA crosses, BB signals |

## Notifications

Enable any combination in `config/settings.yaml`:

- **Discord:** Set `webhook_url` from your server's channel settings
- **Telegram:** Create a bot via @BotFather, get `bot_token` and `chat_id`
- **Email:** Configure SMTP credentials (Gmail app passwords recommended)

Alert format includes a ranked table of top N signals with direction, confidence, predicted magnitude, current price, and signal score.

## License

MIT
