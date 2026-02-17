# PA Bot -- Crypto Price Action Predictor

An automated crypto analysis system that pulls futures market data from BitUnix, computes technical indicators, runs predictions through an LSTM deep-learning model, and broadcasts ranked signals via Discord, Telegram, and/or email. Supports **multi-timeframe analysis** (1h + 15m ensemble) for stronger signals.

## Architecture

```
Single-timeframe mode (default):
  Cron (hourly) -- run_prediction.py
    -> Fetch all BitUnix futures tickers (concurrent)
    -> Pull latest OHLCV candles (1h interval)
    -> Store in SQLite
    -> Compute 34 scale-invariant technical indicators
    -> Feed sliding window into LSTM model (inference only, ~1s total)
    -> Dual output: P(UP/FLAT/DOWN) + predicted % magnitude
    -> Rank coins by signal strength
    -> Dispatch alerts to Discord / Telegram / Email
    -> Log predictions to database

Multi-timeframe mode:
  Cron (every 15m) -- run_prediction.py --multi-timeframe
    -> Run 1h model for directional bias (weight=0.6)
    -> Run 15m model for entry timing (weight=0.4)
    -> Ensemble combines probabilities with agreement classification:
       STRONG (both agree), PARTIAL (one confirms), CONFLICT (disagree)
    -> Agreement multiplier boosts strong signals, penalizes conflicts
    -> Dispatch combined alerts showing both timeframes

Cron (daily @ 00:05 UTC) -- daily_retrain.py
  -> Gap-fill recent candles for all configured intervals
  -> Score predictions and auto-tune FLAT_THRESHOLD
  -> Back up existing model checkpoints
  -> Retrain LSTM for each timeframe on rolling 60-day window
  -> Calibrate probability temperatures
  -> Send accuracy digest via notifications
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
│   │   ├── storage.py         # SQLite backend (OHLCV + predictions, multi-interval)
│   │   └── backfill.py        # Historical data download
│   ├── features/
│   │   └── indicators.py      # 34 scale-invariant technical indicators
│   ├── model/
│   │   ├── architecture.py    # Unidirectional LSTM with feature gating + attention
│   │   ├── dataset.py         # Time-series windowing + walk-forward splits
│   │   ├── trainer.py         # Training loop with early stopping + temp calibration
│   │   ├── predictor.py       # Inference engine
│   │   ├── ensemble.py        # Multi-timeframe prediction combiner
│   │   └── importance.py      # Permutation importance for feature auditing
│   ├── scoring/
│   │   ├── accuracy.py        # Prediction accuracy evaluation
│   │   └── adaptive.py        # Adaptive threshold + sample weight tuning
│   ├── notifications/
│   │   ├── dispatcher.py      # Pluggable notification router
│   │   ├── discord_notifier.py
│   │   ├── telegram_notifier.py
│   │   └── email_notifier.py
│   └── pipeline.py            # Single + multi-timeframe pipeline orchestrator
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
# Single timeframe (1h only)
python scripts/backfill_data.py --candles 2000

# Multi-timeframe (1h + 15m at once)
python scripts/backfill_data.py --candles 2000 --all-timeframes
```

The `--all-timeframes` flag automatically scales: 2000 1h candles + 8000 15m candles (~83 days each).

Options:
- `--candles N` -- number of candles per symbol (default: 2000)
- `--concurrency N` -- parallel API requests (default: 5)
- `--symbols BTCUSDT,ETHUSDT` -- limit to specific pairs
- `--interval N` -- candle interval: 1,3,5,15,30,60,120,240,360,720,D,W,M (default: 60)
- `--all-timeframes` -- backfill both 1h and 15m data

### 4. Train the Model (Initial)

```bash
# Train 1h model (default)
python scripts/train_model.py --epochs 100 --window 168 --patience 10

# Train 15m model (for multi-timeframe)
python scripts/train_model.py --epochs 100 --window 672 --interval 15 --patience 10
```

Options:
- `--window N` -- input window size in candles (default: 168; use 672 for 15m = 7 days)
- `--horizon N` -- prediction horizon in candles (default: 1)
- `--folds N` -- walk-forward cross-validation folds (default: 3)
- `--hidden N` -- LSTM hidden dimension (default: 128)
- `--batch-size N` -- training batch size (default: 64)
- `--rolling-days N` -- only use the most recent N days of data per symbol
- `--interval N` -- candle interval to train on (saves as `model_final_{interval}.pt`)
- `--backup` -- create a timestamped backup of the existing checkpoint

Checkpoints are saved to `data/models/model_final_60.pt` (1h) and `data/models/model_final_15.pt` (15m).

### 5. Run a Prediction

```bash
# Single timeframe (1h)
python scripts/run_prediction.py

# Multi-timeframe ensemble (1h + 15m)
python scripts/run_prediction.py --multi-timeframe
```

### 6. Set Up Cron

**Single-timeframe setup:**

```bash
# Hourly predictions
0 * * * * cd /path/to/pa_bot && python scripts/run_prediction.py >> logs/cron.log 2>&1

# Daily model retrain at 00:05 UTC
5 0 * * * cd /path/to/pa_bot && python scripts/daily_retrain.py >> logs/retrain.log 2>&1
```

**Multi-timeframe setup (recommended):**

```bash
# Run ensemble every 15 minutes (1h direction + 15m entry timing)
*/15 * * * * cd /path/to/pa_bot && python scripts/run_prediction.py --multi-timeframe >> logs/cron.log 2>&1

# Daily retrain both models at 00:05 UTC
5 0 * * * cd /path/to/pa_bot && python scripts/daily_retrain.py >> logs/retrain.log 2>&1
```

On Windows, use Task Scheduler to create equivalent tasks.

The daily retrain:
1. Gap-fills recent candle data for all configured intervals (1h + 15m)
2. Scores predictions against actual outcomes and auto-tunes thresholds
3. Backs up existing model checkpoints with timestamps
4. Retrains LSTM for each timeframe on rolling 60-day window
5. Calibrates probability temperatures on validation set
6. Sends accuracy + retrain digest via notifications

## Model Details

- **Input:** Sliding window of per-window Z-score normalized indicator vectors (168 candles for 1h, 672 for 15m)
- **Feature gating:** Learned sigmoid gate dynamically weights features per timestep
- **Encoder:** 2-layer unidirectional LSTM (128 hidden units) — causal processing only
- **Temporal attention:** Bahdanau-style attention over all LSTM hidden states
- **Classification head:** Softmax over [UP, FLAT, DOWN] with calibrated temperature
- **Regression head:** Predicted % price change magnitude (Huber loss)
- **Loss:** Inverse-frequency weighted cross-entropy + Huber loss
- **Validation:** Temporal walk-forward splitting with per-symbol data isolation
- **Ensemble:** Multi-timeframe combination with agreement-weighted scoring

## Technical Indicators

The feature engineering module computes 41 scale-invariant indicators:

| Category   | Indicators                                                    |
|------------|---------------------------------------------------------------|
| Trend      | EMA(9, 21, 50), MACD(12,26,9), ADX(14)                       |
| Momentum   | RSI(14), Stochastic RSI, Williams %R, ROC(12)                 |
| Volatility | Bollinger Bands(20,2), ATR(14), Keltner Channels              |
| Volume     | OBV, Acc/Dist, Volume SMA Ratio, Volume Z-Score, VWAP         |
| Custom     | % changes (1h/4h/24h/3d/7d), candle ratios, EMA crosses, BB signals |
| Anti-pump  | Price position (2d/7d range), momentum acceleration, ATR expansion, volume-price divergence |

## Notifications

Enable any combination in `config/settings.yaml`:

- **Discord:** Set `webhook_url` from your server's channel settings
- **Telegram:** Create a bot via @BotFather, get `bot_token` and `chat_id`
- **Email:** Configure SMTP credentials (Gmail app passwords recommended)

Alert format includes a ranked table of top N signals with direction, confidence, predicted magnitude, current price, and signal score.

## Local Setup (Windows, Tested Step-by-Step)

These are the exact commands that worked on a fresh Windows machine with Python 3.11.

### 1. Clone the repo

```powershell
cd C:\Users\seanm\Documents\git
git clone https://github.com/DeathByRamen/price_action_bot.git pa_bot
cd pa_bot
```

### 2. Create a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### 3. Install the Visual C++ Redistributable

PyTorch on Windows requires this. Download and run:

https://aka.ms/vs/17/release/vc_redist.x64.exe

Restart your terminal after installing.

### 4. Install dependencies (CPU PyTorch)

```powershell
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Important:** Use `--index-url` (not `--extra-index-url`) for the first torch install to force the CPU-only build. The default pip install pulls the CUDA version which fails without NVIDIA drivers.

Verify torch works:

```powershell
python -c "import torch; print(torch.__version__)"
```

Should print something like `2.10.0+cpu`.

### 5. Set up email notifications

```powershell
copy .env.example .env
```

Edit `.env` and add your Gmail app password:

```
SMTP_PASSWORD=yourapppasswordhere
```

To get a Gmail app password:
1. Go to https://myaccount.google.com/security
2. Enable 2-Step Verification
3. Go to https://myaccount.google.com/apppasswords
4. Create one named "PA Bot", copy the 16-character password (remove spaces)

Then in `config/settings.yaml`, set:
- `email.enabled: true`
- `email.username: "youremail@gmail.com"`
- `email.from_addr: "youremail@gmail.com"`
- `email.to_addrs: ["youremail@gmail.com"]`

Leave `password: ""` in settings.yaml -- it reads from `.env` instead (which is gitignored).

### 6. Backfill historical data

```powershell
python scripts/backfill_data.py --candles 2000
```

Takes 5-15 minutes. Data is written to `data/ohlcv.db` (SQLite).

For multi-timeframe (1h + 15m):

```powershell
python scripts/backfill_data.py --candles 2000 --all-timeframes
```

### 7. Train the model

```powershell
# Train 1h model
python scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 60

# Train 15m model (for multi-timeframe)
python scripts/train_model.py --epochs 100 --window 672 --patience 10 --interval 15
```

Takes 5-30 minutes on CPU. Checkpoints saved to `data/models/`.

### 8. Run a prediction

```powershell
# Single timeframe
python scripts/run_prediction.py

# Multi-timeframe ensemble
python scripts/run_prediction.py --multi-timeframe
```

### 9. Automate with Windows Task Scheduler

Open Task Scheduler and create two tasks:

**Predictions (every 15 min):**
- Trigger: Daily, repeat every 15 minutes
- Program: `C:\Users\seanm\Documents\git\pa_bot\venv\Scripts\python.exe`
- Arguments: `scripts/run_prediction.py --multi-timeframe`
- Start in: `C:\Users\seanm\Documents\git\pa_bot`

**Daily retrain (once per day at 00:05 UTC):**
- Program: `C:\Users\seanm\Documents\git\pa_bot\venv\Scripts\python.exe`
- Arguments: `scripts/daily_retrain.py`
- Start in: `C:\Users\seanm\Documents\git\pa_bot`

### Viewing the database

Install the **SQLite Viewer** extension in VS Code/Cursor (by Florian Klampfer), then click on `data/ohlcv.db` to browse tables.

Or use Python:

```powershell
python -c "import sqlite3; conn = sqlite3.connect('data/ohlcv.db'); print('Candles:', conn.execute('SELECT COUNT(*) FROM ohlcv').fetchone()[0]); print('Symbols:', conn.execute('SELECT COUNT(DISTINCT symbol) FROM ohlcv').fetchone()[0]); conn.close()"
```

## License

MIT
