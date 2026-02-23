# PA Bot — Production Deployment Guide (Hetzner)

This guide covers every step to take the PA Bot from a local development setup to a fully running production system on a Hetzner Cloud VPS.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Create the Hetzner Server](#2-create-the-hetzner-server)
3. [Connect via SSH](#3-connect-via-ssh)
4. [Set Up SSH Key Authentication (Optional)](#4-set-up-ssh-key-authentication-optional)
5. [Install System Dependencies](#5-install-system-dependencies)
6. [Clone the Repository](#6-clone-the-repository)
7. [Set Up Python Environment](#7-set-up-python-environment)
8. [Configure Environment Variables](#8-configure-environment-variables)
9. [Run the Backfill](#9-run-the-backfill)
10. [Train the Models](#10-train-the-models)
11. [Test a Prediction Run](#11-test-a-prediction-run)
12. [Set Up Cron Jobs](#12-set-up-cron-jobs)
13. [Verify Everything Is Running](#13-verify-everything-is-running)
14. [Maintenance & Operations](#14-maintenance--operations)
15. [Updating the Bot](#15-updating-the-bot)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Prerequisites

Before you begin, make sure you have:

- A **Hetzner Cloud account** — sign up at [https://console.hetzner.cloud](https://console.hetzner.cloud)
- A **GitHub account** with access to the `price_action_bot` repository
- A **GitHub Personal Access Token** (PAT) for cloning private repos — [create one here](https://github.com/settings/tokens) with `repo` scope
- Your **notification credentials** ready:
  - Email: Gmail address + App Password ([create one here](https://myaccount.google.com/apppasswords))
  - Discord: Webhook URL (optional)
  - Telegram: Bot token + chat ID (optional)

---

## 2. Create the Hetzner Server

1. Log into [Hetzner Cloud Console](https://console.hetzner.cloud)
2. Click **Add Server**
3. Configure:

| Setting | Value |
|---|---|
| **Location** | Ashburn (US) or nearest to you |
| **Image** | Ubuntu 24.04 |
| **Type** | Shared vCPU > **CX22** (2 vCPU, 4 GB RAM) |
| **SSH Key** | Add one if you have it, or skip for password access |
| **Name** | `pa-bot` |

4. Click **Create & Buy Now** (~$4.50/month)
5. Note the **Public IP address** once it's running
6. If you skipped SSH key setup, check your email for the **root password**

---

## 3. Connect via SSH

Open a terminal (PowerShell on Windows, Terminal on Mac/Linux):

```bash
ssh root@YOUR_SERVER_IP
```

- If it asks about authenticity, type `yes`
- Enter the root password from the Hetzner email
- Nothing appears on screen while typing the password — this is normal

You should now see a prompt like `root@pa-bot:~#`.

---

## 4. Set Up SSH Key Authentication (Optional)

This lets you SSH in without a password every time.

### On your local machine:

**Windows (PowerShell):**

```powershell
# Check if you already have a key
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"

# If the file doesn't exist, generate one:
ssh-keygen -t ed25519
# Press Enter for all prompts (default location, no passphrase)

# Then read the public key:
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

**Mac/Linux:**

```bash
# Check if you already have a key
cat ~/.ssh/id_ed25519.pub

# If the file doesn't exist, generate one:
ssh-keygen -t ed25519
# Press Enter for all prompts

# Then read the public key:
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output (starts with `ssh-ed25519`).

### On the server:

```bash
mkdir -p ~/.ssh
echo "PASTE_YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

Replace `PASTE_YOUR_PUBLIC_KEY_HERE` with your actual public key.

Test from a new terminal: `ssh root@YOUR_SERVER_IP` — should log in without a password.

---

## 5. Install System Dependencies

On the server:

```bash
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git sqlite3
```

Verify Python:

```bash
python3 --version
# Should show Python 3.12.x or similar
```

---

## 6. Clone the Repository

```bash
cd /opt
git clone https://github.com/DeathByRamen/price_action_bot.git pa_bot
```

When prompted:
- **Username:** Your GitHub username
- **Password:** Your GitHub Personal Access Token (NOT your GitHub password)

Save the token for future pulls:

```bash
cd /opt/pa_bot
git config credential.helper store
```

The next time you `git pull`, it will save your credentials so you don't have to re-enter them.

---

## 7. Set Up Python Environment

```bash
cd /opt/pa_bot
python3 -m venv venv
source venv/bin/activate
```

Install PyTorch (CPU version) and all dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

This takes a few minutes. Verify when done:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "from src.features.indicators import get_feature_columns; print('Features:', len(get_feature_columns()))"
```

Expected output:

```
PyTorch: 2.10.0+cpu
Features: 41
```

---

## 8. Configure Environment Variables

Create the `.env` file:

```bash
nano /opt/pa_bot/.env
```

Paste your credentials:

```
SMTP_PASSWORD=your_gmail_app_password_here
DISCORD_WEBHOOK_URL=your_discord_webhook_here
TELEGRAM_BOT_TOKEN=your_telegram_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```

Only include the lines for channels you're using. At minimum you need `SMTP_PASSWORD` for email notifications.

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

Verify the config is correct:

```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv()
import os
print('SMTP:', 'SET' if os.getenv('SMTP_PASSWORD') else 'MISSING')
"
```

---

## 9. Run the Backfill

Create a logs directory and run the backfill in the background:

```bash
mkdir -p logs
nohup python3 scripts/backfill_data.py --all-timeframes > logs/backfill.log 2>&1 &
```

Monitor progress:

```bash
tail -f logs/backfill.log
```

Press `Ctrl+C` to stop watching (the backfill continues in the background).

**Expected duration:** 30-60 minutes (API rate-limited).
**Expected result:** ~862,000 candle rows across ~431 symbols at 1h and 15m intervals.

Check if it's still running:

```bash
ps aux | grep backfill
```

Check the database after completion:

```bash
sqlite3 data/ohlcv.db "SELECT interval, COUNT(*) FROM ohlcv GROUP BY interval;"
```

---

## 10. Train the Models

### Train the 1h model:

```bash
cd /opt/pa_bot
source venv/bin/activate
nohup python3 scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 60 > logs/train_60.log 2>&1 &
```

Monitor: `tail -f logs/train_60.log`

**Expected duration:** 10-20 minutes.

### Train the 15m model (after 1h finishes):

```bash
nohup python3 scripts/train_model.py --epochs 100 --window 672 --patience 10 --interval 15 > logs/train_15.log 2>&1 &
```

Monitor: `tail -f logs/train_15.log`

### Verify models exist:

```bash
ls -la data/models/
```

You should see:

```
model_final_60.pt
model_final_15.pt
```

---

## 11. Test a Prediction Run

Run a single prediction cycle manually to verify everything works end-to-end:

```bash
cd /opt/pa_bot
source venv/bin/activate
python3 scripts/run_prediction.py --multi-timeframe
```

This should:
- Fetch latest candles
- Run both models
- Combine predictions via ensemble
- Send an email notification to your inbox

Check your email. If you received the prediction alert, everything is working.

---

## 12. Set Up Cron Jobs

Open the crontab editor:

```bash
crontab -e
```

If it asks which editor to use, select `1` (nano).

Add these lines at the bottom:

```cron
# PA Bot: Hourly predictions (every hour at minute 5)
5 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/run_prediction.py --multi-timeframe >> logs/prediction.log 2>&1

# PA Bot: Daily retrain — LSTM + TFT + GBM for all timeframes (midnight UTC)
0 0 * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/daily_retrain.py >> logs/retrain.log 2>&1

# PA Bot: Order book + funding rate snapshots (every 15 minutes)
*/15 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/collect_orderbook.py >> logs/orderbook.log 2>&1

# PA Bot: Coinalyze OI + liquidations + L/S ratio (hourly at :10)
10 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/collect_coinalyze.py >> logs/coinalyze.log 2>&1

# PA Bot: Sentiment — Fear & Greed + CryptoPanic news (hourly at :15)
15 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/collect_sentiment.py >> logs/sentiment.log 2>&1

# PA Bot: Binance cross-exchange — funding rates + OI (hourly at :20)
20 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/collect_binance.py >> logs/binance.log 2>&1

# PA Bot: Weekly HPO — Optuna hyperparameter optimization (Sunday 02:00 UTC)
0 2 * * 0 cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/weekly_hpo.py >> logs/hpo.log 2>&1

# PA Bot: Health check (every 30 minutes)
*/30 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/healthcheck.py --notify >> logs/health.log 2>&1

# Log rotation (weekly, Sunday 06:00)
0 6 * * 0 find /opt/pa_bot/logs -name "*.log" -size +50M -exec truncate -s 0 {} \;
```

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

Verify cron is set:

```bash
crontab -l
```

### What this does:

| Job | Schedule | Purpose |
|---|---|---|
| `run_prediction.py` | Every hour at :05 | Fetch candles, run all models, send alerts |
| `daily_retrain.py` | Midnight UTC daily | Score predictions, tune threshold, retrain LSTM + TFT + GBM |
| `collect_orderbook.py` | Every 15 minutes | Snapshot order book depth + funding rates |
| `collect_coinalyze.py` | Hourly at :10 | Coinalyze OI, liquidations, L/S ratio |
| `collect_sentiment.py` | Hourly at :15 | Fear & Greed Index + CryptoPanic news sentiment |
| `collect_binance.py` | Hourly at :20 | Binance funding rates + open interest (cross-exchange) |
| `weekly_hpo.py` | Sunday 02:00 UTC | Optuna hyperparameter optimization for all 3 models |
| `healthcheck.py` | Every 30 minutes | System health monitoring with email alerts |
| Log rotation | Sunday 06:00 | Truncate logs over 50 MB |

### Why :05 and not :00?

The :05 offset gives the exchange a few minutes to finalize the candle data for the top of the hour. Running at exactly :00 might fetch an incomplete candle.

---

## 13. Verify Everything Is Running

### Check cron is executing:

Wait for the next hour to pass, then:

```bash
tail -50 logs/prediction.log
```

You should see prediction output and an email should arrive.

### Check the database is growing:

```bash
sqlite3 data/ohlcv.db "SELECT COUNT(*) FROM predictions;"
```

This number should increase every hour.

### Check system resources:

```bash
htop
```

Press `q` to exit. Memory usage should be well under 4 GB.

---

## 14. Maintenance & Operations

### View recent logs:

```bash
# Prediction logs
tail -100 logs/prediction.log

# Retrain logs
tail -100 logs/retrain.log
```

### Check prediction accuracy:

```bash
cd /opt/pa_bot && source venv/bin/activate
sqlite3 data/ohlcv.db "
SELECT date(scored_at) as day,
       COUNT(*) as total,
       SUM(was_correct) as correct,
       ROUND(AVG(was_correct)*100, 1) as accuracy_pct
FROM predictions
WHERE was_correct IS NOT NULL
GROUP BY day
ORDER BY day DESC
LIMIT 7;
"
```

### View the SQLite database:

```bash
sqlite3 data/ohlcv.db
```

Useful queries:

```sql
-- Table sizes
SELECT 'ohlcv' as tbl, COUNT(*) FROM ohlcv
UNION ALL SELECT 'predictions', COUNT(*) FROM predictions
UNION ALL SELECT 'accuracy_log', COUNT(*) FROM accuracy_log;

-- Latest predictions
SELECT symbol, direction, magnitude, signal_score
FROM predictions ORDER BY id DESC LIMIT 20;

-- Model accuracy over time
SELECT run_date, total_scored, direction_accuracy
FROM accuracy_log ORDER BY run_date DESC LIMIT 14;
```

Type `.quit` to exit SQLite.

### Log rotation (prevent logs from growing forever):

```bash
# Add to crontab (crontab -e):
0 6 * * 0 find /opt/pa_bot/logs -name "*.log" -size +50M -exec truncate -s 0 {} \;
```

This clears logs over 50 MB every Sunday at 6 AM.

### Database maintenance (monthly):

```bash
cd /opt/pa_bot && source venv/bin/activate
sqlite3 data/ohlcv.db "VACUUM;"
```

---

## 15. Updating the Bot

When code changes are pushed to GitHub:

```bash
cd /opt/pa_bot
source venv/bin/activate
git pull
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

If the feature count or model architecture changed (check the error logs for
`Checkpoint expects X features but model has Y`), you must delete old
checkpoints and retrain:

```bash
# 1. Stop cron jobs while retraining
crontab -l > /tmp/cron_backup.txt
crontab -r

# 2. Delete incompatible checkpoints
rm -f data/models/model_final_*.pt
rm -f data/models/model_final_*.pkl

# 3. Seed new data sources (run each once)
python scripts/collect_sentiment.py      # grabs ~10 days of Fear & Greed history
python scripts/collect_binance.py        # grabs current Binance funding rates + OI

# 4. Train LSTM for both timeframes
nohup python scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 60 >> logs/train_60.log 2>&1 &
# Wait for completion: tail -f logs/train_60.log
# Then train 15m:
nohup python scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 15 >> logs/train_15.log 2>&1 &
# Wait for completion: tail -f logs/train_15.log

# 5. Test a prediction
python scripts/run_prediction.py --multi-timeframe

# 6. Restore cron jobs (see Section 12 for full cron schedule)
crontab /tmp/cron_backup.txt
# Or set up fresh: crontab -e
```

**Note:** `train_model.py` only trains the LSTM. The TFT and GBM models will
be trained automatically on the next `daily_retrain.py` run (midnight UTC).
Until then, predictions use the LSTM alone — the ensemble activates once
TFT/GBM checkpoints exist.

**Note:** New data sources (sentiment, Binance cross-exchange) won't have
deep history on first deploy. The model handles this gracefully — those
features default to `0.0`. After 1-2 weeks of hourly collection via cron,
the daily retrain will start incorporating real sentiment and cross-exchange
signal.

Cron jobs pick up code changes automatically on the next run — no restart needed.

---

## 16. Troubleshooting

### "Permission denied" when SSH-ing

```bash
# Reset root password from Hetzner Cloud Console:
# Server > Rescue tab > Reset Root Password
```

### Cron jobs not running

```bash
# Check cron service is running
systemctl status cron

# Check cron logs
grep CRON /var/log/syslog | tail -20

# Common issue: venv not activated in cron
# Make sure cron uses the full path: /opt/pa_bot/venv/bin/python
```

### Out of memory during training

```bash
# Use less data and smaller batches
python3 scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 60 --rolling-days 30 --batch-size 32
```

### API errors during backfill/prediction

```bash
# Check the logs for specific errors
grep ERROR logs/prediction.log | tail -20

# Common: "Parameter error" = symbol not available on spot API
# The bot auto-filters these, but some slip through. Safe to ignore.
```

### Model checkpoint mismatch after code update

```
RuntimeError: Checkpoint expects 38 features but model has 41. Retrain required.
```

This means the feature set changed. Delete old checkpoints and retrain:

```bash
rm data/models/model_final_*.pt
python3 scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 60
python3 scripts/train_model.py --epochs 100 --window 672 --patience 10 --interval 15
```

### Server rebooted (Hetzner maintenance)

Cron jobs survive reboots automatically. Check everything is running:

```bash
crontab -l                              # cron still configured?
tail -5 logs/prediction.log             # last prediction ran when?
sqlite3 data/ohlcv.db "SELECT MAX(ts) FROM ohlcv;"  # latest candle?
```

---

## 17. Health Check Monitoring

The health check script monitors critical system components and can send email alerts:

```bash
# Manual health check
cd /opt/pa_bot && source venv/bin/activate
python3 scripts/healthcheck.py

# Add to cron for automated alerting (every 30 minutes):
*/30 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/healthcheck.py --notify >> logs/health.log 2>&1
```

What it checks:
- Last prediction < 2 hours ago
- Last retrain < 26 hours ago
- Last order book snapshot < 20 minutes ago
- Database size < 5 GB
- Disk free space > 2 GB

---

## 18. Docker Deployment (Alternative)

For a containerized deployment with TimescaleDB and monitoring:

```bash
cd /opt/pa_bot

# Set up environment
cp .env.example .env
nano .env  # fill in credentials + POSTGRES_PASSWORD

# Start the full stack
docker-compose up -d

# Check container status
docker-compose ps

# View app logs
docker-compose logs -f app
```

This starts: PA Bot app + TimescaleDB (PostgreSQL) + Prometheus + Grafana.

Grafana is available at `http://YOUR_SERVER_IP:3000` (default password: admin).

---

## 19. Running a Backtest

Validate the model's performance on historical data:

```bash
cd /opt/pa_bot && source venv/bin/activate

# Single-pass backtest (last 90 days)
python3 scripts/run_backtest.py --interval 60 --days 90 --capital 10000

# Walk-forward backtest (realistic out-of-sample evaluation)
python3 scripts/run_backtest.py --walk-forward --folds 5
```

---

## Quick Reference

| Task | Command |
|---|---|
| SSH into server | `ssh root@YOUR_SERVER_IP` |
| Activate venv | `cd /opt/pa_bot && source venv/bin/activate` |
| Check prediction logs | `tail -50 logs/prediction.log` |
| Check retrain logs | `tail -50 logs/retrain.log` |
| Check health | `python3 scripts/healthcheck.py` |
| Manual prediction run | `python3 scripts/run_prediction.py --multi-timeframe` |
| Manual retrain | `python3 scripts/daily_retrain.py` |
| Run backtest | `python3 scripts/run_backtest.py --walk-forward` |
| Pull latest code | `git pull` |
| View cron schedule | `crontab -l` |
| Edit cron schedule | `crontab -e` |
| Database shell | `sqlite3 data/ohlcv.db` |
| Docker start | `docker-compose up -d` |
| Docker logs | `docker-compose logs -f app` |
| Check disk space | `df -h` |
| Check memory usage | `free -h` |
| Check running processes | `htop` |
