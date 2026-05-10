# PA Bot — Production Deployment Guide

Complete instructions for deploying to a Hetzner Cloud VPS.
Covers fresh install and code updates.

---

## Fresh Install

### 1. Create Server

1. [Hetzner Cloud Console](https://console.hetzner.cloud) → **Add Server**
2. Ubuntu 24.04 / **CX22** (2 vCPU, 4 GB RAM, ~$4.50/mo) / Ashburn or nearest
3. Note the **IP address** and check email for **root password**

### 2. Server Setup

```bash
ssh root@YOUR_SERVER_IP

# System packages
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git sqlite3

# Clone repo
cd /opt
git clone https://github.com/DeathByRamen/price_action_bot.git pa_bot
cd /opt/pa_bot
git config credential.helper store

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Environment Variables

```bash
nano /opt/pa_bot/.env
```

Paste (only include channels you use):

```
SMTP_PASSWORD=your_gmail_app_password
COINALYZE_API_KEY=your_coinalyze_key
CRYPTOPANIC_API_KEY=your_cryptopanic_key
DISCORD_WEBHOOK_URL=your_discord_webhook
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

At minimum you need `SMTP_PASSWORD`. Save with `Ctrl+O`, `Enter`, `Ctrl+X`.

### 4. Backfill Historical Data

```bash
cd /opt/pa_bot && source venv/bin/activate
mkdir -p logs

# OHLCV candles (30-60 min, rate-limited)
nohup python scripts/backfill_data.py --all-timeframes >> logs/backfill.log 2>&1 &
tail -f logs/backfill.log  # Ctrl+C to stop watching

# Coinalyze derivatives (OI, liquidations, L/S ratio)
nohup python scripts/backfill_coinalyze.py >> logs/backfill_coinalyze.log 2>&1 &
tail -f logs/backfill_coinalyze.log
```

Wait for both to finish, then seed the new sources:

```bash
python scripts/collect_sentiment.py    # ~10 days of Fear & Greed
python scripts/collect_binance.py      # current Binance snapshot (may fail if geo-blocked — not critical)
python scripts/collect_orderbook.py    # first order book + funding rate snapshot
```

### 5. Verify Data Before Training

```bash
sqlite3 data/ohlcv.db <<'SQLEOF'
.headers on
.mode column
SELECT 'ohlcv' AS tbl, COUNT(*) AS rows FROM ohlcv
UNION ALL SELECT 'coinalyze_oi', COUNT(*) FROM coinalyze_oi
UNION ALL SELECT 'coinalyze_liq', COUNT(*) FROM coinalyze_liquidations
UNION ALL SELECT 'coinalyze_ls', COUNT(*) FROM coinalyze_long_short
UNION ALL SELECT 'order_book', COUNT(*) FROM order_book_snapshots
UNION ALL SELECT 'funding_rate', COUNT(*) FROM funding_rate_snapshots
UNION ALL SELECT 'fear_greed', COUNT(*) FROM fear_greed_index
UNION ALL SELECT 'predictions', COUNT(*) FROM predictions;

SELECT 'close<=0' AS issue, COUNT(*) FROM ohlcv WHERE close <= 0
UNION ALL SELECT 'high<low', COUNT(*) FROM ohlcv WHERE high < low
UNION ALL SELECT 'duplicates', COUNT(*) FROM (
    SELECT symbol, ts, interval, COUNT(*) AS c FROM ohlcv GROUP BY symbol, ts, interval HAVING c > 1);

SELECT interval, COUNT(DISTINCT symbol) AS symbols, COUNT(*) AS rows,
       MIN(ts) AS earliest, MAX(ts) AS latest FROM ohlcv GROUP BY interval;
SQLEOF
```

**Expected:** OHLCV has 200k+ rows, Coinalyze tables populated, zero bad candles, zero duplicates.

### 6. Train Models

`train_model.py` trains the **LSTM only**. TFT and GBM are trained by `daily_retrain.py`.

```bash
# Train 1h model (10-20 min)
nohup python scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 60 >> logs/train_60.log 2>&1 &
tail -f logs/train_60.log
# Wait for "Training complete" message

# Train 15m model (10-20 min)
nohup python scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 15 >> logs/train_15.log 2>&1 &
tail -f logs/train_15.log

# Verify checkpoints exist
ls -la data/models/model_final_*.pt
```

### 7. Test Prediction

```bash
python scripts/run_prediction.py --multi-timeframe
```

Check your email for the alert. If it arrives, everything works.

### 8. Set Up Cron Jobs

```bash
crontab -e
```

Select nano (option 1), paste this **entire block**:

```cron
# Hourly predictions (at :05 to let candles finalize)
5 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/run_prediction.py --multi-timeframe >> logs/prediction.log 2>&1

# Daily retrain — LSTM + TFT + GBM for all timeframes (midnight UTC)
0 0 * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/daily_retrain.py >> logs/retrain.log 2>&1

# Order book + funding rate snapshots (every 15 min)
*/15 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/collect_orderbook.py >> logs/orderbook.log 2>&1

# Coinalyze OI + liquidations + L/S ratio (hourly at :10)
10 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/collect_coinalyze.py >> logs/coinalyze.log 2>&1

# Sentiment — Fear & Greed + CryptoPanic news (hourly at :15)
15 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/collect_sentiment.py >> logs/sentiment.log 2>&1

# Binance cross-exchange — funding rates + OI (hourly at :20)
20 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/collect_binance.py >> logs/binance.log 2>&1

# Weekly HPO — Optuna hyperparameter optimization (Sunday 02:00 UTC)
0 2 * * 0 cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/weekly_hpo.py >> logs/hpo.log 2>&1

# Health check (every 30 min)
*/30 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/healthcheck.py --notify >> logs/health.log 2>&1

# Log rotation (Sunday 06:00)
0 6 * * 0 find /opt/pa_bot/logs -name "*.log" -size +50M -exec truncate -s 0 {} \;
```

Save: `Ctrl+O`, `Enter`, `Ctrl+X`. Verify: `crontab -l`.

| Job | Schedule | Purpose |
|-----|----------|---------|
| `run_prediction.py` | Hourly :05 | Run models, send alerts |
| `daily_retrain.py` | Midnight | Score predictions, retrain LSTM + TFT + GBM |
| `collect_orderbook.py` | Every 15 min | Order book depth + funding rates |
| `collect_coinalyze.py` | Hourly :10 | OI, liquidations, L/S ratio |
| `collect_sentiment.py` | Hourly :15 | Fear & Greed + news sentiment |
| `collect_binance.py` | Hourly :20 | Cross-exchange funding rates + OI |
| `weekly_hpo.py` | Sunday 02:00 | Hyperparameter optimization |
| `healthcheck.py` | Every 30 min | System health + email alerts |

### 9. Verify

Wait for the next hour, then:

```bash
tail -50 logs/prediction.log          # prediction ran?
sqlite3 data/ohlcv.db "SELECT COUNT(*) FROM predictions;"  # predictions stored?
crontab -l                            # cron configured?
```

---

## Updating After Code Changes

Run these steps **every time** you push code updates to GitHub.

### If features/model architecture did NOT change:

```bash
ssh root@YOUR_SERVER_IP
cd /opt/pa_bot && source venv/bin/activate
git pull
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Cron picks up code changes automatically — no restart needed.

### If features/model architecture DID change:

You'll know because prediction logs show `Checkpoint expects X features but model has Y`.

```bash
ssh root@YOUR_SERVER_IP
cd /opt/pa_bot && source venv/bin/activate

# 1. Stop cron
crontab -l > /tmp/cron_backup.txt
crontab -r

# 2. Pull code and install deps
git pull
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Delete incompatible checkpoints and stale predictions
rm -f data/models/model_final_*.pt
rm -f data/models/model_final_*.pkl
sqlite3 data/ohlcv.db "DELETE FROM predictions;"

# 4. Seed any new data sources
python scripts/collect_sentiment.py
python scripts/collect_binance.py
python scripts/collect_orderbook.py

# 5. Train LSTM for both timeframes
nohup python scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 60 >> logs/train_60.log 2>&1 &
tail -f logs/train_60.log
# Wait for "Training complete", then:
nohup python scripts/train_model.py --epochs 100 --window 168 --patience 10 --interval 15 >> logs/train_15.log 2>&1 &
tail -f logs/train_15.log

# 6. Test prediction
python scripts/run_prediction.py --multi-timeframe

# 7. Restore cron
crontab /tmp/cron_backup.txt
crontab -l  # verify
```

**Note:** `train_model.py` only trains the LSTM. TFT and GBM models are
trained automatically by `daily_retrain.py` at midnight. Predictions use
LSTM-only until those checkpoints exist, then the ensemble activates.

**Note:** New data sources won't have deep history on first deploy. Features
default to `0.0` until the hourly cron collectors accumulate 1-2 weeks of
data. The model handles this gracefully.

---

## Data Quality Check

Run before training or whenever you suspect issues:

```bash
sqlite3 data/ohlcv.db <<'SQLEOF'
.headers on
.mode column

-- Row counts
SELECT 'ohlcv' AS tbl, COUNT(*) AS rows FROM ohlcv
UNION ALL SELECT 'predictions', COUNT(*) FROM predictions
UNION ALL SELECT 'coinalyze_oi', COUNT(*) FROM coinalyze_oi
UNION ALL SELECT 'coinalyze_liq', COUNT(*) FROM coinalyze_liquidations
UNION ALL SELECT 'coinalyze_ls', COUNT(*) FROM coinalyze_long_short
UNION ALL SELECT 'order_book', COUNT(*) FROM order_book_snapshots
UNION ALL SELECT 'funding_rate', COUNT(*) FROM funding_rate_snapshots
UNION ALL SELECT 'fear_greed', COUNT(*) FROM fear_greed_index;

-- Bad candles (all should be 0)
SELECT 'close<=0' AS issue, COUNT(*) AS count FROM ohlcv WHERE close <= 0
UNION ALL SELECT 'high<low', COUNT(*) FROM ohlcv WHERE high < low
UNION ALL SELECT 'volume<0', COUNT(*) FROM ohlcv WHERE volume < 0
UNION ALL SELECT 'duplicates', COUNT(*) FROM (
    SELECT symbol, ts, interval, COUNT(*) AS c FROM ohlcv GROUP BY symbol, ts, interval HAVING c > 1);

-- Data freshness
SELECT 'ohlcv_60m' AS source, MAX(ts) AS latest,
       ROUND((julianday('now') - julianday(MAX(ts))) * 24, 1) AS hours_ago
    FROM ohlcv WHERE interval = '60'
UNION ALL SELECT 'ohlcv_15m', MAX(ts),
       ROUND((julianday('now') - julianday(MAX(ts))) * 24, 1) FROM ohlcv WHERE interval = '15'
UNION ALL SELECT 'coinalyze', MAX(ts),
       ROUND((julianday('now') - julianday(MAX(ts))) * 24, 1) FROM coinalyze_oi
UNION ALL SELECT 'order_book', MAX(ts),
       ROUND((julianday('now') - julianday(MAX(ts))) * 24, 1) FROM order_book_snapshots
UNION ALL SELECT 'fear_greed', MAX(ts),
       ROUND((julianday('now') - julianday(MAX(ts))) * 24, 1) FROM fear_greed_index;

-- BTC candle gap check (expect ~48 for last 48h)
SELECT COUNT(*) AS btc_60m_last_48h FROM ohlcv
WHERE symbol = 'BTCUSDT' AND interval = '60' AND ts >= datetime('now', '-48 hours');
SQLEOF
```

| Check | Healthy | Problem |
|-------|---------|---------|
| Bad candles | All 0 | Corrupt data — investigate |
| Freshness | < 2 hours | > 24 hours = collection stopped |
| BTC gap check | ~48 | < 40 = missing candles |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Checkpoint expects X features but model has Y` | Delete checkpoints + retrain (see update steps above) |
| Cron not running | `systemctl status cron` / check `grep CRON /var/log/syslog \| tail -20` |
| Out of memory | Add `--rolling-days 30 --batch-size 32` to train command |
| Binance 451 error | Geo-blocked on your server's IP — not critical, those features stay 0.0 |
| No email received | Check `SMTP_PASSWORD` in `.env` / check `logs/prediction.log` for errors |
| Retrain log ends with `Killed` / no digest email | OOM: use `--lstm-only` in cron so only LSTM trains (skips TFT/GBM) |
| SSH permission denied | Reset root password from Hetzner Console → Rescue tab |
| Server rebooted | Cron survives reboots. Verify: `crontab -l` and `tail -5 logs/prediction.log` |

---

## Quick Reference

```bash
ssh root@YOUR_SERVER_IP
cd /opt/pa_bot && source venv/bin/activate
```

| Task | Command |
|------|---------|
| View prediction logs | `tail -50 logs/prediction.log` |
| View retrain logs | `tail -50 logs/retrain.log` |
| Manual prediction | `python scripts/run_prediction.py --multi-timeframe` |
| Manual retrain | `python scripts/daily_retrain.py` |
| Health check | `python scripts/healthcheck.py` |
| Pull latest code | `git pull` |
| View cron | `crontab -l` |
| Edit cron | `crontab -e` |
| DB shell | `sqlite3 data/ohlcv.db` |
| Check disk | `df -h` |
| Check memory | `free -h` |
| Check processes | `htop` |
| Weekly accuracy | See query below |

```sql
-- Weekly prediction accuracy
SELECT date(scored_at) AS day, COUNT(*) AS total,
       SUM(was_correct) AS correct,
       ROUND(AVG(was_correct)*100, 1) AS pct
FROM predictions WHERE was_correct IS NOT NULL
GROUP BY day ORDER BY day DESC LIMIT 7;
```
