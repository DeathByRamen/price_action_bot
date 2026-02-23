# PA Bot — Phase 2 Roadmap: Order Book Integration & Beyond

This document captures everything that needs to be implemented after the base system is running in production. It's designed to survive context loss — every detail needed to continue development is here.

---

## Table of Contents

1. [Current State (What Exists Today)](#1-current-state-what-exists-today)
2. [Phase 2A — Order Book Feature Extraction](#2-phase-2a--order-book-feature-extraction)
3. [Phase 2B — Model Integration](#3-phase-2b--model-integration)
4. [Phase 2C — Interval-Specific FLAT Threshold Persistence](#4-phase-2c--interval-specific-flat-threshold-persistence)
5. [Phase 2D — Correlation Guard Integration](#5-phase-2d--correlation-guard-integration)
6. [Phase 2E — Data Quality Monitoring](#6-phase-2e--data-quality-monitoring)
7. [Phase 2F — WebSocket Upgrade (Optional)](#7-phase-2f--websocket-upgrade-optional)
8. [Phase 2G — Futures-Specific Features](#8-phase-2g--futures-specific-features)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Priority & Timeline](#10-priority--timeline)
11. [Key File Locations](#11-key-file-locations)
12. [Technical Decisions Already Made](#12-technical-decisions-already-made)

---

## 1. Current State (What Exists Today)

### Order Book Collection (Phase 1 — COMPLETE)

The infrastructure to collect order book snapshots is built and deployed:

- **API method:** `BitunixClient.get_market_depth()` in `src/api/bitunix_client.py` — calls `GET /api/spot/v1/market/depth` with `symbol` and `precision` parameters
- **Collector method:** `DataCollector.fetch_order_book_snapshots()` in `src/data/collector.py` — fetches depth for all ~431 symbols with bounded concurrency, parses top 15 bid/ask levels, pre-computes spread, mid_price, and imbalance
- **Storage:** `order_book_snapshots` table in SQLite with columns:
  - `symbol`, `ts` (unique together)
  - `bid_prices`, `bid_vols`, `ask_prices`, `ask_vols` (JSON arrays of top 15 levels)
  - `spread` (best_ask - best_bid)
  - `mid_price` ((best_ask + best_bid) / 2)
  - `imbalance` (total_bid_vol / (total_bid_vol + total_ask_vol))
- **Cron script:** `scripts/collect_orderbook.py` — runs every 15 minutes via cron
- **Storage estimate:** ~41,000 rows/day, ~20 MB/day, ~600 MB/month

### Prediction System (Phase 1 — COMPLETE)

- 41 technical indicator features (trend, momentum, volatility, volume, anti-pump/dump)
- Dual-head LSTM with feature gating, temporal attention, residual connection
- Multi-timeframe ensemble (1h + 15m models combined via log-odds)
- Adaptive feedback loop: FLAT threshold auto-tuning + per-symbol sample weighting
- Validation gate with auto-rollback on model regression
- Auto-scaling FLAT threshold by interval (0.5% for 1h, 0.125% for 15m)
- Hourly predictions + daily retrain on cron
- Email notifications

### What We're Waiting For

Order book snapshots need to accumulate for **2-4 weeks** before Phase 2A can begin. This gives:
- ~14-28 days x 96 snapshots/day x 431 symbols = 580K-1.16M rows
- Enough history to compute rolling features (velocity, wall persistence, etc.)
- Enough data for the model to learn meaningful patterns

**Target start date for Phase 2A: approximately 2-4 weeks after cron deployment.**

---

## 2. Phase 2A — Order Book Feature Extraction

### What to Build

Create a new module `src/features/orderbook_features.py` that takes raw order book snapshots and computes features aligned to OHLCV candle timestamps.

### Features to Extract (8 new indicators)

| Feature | Formula | What It Detects |
|---|---|---|
| `ob_imbalance` | `total_bid_vol / (total_bid_vol + total_ask_vol)` | Buy/sell pressure. >0.6 = buy heavy, <0.4 = sell heavy |
| `ob_imbalance_delta` | Change in imbalance over last 4 snapshots (1 hour) | Is buy pressure building or fading? |
| `ob_spread_pct` | `spread / mid_price` | Market maker confidence. Tight = stable, wide = uncertain |
| `ob_bid_wall_ratio` | `max(bid_vols) / mean(bid_vols)` | Detects large bid walls. >3.0 = significant support |
| `ob_ask_wall_ratio` | `max(ask_vols) / mean(ask_vols)` | Detects large ask walls. >3.0 = significant resistance |
| `ob_bid_wall_dist` | `(mid_price - wall_bid_price) / ATR` | How far below price is the nearest bid wall (in ATR units) |
| `ob_ask_wall_dist` | `(wall_ask_price - mid_price) / ATR` | How far above price is the nearest ask wall (in ATR units) |
| `ob_depth_concentration` | Herfindahl index of volume across levels | Is liquidity concentrated (high) or spread evenly (low)? |

### Implementation Details

```python
# src/features/orderbook_features.py

async def compute_orderbook_features(
    storage: Storage,
    symbol: str,
    candle_timestamps: list[str],
    lookback_snapshots: int = 4,  # 4 x 15min = 1 hour of OB history per candle
) -> pd.DataFrame:
    """
    For each candle timestamp, find the closest order book snapshot(s)
    and compute features.

    Returns a DataFrame indexed by candle timestamp with OB feature columns.
    """
```

**Alignment strategy:** For each OHLCV candle timestamp, find the most recent order book snapshot that occurred at or before that timestamp. If multiple snapshots fall within one candle period, use the latest one. For delta features (like `ob_imbalance_delta`), look back `lookback_snapshots` snapshots from the aligned one.

**Handling missing data:** If no order book snapshot exists for a candle (e.g., early in collection), fill with neutral values:
- `ob_imbalance` = 0.5 (neutral)
- `ob_imbalance_delta` = 0.0 (no change)
- `ob_spread_pct` = NaN (will be handled by Z-score normalization)
- Wall ratios = 1.0 (no wall)
- Wall distances = NaN
- `ob_depth_concentration` = 0.0

### Files to Modify

1. **Create** `src/features/orderbook_features.py` — the feature extraction module
2. **Modify** `src/features/indicators.py`:
   - Add the 8 new OB feature names to `get_feature_columns()`
   - Update `MAX_WARMUP_PERIODS` if needed
3. **Modify** `src/pipeline.py` — call `compute_orderbook_features()` and merge with OHLCV data before prediction
4. **Modify** `scripts/train_model.py` — load OB features during training data preparation
5. **Modify** `scripts/daily_retrain.py` — same as above for daily retrain

---

## 3. Phase 2B — Model Integration

### The Problem

Adding 8 new features means the model now expects 49 inputs instead of 41. The existing trained checkpoint (`model_final_60.pt`) has `num_features=41` baked into its weights. Loading it with 49 features will raise a `RuntimeError`.

### The Solution

**Option A: Full retrain (recommended for first integration)**
1. Delete existing checkpoints
2. Retrain from scratch with all 49 features
3. The model's feature gate will automatically learn how much weight to give OB features vs. technical indicators

**Option B: Transfer learning (for subsequent retrains)**
1. Load the old 41-feature model
2. Expand the `feature_gate` input layer from 41 to 49 with zero-initialized weights for the new columns
3. Expand `input_proj` similarly
4. Fine-tune for a few epochs — existing feature weights are preserved, OB features are learned from scratch

### Implementation for Option B

```python
# In trainer.py or a new utility

def expand_model_features(
    old_checkpoint_path: str,
    new_num_features: int,
    old_num_features: int = 41,
) -> dict:
    """
    Load an old checkpoint and expand its weight matrices to accommodate
    new features. New feature weights are zero-initialized.
    """
    checkpoint = torch.load(old_checkpoint_path)
    state = checkpoint["model_state_dict"]
    
    # Expand feature_gate.0.weight: (4*old_F, old_F) -> (4*new_F, new_F)
    # Expand input_proj.weight: (H, old_F) -> (H, new_F)
    # ... etc
```

### Validation

After integration, run permutation importance on the new OB features to verify they're contributing. If any OB feature has near-zero importance after 1 week, it's not useful and should be removed to reduce noise.

### Checkpoint Versioning

The existing checkpoint metadata (`num_features`, `feature_cols_hash`) will automatically detect the mismatch and force a retrain if needed. No code changes required for safety — it's already built in.

---

## 4. Phase 2C — Interval-Specific FLAT Threshold Persistence

### The Problem

Currently, the FLAT threshold auto-tuner in `daily_retrain.py` computes a single threshold from all scored predictions. But 1h and 15m predictions need very different thresholds (0.5% vs 0.125%). The auto-tuner should track and tune thresholds independently per interval.

### What to Build

1. **New table or column** in `accuracy_log` to store the tuned threshold per interval
2. **Modify** `src/scoring/adaptive.py` → `compute_optimal_threshold()` to accept an `interval` parameter and only use scored predictions from that interval
3. **Modify** `scripts/daily_retrain.py` to tune thresholds per-interval before training each timeframe
4. **Modify** `config/settings.yaml` to store `current_flat_threshold` per interval instead of globally

### Current Workaround

The auto-scaling formula (`threshold * interval_mins / 60`) in `train_model.py` and `daily_retrain.py` handles the initial case, but the auto-tuner doesn't know about intervals yet. Once predictions accumulate for a week, the tuner should take over with data-driven thresholds.

---

## 5. Phase 2D — Correlation Guard Integration

### The Problem

`check_feature_correlation()` exists in `src/features/indicators.py` but is never called anywhere in the pipeline. Highly correlated features waste model capacity and can destabilize training.

### What to Build

1. **Modify** `scripts/train_model.py` — call `check_feature_correlation()` after loading training data and log warnings
2. **Modify** `scripts/daily_retrain.py` — same, and include warnings in the daily digest
3. **Consider** auto-dropping features with Pearson r > 0.98 (not just warning)

### Implementation

```python
# In train_model.py, after loading symbol_data:

from src.features.indicators import check_feature_correlation

# Use data from a few symbols to check correlations
sample_df = next(iter(symbol_data.values()))
correlated_pairs = check_feature_correlation(sample_df, threshold=0.95)
if correlated_pairs:
    logging.warning("Found %d highly correlated feature pairs", len(correlated_pairs))
```

This is especially important after adding OB features, as some may correlate with existing volume indicators.

---

## 6. Phase 2E — Data Quality Monitoring

### The Problem

Bad data flows silently into training and prediction. Examples:
- A symbol has zero volume for 24+ hours (delisted or broken feed)
- Price drops to 0 (API glitch)
- Timestamp gaps (missed candles)
- Order book snapshots returning empty data

### What to Build

Create `src/data/quality.py` with checks that run during:

1. **Backfill/gap-fill** — validate candles before insertion:
   - `close > 0` and `volume >= 0`
   - Timestamp is within expected range
   - No duplicate timestamps per symbol/interval

2. **Training data load** — flag and optionally exclude bad symbols:
   - Symbols with >10% zero-volume candles in the training window
   - Symbols with price jumps >50% in a single candle (likely data error)
   - Symbols with fewer than `MIN_CANDLES` after indicator computation

3. **Daily digest** — include a data quality section:
   - Number of symbols excluded for quality reasons
   - Symbols with stale data (no new candles in 24h)
   - Order book collection success rate

### Files to Create/Modify

1. **Create** `src/data/quality.py` — data quality check functions
2. **Modify** `scripts/daily_retrain.py` — run quality checks and include in digest
3. **Modify** `src/data/collector.py` — validate candles before storage

---

## 7. Phase 2F — WebSocket Upgrade (Optional)

### When to Consider

Only if 15-minute REST snapshots prove insufficient. Signs this is needed:
- Permutation importance shows OB features have low importance despite being theoretically useful
- The 15-minute granularity is too coarse to capture wall formation/destruction dynamics
- You want sub-minute order flow data

### What to Build

A persistent WebSocket daemon that:
1. Connects to `wss://fapi.bitunix.com/public/`
2. Subscribes to `depth_book15` for top symbols (max 300 per connection)
3. Buffers incoming updates
4. Flushes snapshots to SQLite every N seconds
5. Handles reconnection, heartbeat pings, and error recovery
6. Runs as a systemd service (not cron)

### BitUnix WebSocket API Details

```json
// Subscribe
{
    "op": "subscribe",
    "args": [{"symbol": "BTCUSDT", "ch": "depth_book15"}]
}

// Ping (required to keep connection alive)
{
    "op": "ping",
    "ping": 1732519687
}

// Incoming data format
{
    "ch": "depth_book15",
    "symbol": "BTCUSDT",
    "ts": 1732178884994,
    "data": {
        "b": [["7403.89", "0.002"], ...],  // bids: [price, quantity]
        "a": [["7405.96", "3.340"], ...]   // asks: [price, quantity]
    }
}
```

- Domain: `wss://fapi.bitunix.com/public/`
- Max 300 channel subscriptions per connection
- For 431 symbols, need 2 connections
- Channel options: `depth_book1` (1 level), `depth_book5` (5 levels), `depth_book15` (15 levels), `depth_books` (full snapshot + deltas)

### Additional Infrastructure Needed

- `scripts/orderbook_daemon.py` — the WebSocket daemon
- `systemd` service file for auto-restart
- Monitoring to detect if the daemon dies

**Recommendation:** Don't build this unless REST snapshots prove insufficient. The complexity isn't justified until we know OB features have value.

---

## 8. Phase 2G — Futures-Specific Features

### Status: PARTIALLY COMPLETE (Funding Rate Collection)

### The Problem

The system analyzes futures markets but uses only spot-derived data. The most predictive signals for crypto futures are:

1. **Funding rate** — The periodic payment between longs and shorts. High positive funding = overcrowded longs (bearish). High negative = overcrowded shorts (bullish).
2. **Open interest** — Total outstanding futures contracts. Rising OI + rising price = strong trend. Rising OI + falling price = shorts building.
3. **Liquidation data** — Cascade liquidations cause violent moves. A cluster of long liquidations creates selling pressure.

### BitUnix API Availability (Confirmed)

- **Funding rate**: `GET /api/v1/futures/market/funding_rate` — **AVAILABLE** (public, no auth). Returns: `symbol`, `markPrice`, `lastPrice`, `fundingRate`, `nextFundingTime`, `fundingInterval` (typically 8h).
- **Open interest**: **NOT AVAILABLE** — no dedicated endpoint found. Not included in tickers response.
- **Liquidation data**: **NOT AVAILABLE** — no public endpoint.
- **WebSocket `price` channel**: Pushes `mp` (mark price), `ip` (index price), `fr` (funding rate), `ft` / `nft` (settlement times) in real-time.

### What Has Been Built (Phase 2G Collection — COMPLETE)

1. **`BitunixClient.get_funding_rate(symbol)`** — REST client method for per-symbol funding rate.
2. **`BitunixClient.get_all_funding_rates()`** — Bulk fetch attempt (falls back to per-symbol).
3. **`DataCollector.fetch_funding_rate_snapshots(symbols)`** — Concurrent funding rate collection for all symbols, returns rows ready for DB insertion.
4. **`Storage` table `funding_rate_snapshots`** — Stores `(symbol, ts, funding_rate, mark_price, last_price, next_funding_ts, funding_interval_hours)` with `UNIQUE(symbol, ts)`.
5. **`Storage.insert_funding_rate_snapshots()`** / **`get_funding_rate_snapshots()`** — Bulk insert and query methods.
6. **`scripts/collect_orderbook.py`** — Updated to collect **both** order book depth **and** funding rate snapshots in parallel, on the same 15-minute cron schedule.

### What Remains (Phase 2G Feature Extraction — TODO)

When enough funding rate data has been accumulated (2-4 weeks), extract these features:

- `funding_rate` — raw funding rate (already bounded, typically -0.1% to +0.1%)
- `funding_rate_zscore` — Z-score of funding rate vs. 7-day rolling average
- `funding_rate_momentum` — rate of change of funding rate between snapshots
- `mark_spot_divergence` — (mark_price - last_price) / last_price — captures market sentiment divergence

### Not Available on BitUnix

- `open_interest_change` — % change in OI over last 4/24 hours — **endpoint does not exist**
- `long_short_ratio` — ratio of long vs short positions — **endpoint does not exist**
- Liquidation cascade detection — **no public endpoint**

These could be sourced from third-party providers (CoinGlass API, Coinalyze) in a future phase if high-value.

---

## 9. Architecture Diagram

### Current System (Phase 1)

```
                    OHLCV Candles (SQLite)
                           │
                           ▼
                 ┌───────────────────┐
                 │  41 Technical     │
                 │  Indicators       │
                 │  (indicators.py)  │
                 └────────┬──────────┘
                          │
                          ▼
                 ┌───────────────────┐
                 │  LSTM Model       │
                 │  (architecture.py)│
                 │  Feature Gate     │
                 │  Temporal Attn    │
                 │  Dual Heads       │
                 └────────┬──────────┘
                          │
                          ▼
                 ┌───────────────────┐
                 │  Predictions      │
                 │  UP/FLAT/DOWN     │
                 │  + Magnitude      │
                 └──────────────────┘
```

### Phase 2 System (After Order Book + Funding Rate Integration)

```
  OHLCV Candles     Order Book Snapshots     Funding Rate Snapshots
  (SQLite)          (SQLite)                 (SQLite)
       │                    │                        │
       ▼                    ▼                        ▼
  ┌──────────────┐  ┌────────────────────┐  ┌────────────────────┐
  │ 41 Technical │  │ 8 Order Book       │  │ 4 Funding Rate     │
  │ Indicators   │  │ Features           │  │ Features           │
  │ (indicators) │  │ (orderbook_feats)  │  │ (funding_feats)    │
  └──────┬───────┘  └────────┬───────────┘  └────────┬───────────┘
         │                   │                       │
         └───────────┬───────┴───────────────────────┘
                     │
                     ▼ (~53 features total)
            ┌───────────────────┐
            │  LSTM Model       │
            │  Feature Gate     │◄── Learns which features matter
            │  Temporal Attn    │    (OB near support/resistance,
            │  Dual Heads       │     funding near extremes,
            │  Residual Conn    │     technical in trends)
            └────────┬──────────┘
                     │
                     ▼
            ┌───────────────────┐
            │  Predictions      │
            │  Better timed     │
            │  Liquidity-aware  │
            │  Sentiment-aware  │
            └──────────────────┘
```

---

## 10. Priority & Timeline

| Phase | What | When | Effort | Impact |
|---|---|---|---|---|
| **2A** | Order book feature extraction | After 2-4 weeks of OB data collection | 1-2 days | High |
| **2B** | Model integration (retrain with 49 features) | Immediately after 2A | 0.5 days | High |
| **2C** | Interval-specific threshold persistence | After 1 week of scored predictions | 0.5 days | Medium |
| **2D** | Correlation guard integration | Any time | 1 hour | Low-Medium |
| **2E** | Data quality monitoring | Any time | 0.5 days | Medium |
| **2F** | WebSocket upgrade | Only if REST is insufficient | 2-3 days | Uncertain |
| **2G** | Futures-specific features — **Collection DONE**, feature extraction TODO | After 2-4 weeks of funding rate data | 0.5 days | High |

**Recommended order:** 2D (quick win) → 2C → 2E → 2A + 2G-features → 2B → 2F

---

## 11. Key File Locations

| File | Purpose |
|---|---|
| `src/api/bitunix_client.py` | BitUnix REST API client (includes `get_market_depth()`, `get_funding_rate()`, `get_all_funding_rates()`) |
| `src/data/storage.py` | SQLite backend (includes `order_book_snapshots` + `funding_rate_snapshots` tables and CRUD methods) |
| `src/data/collector.py` | Data fetching orchestration (includes `fetch_order_book_snapshots()`, `fetch_funding_rate_snapshots()`) |
| `src/features/indicators.py` | Technical indicator computation (41 features, `get_feature_columns()`) |
| `src/model/architecture.py` | LSTM model (feature gate, temporal attention, dual heads) |
| `src/model/dataset.py` | Training dataset with per-symbol isolation, Z-score normalization |
| `src/model/trainer.py` | Training loop, checkpointing, temperature calibration |
| `src/model/predictor.py` | Inference engine with batch prediction |
| `src/model/ensemble.py` | Multi-timeframe combination (log-odds, adaptive weights) |
| `src/pipeline.py` | Main prediction pipeline orchestrator |
| `src/scoring/accuracy.py` | Prediction scoring against actual outcomes |
| `src/scoring/adaptive.py` | FLAT threshold auto-tuning + sample weighting |
| `scripts/collect_orderbook.py` | Cron script for OB + funding rate snapshot collection |
| `scripts/run_prediction.py` | Cron script for hourly predictions |
| `scripts/daily_retrain.py` | Cron script for daily retrain + feedback loop |
| `scripts/train_model.py` | Manual training script |
| `scripts/backfill_data.py` | Historical data backfill |
| `config/settings.yaml` | All configuration (model, pipeline, notifications, timeframes) |
| `docs/WORKFLOW.md` | Full system workflow documentation |
| `docs/DEPLOYMENT.md` | Production deployment guide (Hetzner) |

---

## 12. Technical Decisions Already Made

These decisions were made during Phase 1 development and should be maintained:

1. **REST over WebSocket for OB collection** — simpler, cron-friendly, 15-min granularity is sufficient for structural features. Upgrade to WebSocket only if proven insufficient.

2. **Store raw bid/ask arrays as JSON** — flexible for future feature extraction. Pre-computed metrics (spread, mid_price, imbalance) stored as columns for fast querying.

3. **15 levels of depth** — captures enough of the book to detect walls and concentration without excessive storage.

4. **Auto-scaling FLAT threshold** — `threshold * (interval_mins / 60)`. A 0.5% move in 1 hour is equivalent to a 0.125% move in 15 minutes. The adaptive tuner will further refine per-interval thresholds once enough data accumulates.

5. **Feature gate handles feature selection** — when OB features are added, the model's learned sigmoid gate will automatically discover how much weight to give them vs. technical indicators. No manual feature selection needed.

6. **Per-window Z-score normalization** — each 168-step window is independently normalized. This means OB features will be on the same scale as technical features regardless of their raw range. No additional normalization needed.

7. **Validation gate prevents regression** — when retraining with new features, the gate ensures the new model must outperform the old one before deployment. If OB features hurt performance (unlikely but possible), the old model is automatically preserved.

8. **Checkpoint metadata for compatibility** — `num_features` and `feature_cols_hash` are stored in every checkpoint. Adding OB features will trigger a clean retrain rather than loading incompatible weights.

9. **Funding rate collected alongside order book** — Both are fetched in parallel on the same 15-minute cron. Funding rate is the only futures-specific data point available from BitUnix (no OI or liquidation endpoints exist). Collecting early maximizes historical depth for feature extraction in Phase 2G.

10. **Funding rate stored as raw snapshots** — `(symbol, ts, funding_rate, mark_price, last_price, next_funding_ts, funding_interval_hours)`. Feature extraction (z-scores, momentum, mark-spot divergence) will be computed at model integration time, keeping raw data flexible.

---

## Appendix: How to Verify Order Book Collection Is Working

After cron is set up, run these checks:

```bash
# How many snapshots collected so far
sqlite3 /opt/pa_bot/data/ohlcv.db "SELECT COUNT(*) FROM order_book_snapshots;"

# Snapshots per day
sqlite3 /opt/pa_bot/data/ohlcv.db "
SELECT date(ts) as day, COUNT(*) as snapshots, COUNT(DISTINCT symbol) as symbols
FROM order_book_snapshots
GROUP BY day
ORDER BY day DESC
LIMIT 7;
"

# Sample snapshot for BTC
sqlite3 /opt/pa_bot/data/ohlcv.db "
SELECT ts, spread, mid_price, imbalance
FROM order_book_snapshots
WHERE symbol = 'BTCUSDT'
ORDER BY ts DESC
LIMIT 5;
"

# Check collection is still running
tail -20 /opt/pa_bot/logs/orderbook.log
```

Expected: ~41,000 snapshots/day (431 symbols x 96 runs/day).

## Appendix: How to Verify Funding Rate Collection Is Working

```bash
# How many funding rate snapshots collected so far
sqlite3 /opt/pa_bot/data/ohlcv.db "SELECT COUNT(*) FROM funding_rate_snapshots;"

# Funding rate snapshots per day
sqlite3 /opt/pa_bot/data/ohlcv.db "
SELECT date(ts) as day, COUNT(*) as snapshots, COUNT(DISTINCT symbol) as symbols
FROM funding_rate_snapshots
GROUP BY day
ORDER BY day DESC
LIMIT 7;
"

# Sample funding rates for BTC
sqlite3 /opt/pa_bot/data/ohlcv.db "
SELECT ts, funding_rate, mark_price, last_price, next_funding_ts
FROM funding_rate_snapshots
WHERE symbol = 'BTCUSDT'
ORDER BY ts DESC
LIMIT 10;
"

# Extreme funding rates (potential reversal signals)
sqlite3 /opt/pa_bot/data/ohlcv.db "
SELECT symbol, ts, funding_rate
FROM funding_rate_snapshots
WHERE ABS(funding_rate) > 0.001
ORDER BY ts DESC
LIMIT 20;
"
```

Expected: Same volume as order book snapshots (~41,000/day). Funding rate values typically range from -0.001 to +0.001 (with extremes at -0.01 to +0.01 during volatile periods).
