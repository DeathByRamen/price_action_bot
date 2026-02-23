# PA Bot — Complete System Workflow

This document explains the entire prediction pipeline end-to-end: how data flows in, how the model learns, how predictions are generated, how accuracy is measured, and how the system automatically retrains and self-corrects based on its own performance.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Phase 1 — Data Collection (Backfill)](#2-phase-1--data-collection-backfill)
3. [Phase 2 — Initial Model Training](#3-phase-2--initial-model-training)
4. [Phase 3 — Hourly Prediction Pipeline](#4-phase-3--hourly-prediction-pipeline)
5. [Phase 4 — Scoring: How We Know If Predictions Were Right](#5-phase-4--scoring-how-we-know-if-predictions-were-right)
6. [Phase 5 — The Adaptive Feedback Loop (Self-Correction)](#6-phase-5--the-adaptive-feedback-loop-self-correction)
7. [Phase 6 — Daily Retrain (Putting It All Together)](#7-phase-6--daily-retrain-putting-it-all-together)
8. [Multi-Timeframe Ensemble](#8-multi-timeframe-ensemble)
9. [What You Receive as the End User](#9-what-you-receive-as-the-end-user)
10. [Visual Flowchart](#10-visual-flowchart)

---

## 1. High-Level Overview

The system operates on two recurring schedules:

| Schedule | Script | What It Does |
|---|---|---|
| **Every hour** | `run_prediction.py` | Fetches latest candles, runs the model, generates predictions, sends alerts |
| **Once daily** (midnight UTC) | `daily_retrain.py` | Scores yesterday's predictions, measures accuracy, auto-tunes parameters, retrains the model on fresh data |

The daily retrain is where the magic happens — it creates a **closed feedback loop** where the system measures its own mistakes and adjusts itself to do better tomorrow.

---

## 2. Phase 1 — Data Collection (Backfill)

**Script:** `scripts/backfill_data.py`
**Run once** when setting up the system.

### What happens:

1. **Discover symbols** — Queries BitUnix's futures API for all available trading pairs (e.g., BTCUSDT, ETHUSDT, etc.), then cross-references with the spot API to filter out pairs that don't have spot candle data available. Result: ~431 tradeable symbols.

2. **Fetch historical candles** — For each symbol, downloads up to 2,000 hourly (1h) candles from the spot kline history API. That's roughly 83 days of price history per symbol.

3. **Store in SQLite** — Each candle (timestamp, open, high, low, close, volume) is stored in the `ohlcv` table, keyed by `(symbol, timestamp, interval)`.

4. **Resume logic** — If the backfill is interrupted, it checks how many candles already exist per symbol and only fetches the remaining ones. You never re-download data you already have.

For multi-timeframe mode, it also backfills 15-minute candles (8,000 candles per symbol to cover the same time period).

### Data at this point:
```
ohlcv table: ~431 symbols x ~2,000 candles each = ~862,000 rows
```

---

## 3. Phase 2 — Initial Model Training

**Script:** `scripts/train_model.py`
**Run once** after backfill (and again manually when you want a fresh start).

### Step-by-step:

#### 3a. Load raw candles from the database
For each symbol, pull the last N days of candles (default: all available, or `--rolling-days 60` for 2 months).

#### 3b. Compute technical indicators (feature engineering)
For each symbol's candle data, compute **41 technical indicators** across 6 categories:

| Category | Examples | Why It Matters |
|---|---|---|
| **Trend** | EMA distances (9/21/50), MACD, ADX | Is the price trending or ranging? |
| **Momentum** | RSI, Stochastic RSI, Williams %R, ROC | Is momentum building or fading? |
| **Volatility** | Bollinger Bands, ATR, Keltner Channels | How wild are the price swings? |
| **Volume** | OBV rate-of-change, volume Z-score, VWAP | Is volume confirming the move? |
| **Custom** | Candle body/wick ratios, EMA crossovers, Bollinger distance | Pattern recognition signals |
| **Anti-pump/dump** | 3-day/7-day change, price position, momentum acceleration, ATR expansion, volume-price ratio | Detect unsustainable moves |

**Critical design choice:** Every feature is either a ratio, percentage, or normalized by ATR/price. This means BTC at $60,000 and an altcoin at $0.003 produce features on the same scale — the model doesn't have to learn that "60,000" and "0.003" are both valid prices.

#### 3c. Create the training dataset
The data is organized **per-symbol** (windows never cross symbol boundaries) with:
- **Window size:** 168 timesteps (7 days of hourly candles)
- **Labels:** For each window, look 1 hour ahead and classify the price change:
  - **UP:** change > +0.5% (the FLAT threshold)
  - **DOWN:** change < -0.5%
  - **FLAT:** change between -0.5% and +0.5%
- **Per-window Z-score normalization:** Each 168-step window is independently normalized (subtract mean, divide by standard deviation). This matches exactly what happens during inference.

#### 3d. Walk-forward cross-validation
The data is split **temporally** (by timestamp, not randomly) to prevent data leakage:
- Training data: everything before timestamp T
- Validation data: everything after timestamp T

This simulates real-world conditions where the model only ever sees past data.

#### 3e. Train the LSTM model
The model architecture:

```
Input (168 timesteps x 41 features)
    │
    ▼
Feature Gate (sigmoid) ─── Learns which features matter in the current context
    │                      Some features get amplified, others suppressed
    ▼
LSTM Encoder (2 layers, 128 hidden units, unidirectional)
    │                      Processes the sequence causally (no future info)
    ▼
Temporal Attention ─────── Weights all 168 timesteps by importance
    │                      A volume spike 12h ago can be weighted alongside current RSI
    ▼
Residual Connection ────── Skip connection from input for gradient flow
    │
    ├──► Classification Head → [P(UP), P(FLAT), P(DOWN)]  (3 probabilities)
    │
    └──► Regression Head → predicted % magnitude of the move
```

Training uses:
- **Combined loss:** CrossEntropy for direction + Huber loss for magnitude
- **Inverse-frequency class weights:** If UP has 9,733 samples and DOWN has 8,081, DOWN gets slightly higher weight so the model doesn't just always predict the majority class
- **Class weight clamping (max 3.0):** Prevents explosive weights for rare classes
- **AdamW optimizer** with weight decay (L2 regularization)
- **Gradient clipping (max norm 1.0):** Prevents exploding gradients
- **ReduceLROnPlateau scheduler:** Halves the learning rate when validation loss plateaus
- **Early stopping (patience 10):** Stops training if no improvement for 10 epochs

#### 3f. Calibrate probability temperature
After training, the model's `temperature` parameter is optimized on the validation set to make probabilities well-calibrated. This means when the model says "70% UP", it should actually be correct about 70% of the time — not just confident-sounding.

#### 3g. Save the checkpoint
The final model is saved as `model_final_60.pt` (for the 1h model) with metadata:
- `num_features`: 41 (for compatibility checking on load)
- `hidden_dim`: 128
- `feature_cols_hash`: MD5 hash of feature column names (detects if features change)
- `created_at`: UTC timestamp

---

## 4. Phase 3 — Hourly Prediction Pipeline

**Script:** `scripts/run_prediction.py`
**Runs every hour** via cron or Task Scheduler.

### Step-by-step:

1. **Discover symbols** — Same filtering as backfill (futures pairs with spot data).

2. **Fetch latest candles** — Download the last 10 candles for each symbol to fill any gaps since the previous run.

3. **Load historical candles** — Pull `168 + 168 + 10 = 346` candles per symbol from the database (window size + warm-up for indicators + buffer).

4. **Batch inference** — All symbols are processed in a single forward pass through the model:
   - Compute indicators for each symbol
   - Extract the most recent 168-step window
   - Z-score normalize each window independently
   - Stack all windows into one batch tensor
   - Run one forward pass → get probabilities and magnitudes for all ~431 symbols at once

5. **Compute conviction scores** — For each prediction:
   - **Conviction** = 1 - (Shannon entropy / max entropy). A uniform [33%, 33%, 33%] distribution gives conviction = 0 (random). A sharp [90%, 5%, 5%] gives conviction ≈ 0.8 (confident).
   - **Signal score** = conviction × directional probability × |magnitude|

6. **Rank and filter** — Sort by signal score descending. Take the top N (default 10).

7. **Send notifications** — Dispatch the top signals via email/Discord/Telegram.

8. **Log to database** — Every prediction (all ~431, not just top 10) is stored in the `predictions` table with its timestamp, direction, probabilities, magnitude, and score.

### Why log all predictions?
Because the scoring system needs them. Tomorrow, when the actual price is known, every prediction will be scored — not just the ones that were sent as alerts. This gives the feedback loop maximum data to learn from.

---

## 5. Phase 4 — Scoring: How We Know If Predictions Were Right

**Module:** `src/scoring/accuracy.py`
**Called by:** `daily_retrain.py` (Step 2)

### How a prediction gets scored:

For each unscored prediction in the database:

1. **Look up the price at prediction time** — Get the close price of the candle at the timestamp when the prediction was made (filtered by the correct interval: 1h or 15m).

2. **Look up the actual next candle** — Get the close price of the *next* candle after the prediction timestamp.

3. **Calculate actual % change:**
   ```
   actual_magnitude = (next_close - pred_close) / pred_close
   ```

4. **Classify actual direction** using the current FLAT threshold:
   ```
   if actual_magnitude > 0.5%  → actual_direction = "UP"
   if actual_magnitude < -0.5% → actual_direction = "DOWN"
   otherwise                   → actual_direction = "FLAT"
   ```

5. **Compare:** `was_correct = (predicted_direction == actual_direction)`

6. **Store the result** — The prediction row is updated with `actual_direction`, `actual_magnitude`, `was_correct`, and `scored_at`.

### Accuracy Report

After scoring, an aggregate report is computed:

```
Predictions scored:   847
Direction accuracy:   42.3%
  UP  prec/recall:    45% / 51%
  DOWN prec/recall:   38% / 35%
  FLAT prec/recall:   41% / 39%
Magnitude MAE:        0.0082
```

**Important detail:** The report re-classifies `actual_direction` using the *current* FLAT threshold. This prevents metrics from being skewed by historical threshold values that have since been auto-tuned.

---

## 6. Phase 5 — The Adaptive Feedback Loop (Self-Correction)

This is the core intelligence of the system. Two mechanisms work together to improve performance over time:

### 6a. FLAT Threshold Auto-Tuning

**Module:** `src/scoring/adaptive.py` → `compute_optimal_threshold()`

**The problem:** The boundary between "flat" and "directional" moves (the FLAT threshold) is not universal. In high-volatility regimes, a 0.5% move might be noise. In low-volatility regimes, 0.5% might be significant.

**How it works:**

1. **Gather recent data** — Pull all scored predictions from the last 7 days.

2. **Split into tune/eval sets** — 70% for tuning, 30% for validation. This prevents overfitting the threshold to a single dataset.

3. **Grid search** — Test 50 candidate thresholds between 0.2% and 1.5%:
   - For each candidate, re-classify both predictions and actuals using that threshold
   - Measure direction accuracy
   - Track the threshold that gives the highest accuracy on the tune set

4. **Validate** — Check if the new optimal threshold actually improves accuracy on the held-out eval set compared to the current threshold. If it doesn't improve, **keep the current threshold** (prevents regression).

5. **EMA smoothing** — If the new threshold passes validation:
   ```
   new_threshold = 0.3 × optimal + 0.7 × current
   ```
   The smoothing factor (0.3) prevents sudden jumps that could destabilize training. The threshold evolves gradually.

6. **Clamp to bounds** — Final threshold is clamped between 0.2% and 1.5% as safety rails.

**Result:** The FLAT threshold automatically adapts to the current market volatility regime. During calm markets, it tightens (smaller moves count as directional). During volatile markets, it widens (larger moves needed to be called directional).

### 6b. Per-Symbol Sample Weighting

**Module:** `src/scoring/adaptive.py` → `compute_sample_weights()`

**The problem:** The model might consistently get certain symbols wrong (e.g., always predicting UP for a coin that keeps going DOWN). Standard training treats all symbols equally.

**How it works:**

1. **Gather recent errors** — Pull all scored predictions from the last 7 days. Group by symbol.

2. **Calculate error rate per symbol:**
   ```
   error_rate(BTCUSDT) = 1 - (correct_predictions / total_predictions)
   ```

3. **Convert to training weight:**
   ```
   weight = 1.0 + (symbol_error_rate / mean_error_rate_across_all_symbols)
   ```
   Symbols with above-average error rates get weights > 2.0. Symbols with below-average error rates stay near 1.0.

4. **Cap at 5.0** — No single symbol can dominate the training gradient.

5. **Minimum data requirement** — A symbol needs at least 5 scored predictions before it gets a custom weight. Otherwise, it defaults to 1.0.

**How these weights are used during training:**
During the next retrain, these weights feed into a `WeightedRandomSampler`. This means the DataLoader is more likely to draw training examples from poorly-predicted symbols, causing the model to focus more gradient on its weak spots.

**Example:**

| Symbol | Predictions | Correct | Error Rate | Weight |
|---|---|---|---|---|
| BTCUSDT | 24 | 18 | 25% | 0.8 |
| DOGEUSDT | 24 | 6 | 75% | 2.4 |
| SOLUSDT | 24 | 12 | 50% | 1.6 |

DOGEUSDT would appear ~3x more often in training batches than BTCUSDT, forcing the model to learn its patterns better.

---

## 7. Phase 6 — Daily Retrain (Putting It All Together)

**Script:** `scripts/daily_retrain.py`
**Runs once daily** at midnight UTC.

This is the orchestrator that ties every feedback mechanism together into a single daily cycle. Here's the complete step-by-step:

### Step 1: Gap-Fill Candles
Fetch the last 48 hours of candles for all symbols and all active intervals (1h and 15m). This ensures no gaps exist from overnight or missed hourly runs.

### Step 2: Score Yesterday's Predictions
Call `score_predictions()` to compare every unscored prediction against what actually happened. Each prediction gets marked as correct or incorrect with the actual magnitude stored.

### Step 3: Compute Accuracy Report
Generate aggregate metrics (direction accuracy, per-class precision/recall, magnitude MAE, per-symbol hit rates). This is what gets sent to you in the daily digest email.

### Step 4: Auto-Tune the FLAT Threshold
Run the threshold optimization described in [Section 6a](#6a-flat-threshold-auto-tuning). If a better threshold is found and validated, it's adopted (with EMA smoothing). If not, the current threshold is kept.

### Step 5: Compute Sample Weights
Run the sample weight computation described in [Section 6b](#6b-per-symbol-sample-weighting). Symbols the model keeps getting wrong receive higher weights for the upcoming training.

### Step 6: Back Up the Current Model
Copy the existing `model_final_60.pt` to `model_final_60_backup.pt` before overwriting, just in case.

### Step 7: Retrain the Model
This is a full training run using the updated parameters:

1. **Load data** — Last 60 days of candles for all symbols (configurable via `rolling_days`).
2. **Compute indicators** — All 41 features, freshly computed.
3. **Create dataset** — Using the newly tuned FLAT threshold (from Step 4) for labeling.
4. **Walk-forward split** — Temporal train/val split.
5. **Train** — Up to 50 epochs with early stopping (patience 8), using the adaptive sample weights from Step 5.
6. **Calibrate temperature** — Optimize probability calibration on validation set.

### Step 8: Validation Gate (Auto-Rollback)
**This is a critical safety mechanism.** Before deploying the newly trained model:

1. Evaluate the **new** model on the validation set → get `new_val_loss`
2. Load the **old** deployed model and evaluate on the same validation set → get `old_val_loss`
3. Compare:
   - If `new_val_loss <= old_val_loss × 1.05` → **Deploy the new model** (it's at least as good)
   - If `new_val_loss > old_val_loss × 1.05` → **Keep the old model** (the new one regressed)

The 5% tolerance allows for minor statistical fluctuations without blocking improvements. But if a retrain produces a significantly worse model (bad data day, anomalous market, etc.), the old model is automatically preserved.

### Step 9: Permutation Importance
Run feature importance analysis on the validation set:
- For each of the 41 features, shuffle its values and measure how much the model's accuracy drops
- Features that cause a big accuracy drop when shuffled are important
- Features that cause no drop are candidates for removal

Results are saved to the `feature_importance` table in the database. After 3 consecutive daily runs where a feature scores below 0.001 importance, it's flagged in the digest email as a candidate for removal.

### Step 10: Send Daily Digest
Email (and/or Discord/Telegram) a summary containing:
- Accuracy report (from Step 3)
- Threshold changes (from Step 4)
- Model deployment status (deployed new vs. rolled back)
- Low-importance feature warnings (from Step 9)

---

## 8. Multi-Timeframe Ensemble

When multi-timeframe mode is enabled (the default), the system runs **two models in parallel**:

| Model | Interval | Window | Purpose |
|---|---|---|---|
| Primary | 1h (60m) | 168 candles (7 days) | **Directional bias** — where is the trend going? |
| Secondary | 15m | 672 candles (7 days) | **Entry timing** — is now a good time to enter? |

### How predictions are combined:

1. **Log-odds probability combination** — Each model's probabilities (UP/FLAT/DOWN) are converted to log-odds space, weighted (default 60/40), summed, and converted back. This is statistically sound — unlike naive averaging, it properly handles the non-linear nature of probabilities.

2. **Adaptive weighting** — The 60/40 split isn't fixed. Each day, the system looks at each model's accuracy over the past 7 days and adjusts weights proportionally. If the 15m model has been more accurate lately, it gets more influence.

3. **Agreement classification** — The system labels each combined prediction:
   - **STRONG:** Both models predict the same direction (e.g., both say UP)
   - **PARTIAL:** One is directional, the other is FLAT, or magnitudes disagree weakly
   - **CONFLICT:** Both are directional but in opposite directions with significant magnitude
   - **PULLBACK:** The secondary disagrees but with tiny magnitude (<1%) — interpreted as a pullback within the primary trend, not a genuine conflict

4. **Score multiplier:**
   - STRONG signals get 1.5x score boost
   - PARTIAL signals get 1.0x (neutral)
   - CONFLICT signals get 0.3x penalty (pushed to bottom of rankings)

### Why two timeframes?

The 1h model sees the forest (multi-day trend direction). The 15m model sees the trees (short-term momentum, pullbacks, entry points). A coin might be in a strong uptrend on the 1h chart but temporarily pulling back on the 15m chart. The ensemble recognizes "PULLBACK UP" instead of sending conflicting signals.

---

## 9. What You Receive as the End User

### Hourly (every prediction run):

An email/notification with the top 10 signals:

```
PA Bot Multi-Timeframe Predictions (60m + 15m)
Top 10 signals by conviction:

  #  Symbol        60m  15m  Signal        Prob    Mag%        Price   Score
--- ------------ ---- ----  ------------ ------ ------- ------------ -------
  1  BTCUSDT        UP   UP  STRONG UP    82.3%  +1.42%   67543.2100  0.0847
  2  ETHUSDT        UP   UP  STRONG UP    75.1%  +0.98%    3521.4500  0.0623
  3  SOLUSDT      DOWN FLAT  WEAK DOWN    68.5%  -1.15%     145.3200  0.0412
  ...
```

### Daily (from the retrain):

An accuracy digest email:

```
PA Bot Daily Accuracy Report

Predictions scored:   847
Direction accuracy:   42.3%
  UP  prec/recall:    45% / 51%
  DOWN prec/recall:   38% / 35%
  FLAT prec/recall:   41% / 39%
Magnitude MAE:        0.0082
FLAT threshold:       0.0053 (adjusted from 0.0050)

Top performers: BTCUSDT (67%), ETHUSDT (58%), BNBUSDT (54%)
Worst performers: SHIBUSDT (18%), PEPEUSDT (21%), FLOKIUSDT (23%)

Low-importance features (60m) — candidates for removal:
  - `ema_21_50_cross`: avg importance = 0.000312
```

---

## 10. Visual Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                     HOURLY CYCLE (every hour)                    │
│                                                                  │
│  Fetch Latest   ──►  Compute     ──►  LSTM Model  ──►  Rank &   │
│  Candles             Indicators       Inference        Notify    │
│  (gap-fill)          (41 features)    (batch)          (top 10)  │
│                                                                  │
│  All predictions stored to DB ──────────────────────────►  DB    │
└─────────────────────────────────────────────────────────────────┘
                                                              │
                                                              │ predictions
                                                              │ wait for
                                                              │ actual outcomes
                                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DAILY CYCLE (midnight UTC)                     │
│                                                                  │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐              │
│  │ Gap-Fill │───►│  Score     │───►│  Accuracy    │              │
│  │ Candles  │    │ Predictions│    │  Report      │              │
│  └──────────┘    └───────────┘    └──────┬───────┘              │
│                                          │                       │
│                         ┌────────────────┼────────────────┐      │
│                         ▼                ▼                ▼      │
│                   ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│                   │  Tune    │    │  Compute │    │  Feature  │  │
│                   │  FLAT    │    │  Sample  │    │Importance │  │
│                   │Threshold │    │  Weights │    │  Audit    │  │
│                   └────┬─────┘    └────┬─────┘    └────┬─────┘  │
│                        │               │               │         │
│                        ▼               ▼               │         │
│                   ┌─────────────────────────┐          │         │
│                   │     RETRAIN MODEL       │          │         │
│                   │                         │          │         │
│                   │  • New FLAT threshold    │          │         │
│                   │    for labeling          │          │         │
│                   │  • Sample weights for    │          │         │
│                   │    focused learning      │          │         │
│                   │  • Walk-forward CV       │          │         │
│                   │  • Temperature calib.    │          │         │
│                   └────────┬────────────────┘          │         │
│                            │                           │         │
│                            ▼                           │         │
│                   ┌─────────────────┐                  │         │
│                   │ VALIDATION GATE │                  │         │
│                   │                 │                  │         │
│                   │ new_loss <=     │                  │         │
│                   │ old_loss × 1.05?│                  │         │
│                   └───┬─────────┬───┘                  │         │
│                   YES │         │ NO                   │         │
│                       ▼         ▼                      │         │
│               Deploy New    Keep Old                   │         │
│               Model         Model (rollback)           │         │
│                       │         │                      │         │
│                       └────┬────┘                      │         │
│                            │                           │         │
│                            ▼                           ▼         │
│                   ┌─────────────────────────────────────┐        │
│                   │        SEND DAILY DIGEST            │        │
│                   │  • Accuracy metrics                 │        │
│                   │  • Threshold changes                │        │
│                   │  • Deploy/rollback status            │        │
│                   │  • Low-importance feature warnings   │        │
│                   └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │  Model checkpoint updated
                            │  (or preserved if rollback)
                            ▼
                  Next hourly cycle uses
                  the latest model automatically
```

### The Feedback Loop Summarized

```
Predictions → Wait for outcomes → Score accuracy
     ▲                                    │
     │                                    ▼
     │                           Tune threshold
     │                           Compute weights
     │                                    │
     │                                    ▼
     │                           Retrain model
     │                           (with new labels + focused weights)
     │                                    │
     │                                    ▼
     │                           Validate & deploy
     │                                    │
     └────────────────────────────────────┘
              Better model makes
              better predictions tomorrow
```

The system gets smarter every day — not by magic, but by systematically measuring where it fails, adjusting what "flat" means for the current market, focusing training on its weakest symbols, and only deploying models that are provably at least as good as the previous one.

---

## 11. Advanced Capabilities (Road to 10/10)

The following modules extend the base system described above. They are implemented and ready to integrate as the system matures:

### Backtesting Engine (`src/backtesting/`)
Before deploying any strategy changes, the backtesting engine simulates them on historical data. It models transaction costs (maker/taker fees, slippage, funding), computes standard quant metrics (Sharpe, Sortino, Calmar, max drawdown, profit factor), and supports walk-forward evaluation where the model never sees future data. Run via `scripts/run_backtest.py`.

### Risk Management (`src/risk/`)
Controls position sizing (Kelly criterion, volatility targeting), enforces portfolio-level limits (max exposure, correlated position caps), and implements a drawdown circuit breaker that reduces or halts trading during equity declines.

### Multi-Model Ensemble (`src/model/multi_ensemble.py`)
Combines predictions from LSTM, Temporal Fusion Transformer (`src/model/tft.py`), and gradient boosting (`src/model/gbm.py`). Weights are proportional to each model's recent Sharpe ratio, and diversity is enforced (only models with low prediction correlation are included).

### Regime Detection (`src/model/regime.py`)
An HMM trained on BTC/ETH data classifies the market into 4 states: trending up, trending down, ranging, or high-volatility. Each regime adjusts position sizing, FLAT threshold, and model selection.

### P&L-Based Optimization (`src/scoring/pnl_optimizer.py`)
Replaces accuracy-based threshold tuning with Sharpe-based tuning. A prediction that is wrong but causes a small loss is weighted differently than one causing a large loss. Optimizes the FLAT threshold to maximize simulated Sharpe over a recent window.

### Uncertainty Quantification (`src/model/uncertainty.py`)
MC Dropout runs inference N times with dropout enabled to measure prediction variance. High variance = low confidence = smaller positions or no trade. Deep ensemble disagreement provides an alternative uncertainty estimate.

### Drift Monitoring (`src/model/drift.py`)
Tracks prediction distribution shifts (KL divergence) and calibration quality (Expected Calibration Error). Alerts when the model's behavior changes significantly, triggering recalibration or retraining.

### A/B Testing (`src/model/ab_testing.py`)
Runs a shadow model alongside production, comparing Sharpe ratios without affecting live signals. Supports automated promotion with gradual rollout (10% → 25% → 50% → 100%).

### Production Infrastructure
- **Health checks** (`scripts/healthcheck.py`): monitors predictions, retraining, OB snapshots, DB size, disk space
- **Prometheus metrics** (`src/monitoring/metrics.py`): prediction latency, accuracy, API health
- **Docker Compose** (`docker-compose.yml`): full stack with TimescaleDB + Prometheus + Grafana
- **PostgreSQL backend** (`src/data/postgres_storage.py`): TimescaleDB with hypertables and automatic compression
- **CI/CD** (`.github/workflows/`): lint + test on push, auto-deploy on merge to main
