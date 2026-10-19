# AI Model Training — Complete Documentation
**Project:** NIFTY 50 Quant AI Trading System  
**Last Updated:** May 2026  

---

## Table of Contents
1. [Big Picture Overview](#1-big-picture-overview)
2. [Step 1 — Data Collection](#2-step-1--data-collection)
3. [Step 2 — Feature Engineering (28 Features)](#3-step-2--feature-engineering-28-features)
4. [Step 3 — Trade Labeling (What the Model Learns)](#4-step-3--trade-labeling)
5. [Step 4 — Data Cleaning & Filters](#5-step-4--data-cleaning--filters)
6. [Step 5 — XGBoost Training](#6-step-5--xgboost-training)
7. [Step 6 — GRU (LSTM) Training](#7-step-6--gru-lstm-training)
8. [Step 7 — Ensemble (Live Inference)](#8-step-7--ensemble-live-inference)
9. [Step 8 — Live Feedback Loop (Auto-Retrain)](#9-step-8--live-feedback-loop)
10. [Hard Safety Rules (Post-Model Filters)](#10-hard-safety-rules)
11. [How to Run Everything](#11-how-to-run-everything)
12. [Files Reference](#12-files-reference)

---

## 1. Big Picture Overview

```
Angel One API
    │
    ▼
data_get.py          ← fetches 10 years of 5-min NIFTY candles
    │
    ▼
nifty_5m_data.csv    ← ~185,000 candles (2016–2026)
    │
    ├──► model_trainer.py   ── trains XGBoost  ──► trading_model.pkl
    │                                               feature_list.pkl
    │
    └──► lstm_trainer.py    ── trains GRU      ──► lstm_model.pt
                                                    lstm_scaler.pkl
                                                         │
                                                         ▼
                                              indian_ai_engine.py
                                         (80% XGBoost + 20% GRU ensemble)
                                                         │
                                                    BUY / SELL / WAIT
                                                    + Confidence %
```

The model is a **3-class classifier**: predict whether a trade entered NOW will hit its **Take Profit (BUY)**, **Take Profit (SELL)**, or **neither (WAIT)** within the next 2 hours.

---

## 2. Step 1 — Data Collection

**File:** `data_get.py`  
**Run:** `python data_get.py`

| Parameter | Value |
|---|---|
| Source | Angel One SmartAPI (`getCandleData`) |
| Instrument | NIFTY 50 Index (token: 99926000, NSE) |
| Interval | **5-minute candles** |
| History target | **10 years (3650 days)** |
| Columns saved | `date` (unix), `open`, `high`, `low`, `close`, `volume` |
| Chunk size | 25 days per API request (Angel One limit) |
| Rate limit pause | 1.2 seconds between chunks |

**Smart resume logic:**
- If `nifty_5m_data.csv` already exists, script only fetches:
  - **Backfill:** data older than what's in the CSV (up to 10y limit)
  - **Forward-fill:** new candles since last run
- Drop `nifty_manual.csv` (same columns) to merge externally sourced historical data

**Practical note:** Angel One's 5-min API typically covers ~2 years back from today. The existing CSV (2021–2026) was accumulated over multiple runs. For pre-2021 data, export from Angel One's historical portal or another vendor and place in `nifty_manual.csv`.

---

## 3. Step 2 — Feature Engineering (28 Features)

**File:** `model_trainer.py` → `add_features()`

Every 5-min candle gets 28 computed features before training. These must **exactly match** what `indian_quant_engine.py` computes in real-time.

### Group 1 — RSI (4 features)
| Feature | Description |
|---|---|
| `RSI` | 14-period Relative Strength Index on close price |
| `RSI_Delta` | Change in RSI vs previous candle (momentum of momentum) |
| `RSI_Lag1` | RSI value 1 candle ago |
| `RSI_Lag2` | RSI value 2 candles ago |

> RSI lags let the model detect whether RSI is rising, falling, or reversing — not just where it is.

### Group 2 — MACD (2 features)
| Feature | Description |
|---|---|
| `MACD_Hist` | MACD Histogram (12 EMA − 26 EMA − 9 signal) |
| `MACD_Hist_Delta` | Change in MACD histogram (acceleration) |

### Group 3 — EMA (4 features)
| Feature | Description |
|---|---|
| `EMA_dist_20` | % distance of close from 20-period EMA |
| `EMA_dist_50` | % distance of close from 50-period EMA |
| `EMA_dist_200` | % distance of close from 200-period EMA |
| `EMA_alignment` | `+1` = full bull (20>50>200), `-1` = full bear, `0` = mixed |

### Group 4 — Volatility (3 features)
| Feature | Description |
|---|---|
| `BB_pct` | Bollinger %B — where price sits within the bands (0=lower, 1=upper) |
| `BB_width` | Band width as % of MA20 — measures how squeezed/expanded price is |
| `ATR_pct` | 14-period ATR as % of price — normalised volatility |

### Group 5 — Candle Structure (3 features)
| Feature | Description |
|---|---|
| `Candle_Body` | Body size / total candle range (0=doji, 1=marubozu) |
| `Upper_Wick` | Upper shadow / total range |
| `Lower_Wick` | Lower shadow / total range |

### Group 6 — Session Timing (3 features)
| Feature | Description |
|---|---|
| `Hour` | Hour of day in IST (9–15) |
| `Minute` | Minute (0, 5, 10 ... 55) |
| `Day_of_week` | 0=Monday … 4=Friday |

> Nifty behaves very differently at 9:15 open, 12:00 noon, and 15:25 closing candle.

### Group 7 — Volume (1 feature)
| Feature | Description |
|---|---|
| `Vol_Ratio` | Current volume / 20-period EMA of volume |

### Group 8 — Regime (1 feature)
| Feature | Description |
|---|---|
| `Daily_Regime` | `1` if close > EMA_200 (bull market), `0` if below (bear) |

### Group 9 — Multi-Timeframe Confluence (5 features)
| Feature | Description |
|---|---|
| `RSI_15m` | RSI computed on 15-min bars (resampled from 5-min) |
| `RSI_1h` | RSI computed on 1-hour bars |
| `EMA_align_15m` | EMA alignment (+1/0/-1) on 15-min chart |
| `EMA_align_1h` | EMA alignment (+1/0/-1) on 1-hour chart |
| `BB_pct_1h` | Bollinger %B on 1-hour chart |

### Group 10 — India VIX (2 features)
| Feature | Description |
|---|---|
| `VIX` | India VIX level — fear/volatility gauge (high = risky market) |
| `VIX_trend` | `+1` if VIX rising (fear increasing), `-1` if falling |

> VIX is downloaded from Yahoo Finance (^INDIAVIX) from 2015 onward, cached in `_vix_cache.csv`, and merged by date.

---

## 4. Step 3 — Trade Labeling

**File:** `model_trainer.py` → `make_targets()`

This is the most important part. The model is trained to answer: **"If I enter a trade at this candle, will I make money in the next 2 hours?"**

### Triple-Barrier Labeling with Directional Timeout

For every candle `i`, we set:
```
TP (BUY)  = entry + 3.0 × ATR
SL (BUY)  = entry − 1.5 × ATR

TP (SELL) = entry − 3.0 × ATR  
SL (SELL) = entry + 1.5 × ATR
```

Then scan the next **24 candles (2 hours)**:

| What happens first | Label assigned |
|---|---|
| Long TP hit before long SL | **BUY** (class 2) |
| Short TP hit before short SL | **SELL** (class 0) |
| Neither hits → timeout → price drifted >0.5×ATR up | **BUY** |
| Neither hits → timeout → price drifted >0.5×ATR down | **SELL** |
| Neither hits → timeout → price barely moved | **WAIT** (class 1) |

**Risk:Reward ratio:** 3.0 : 1.5 = **2:1** — the model only gets a BUY label when the price moves twice as far in our favour as our risk.

---

## 5. Step 4 — Data Cleaning & Filters

After labeling, before training:

### Chop Filter
Drop the **bottom 20% of candles by ATR_pct** (flat/low-volatility candles). The model cannot predict direction in a choppy market — including these candles only adds noise.

```
Before filter: ~180,000 candles
After filter:  ~144,000 candles (20% dropped)
```

### Train / Test Split (Time-Based, No Shuffle)
```
Train: first 80% of candles  (chronological)
Test:  last  20% of candles  (out-of-sample future data)
```
**No random shuffle** — shuffling would leak future data into training (lookahead bias).

### Sample Weights
Two weighting schemes are applied together:

**Class balancing** — WAIT tends to be the majority class. Each class gets weight = `max_count / class_count` so BUY/SELL are not drowned out.

**Time decay** — the most recent 25% of training candles get **2× weight**. Markets evolve; recent patterns matter more than 2016 patterns.

---

## 6. Step 5 — XGBoost Training

**File:** `model_trainer.py`  
**Output:** `trading_model.pkl`, `feature_list.pkl`

XGBoost is the primary model (80% of ensemble weight in live inference).

### Hyperparameters (v4.0 — tuned for 10-year dataset)

| Parameter | Value | Why |
|---|---|---|
| `n_estimators` | 1200 | More trees for larger dataset |
| `max_depth` | 5 | Limits tree complexity |
| `learning_rate` | 0.015 | Slower, more precise |
| `subsample` | 0.75 | Each tree sees 75% of rows |
| `colsample_bytree` | 0.75 | Each tree uses 75% of features |
| `min_child_weight` | 25 | Each leaf needs 25+ samples — prevents candle-level noise fitting |
| `gamma` | 3 | Split only if gain > 3 (strict) |
| `reg_alpha` | 0.3 | L1 regularization |
| `reg_lambda` | 2.5 | L2 regularization |
| `early_stopping_rounds` | 60 | Stop if test loss doesn't improve for 60 rounds |

### Output
```
trading_model.pkl  ← XGBoost classifier (3 classes: SELL=0, WAIT=1, BUY=2)
feature_list.pkl   ← ordered list of 28 feature names (must match live engine)
```

### Live prediction format
```python
proba = model.predict_proba(X)[0]
# proba[0] = P(SELL)
# proba[1] = P(WAIT)
# proba[2] = P(BUY)
```

---

## 7. Step 6 — GRU (LSTM) Training

**File:** `lstm_trainer.py`  
**Output:** `lstm_model.pt`, `lstm_scaler.pkl`

XGBoost treats each candle independently. GRU sees **sequences of 30 candles** (2.5 hours of history) and learns temporal patterns — momentum shifts, trend continuations, divergences.

### Architecture

```
Input:  (batch, 30 candles, 28 features)
         ↓
GRU(hidden=96, 1 layer)   ← sequential pattern learning
         ↓
Global Average Pooling     ← averages all 30 timesteps (not just last)
         ↓
BatchNorm(96)
         ↓
Dropout(0.45)
         ↓
Linear(96 → 3)             ← SELL / WAIT / BUY
```

### Hyperparameters (v2.0 — tuned for 10-year dataset)

| Parameter | Value | Why |
|---|---|---|
| `SEQ_LEN` | 30 | 30 × 5min = 2.5 hours of context |
| `HIDDEN` | 96 | Slightly larger than before (more data available) |
| `DROPOUT` | 0.45 | High — GRU overfits easily on financial data |
| `BATCH_SIZE` | 256 | Large batches for stable gradients |
| `EPOCHS` | 200 | More data needs more epochs |
| `PATIENCE` | 30 | Early stop if val loss stagnates for 30 epochs |
| `LR` | 0.0005 | Adam optimizer |

### Scaler
A `StandardScaler` is fit on the **training set only** and saved as `lstm_scaler.pkl`. The same scaler is applied in live inference before feeding the sequence to the GRU.

---

## 8. Step 7 — Ensemble (Live Inference)

**File:** `indian_ai_engine.py`

The two models run in parallel every 5 seconds and their probabilities are **weighted-averaged**:

```
final_P = 0.80 × XGBoost_P  +  0.20 × GRU_P
```

**Why 80/20 and not 50/50?**  
In the 5-year retraining evaluation:
- XGBoost accuracy: **~49%** on 3-class problem (random = 33%)
- GRU accuracy: **~37%** (was overfitting on limited data)

With 10-year data, GRU accuracy should improve. The weights can be re-evaluated after the 10-year retrain.

### Signal generation logic

```python
if P(BUY)  >= 0.42  AND  P(BUY) > P(SELL)  AND  RSI < 70:
    signal = "BUY"
    SL     = price − 1.5 × ATR
    TP     = price + 3.0 × ATR

elif P(SELL) >= 0.42  AND  P(SELL) > P(BUY)  AND  RSI > 30:
    signal = "SELL"
    SL     = price + 1.5 × ATR
    TP     = price − 3.0 × ATR

else:
    signal = "WAIT"
```

**Confidence threshold:** 42% — the model must be at least 42% confident to enter (3-class random baseline is 33%).

---

## 9. Step 8 — Live Feedback Loop

**File:** `indian_execution_manager.py` → auto-triggers `model_trainer.py`

Every real/paper trade that closes gets written to `live_feedback.csv` with:
- Entry features (all 28 indicators at entry time)
- Outcome: WIN or LOSS
- Entry price, exit price, P&L

**After every 20 closed trades**, XGBoost is **automatically retrained** using:
```
Training data = historical CSV data  +  live_feedback.csv (5× weight)
```

Each live trade counts as 5 historical candles — the model biases toward what actually worked/failed in the current market regime.

---

## 10. Hard Safety Rules

These are NOT part of the model — they are hard overrides applied **after** the model predicts:

| Rule | Condition | Effect |
|---|---|---|
| RSI Overbought Block | RSI ≥ 70 | Block BUY even if model says BUY |
| RSI Oversold Block | RSI ≤ 30 | Block SELL even if model says SELL |
| Confidence Gate | Confidence < 42% | Force WAIT |
| Daily Loss Limit | Session loss ≥ 3% of capital | Halt all trading for the day |
| One Trade at a Time | Position already open | No new signals executed |
| Signal Age | Signal older than 2 minutes | Ignored (stale data) |

---

## 11. How to Run Everything

### First time (or 10-year retrain)

```bash
cd ai_engine

# 1. Fetch 10 years of 5-min candles from Angel One
python data_get.py

# 2. Train XGBoost  (~5–10 min on 150k candles)
python model_trainer.py

# 3. Train GRU  (~15–30 min, depends on CPU/GPU)
python lstm_trainer.py

# 4. Health check — verify all models load correctly
python test.py
```

### Daily (live trading)

```bash
# Terminal 1 — feeds live indicators to Redis every 30s
python indian_quant_engine.py

# Terminal 2 — generates BUY/SELL/WAIT signal every 5s
python indian_ai_engine.py

# Terminal 3 — executes trades, monitors SL/TP
python indian_execution_manager.py

# Backend + Frontend
cd backend_manager && node app.js
cd frontend_manager && npm run dev
```

### Backtest after retraining

```bash
python backtester.py
# Runs on last 20% of data (out-of-sample)
# Outputs: backtest_equity_curve.png + backtest_trades.csv
```

---

## 12. Files Reference

| File | Purpose |
|---|---|
| `data_get.py` | Fetch 10y of 5-min NIFTY candles from Angel One |
| `model_trainer.py` | XGBoost training pipeline (features + labels + train) |
| `lstm_trainer.py` | GRU training pipeline (sequence model) |
| `indian_quant_engine.py` | Live: computes 28 indicators every 30s, writes to Redis |
| `indian_ai_engine.py` | Live: reads indicators, runs ensemble, writes signal to Redis |
| `indian_execution_manager.py` | Live: reads signal, places paper/live orders, manages SL/TP |
| `backtester.py` | Offline: simulates trades on holdout data, equity curve |
| `replay_engine.py` | Offline: replays historical candles through the full live pipeline |
| `monitor.py` | Terminal dashboard: live signal + position + indicators |
| `test.py` | Health check: verifies all models + Redis + features |
| `options_helper.py` | Live mode: resolves CE/PE strike, lot sizing for Angel One |
| `nifty_5m_data.csv` | ~185,000 candles of 5-min NIFTY data (10 years) |
| `trading_model.pkl` | Trained XGBoost model |
| `feature_list.pkl` | Ordered list of 28 feature names |
| `lstm_model.pt` | Trained GRU model (PyTorch) |
| `lstm_scaler.pkl` | StandardScaler for GRU input normalisation |
| `live_feedback.csv` | Closed trade outcomes — used for auto-retrain |
| `_vix_cache.csv` | India VIX daily data from 2015 (auto-refreshed) |
