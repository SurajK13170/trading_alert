"""
NIFTY 50 - GRU TRAINER v3.0
------------------------------
Key fix over v2.0: restrict input to CONTINUOUS features only.

Root cause of v2.0 overfitting: the GRU received all 45 features,
including categorical/discrete ones (EMA_alignment=-1/0/1, Session_bucket=0-5,
Is_expiry_day=0/1, Market_Regime=0-4, etc.). These have so few distinct values
that the GRU's sequential learning collapses to memorising category combinations
rather than learning genuine price-action momentum patterns.

Fix: use only the 29 CONTINUOUS, time-varying features. The 16 excluded features
(categorical, temporal, slow-daily) are still used by XGBoost (which handles
them well via tree splits), so no predictive value is lost from the ensemble.

Architecture improvements:
  - LayerNorm instead of BatchNorm1d (more stable for variable batch sizes)
  - CosineAnnealingLR (better convergence than ReduceLROnPlateau)
  - Gradient clipping already present — kept
  - Hidden reduced 96→64 (fewer features = simpler model needed)
  - Dropout increased 0.45→0.55

Saved outputs:
  lstm_model.pt    — PyTorch state dict + GRU_FEATURES list in checkpoint
  lstm_scaler.pkl  — StandardScaler (fit on train only)

Run: python lstm_trainer.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("[ERR] PyTorch not installed.  Run: pip install torch")
    raise SystemExit(1)

from model_trainer import load_and_clean, add_features, make_targets, FEATURES

# ----------------------------------------------------------------
HOURLY_CSV  = "nifty_5m_data.csv"
MODEL_OUT   = "lstm_model.pt"
SCALER_OUT  = "lstm_scaler.pkl"

# Continuous, time-varying features only — suitable for a sequential model.
# Excluded: EMA_alignment/-15m/-1h (discrete -1/0/1), Hour/Minute/Day_of_week/
#   Session_bucket/Days_to_expiry/Is_expiry_day (temporal categoricals),
#   Daily_Regime/Market_Regime (regime classifiers), VIX_trend (discrete),
#   RS_ratio/RS_slope (daily granularity — too slow for 30-bar window),
#   Dist_round_50/Dist_round_100 (semi-discrete, low sequence signal).
GRU_FEATURES = [
    # RSI — smooth oscillator, strong sequential signal
    "RSI", "RSI_Delta", "RSI_Lag1", "RSI_Lag2",
    # MACD — trend + momentum acceleration
    "MACD_Hist", "MACD_Hist_Delta",
    # EMA distance % — continuous trend strength
    "EMA_dist_20", "EMA_dist_50", "EMA_dist_200",
    # Bollinger + ATR — volatility regime
    "BB_pct", "BB_width", "ATR_pct",
    # Volatility percentile ranks (rolling 100-bar)
    "ATR_pctile", "BB_width_pctile",
    # Candle structure — price action quality
    "Candle_Body", "Upper_Wick", "Lower_Wick",
    # Volume ratio — participation
    "Vol_Ratio",
    # Trend strength
    "ADX",
    # Multi-timeframe continuous signals
    "RSI_15m", "RSI_1h", "BB_pct_1h",
    # Fear gauge
    "VIX",
    # Intraday price levels (continuous % distances)
    "VWAP_dist",
    "PDH_dist", "PDL_dist", "PDC_dist",
    # Opening range (continuous)
    "OR_size", "OR_position",
]
# 29 features — all continuous, all meaningful over a 30-candle sequence

SEQ_LEN     = 30      # 30 × 5min = 2.5 hours of context
HIDDEN      = 64      # reduced from 96 (fewer features, simpler needed)
DROPOUT     = 0.40    # reduced from 0.55 — model was stopping at epoch 37/200,
                      # meaning regularisation was too aggressive to learn properly
BATCH_SIZE  = 256
EPOCHS      = 200
LR          = 0.0005
PATIENCE    = 30
TRAIN_SPLIT = 0.80
# ----------------------------------------------------------------


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Sliding window: each sample = last seq_len rows → label at current row."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


class NiftyGRU(nn.Module):
    """
    GRU with global average pooling + LayerNorm.

    LayerNorm (over hidden dim) is more stable than BatchNorm1d when batch
    sizes vary at inference time (e.g. single-sample live prediction).
    Global average pooling across all 30 timesteps gives a richer summary
    than just the last hidden state and significantly reduces overfitting.

    Parameter count: ~29*64*3 + 64*64*3 + 64*3 ≈ 18k params (very lean).
    """
    def __init__(self, input_size: int, hidden: int,
                 dropout: float, num_classes: int = 3):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)           # (batch, seq_len, hidden)
        pooled = out.mean(dim=1)       # global average over time axis
        pooled = self.norm(pooled)
        pooled = self.drop(pooled)
        return self.head(pooled)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, preds_all, labels_all = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += criterion(logits, yb).item()
        preds_all.extend(torch.argmax(logits, 1).cpu().tolist())
        labels_all.extend(yb.cpu().tolist())
    return total / len(loader), preds_all, labels_all


def train():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  NIFTY 50 GRU TRAINER v3.0")
    print(f"  Continuous features : {len(GRU_FEATURES)}  (was 45 — categorical excluded)")
    print(f"  Sequence length     : {SEQ_LEN} candles (2.5 hours)")
    print(f"  Architecture        : GRU({HIDDEN}) → LayerNorm → Dropout({DROPOUT}) → Dense(3)")
    print(f"  Max epochs          : {EPOCHS}  |  Early stop: {PATIENCE}")
    print(f"{sep}\n")

    # Validate all GRU features are present in FEATURES
    missing = [f for f in GRU_FEATURES if f not in FEATURES]
    if missing:
        print(f"[ERR] GRU_FEATURES references unknown features: {missing}")
        print("      Check that model_trainer.py FEATURES list includes all of these.")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}\n")

    # 1. Load, feature, label
    print("[1/4] Loading data and computing features...")
    df = load_and_clean(HOURLY_CSV)
    df = add_features(df)
    df = make_targets(df)
    df.dropna(subset=GRU_FEATURES + ["target"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    dist      = df["target"].value_counts().sort_index()
    label_map = {0: "SELL", 1: "WAIT", 2: "BUY"}
    print("Target distribution:")
    for k, v in dist.items():
        print(f"  {label_map[k]:>4}: {v:>5}  ({v/len(df)*100:.1f}%)")
    print(f"  Total: {len(df):,}\n")

    # 2. Extract GRU features only
    X_raw = df[GRU_FEATURES].values.astype(np.float32)
    y_raw = df["target"].values.astype(np.int64)

    # 3. Time-based split
    split  = int(len(X_raw) * TRAIN_SPLIT)
    X_tr_r = X_raw[:split];  y_tr = y_raw[:split]
    X_te_r = X_raw[split:];  y_te = y_raw[split:]

    # 4. Scale on train only (prevent data leakage)
    print("[2/4] Fitting scaler on training set...")
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_r)
    X_te   = scaler.transform(X_te_r)

    # 5. Build sequences
    print("[3/4] Building sequences...")
    X_tr_s, y_tr_s = make_sequences(X_tr, y_tr, SEQ_LEN)
    X_te_s, y_te_s = make_sequences(X_te, y_te, SEQ_LEN)
    print(f"  Train: {len(X_tr_s):,} sequences | shape: {X_tr_s.shape}")
    print(f"  Test : {len(X_te_s):,} sequences | shape: {X_te_s.shape}\n")

    # 6. Class weights — capped to prevent WAIT over-prediction.
    # Uncapped WAIT weight = ~6.4x (88k BUY / 13k WAIT). At that weight the GRU
    # buries 10k+ real BUY/SELL signals into WAIT. Cap at 3x so WAIT is respected
    # but doesn't dominate. SELL and BUY stay at 1.0 (already balanced).
    counts      = np.bincount(y_tr_s)
    raw_weights = counts.max() / counts
    raw_weights = np.clip(raw_weights, 1.0, 3.0)   # cap any class at 3× the majority
    weights     = torch.FloatTensor(raw_weights).to(device)
    print(f"  Class weights (capped 3x): "
          f"SELL={raw_weights[0]:.2f}  WAIT={raw_weights[1]:.2f}  BUY={raw_weights[2]:.2f}")

    # 7. DataLoaders
    tr_dl = DataLoader(SequenceDataset(X_tr_s, y_tr_s),
                       batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    te_dl = DataLoader(SequenceDataset(X_te_s, y_te_s),
                       batch_size=BATCH_SIZE, shuffle=False)

    # 8. Model + optimiser + scheduler
    model     = NiftyGRU(len(GRU_FEATURES), HIDDEN, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # CosineAnnealingLR: smoothly decays LR from LR→0 over T_max epochs.
    # Better than ReduceLROnPlateau for financial data — avoids getting stuck
    # in shallow local minima when val loss plateaus temporarily.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 20
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Parameters: {total_params:,}  (lean → less overfitting)\n")

    # 9. Training loop with early stopping
    best_loss    = float("inf")
    best_weights = None
    no_improve   = 0

    print(f"[4/4] Training...\n")
    print(f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} {'ValAcc':>8} {'LR':>10}")
    print("-" * 48)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, tr_dl, optimizer, criterion, device)
        va_loss, va_preds, va_labels = eval_epoch(model, te_dl, criterion, device)
        scheduler.step()

        va_acc  = np.mean(np.array(va_preds) == np.array(va_labels)) * 100
        cur_lr  = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {tr_loss:>10.4f} {va_loss:>10.4f} "
                  f"{va_acc:>7.1f}% {cur_lr:>10.6f}")

        if va_loss < best_loss - 1e-4:
            best_loss    = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n[STOP] Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # 10. Restore best checkpoint
    model.load_state_dict(best_weights)

    # 11. Final evaluation
    _, final_preds, final_labels = eval_epoch(model, te_dl, criterion, device)
    print(f"\n{sep}")
    print("CLASSIFICATION REPORT (GRU v3 — continuous features only)")
    print(sep)
    print(classification_report(final_labels, final_preds,
                                target_names=["SELL", "WAIT", "BUY"]))

    print("CONFUSION MATRIX  [rows=actual, cols=predicted]")
    print("       SELL  WAIT  BUY")
    cm = confusion_matrix(final_labels, final_preds)
    for lbl, row in zip(["SELL", "WAIT", "BUY"], cm):
        print(f"  {lbl:>4}  {row}")

    # 12. Save model + scaler
    # GRU_FEATURES is saved IN the checkpoint so indian_ai_engine.py
    # knows exactly which features to extract from Redis for GRU input.
    torch.save({
        "state_dict":  model.state_dict(),
        "input_size":  len(GRU_FEATURES),
        "hidden":      HIDDEN,
        "dropout":     DROPOUT,
        "seq_len":     SEQ_LEN,
        "gru_features": GRU_FEATURES,   # ← key: ai_engine reads this
        "arch":        "GRU-v3",
    }, MODEL_OUT)

    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n[SAVED] GRU model  -> {MODEL_OUT}  ({len(GRU_FEATURES)} features)")
    print(f"[SAVED] Scaler     -> {SCALER_OUT}")
    print(f"\n[NEXT]  Re-enable GRU in indian_ai_engine.py:")
    print(f"        Set LSTM_WEIGHT = 0.20 and XGB_WEIGHT = 0.80")
    print(f"{sep}\n")


if __name__ == "__main__":
    train()
