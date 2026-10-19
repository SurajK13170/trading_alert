"""
NIFTY 50 - WALK-FORWARD VALIDATOR
-----------------------------------
Tests whether the model's edge holds CONSISTENTLY across time,
not just on one lucky train/test split.

Method (Expanding Window):
  Window 1: Train 2016-2020 (5y)  → Test 2021 (1y)
  Window 2: Train 2016-2021 (6y)  → Test 2022 (1y)
  Window 3: Train 2016-2022 (7y)  → Test 2023 (1y)
  Window 4: Train 2016-2023 (8y)  → Test 2024 (1y)
  Window 5: Train 2016-2024 (9y)  → Test 2025+ (remaining)

For each window:
  - Train a fresh XGBoost (same hyperparams as production)
  - Test on the next unseen year
  - Report: accuracy, win rate, gross return, Sharpe, max DD

If accuracy and win rate are consistent across all windows,
the edge is real. If they vary wildly, the model is curve-fitting.

Run: python walk_forward.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_trainer import (
    load_and_clean, add_features, make_targets, FEATURES,
    CHOP_FILTER_PCTILE, TIME_DECAY_RECENT, CalibratedXGB,
)

# ----------------------------------------------------------------
CSV_PATH        = "nifty_5m_data.csv"
CONF_THRESHOLD  = 0.55
ATR_SL_MULT     = 1.0
ATR_TP_MULT     = 2.0
INITIAL_CAP     = 100_000
RISK_PCT        = 1.0
SLIPPAGE_PTS    = 2.0
BROKERAGE       = 40.0
CHART_OUT       = "walk_forward_results.png"

# Minimum years of training data before first test window
MIN_TRAIN_YEARS = 5
# Each test window size in months
TEST_WINDOW_MONTHS = 12
# ----------------------------------------------------------------


def _train_xgb_on_df(df_train: pd.DataFrame) -> CalibratedXGB:
    """Train a production-identical XGBoost + isotonic calibration on df_train."""
    df = df_train.copy()

    # Chop filter
    chop_thresh = df["ATR_pct"].quantile(CHOP_FILTER_PCTILE)
    df = df[df["ATR_pct"] >= chop_thresh].reset_index(drop=True)

    X = df[FEATURES]
    y = df["target"]

    if len(df) < 500:
        raise ValueError(f"Not enough training samples: {len(df)}")

    # 90% train / 10% calibration within this fold (no further test leak)
    split_calib = int(len(X) * 0.90)
    X_tr, X_calib = X.iloc[:split_calib], X.iloc[split_calib:]
    y_tr, y_calib = y.iloc[:split_calib], y.iloc[split_calib:]

    # Sample weights: class balance + time decay
    counts        = y_tr.value_counts()
    max_count     = counts.max()
    class_weights = {cls: max_count / cnt for cls, cnt in counts.items()}
    class_weights[1] = class_weights.get(1, 1) * 1.5
    sw = y_tr.map(class_weights).values.astype(float)
    recent = int(len(sw) * 0.75)
    sw[recent:] *= TIME_DECAY_RECENT

    model = xgb.XGBClassifier(
        n_estimators          = 1200,
        max_depth             = 5,
        learning_rate         = 0.015,
        subsample             = 0.75,
        colsample_bytree      = 0.75,
        min_child_weight      = 25,
        gamma                 = 3,
        reg_alpha             = 0.3,
        reg_lambda            = 2.5,
        objective             = "multi:softprob",
        num_class             = 3,
        eval_metric           = "mlogloss",
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 60,
    )
    model.fit(
        X_tr, y_tr,
        sample_weight = sw,
        eval_set      = [(X_calib, y_calib)],
        verbose       = False,
    )

    raw_calib = model.predict_proba(X_calib)
    y_calib_arr = y_calib.values
    calibrators = []
    for k in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_calib[:, k], (y_calib_arr == k).astype(float))
        calibrators.append(ir)

    return CalibratedXGB(model, calibrators)


def _backtest_fold(df_test: pd.DataFrame, model: CalibratedXGB) -> dict:
    """
    Simulate trades on df_test using the given model.
    Returns metrics dict with gross + net PnL, win rate, Sharpe, max DD.
    """
    df   = df_test.reset_index(drop=True)
    n    = len(df)

    capital  = float(INITIAL_CAP)
    equity   = [capital]
    trades   = []

    in_trade    = False
    side        = None
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    qty         = 0

    for i in range(n - 1):
        row      = df.iloc[i]
        next_row = df.iloc[i + 1]

        if in_trade:
            h, l = float(row["high"]), float(row["low"])
            if side == "BUY":
                if l <= sl_price:
                    gross = (sl_price - entry_price) * qty
                    costs = _costs(entry_price, sl_price, qty)
                    capital += gross - costs
                    trades.append({"pnl": gross - costs, "gross": gross})
                    in_trade = False
                elif h >= tp_price:
                    gross = (tp_price - entry_price) * qty
                    costs = _costs(entry_price, tp_price, qty)
                    capital += gross - costs
                    trades.append({"pnl": gross - costs, "gross": gross})
                    in_trade = False
            elif side == "SELL":
                if h >= sl_price:
                    gross = (entry_price - sl_price) * qty
                    costs = _costs(entry_price, sl_price, qty)
                    capital += gross - costs
                    trades.append({"pnl": gross - costs, "gross": gross})
                    in_trade = False
                elif l <= tp_price:
                    gross = (entry_price - tp_price) * qty
                    costs = _costs(entry_price, tp_price, qty)
                    capital += gross - costs
                    trades.append({"pnl": gross - costs, "gross": gross})
                    in_trade = False

        if not in_trade:
            try:
                X    = row[FEATURES].values.reshape(1, -1)
                prob = model.predict_proba(X)[0]
            except Exception:
                equity.append(capital)
                continue

            p_sell, p_buy = float(prob[0]), float(prob[2])
            entry = float(next_row["open"])
            atr   = float(row["ATR"])

            if p_buy >= CONF_THRESHOLD and p_buy > p_sell:
                sl = entry - atr * ATR_SL_MULT
                tp = entry + atr * ATR_TP_MULT
                q  = int((capital * RISK_PCT / 100) / max(abs(entry - sl), 1))
                if q > 0:
                    in_trade, side            = True, "BUY"
                    entry_price, sl_price, tp_price = entry, sl, tp
                    qty = q
            elif p_sell >= CONF_THRESHOLD and p_sell > p_buy:
                sl = entry + atr * ATR_SL_MULT
                tp = entry - atr * ATR_TP_MULT
                q  = int((capital * RISK_PCT / 100) / max(abs(entry - sl), 1))
                if q > 0:
                    in_trade, side            = True, "SELL"
                    entry_price, sl_price, tp_price = entry, sl, tp
                    qty = q

        equity.append(capital)

    if not trades:
        return {"trades": 0, "win_rate": 0, "net_return": 0,
                "gross_return": 0, "sharpe": 0, "max_dd": 0}

    wins        = [t for t in trades if t["pnl"] > 0]
    eq          = np.array(equity)
    peak        = np.maximum.accumulate(eq)
    dd          = (peak - eq) / peak * 100
    rets        = np.diff(eq) / eq[:-1]
    rets_nz     = rets[rets != 0]
    sharpe      = (rets_nz.mean() / rets_nz.std() * np.sqrt(1512)) if len(rets_nz) > 1 else 0.0

    total_net   = sum(t["pnl"]   for t in trades)
    total_gross = sum(t["gross"] for t in trades)

    return {
        "trades":        len(trades),
        "win_rate":      len(wins) / len(trades) * 100,
        "net_return":    total_net   / INITIAL_CAP * 100,
        "gross_return":  total_gross / INITIAL_CAP * 100,
        "cost_drag":     (total_gross - total_net) / INITIAL_CAP * 100,
        "sharpe":        round(sharpe, 3),
        "max_dd":        round(float(np.max(dd)), 2),
        "final_capital": INITIAL_CAP + total_net,
        "equity":        equity,
    }


def _costs(entry, exit_p, qty):
    slip    = 2 * SLIPPAGE_PTS * qty
    sttt    = 0.0001 * exit_p * qty
    exch    = 0.0000053 * (entry + exit_p) * qty
    gst     = 0.18 * BROKERAGE
    stamp   = 0.00003 * entry * qty
    return slip + BROKERAGE + sttt + exch + gst + stamp


def _classify_accuracy(df_test: pd.DataFrame, model: CalibratedXGB) -> float:
    df = df_test.copy()
    chop_thresh = df["ATR_pct"].quantile(CHOP_FILTER_PCTILE)
    df = df[df["ATR_pct"] >= chop_thresh]
    if df.empty:
        return 0.0
    X = df[FEATURES]
    y = df["target"]
    preds = model.predict(X)
    return accuracy_score(y, preds) * 100


def run_walk_forward():
    sep = "=" * 65

    print(f"\n{sep}")
    print("  NIFTY 50 — WALK-FORWARD VALIDATION")
    print(f"  Method  : Expanding window, {TEST_WINDOW_MONTHS}-month test folds")
    print(f"  Min train: {MIN_TRAIN_YEARS} years before first test")
    print(f"  Costs   : slippage={SLIPPAGE_PTS}pts, brokerage=₹{BROKERAGE:.0f}/trade")
    print(f"{sep}\n")

    # 1. Load + feature full dataset ONCE (expensive, done once)
    print("[1/3] Loading and featurising full dataset (this takes a few minutes)...")
    df_full = load_and_clean(CSV_PATH)
    df_full = add_features(df_full)
    df_full = make_targets(df_full)
    df_full.dropna(subset=FEATURES + ["target", "ATR", "open", "high", "low", "close"],
                   inplace=True)
    df_full.reset_index(drop=True, inplace=True)
    print(f"[OK] Full dataset: {len(df_full):,} rows | "
          f"{df_full['date'].iloc[0].date()} → {df_full['date'].iloc[-1].date()}\n")

    # 2. Build walk-forward windows
    df_full["_year_month"] = df_full["date"].dt.to_period("M")
    all_months   = df_full["_year_month"].unique()
    all_months   = sorted(all_months)

    start_month  = all_months[0]
    min_train_m  = MIN_TRAIN_YEARS * 12

    # Find test window starts
    windows = []
    test_start_idx = min_train_m  # first test window starts after MIN_TRAIN_YEARS
    while test_start_idx + TEST_WINDOW_MONTHS <= len(all_months):
        train_end_month = all_months[test_start_idx - 1]
        test_start_month = all_months[test_start_idx]
        test_end_idx = min(test_start_idx + TEST_WINDOW_MONTHS - 1, len(all_months) - 1)
        test_end_month = all_months[test_end_idx]
        windows.append({
            "train_from": start_month,
            "train_to":   train_end_month,
            "test_from":  test_start_month,
            "test_to":    test_end_month,
        })
        test_start_idx += TEST_WINDOW_MONTHS

    print(f"[2/3] Running {len(windows)} walk-forward windows...\n")

    results = []
    for i, w in enumerate(windows, 1):
        print(f"  Window {i}/{len(windows)}: "
              f"Train {w['train_from']}→{w['train_to']} | "
              f"Test {w['test_from']}→{w['test_to']}")

        df_train = df_full[
            (df_full["_year_month"] >= w["train_from"]) &
            (df_full["_year_month"] <= w["train_to"])
        ].copy()
        df_test  = df_full[
            (df_full["_year_month"] >= w["test_from"]) &
            (df_full["_year_month"] <= w["test_to"])
        ].copy()

        if len(df_train) < 5000 or len(df_test) < 100:
            print(f"    [SKIP] Not enough data (train={len(df_train)}, test={len(df_test)})\n")
            continue

        # Train fresh model
        try:
            model = _train_xgb_on_df(df_train)
        except Exception as e:
            print(f"    [ERR] Training failed: {e}\n")
            continue

        # Accuracy on test set
        acc = _classify_accuracy(df_test, model)

        # Backtest on test set
        metrics = _backtest_fold(df_test, model)

        row = {
            "window":        i,
            "train_from":    str(w["train_from"]),
            "train_to":      str(w["train_to"]),
            "test_from":     str(w["test_from"]),
            "test_to":       str(w["test_to"]),
            "train_rows":    len(df_train),
            "test_rows":     len(df_test),
            "accuracy":      round(acc, 1),
            **{k: v for k, v in metrics.items() if k != "equity"},
        }
        results.append({**row, "_equity": metrics.get("equity", [])})

        print(f"    Accuracy  : {acc:.1f}%   (random=33%)")
        print(f"    Trades    : {metrics['trades']}   Win rate: {metrics['win_rate']:.1f}%")
        print(f"    Gross ret : {metrics['gross_return']:+.2f}%")
        print(f"    Cost drag : -{metrics['cost_drag']:.2f}%")
        print(f"    NET ret   : {metrics['net_return']:+.2f}%   "
              f"Sharpe: {metrics['sharpe']:.2f}   Max DD: {metrics['max_dd']:.1f}%")
        print()

    if not results:
        print("[ERR] No windows completed. Check data.")
        return

    # 3. Summary table
    print(f"\n{sep}")
    print("  WALK-FORWARD SUMMARY")
    print(sep)
    print(f"  {'Win':<6} {'Period':<22} {'Acc':>5} {'Trades':>7} {'WR%':>6} "
          f"{'Gross':>8} {'Costs':>7} {'NET':>8} {'Sharpe':>7} {'MaxDD':>7}")
    print("  " + "-" * 63)
    for r in results:
        period = f"{r['test_from']}→{r['test_to']}"
        print(f"  {r['window']:<6} {period:<22} {r['accuracy']:>4.1f}% "
              f"{r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['gross_return']:>+7.1f}% "
              f"{r['cost_drag']:>6.1f}% "
              f"{r['net_return']:>+7.1f}% "
              f"{r['sharpe']:>6.2f} "
              f"{r['max_dd']:>6.1f}%")

    accs        = [r["accuracy"]    for r in results]
    win_rates   = [r["win_rate"]    for r in results]
    net_returns = [r["net_return"]  for r in results]
    sharpes     = [r["sharpe"]      for r in results]

    print("  " + "-" * 63)
    print(f"  {'AVG':<29} {np.mean(accs):>4.1f}% "
          f"{'':>6} {np.mean(win_rates):>5.1f}% "
          f"{'':>8} {'':>7} "
          f"{np.mean(net_returns):>+7.1f}% "
          f"{np.mean(sharpes):>6.2f}")
    print(f"  {'STD':<29} {np.std(accs):>4.1f}% "
          f"{'':>6} {np.std(win_rates):>5.1f}% "
          f"{'':>8} {'':>7} "
          f"{np.std(net_returns):>7.1f}%")

    profitable = sum(1 for r in results if r["net_return"] > 0)
    print(f"\n  Profitable windows : {profitable}/{len(results)}")
    print(f"  Consistency note   : Low STD across windows = real edge. "
          f"High STD = luck / curve-fitting.")
    print(sep)

    # 4. Verdict
    avg_acc = np.mean(accs)
    avg_wr  = np.mean(win_rates)
    avg_net = np.mean(net_returns)
    print("\n  VERDICT:")
    if avg_acc >= 45 and avg_wr >= 50 and profitable >= len(results) * 0.7:
        print("  ✓ EDGE CONFIRMED — model generalises well across time windows.")
        print(f"    Avg accuracy {avg_acc:.1f}% > 33% random | "
              f"Win rate {avg_wr:.1f}% | "
              f"{profitable}/{len(results)} windows profitable")
    elif avg_acc >= 40 and profitable >= len(results) * 0.5:
        print("  ~ MARGINAL EDGE — model has some predictive power but is inconsistent.")
        print(f"    Avg accuracy {avg_acc:.1f}% | "
              f"Win rate {avg_wr:.1f}% | "
              f"{profitable}/{len(results)} windows profitable")
        print("    → Add more features or data, or reduce confidence threshold.")
    else:
        print("  ✗ WEAK/NO EDGE — model may be overfitting to historical data.")
        print(f"    Avg accuracy {avg_acc:.1f}% | "
              f"{profitable}/{len(results)} windows profitable")
        print("    → Do NOT go live. Review labeling, features, or hyperparameters.")
    print()

    # 5. Plot
    _plot_walk_forward(results)

    # 6. Save results CSV
    out_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_equity"}
                            for r in results])
    out_df.to_csv("walk_forward_results.csv", index=False)
    print(f"[SAVED] Results -> walk_forward_results.csv")
    print(f"[SAVED] Chart   -> {CHART_OUT}\n")


def _plot_walk_forward(results: list):
    n   = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes.flat:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333333")

    labels = [f"W{r['window']}\n{r['test_from'][:7]}" for r in results]
    x      = np.arange(n)

    def bar(ax, vals, title, ylabel, color, ref=None):
        bars = ax.bar(x, vals, color=color, alpha=0.8, width=0.6)
        if ref is not None:
            ax.axhline(ref, color="#ffffff", linewidth=1, linestyle="--",
                       label=f"ref={ref}")
            ax.legend(facecolor="#222222", labelcolor="white", fontsize=8)
        ax.set_title(title, color="white", fontsize=10)
        ax.set_ylabel(ylabel, color="#aaaaaa")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(axis="y", color="#222222", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                    f"{v:.1f}", ha="center", va="bottom", color="white", fontsize=8)

    bar(axes[0][0], [r["accuracy"]   for r in results], "Model Accuracy %",
        "Accuracy %", "#00e5ff", ref=33)
    bar(axes[0][1], [r["win_rate"]   for r in results], "Backtest Win Rate %",
        "Win Rate %",  "#00ff88", ref=50)
    bar(axes[1][0], [r["net_return"] for r in results], "NET Return % (after costs)",
        "Return %",    "#ffaa00", ref=0)
    bar(axes[1][1], [r["sharpe"]     for r in results], "Sharpe Ratio",
        "Sharpe",      "#cc88ff", ref=1)

    fig.suptitle("Walk-Forward Validation — NIFTY 50 AI Model",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout(pad=2.0)
    plt.savefig(CHART_OUT, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()


if __name__ == "__main__":
    run_walk_forward()
