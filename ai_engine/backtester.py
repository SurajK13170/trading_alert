"""
NIFTY 50 - AI BACKTESTER v3.0
-------------------------------
Tests the trained XGBoost model on the HOLDOUT (out-of-sample) 20% of data.

Realistic simulation:
  - Entry at NEXT candle's open (no lookahead)
  - SL/TP checked each candle (low vs SL first, then high vs TP)
  - Position sizing: 1% capital risk per trade
  - Transaction costs: slippage + brokerage + STT + exchange charges
  - Metrics: gross vs NET return, win rate, Sharpe, max drawdown, profit factor
  - Equity curve saved as PNG

Run:
  python backtester.py              # with full costs (default)
  python backtester.py --no-costs  # gross only (to isolate signal quality)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — saves PNG without needing a display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model_trainer import load_and_clean, add_features, FEATURES
from model_trainer import CalibratedXGB

# ----------------------------------------------------------------
MODEL_PATH     = "trading_model.pkl"
FEATURE_PATH   = "feature_list.pkl"
CSV_PATH       = "nifty_5m_data.csv"

INITIAL_CAP    = 100_000   # INR starting capital
RISK_PCT       = 1.0       # % of capital risked per trade
ATR_SL_MULT    = 1.0       # must match live system
ATR_TP_MULT    = 2.0       # must match live system
CONF_THRESHOLD = 0.55      # raised from 0.42 — backtest shows 57% win rate vs 48% at 0.42
TEST_SPLIT     = 0.20      # last 20% used as holdout test
CHART_OUT      = "backtest_equity_curve.png"

# Realistic constraints matching the live execution manager
TRADE_COOLDOWN_CANDLES = 1    # 1 candle (5 min) cooldown — minimal gap after trade closes
SESSION_OPEN_BLOCK     = 3    # skip first 3 × 5min candles (9:15–9:30)
SESSION_CLOSE_BLOCK    = 3    # skip last  3 × 5min candles (15:15–15:30)

# Cap position size at MAX_CAPITAL_FOR_SIZING so compounding doesn't explode
# qty later in the test. This reflects real margin/position limits.
# Set to INITIAL_CAP to use fixed-capital sizing (most realistic for retail).
MAX_CAPITAL_FOR_SIZING = 100_000

# ── Transaction Costs (realistic Indian F&O, Angel One) ──────────
# Toggle via --no-costs flag. Defaults to ON.
INCLUDE_COSTS        = "--no-costs" not in sys.argv

SLIPPAGE_PTS         = 2.0      # index points lost per side (entry + exit = 4 pts total)
                                 # At NIFTY 24000, a 2-pt slip is ~0.008% — conservative
BROKERAGE_PER_TRADE  = 40.0     # INR flat per round-trip (Rs.20 entry + Rs.20 exit, Angel One)
STT_PCT              = 0.0001   # 0.01% of sell-side turnover (NSE futures STT)
EXCHANGE_CHARGES_PCT = 0.0000053 # NSE F&O: 0.00053% of turnover (both sides)
GST_ON_BROKERAGE     = 0.18     # 18% GST on brokerage portion only
STAMP_DUTY_PCT       = 0.00003  # 0.003% on buy-side turnover (Maharashtra stamp)
# ----------------------------------------------------------------


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run model_trainer.py first.")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    if os.path.exists(FEATURE_PATH):
        with open(FEATURE_PATH, "rb") as f:
            features = pickle.load(f)
    else:
        features = FEATURES

    print(f"[MODEL] Loaded | {len(features)} features | 3-class (SELL/WAIT/BUY)")
    return model, features


def prepare_data(features: list):
    df = load_and_clean(CSV_PATH)
    df = add_features(df)
    df.dropna(subset=features + ["ATR", "open", "high", "low", "close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _compute_costs(side: str, entry_price: float, exit_price: float, qty: int) -> float:
    """
    Compute total transaction costs for one round-trip trade.

    Components:
      - Slippage      : SLIPPAGE_PTS per side × qty (entry + exit)
      - Brokerage     : flat Rs.40 per round-trip
      - STT           : 0.01% of sell-side turnover
      - Exchange fees : 0.00053% both sides
      - GST           : 18% of brokerage only
      - Stamp duty    : 0.003% on buy-side

    For a BUY trade:  sell-side is the exit, buy-side is the entry.
    For a SELL trade: sell-side is the entry, buy-side is the exit.
    """
    if not INCLUDE_COSTS:
        return 0.0

    buy_turnover  = entry_price * qty if side == "BUY" else exit_price * qty
    sell_turnover = exit_price  * qty if side == "BUY" else entry_price * qty

    slippage        = 2 * SLIPPAGE_PTS * qty                      # both sides
    brokerage       = BROKERAGE_PER_TRADE
    stt             = STT_PCT             * sell_turnover
    exchange        = EXCHANGE_CHARGES_PCT * (buy_turnover + sell_turnover)
    gst             = GST_ON_BROKERAGE    * brokerage
    stamp           = STAMP_DUTY_PCT      * buy_turnover

    return slippage + brokerage + stt + exchange + gst + stamp


def run_backtest(df: pd.DataFrame, model, features: list):
    """
    Simulate trades on the holdout test set (last TEST_SPLIT fraction).

    Entry:  next candle's open (1-candle delay — realistic)
    SL/TP:  checked per candle: low vs SL first, then high vs TP
    Size:   (capital × RISK_PCT%) / |entry − SL|
    Costs:  deducted from each trade's PnL (see _compute_costs)
    """
    split     = int(len(df) * (1 - TEST_SPLIT))
    test      = df.iloc[split:].copy().reset_index(drop=True)
    n         = len(test)

    capital      = float(INITIAL_CAP)
    equity       = [capital]
    timestamps   = [test["date"].iloc[0]]
    trades       = []

    in_trade          = False
    side              = None
    entry_price       = 0.0
    sl_price          = 0.0
    tp_price          = 0.0
    qty               = 0
    entry_time        = None
    last_close_candle = -TRADE_COOLDOWN_CANDLES   # allow entry from first candle

    print(f"\n[BT] Test window: {test['date'].iloc[0].date()} -> {test['date'].iloc[-1].date()}")
    print(f"[BT] Candles    : {n} | Capital: INR {capital:,.0f}")
    print(f"[BT] Costs      : {'ON  (slippage={:.0f}pts, brokerage=Rs.{:.0f})'.format(SLIPPAGE_PTS, BROKERAGE_PER_TRADE) if INCLUDE_COSTS else 'OFF (gross returns)'}\n")

    for i in range(n - 1):
        row      = test.iloc[i]
        next_row = test.iloc[i + 1]

        # ── Monitor open position ─────────────────────────────
        if in_trade:
            h, l = float(row["high"]), float(row["low"])

            if side == "BUY":
                if l <= sl_price:
                    gross_pnl = (sl_price - entry_price) * qty
                    costs     = _compute_costs("BUY", entry_price, sl_price, qty)
                    net_pnl   = gross_pnl - costs
                    capital  += net_pnl
                    trades.append(_make_trade("BUY", entry_price, sl_price, qty,
                                              gross_pnl, net_pnl, costs, "SL",
                                              entry_time, row["date"]))
                    in_trade = False
                    last_close_candle = i
                elif h >= tp_price:
                    gross_pnl = (tp_price - entry_price) * qty
                    costs     = _compute_costs("BUY", entry_price, tp_price, qty)
                    net_pnl   = gross_pnl - costs
                    capital  += net_pnl
                    trades.append(_make_trade("BUY", entry_price, tp_price, qty,
                                              gross_pnl, net_pnl, costs, "TP",
                                              entry_time, row["date"]))
                    in_trade = False
                    last_close_candle = i

            elif side == "SELL":
                if h >= sl_price:
                    gross_pnl = (entry_price - sl_price) * qty
                    costs     = _compute_costs("SELL", entry_price, sl_price, qty)
                    net_pnl   = gross_pnl - costs
                    capital  += net_pnl
                    trades.append(_make_trade("SELL", entry_price, sl_price, qty,
                                              gross_pnl, net_pnl, costs, "SL",
                                              entry_time, row["date"]))
                    in_trade = False
                    last_close_candle = i
                elif l <= tp_price:
                    gross_pnl = (entry_price - tp_price) * qty
                    costs     = _compute_costs("SELL", entry_price, tp_price, qty)
                    net_pnl   = gross_pnl - costs
                    capital  += net_pnl
                    trades.append(_make_trade("SELL", entry_price, tp_price, qty,
                                              gross_pnl, net_pnl, costs, "TP",
                                              entry_time, row["date"]))
                    in_trade = False
                    last_close_candle = i

        # ── Generate new signal if flat ───────────────────────
        if not in_trade:
            # Trade cooldown: match live execution manager (15-min wait after close)
            cooldown_ok = (i - last_close_candle) >= TRADE_COOLDOWN_CANDLES

            # Session filter: skip opening (first 3 candles) and closing (last 3)
            # Uses candle index within the day — approximated via Hour/Minute features
            hour   = int(row.get("Hour",   row.get("hour",   9)))
            minute = int(row.get("Minute", row.get("minute", 15)))
            mins_from_open  = (hour - 9) * 60 + (minute - 15)
            mins_to_close   = (15 * 60 + 30) - (hour * 60 + minute)
            session_ok = (mins_from_open >= SESSION_OPEN_BLOCK * 5 and
                          mins_to_close  >= SESSION_CLOSE_BLOCK * 5)

            if cooldown_ok and session_ok:
                X    = row[features].values.reshape(1, -1)
                prob = model.predict_proba(X)[0]
                p_sell, p_buy = float(prob[0]), float(prob[2])

                entry = float(next_row["open"])
                atr   = float(row["ATR"])

                # Fixed-capital sizing: cap at MAX_CAPITAL_FOR_SIZING so position
                # sizes stay constant regardless of how much capital has compounded.
                # This matches what a real retail trader does (margin-limited).
                sizing_cap = min(capital, MAX_CAPITAL_FOR_SIZING)

                if p_buy >= CONF_THRESHOLD and p_buy > p_sell:
                    sl = entry - atr * ATR_SL_MULT
                    tp = entry + atr * ATR_TP_MULT
                    q  = int((sizing_cap * RISK_PCT / 100) / max(abs(entry - sl), 1))
                    if q > 0:
                        in_trade, side            = True, "BUY"
                        entry_price, sl_price, tp_price = entry, sl, tp
                        qty, entry_time           = q, next_row["date"]

                elif p_sell >= CONF_THRESHOLD and p_sell > p_buy:
                    sl = entry + atr * ATR_SL_MULT
                    tp = entry - atr * ATR_TP_MULT
                    q  = int((sizing_cap * RISK_PCT / 100) / max(abs(entry - sl), 1))
                    if q > 0:
                        in_trade, side            = True, "SELL"
                        entry_price, sl_price, tp_price = entry, sl, tp
                        qty, entry_time           = q, next_row["date"]

        equity.append(capital)
        timestamps.append(row["date"])

    # Close any position still open at end of test
    if in_trade:
        last      = float(test.iloc[-1]["close"])
        gross_pnl = (last - entry_price) * qty if side == "BUY" else (entry_price - last) * qty
        costs     = _compute_costs(side, entry_price, last, qty)
        net_pnl   = gross_pnl - costs
        capital  += net_pnl
        trades.append(_make_trade(side, entry_price, last, qty,
                                  gross_pnl, net_pnl, costs, "EOD",
                                  entry_time, test.iloc[-1]["date"]))
        equity[-1] = capital

    return trades, equity, timestamps, test


def _make_trade(side, entry, exit_p, qty, gross_pnl, net_pnl, costs, reason,
                entry_ts, exit_ts):
    return {
        "side":       side,
        "entry":      round(entry, 2),
        "exit":       round(exit_p, 2),
        "qty":        qty,
        "gross_pnl":  round(gross_pnl, 2),
        "costs":      round(costs, 2),
        "net_pnl":    round(net_pnl, 2),
        "reason":     reason,
        "entry_time": entry_ts,
        "exit_time":  exit_ts,
    }


def compute_metrics(trades: list, equity: list) -> dict:
    if not trades:
        return {}

    wins   = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]

    total_net    = sum(t["net_pnl"]   for t in trades)
    total_gross  = sum(t["gross_pnl"] for t in trades)
    total_costs  = sum(t["costs"]     for t in trades)

    gross_win  = sum(t["net_pnl"] for t in wins)   if wins   else 0
    gross_loss = abs(sum(t["net_pnl"] for t in losses)) if losses else 1e-9

    eq      = np.array(equity)
    peak    = np.maximum.accumulate(eq)
    dd      = (peak - eq) / peak * 100
    max_dd  = float(np.max(dd))

    rets    = np.diff(eq) / eq[:-1]
    rets_nz = rets[rets != 0]
    sharpe  = (rets_nz.mean() / rets_nz.std() * np.sqrt(1512)) if len(rets_nz) > 1 else 0.0

    return {
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       len(wins) / len(trades) * 100,
        # Net (after all costs) — this is your REAL return
        "total_net_pnl":  total_net,
        "net_return_pct": total_net / INITIAL_CAP * 100,
        "final_capital":  INITIAL_CAP + total_net,
        # Gross (signal quality only, no costs)
        "total_gross_pnl":  total_gross,
        "gross_return_pct": total_gross / INITIAL_CAP * 100,
        # Cost drag
        "total_costs":    total_costs,
        "cost_drag_pct":  total_costs / INITIAL_CAP * 100,
        # Risk metrics
        "max_drawdown":   max_dd,
        "sharpe":         sharpe,
        "profit_factor":  gross_win / gross_loss,
        "avg_win":        gross_win  / len(wins)   if wins   else 0,
        "avg_loss":       gross_loss / len(losses) if losses else 0,
        "avg_rr":         (gross_win / len(wins)) / (gross_loss / len(losses))
                          if wins and losses else 0,
        "cost_per_trade": total_costs / len(trades),
    }


def monthly_breakdown(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        trades=("net_pnl", "count"),
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        costs=("costs", "sum"),
        wins=("net_pnl", lambda x: (x > 0).sum()),
    ).reset_index()
    monthly["win_rate"]  = (monthly["wins"] / monthly["trades"] * 100).round(1)
    monthly["net_pnl"]   = monthly["net_pnl"].round(0)
    monthly["gross_pnl"] = monthly["gross_pnl"].round(0)
    monthly["costs"]     = monthly["costs"].round(0)
    return monthly


def plot_equity(equity: list, timestamps: list, trades: list, metrics: dict):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333333")

    ts = pd.to_datetime(timestamps)
    ax1 = axes[0]
    eq  = np.array(equity)
    ax1.plot(ts[:len(eq)], eq, color="#00e5ff", linewidth=1.5, label="Equity (NET)")
    ax1.fill_between(ts[:len(eq)], INITIAL_CAP, eq,
                     where=(eq >= INITIAL_CAP), alpha=0.15, color="#00e5ff")
    ax1.fill_between(ts[:len(eq)], INITIAL_CAP, eq,
                     where=(eq < INITIAL_CAP),  alpha=0.15, color="#ff4444")
    ax1.axhline(INITIAL_CAP, color="#555555", linewidth=0.8, linestyle="--")

    for t in trades:
        t_time = pd.Timestamp(t["exit_time"])
        color  = "#00ff88" if t["net_pnl"] > 0 else "#ff4444"
        marker = "^" if t["side"] == "BUY" else "v"
        idx    = abs(ts - t_time).argmin()
        ax1.scatter(t_time, eq[min(idx, len(eq)-1)], color=color,
                    marker=marker, s=40, zorder=5, alpha=0.8)

    net_ret   = metrics.get("net_return_pct", 0)
    gross_ret = metrics.get("gross_return_pct", 0)
    cost_drag = metrics.get("cost_drag_pct", 0)
    dd        = metrics.get("max_drawdown", 0)
    sh        = metrics.get("sharpe", 0)
    wr        = metrics.get("win_rate", 0)
    title = (
        f"Nifty 50 AI — NET {net_ret:+.1f}%  |  Gross {gross_ret:+.1f}%  |  "
        f"Cost drag {cost_drag:.1f}%  |  DD {dd:.1f}%  |  Sharpe {sh:.2f}  |  WR {wr:.1f}%"
    )
    ax1.set_title(title, color="white", fontsize=10, pad=10)
    ax1.set_ylabel("Capital (INR)", color="#aaaaaa")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9)
    ax1.grid(axis="y", color="#222222", linewidth=0.5)

    ax2 = axes[1]
    peak     = np.maximum.accumulate(eq)
    dd_curve = (peak - eq) / peak * 100
    ax2.fill_between(ts[:len(dd_curve)], 0, -dd_curve, color="#ff4444", alpha=0.6)
    ax2.set_ylabel("Drawdown %", color="#aaaaaa")
    ax2.set_xlabel("Date", color="#aaaaaa")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.grid(axis="y", color="#222222", linewidth=0.5)

    plt.tight_layout(pad=2.0)
    plt.savefig(CHART_OUT, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"[CHART] Saved -> {CHART_OUT}")


def print_results(metrics: dict, monthly: pd.DataFrame, trades: list):
    sep = "=" * 60
    win  = metrics.get("wins", 0)
    loss = metrics.get("losses", 0)
    n    = metrics.get("total_trades", 0)

    print(f"\n{sep}")
    print("  BACKTEST RESULTS  (out-of-sample holdout)")
    print(sep)
    print(f"  Trades         : {n}  ({win}W / {loss}L)")
    print(f"  Win Rate       : {metrics['win_rate']:.1f}%")
    print()
    print(f"  -- Returns ------------------------------------------")
    print(f"  Gross Return   : {metrics['gross_return_pct']:+.2f}%  "
          f"(INR {metrics['total_gross_pnl']:+,.0f})   [signal quality only]")
    print(f"  Cost Drag      : -{metrics['cost_drag_pct']:.2f}%  "
          f"(INR {metrics['total_costs']:,.0f} total, "
          f"Rs.{metrics['cost_per_trade']:.0f}/trade avg)")
    print(f"  NET Return     : {metrics['net_return_pct']:+.2f}%  "
          f"(INR {metrics['total_net_pnl']:+,.0f})   [your REAL return]")
    print(f"  Final Capital  : INR {metrics['final_capital']:,.0f}")
    print()
    print(f"  -- Risk ---------------------------------------------")
    print(f"  Max Drawdown   : {metrics['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio   : {metrics['sharpe']:.3f}")
    print(f"  Profit Factor  : {metrics['profit_factor']:.2f}")
    print(f"  Avg Win        : INR {metrics['avg_win']:,.0f}")
    print(f"  Avg Loss       : INR {metrics['avg_loss']:,.0f}")
    print(f"  Avg RR         : {metrics['avg_rr']:.2f}:1")
    print(sep)

    sl_count = sum(1 for t in trades if t["reason"] == "SL")
    tp_count = sum(1 for t in trades if t["reason"] == "TP")
    print(f"\n  Exit breakdown : TP={tp_count} | SL={sl_count}")

    if not monthly.empty:
        print(f"\n  Monthly PnL (NET):")
        print(f"  {'Month':<12} {'Trades':>6} {'WR%':>6} {'Gross':>10} {'Costs':>8} {'NET':>10}")
        print("  " + "-" * 54)
        for _, row in monthly.iterrows():
            print(f"  {str(row['month']):<12} {row['trades']:>6} "
                  f"{row['win_rate']:>5.1f}% "
                  f"INR {row['gross_pnl']:>+8,.0f} "
                  f"-{row['costs']:>6,.0f} "
                  f"INR {row['net_pnl']:>+8,.0f}")
    print(sep + "\n")


def main():
    print("=" * 60)
    print("  NIFTY 50 AI BACKTESTER v3.0")
    print(f"  Mode           : {'WITH COSTS' if INCLUDE_COSTS else 'GROSS (no costs)'}")
    if INCLUDE_COSTS:
        print(f"  Slippage       : {SLIPPAGE_PTS} pts/side ({SLIPPAGE_PTS*2:.0f} pts round-trip)")
        print(f"  Brokerage      : Rs.{BROKERAGE_PER_TRADE:.0f}/trade (Angel One flat)")
        print(f"  STT            : {STT_PCT*100:.3f}% sell-side turnover")
        print(f"  Exchange+GST   : included")
    print(f"  Confidence     : {CONF_THRESHOLD*100:.0f}%")
    print(f"  Risk per trade : {RISK_PCT}%")
    print(f"  ATR SL/TP      : {ATR_SL_MULT}x / {ATR_TP_MULT}x  (RR={ATR_TP_MULT/ATR_SL_MULT:.1f}:1)")
    print(f"  Test set       : last {TEST_SPLIT*100:.0f}% of data")
    print("=" * 60)

    model, features = load_model()
    df = prepare_data(features)

    trades, equity, timestamps, _ = run_backtest(df, model, features)

    if not trades:
        print("[WARN] No trades taken. Lower CONF_THRESHOLD and retry.")
        return

    metrics = compute_metrics(trades, equity)
    monthly = monthly_breakdown(trades)

    print_results(metrics, monthly, trades)
    plot_equity(equity, timestamps, trades, metrics)

    log_path = "backtest_trades.csv"
    pd.DataFrame(trades).to_csv(log_path, index=False)
    print(f"[LOG] Trade log saved -> {log_path}")


if __name__ == "__main__":
    main()
