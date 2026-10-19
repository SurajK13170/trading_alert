"""
NIFTY 50 — LIVE-FILTER BACKTEST
---------------------------------
Mirrors the EXACT live system logic: same RSI blocks, ADX filter, MTF confluence
check, session blocks, cooldown, and dynamic SL/TP as indian_ai_engine.py.

The standard backtester.py has NO RSI/ADX/MTF filters — so it doesn't reflect what
the live system actually trades. This script closes that gap.

Run:
    python live_sim_backtest.py                  # full 2016-2026, all filters
    python live_sim_backtest.py --from 2021      # from 2021 onward
    python live_sim_backtest.py --from 2021 --to 2024
    python live_sim_backtest.py --no-costs       # gross only
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_trainer import load_and_clean, add_features, FEATURES, CalibratedXGB

# ── Settings — must match live system exactly ──────────────────────
MODEL_PATH   = "trading_model.pkl"
FEATURE_PATH = "feature_list.pkl"
CSV_PATH     = "nifty_5m_data.csv"

INITIAL_CAP          = 100_000
RISK_PCT             = 1.0
MAX_CAPITAL_SIZING   = 100_000

CONF_THRESHOLD       = 0.55

# Live filters (match indian_ai_engine.py)
RSI_OVERBOUGHT_BLOCK = 80.0   # BUY blocked if RSI >= this
RSI_OVERSOLD_BLOCK   = 20.0   # SELL blocked if RSI <= this
ADX_MIN              = 10.0   # signal blocked if ADX < this
MTF_MIN_CONFLUENCE   = 1      # need at least 1/3 timeframes aligned
SESSION_OPEN_BLOCK   = 3      # skip first 3 candles (15 min)
SESSION_CLOSE_BLOCK  = 3      # skip last 3 candles (15 min)
COOLDOWN_CANDLES     = 1      # 5 min cooldown after trade closes

# Transaction costs (match backtester.py)
INCLUDE_COSTS        = "--no-costs" not in sys.argv
SLIPPAGE_PTS         = 2.0
BROKERAGE            = 40.0
STT_PCT              = 0.0001
EXCHANGE_PCT         = 0.0000053
GST_RATE             = 0.18
STAMP_PCT            = 0.00003

# Date range (optional CLI args)
FROM_YEAR = None
TO_YEAR   = None
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--from" and i < len(sys.argv):
        FROM_YEAR = int(sys.argv[i + 1])
    if arg == "--to"   and i < len(sys.argv):
        TO_YEAR   = int(sys.argv[i + 1])

CHART_OUT = "live_sim_equity.png"


def _compute_costs(entry, exit_p, qty, side):
    if not INCLUDE_COSTS:
        return 0.0
    slip      = SLIPPAGE_PTS * 2 * qty
    brok      = BROKERAGE
    gst       = brok * GST_RATE
    stt       = exit_p * qty * STT_PCT
    exchange  = (entry + exit_p) * qty * EXCHANGE_PCT
    stamp     = entry * qty * STAMP_PCT
    return slip + brok + gst + stt + exchange + stamp


def _check_filters(row, signal):
    """Apply all live system filters. Returns (passes, reason_if_blocked)."""
    rsi     = float(row.get("RSI", 50))
    adx     = float(row.get("ADX", 0))
    ema_5m  = int(row.get("EMA_alignment",  0))
    ema_15m = int(row.get("EMA_align_15m",  0))
    ema_1h  = int(row.get("EMA_align_1h",   0))
    hour    = int(row.get("Hour",   9))
    minute  = int(row.get("Minute", 15))

    direction = 1 if signal == "BUY" else -1

    # RSI extreme block
    if signal == "BUY"  and rsi >= RSI_OVERBOUGHT_BLOCK:
        return False, f"RSI={rsi:.1f} >= {RSI_OVERBOUGHT_BLOCK}"
    if signal == "SELL" and rsi <= RSI_OVERSOLD_BLOCK:
        return False, f"RSI={rsi:.1f} <= {RSI_OVERSOLD_BLOCK}"

    # ADX filter
    if adx < ADX_MIN:
        return False, f"ADX={adx:.1f} < {ADX_MIN}"

    # Session open block
    mins_open = (hour - 9) * 60 + (minute - 15)
    if 0 <= mins_open < SESSION_OPEN_BLOCK * 5:
        return False, "Opening block"

    # Session close block
    mins_close = (15 * 60 + 30) - (hour * 60 + minute)
    if 0 <= mins_close <= SESSION_CLOSE_BLOCK * 5:
        return False, "Closing block"

    # MTF confluence
    score = sum([ema_5m == direction, ema_15m == direction, ema_1h == direction])
    if score < MTF_MIN_CONFLUENCE:
        return False, f"MTF {score}/3"

    return True, ""


def _sl_tp_mult(row):
    """Dynamic SL/TP multipliers — exact copy of indian_ai_engine.py logic."""
    bb_pctile    = float(row.get("BB_width_pctile", 0.5))
    atr_pctile   = float(row.get("ATR_pctile",      0.5))
    market_regime = int(row.get("Market_Regime",    0))
    adx          = float(row.get("ADX", 0))

    if bb_pctile < 0.15:
        return 1.5, 4.0, "SQUEEZE"
    elif market_regime == 4 or atr_pctile > 0.80:
        return 1.5, 3.0, "HIGH-VOL"
    elif market_regime in (1, 2) and adx > 30:
        return 1.0, 3.5, "TRENDING"
    elif market_regime == 0 and atr_pctile < 0.30:
        return 0.8, 2.0, "SIDEWAYS"
    else:
        return 1.0, 2.0, "DEFAULT"


def run():
    sep = "=" * 64

    # ── Load model ─────────────────────────────────────────────
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    features = pickle.load(open(FEATURE_PATH, "rb")) if os.path.exists(FEATURE_PATH) else FEATURES

    # ── Load data ──────────────────────────────────────────────
    print(f"\n[LOAD] Reading {CSV_PATH}...")
    df = load_and_clean(CSV_PATH)
    df = add_features(df)
    df.dropna(subset=features + ["ATR"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Date filter
    df["_year"] = pd.to_datetime(df["date"]).dt.year
    if FROM_YEAR:
        df = df[df["_year"] >= FROM_YEAR].reset_index(drop=True)
    if TO_YEAR:
        df = df[df["_year"] <= TO_YEAR].reset_index(drop=True)

    if df.empty:
        print("[ERR] No data in selected date range."); return

    date_start = pd.to_datetime(df["date"].iloc[0]).strftime("%Y-%m-%d")
    date_end   = pd.to_datetime(df["date"].iloc[-1]).strftime("%Y-%m-%d")

    print(f"\n{sep}")
    print("  NIFTY 50 - LIVE-FILTER BACKTEST")
    print(f"  Period         : {date_start}  ->  {date_end}")
    print(f"  Candles        : {len(df):,}")
    print(f"  Confidence     : {CONF_THRESHOLD*100:.0f}%")
    print(f"  RSI block      : BUY>={RSI_OVERBOUGHT_BLOCK}  SELL<={RSI_OVERSOLD_BLOCK}")
    print(f"  ADX minimum    : {ADX_MIN}")
    print(f"  MTF confluence : {MTF_MIN_CONFLUENCE}/3 timeframes")
    print(f"  Cooldown       : {COOLDOWN_CANDLES} candle(s) = {COOLDOWN_CANDLES*5} min")
    print(f"  Session block  : first/last {SESSION_OPEN_BLOCK*5} min")
    print(f"  Costs          : {'ON' if INCLUDE_COSTS else 'OFF (--no-costs)'}")
    print(f"  Capital        : Rs{INITIAL_CAP:,} (fixed sizing)")
    print(f"{sep}\n")

    # ── Backtest loop ──────────────────────────────────────────
    capital   = INITIAL_CAP
    in_trade  = False
    side      = None
    entry_p   = sl_p = tp_p = 0.0
    qty       = 0
    last_close_candle = -COOLDOWN_CANDLES

    trades = []
    equity = []

    for i in range(len(df) - 1):
        row      = df.iloc[i]
        next_row = df.iloc[i + 1]

        equity.append({"date": row["date"], "capital": capital})

        # ── Monitor open trade ─────────────────────────────────
        if in_trade:
            hi, lo = float(row["high"]), float(row["low"])
            if side == "BUY":
                sl_hit = lo <= sl_p
                tp_hit = hi >= tp_p
            else:
                sl_hit = hi >= sl_p
                tp_hit = lo <= tp_p

            if sl_hit or tp_hit:
                exit_p  = sl_p if sl_hit else tp_p
                gross   = (exit_p - entry_p) * qty if side == "BUY" else (entry_p - exit_p) * qty
                costs   = _compute_costs(entry_p, exit_p, qty, side)
                net     = gross - costs
                capital += net
                trades.append({
                    "date": row["date"], "side": side,
                    "entry": entry_p, "exit": exit_p,
                    "qty": qty, "gross": gross, "costs": costs, "net": net,
                    "result": "TP" if tp_hit else "SL",
                    "regime": row.get("Market_Regime", 0),
                })
                in_trade = False
                last_close_candle = i

        # ── Generate signal (only if flat + cooldown passed) ───
        if not in_trade and (i - last_close_candle) >= COOLDOWN_CANDLES:
            X     = row[features].values.reshape(1, -1)
            prob  = model.predict_proba(X)[0]
            p_sell, p_buy = float(prob[0]), float(prob[2])
            rsi   = float(row.get("RSI", 50))
            atr   = float(row.get("ATR", 1))
            entry = float(next_row["open"])

            sl_m, tp_m, _ = _sl_tp_mult(row)
            sizing = min(capital, MAX_CAPITAL_SIZING)

            if p_buy >= CONF_THRESHOLD and p_buy > p_sell and rsi < RSI_OVERBOUGHT_BLOCK:
                ok, _ = _check_filters(row, "BUY")
                if ok:
                    sl_c = entry - atr * sl_m
                    tp_c = entry + atr * tp_m
                    q = int((sizing * RISK_PCT / 100) / max(abs(entry - sl_c), 1))
                    if q > 0:
                        in_trade, side = True, "BUY"
                        entry_p, sl_p, tp_p, qty = entry, sl_c, tp_c, q

            elif p_sell >= CONF_THRESHOLD and p_sell > p_buy and rsi > RSI_OVERSOLD_BLOCK:
                ok, _ = _check_filters(row, "SELL")
                if ok:
                    sl_c = entry + atr * sl_m
                    tp_c = entry - atr * tp_m
                    q = int((sizing * RISK_PCT / 100) / max(abs(entry - sl_c), 1))
                    if q > 0:
                        in_trade, side = True, "SELL"
                        entry_p, sl_p, tp_p, qty = entry, sl_c, tp_c, q

    # Close any open trade at last price
    if in_trade:
        exit_p  = float(df.iloc[-1]["close"])
        gross   = (exit_p - entry_p) * qty if side == "BUY" else (entry_p - exit_p) * qty
        costs   = _compute_costs(entry_p, exit_p, qty, side)
        net     = gross - costs
        capital += net
        trades.append({
            "date": df.iloc[-1]["date"], "side": side,
            "entry": entry_p, "exit": exit_p, "qty": qty,
            "gross": gross, "costs": costs, "net": net, "result": "EOD",
        })

    # ── Results ────────────────────────────────────────────────
    eq_df = pd.DataFrame(equity)
    tr_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    if tr_df.empty:
        print("[RESULT] No trades taken in this period."); return

    wins       = (tr_df["net"] > 0).sum()
    losses     = (tr_df["net"] <= 0).sum()
    total      = len(tr_df)
    win_rate   = wins / total * 100
    net_return = (capital - INITIAL_CAP) / INITIAL_CAP * 100
    gross_pnl  = tr_df["gross"].sum()
    total_cost = tr_df["costs"].sum()
    net_pnl    = tr_df["net"].sum()
    avg_win    = tr_df[tr_df["net"] > 0]["net"].mean() if wins > 0 else 0
    avg_loss   = tr_df[tr_df["net"] <= 0]["net"].mean() if losses > 0 else 0

    # Max drawdown
    eq_vals = eq_df["capital"].values
    peak    = np.maximum.accumulate(eq_vals)
    dd      = (eq_vals - peak) / peak * 100
    max_dd  = float(dd.min())

    # Sharpe (daily returns approximation)
    daily_pnl = tr_df.groupby(pd.to_datetime(tr_df["date"]).dt.date)["net"].sum()
    sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0

    profit_factor = (tr_df[tr_df["net"] > 0]["net"].sum() /
                     abs(tr_df[tr_df["net"] <= 0]["net"].sum() + 1e-9))

    print(f"{sep}")
    print("  RESULTS — WITH LIVE FILTERS")
    print(f"{sep}")
    print(f"  Period         : {date_start}  ->  {date_end}")
    print(f"  Trades         : {total}  ({wins}W / {losses}L)")
    print(f"  Win Rate       : {win_rate:.1f}%")
    print()
    print(f"  Gross Return   : +{gross_pnl/INITIAL_CAP*100:.2f}%  (Rs{gross_pnl:,.0f})")
    print(f"  Cost Drag      : -{total_cost/INITIAL_CAP*100:.2f}%  (Rs{total_cost:,.0f}  "
          f"avg Rs{total_cost/total:.0f}/trade)")
    print(f"  NET Return     : {net_pnl/INITIAL_CAP*100:+.2f}%  (Rs{net_pnl:,.0f})")
    print(f"  Final Capital  : Rs{capital:,.0f}")
    print()
    print(f"  Max Drawdown   : {max_dd:.2f}%")
    print(f"  Sharpe Ratio   : {sharpe:.3f}")
    print(f"  Profit Factor  : {profit_factor:.2f}")
    print(f"  Avg Win        : Rs{avg_win:,.0f}")
    print(f"  Avg Loss       : Rs{avg_loss:,.0f}")
    print()

    # ── Year-by-year breakdown ─────────────────────────────────
    print(f"  {'Year':<6}  {'Trades':>7}  {'WR%':>6}  {'NET Rs':>12}  {'NET%':>8}  {'MaxDD':>8}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*8}")

    tr_df["year"] = pd.to_datetime(tr_df["date"]).dt.year
    years = sorted(tr_df["year"].unique())

    for yr in years:
        y  = tr_df[tr_df["year"] == yr]
        yw = (y["net"] > 0).sum()
        yt = len(y)
        yn = y["net"].sum()
        ywr = yw / yt * 100 if yt > 0 else 0
        ypct = yn / INITIAL_CAP * 100
        # Max DD for this year
        yr_eq = eq_df[pd.to_datetime(eq_df["date"]).dt.year == yr]["capital"].values
        if len(yr_eq) > 1:
            ypk = np.maximum.accumulate(yr_eq)
            ydd = float(((yr_eq - ypk) / ypk * 100).min())
        else:
            ydd = 0.0
        flag = " [LOSS]" if yn < 0 else ""
        print(f"  {yr:<6}  {yt:>7}  {ywr:>5.1f}%  {yn:>+12,.0f}  {ypct:>+7.2f}%  {ydd:>7.2f}%{flag}")

    print(f"{sep}\n")

    # ── Equity curve ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    eq_df["date_ts"] = pd.to_datetime(eq_df["date"])
    ax.plot(eq_df["date_ts"], eq_df["capital"], linewidth=1.2, color="#2196F3")
    ax.axhline(INITIAL_CAP, color="gray", linestyle="--", linewidth=0.8, label="Start capital")
    ax.set_title(f"NIFTY 50 Live-Filter Backtest  ({date_start} → {date_end})\n"
                 f"Trades={total}  WR={win_rate:.1f}%  NET={net_pnl/INITIAL_CAP*100:+.1f}%  "
                 f"MaxDD={max_dd:.2f}%  Sharpe={sharpe:.2f}", fontsize=11)
    ax.set_ylabel("Capital (Rs)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rs{x/1e5:.1f}L"))
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_OUT, dpi=150)
    print(f"[CHART] Saved -> {CHART_OUT}")
    print(f"[DONE]  live_sim_backtest complete.\n")


if __name__ == "__main__":
    run()
