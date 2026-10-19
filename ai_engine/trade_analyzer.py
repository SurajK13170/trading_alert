"""
NIFTY 50 — WINNING TRADE PATTERN ANALYZER
-------------------------------------------
Runs the same live-filter backtest but saves every trade's feature values.
Then finds what conditions are most common in WINNING trades.

Run:
    python trade_analyzer.py
    python trade_analyzer.py --from 2021
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_trainer import load_and_clean, add_features, FEATURES, CalibratedXGB

MODEL_PATH   = "trading_model.pkl"
FEATURE_PATH = "feature_list.pkl"
CSV_PATH     = "nifty_5m_data.csv"

INITIAL_CAP        = 100_000
MAX_CAPITAL_SIZING = 100_000
RISK_PCT           = 1.0
CONF_THRESHOLD     = 0.55
RSI_OB             = 80.0
RSI_OS             = 20.0
ADX_MIN            = 10.0
MTF_MIN            = 1
SESSION_OPEN_BLK   = 3
SESSION_CLOSE_BLK  = 3
COOLDOWN_CANDLES   = 1

SLIPPAGE_PTS = 2.0
BROKERAGE    = 40.0
STT_PCT      = 0.0001
EXCHANGE_PCT = 0.0000053
GST_RATE     = 0.18
STAMP_PCT    = 0.00003

FROM_YEAR = None
TO_YEAR   = None
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--from" and i < len(sys.argv):
        FROM_YEAR = int(sys.argv[i + 1])
    if arg == "--to" and i < len(sys.argv):
        TO_YEAR = int(sys.argv[i + 1])


def _costs(entry, exit_p, qty):
    slip     = SLIPPAGE_PTS * 2 * qty
    gst      = BROKERAGE * GST_RATE
    stt      = exit_p * qty * STT_PCT
    exchange = (entry + exit_p) * qty * EXCHANGE_PCT
    stamp    = entry * qty * STAMP_PCT
    return slip + BROKERAGE + gst + stt + exchange + stamp


def _check_filters(row, signal):
    rsi    = float(row.get("RSI", 50))
    adx    = float(row.get("ADX", 0))
    ema5   = int(row.get("EMA_alignment",  0))
    ema15  = int(row.get("EMA_align_15m",  0))
    ema1h  = int(row.get("EMA_align_1h",   0))
    hour   = int(row.get("Hour",  9))
    minute = int(row.get("Minute", 15))
    d      = 1 if signal == "BUY" else -1

    if signal == "BUY"  and rsi >= RSI_OB: return False
    if signal == "SELL" and rsi <= RSI_OS: return False
    if adx < ADX_MIN: return False

    mins_open  = (hour - 9) * 60 + (minute - 15)
    if 0 <= mins_open < SESSION_OPEN_BLK * 5: return False
    mins_close = (15 * 60 + 30) - (hour * 60 + minute)
    if 0 <= mins_close <= SESSION_CLOSE_BLK * 5: return False

    score = sum([ema5 == d, ema15 == d, ema1h == d])
    if score < MTF_MIN: return False
    return True


def _sl_tp(row):
    bb  = float(row.get("BB_width_pctile", 0.5))
    atp = float(row.get("ATR_pctile",      0.5))
    mr  = int(row.get("Market_Regime",    0))
    adx = float(row.get("ADX", 0))
    if bb < 0.15:                           return 1.5, 4.0, "SQUEEZE"
    elif mr == 4 or atp > 0.80:             return 1.5, 3.0, "HIGH-VOL"
    elif mr in (1, 2) and adx > 30:         return 1.0, 3.5, "TRENDING"
    elif mr == 0 and atp < 0.30:            return 0.8, 2.0, "SIDEWAYS"
    else:                                   return 1.0, 2.0, "DEFAULT"


REGIME_NAMES = {0: "SIDEWAYS", 1: "BULL", 2: "STRONG-BULL", 3: "BEAR", 4: "HIGH-VOL"}


def run():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    features = pickle.load(open(FEATURE_PATH, "rb")) if os.path.exists(FEATURE_PATH) else FEATURES

    df = load_and_clean(CSV_PATH)
    df = add_features(df)
    df.dropna(subset=features + ["ATR"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["_year"] = pd.to_datetime(df["date"]).dt.year
    if FROM_YEAR: df = df[df["_year"] >= FROM_YEAR].reset_index(drop=True)
    if TO_YEAR:   df = df[df["_year"] <= TO_YEAR].reset_index(drop=True)

    print(f"[INFO] Running on {len(df):,} candles ({df['date'].iloc[0]} -> {df['date'].iloc[-1]})\n")

    capital          = INITIAL_CAP
    in_trade         = False
    side             = None
    entry_p = sl_p = tp_p = 0.0
    qty              = 0
    last_close       = -COOLDOWN_CANDLES
    entry_snapshot   = {}
    trades           = []

    for i in range(len(df) - 1):
        row      = df.iloc[i]
        next_row = df.iloc[i + 1]

        if in_trade:
            hi, lo = float(row["high"]), float(row["low"])
            sl_hit = (lo <= sl_p) if side == "BUY" else (hi >= sl_p)
            tp_hit = (hi >= tp_p) if side == "BUY" else (lo <= tp_p)
            if sl_hit or tp_hit:
                exit_p = sl_p if sl_hit else tp_p
                gross  = (exit_p - entry_p) * qty if side == "BUY" else (entry_p - exit_p) * qty
                costs  = _costs(entry_p, exit_p, qty)
                net    = gross - costs
                capital += net
                t = {**entry_snapshot,
                     "result":   "WIN" if net > 0 else "LOSS",
                     "net":      net,
                     "exit":     "TP" if tp_hit else "SL",
                     "duration": i - entry_snapshot["entry_idx"],
                }
                trades.append(t)
                in_trade = False
                last_close = i

        if not in_trade and (i - last_close) >= COOLDOWN_CANDLES:
            X    = row[features].values.reshape(1, -1)
            prob = model.predict_proba(X)[0]
            p_sell, p_buy = float(prob[0]), float(prob[2])
            rsi  = float(row.get("RSI",  50))
            adx  = float(row.get("ADX",  0))
            atr  = float(row.get("ATR",  1))
            hour = int(row.get("Hour",   9))
            minute = int(row.get("Minute", 15))
            entry = float(next_row["open"])
            sl_m, tp_m, regime_tag = _sl_tp(row)
            sizing = min(capital, MAX_CAPITAL_SIZING)

            sig = None
            conf = 0.0
            if p_buy >= CONF_THRESHOLD and p_buy > p_sell and rsi < RSI_OB:
                if _check_filters(row, "BUY"):
                    sig, conf = "BUY", p_buy
            elif p_sell >= CONF_THRESHOLD and p_sell > p_buy and rsi > RSI_OS:
                if _check_filters(row, "SELL"):
                    sig, conf = "SELL", p_sell

            if sig:
                sl_c = entry - atr * sl_m if sig == "BUY" else entry + atr * sl_m
                tp_c = entry + atr * tp_m if sig == "BUY" else entry - atr * tp_m
                q    = int((sizing * RISK_PCT / 100) / max(abs(entry - sl_c), 1))
                if q > 0:
                    in_trade = True
                    side     = sig
                    entry_p, sl_p, tp_p, qty = entry, sl_c, tp_c, q
                    entry_snapshot = {
                        "entry_idx":      i,
                        "date":           row["date"],
                        "side":           sig,
                        "conf":           round(conf * 100, 1),
                        "rsi":            round(rsi, 1),
                        "adx":            round(adx, 1),
                        "atr_pctile":     round(float(row.get("ATR_pctile",    0.5)), 2),
                        "bb_pctile":      round(float(row.get("BB_width_pctile", 0.5)), 2),
                        "market_regime":  REGIME_NAMES.get(int(row.get("Market_Regime", 0)), "?"),
                        "regime_tag":     regime_tag,
                        "hour":           hour,
                        "minute":         minute,
                        "ema5":           int(row.get("EMA_alignment",  0)),
                        "ema15":          int(row.get("EMA_align_15m",  0)),
                        "ema1h":          int(row.get("EMA_align_1h",   0)),
                        "mtf_score":      sum([
                                            int(row.get("EMA_alignment", 0)) == (1 if sig=="BUY" else -1),
                                            int(row.get("EMA_align_15m", 0)) == (1 if sig=="BUY" else -1),
                                            int(row.get("EMA_align_1h",  0)) == (1 if sig=="BUY" else -1),
                                          ]),
                        "entry_price":    round(entry, 2),
                    }

    tr = pd.DataFrame(trades)
    if tr.empty:
        print("[ERR] No trades found."); return

    wins  = tr[tr["result"] == "WIN"]
    losses= tr[tr["result"] == "LOSS"]
    total = len(tr)
    wr    = len(wins) / total * 100

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  WINNING TRADE PATTERN ANALYSIS  ({total} trades, {wr:.1f}% WR)")
    print(f"{sep}\n")

    # ── 1. Confidence breakdown ────────────────────────────────
    print("  [1] WIN RATE BY CONFIDENCE BUCKET")
    print(f"  {'Conf%':>8}  {'Trades':>7}  {'Wins':>6}  {'WR%':>6}  {'Avg Net':>10}")
    bins = [(55,60),(60,65),(65,70),(70,75),(75,80),(80,101)]
    for lo, hi in bins:
        b = tr[(tr["conf"] >= lo) & (tr["conf"] < hi)]
        if len(b) == 0: continue
        bw = (b["result"]=="WIN").sum()
        print(f"  {lo:>4}-{hi:<3}%  {len(b):>7}  {bw:>6}  {bw/len(b)*100:>5.1f}%  "
              f"Rs{b['net'].mean():>+8,.0f}")
    print()

    # ── 2. Direction breakdown ─────────────────────────────────
    print("  [2] WIN RATE BY DIRECTION")
    for d in ["BUY", "SELL"]:
        b  = tr[tr["side"] == d]
        bw = (b["result"]=="WIN").sum()
        print(f"  {d:<6}  Trades={len(b):>5}  WR={bw/len(b)*100:.1f}%  "
              f"Avg Win=Rs{b[b['result']=='WIN']['net'].mean():,.0f}  "
              f"Avg Loss=Rs{b[b['result']=='LOSS']['net'].mean():,.0f}")
    print()

    # ── 3. Market regime breakdown ────────────────────────────
    print("  [3] WIN RATE BY MARKET REGIME")
    print(f"  {'Regime':>12}  {'Trades':>7}  {'WR%':>6}  {'Avg Net':>10}")
    for reg in tr["market_regime"].unique():
        b  = tr[tr["market_regime"] == reg]
        bw = (b["result"]=="WIN").sum()
        print(f"  {reg:>12}  {len(b):>7}  {bw/len(b)*100:>5.1f}%  Rs{b['net'].mean():>+8,.0f}")
    print()

    # ── 4. Regime tag (SQUEEZE / HIGH-VOL / TRENDING etc.) ────
    print("  [4] WIN RATE BY SL/TP REGIME TAG")
    print(f"  {'Tag':>12}  {'Trades':>7}  {'WR%':>6}  {'Avg Net':>10}")
    for tag in tr["regime_tag"].unique():
        b  = tr[tr["regime_tag"] == tag]
        bw = (b["result"]=="WIN").sum()
        print(f"  {tag:>12}  {len(b):>7}  {bw/len(b)*100:>5.1f}%  Rs{b['net'].mean():>+8,.0f}")
    print()

    # ── 5. Time of day breakdown ──────────────────────────────
    print("  [5] WIN RATE BY HOUR (IST)")
    print(f"  {'Hour':>6}  {'Trades':>7}  {'WR%':>6}  {'Avg Net':>10}")
    for h in sorted(tr["hour"].unique()):
        b  = tr[tr["hour"] == h]
        bw = (b["result"]=="WIN").sum()
        bar = "#" * int(bw/len(b)*20)
        print(f"  {h:>4}:xx  {len(b):>7}  {bw/len(b)*100:>5.1f}%  Rs{b['net'].mean():>+8,.0f}  {bar}")
    print()

    # ── 6. MTF alignment score ────────────────────────────────
    print("  [6] WIN RATE BY MTF CONFLUENCE SCORE")
    print(f"  {'MTF Score':>10}  {'Trades':>7}  {'WR%':>6}  {'Avg Net':>10}")
    for sc in sorted(tr["mtf_score"].unique()):
        b  = tr[tr["mtf_score"] == sc]
        bw = (b["result"]=="WIN").sum()
        print(f"  {sc}/3{' '*7}  {len(b):>7}  {bw/len(b)*100:>5.1f}%  Rs{b['net'].mean():>+8,.0f}")
    print()

    # ── 7. ADX buckets ────────────────────────────────────────
    print("  [7] WIN RATE BY ADX BUCKET")
    print(f"  {'ADX Range':>12}  {'Trades':>7}  {'WR%':>6}  {'Avg Net':>10}")
    adx_bins = [(10,15),(15,20),(20,25),(25,30),(30,40),(40,200)]
    for lo, hi in adx_bins:
        b = tr[(tr["adx"] >= lo) & (tr["adx"] < hi)]
        if len(b) == 0: continue
        bw = (b["result"]=="WIN").sum()
        print(f"  ADX {lo:>3}-{hi:<4}  {len(b):>7}  {bw/len(b)*100:>5.1f}%  Rs{b['net'].mean():>+8,.0f}")
    print()

    # ── 8. ATR percentile ─────────────────────────────────────
    print("  [8] WIN RATE BY ATR PERCENTILE (volatility)")
    print(f"  {'ATR Pctile':>12}  {'Trades':>7}  {'WR%':>6}  {'Avg Net':>10}")
    atr_bins = [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]
    labels   = ["Low","Low-Mid","Mid","Mid-High","High"]
    for (lo, hi), lbl in zip(atr_bins, labels):
        b = tr[(tr["atr_pctile"] >= lo) & (tr["atr_pctile"] < hi)]
        if len(b) == 0: continue
        bw = (b["result"]=="WIN").sum()
        print(f"  {lbl:>12}  {len(b):>7}  {bw/len(b)*100:>5.1f}%  Rs{b['net'].mean():>+8,.0f}")
    print()

    # ── 9. TOP PATTERN — best combo ───────────────────────────
    print("  [9] TOP WINNING CONDITIONS (MTF+Regime+Direction)")
    print(f"  {'Condition':>40}  {'Trades':>6}  {'WR%':>6}")
    grp = tr.groupby(["side","market_regime","mtf_score"]).apply(
        lambda x: pd.Series({"trades": len(x), "wr": (x["result"]=="WIN").sum()/len(x)*100})
    ).reset_index()
    grp = grp[grp["trades"] >= 20].sort_values("wr", ascending=False).head(10)
    for _, row in grp.iterrows():
        cond = f"{row['side']} | {row['market_regime']} | MTF={int(row['mtf_score'])}/3"
        print(f"  {cond:>40}  {int(row['trades']):>6}  {row['wr']:>5.1f}%")
    print()

    # ── Summary: what good trades look like ───────────────────
    print(f"{sep}")
    print("  SUMMARY — What winning trades have in common:")
    print(f"{sep}")

    best_conf    = tr.groupby(pd.cut(tr["conf"], [55,60,65,70,75,80,101]))["result"].apply(
                       lambda x: (x=="WIN").sum()/len(x)*100).idxmax()
    best_hour    = tr.groupby("hour")["result"].apply(
                       lambda x: (x=="WIN").sum()/len(x)*100 if len(x)>=30 else 0).idxmax()
    best_regime  = tr.groupby("market_regime")["result"].apply(
                       lambda x: (x=="WIN").sum()/len(x)*100).idxmax()
    best_tag     = tr.groupby("regime_tag")["result"].apply(
                       lambda x: (x=="WIN").sum()/len(x)*100).idxmax()

    hi_conf_wr   = tr[tr["conf"] >= 65]
    hi_conf_wr_v = (hi_conf_wr["result"]=="WIN").sum() / len(hi_conf_wr) * 100 if len(hi_conf_wr) > 0 else 0

    hi_adx       = tr[tr["adx"] >= 25]
    hi_adx_wr    = (hi_adx["result"]=="WIN").sum() / len(hi_adx) * 100 if len(hi_adx) > 0 else 0

    mtf3         = tr[tr["mtf_score"] == 3]
    mtf3_wr      = (mtf3["result"]=="WIN").sum() / len(mtf3) * 100 if len(mtf3) > 0 else 0

    print(f"  * Best confidence range  : {best_conf} (highest WR)")
    print(f"  * Conf >= 65% WR         : {hi_conf_wr_v:.1f}%  ({len(hi_conf_wr)} trades)")
    print(f"  * ADX >= 25 WR           : {hi_adx_wr:.1f}%  ({len(hi_adx)} trades)")
    print(f"  * MTF = 3/3 aligned WR   : {mtf3_wr:.1f}%  ({len(mtf3)} trades)")
    print(f"  * Best hour (IST)        : {best_hour}:xx")
    print(f"  * Best market regime     : {best_regime}")
    print(f"  * Best SL/TP regime      : {best_tag}")
    print(f"{sep}\n")

    # ── Chart: WR by hour ─────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Winning Trade Pattern Analysis", fontsize=14)

    # Hour WR
    ax = axes[0][0]
    hg = tr.groupby("hour")["result"].apply(lambda x: (x=="WIN").sum()/len(x)*100)
    ax.bar(hg.index, hg.values, color=["green" if v >= wr else "salmon" for v in hg.values])
    ax.axhline(wr, color="gray", linestyle="--", linewidth=0.8, label=f"Overall {wr:.1f}%")
    ax.set_title("Win Rate by Hour (IST)"); ax.set_ylabel("Win Rate %"); ax.set_xlabel("Hour")
    ax.legend()

    # Confidence WR
    ax = axes[0][1]
    conf_bins = pd.cut(tr["conf"], [55,60,65,70,75,80,101])
    cg = tr.groupby(conf_bins)["result"].apply(lambda x: (x=="WIN").sum()/len(x)*100)
    ax.bar([str(c) for c in cg.index], cg.values,
           color=["green" if v >= wr else "salmon" for v in cg.values])
    ax.axhline(wr, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Win Rate by Confidence"); ax.set_ylabel("Win Rate %")
    ax.tick_params(axis='x', rotation=20)

    # ADX WR
    ax = axes[0][2]
    adx_labels = [f"{lo}-{hi}" for lo,hi in adx_bins]
    adx_wrs = []
    for lo, hi in adx_bins:
        b = tr[(tr["adx"] >= lo) & (tr["adx"] < hi)]
        adx_wrs.append((b["result"]=="WIN").sum()/len(b)*100 if len(b) > 0 else 0)
    ax.bar(adx_labels, adx_wrs, color=["green" if v >= wr else "salmon" for v in adx_wrs])
    ax.axhline(wr, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Win Rate by ADX"); ax.set_ylabel("Win Rate %"); ax.set_xlabel("ADX")

    # Market regime WR
    ax = axes[1][0]
    rg = tr.groupby("market_regime")["result"].apply(lambda x: (x=="WIN").sum()/len(x)*100)
    ax.bar(rg.index, rg.values, color=["green" if v >= wr else "salmon" for v in rg.values])
    ax.axhline(wr, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Win Rate by Market Regime"); ax.set_ylabel("Win Rate %")

    # MTF score WR
    ax = axes[1][1]
    mg = tr.groupby("mtf_score")["result"].apply(lambda x: (x=="WIN").sum()/len(x)*100)
    ax.bar([f"MTF {s}/3" for s in mg.index], mg.values,
           color=["green" if v >= wr else "salmon" for v in mg.values])
    ax.axhline(wr, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Win Rate by MTF Confluence"); ax.set_ylabel("Win Rate %")

    # Regime tag WR
    ax = axes[1][2]
    tg = tr.groupby("regime_tag")["result"].apply(lambda x: (x=="WIN").sum()/len(x)*100)
    ax.bar(tg.index, tg.values, color=["green" if v >= wr else "salmon" for v in tg.values])
    ax.axhline(wr, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Win Rate by SL/TP Regime"); ax.set_ylabel("Win Rate %")

    plt.tight_layout()
    plt.savefig("trade_patterns.png", dpi=150)
    print("[CHART] Saved -> trade_patterns.png\n")


if __name__ == "__main__":
    run()
