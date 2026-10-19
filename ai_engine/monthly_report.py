"""
Last 1 month backtest — clean trade-by-trade report.
Run:  python monthly_report.py
"""
import os, pickle
import numpy as np
import pandas as pd
from model_trainer import load_and_clean, add_features, FEATURES, CalibratedXGB

MODEL_PATH   = "trading_model.pkl"
FEATURE_PATH = "feature_list.pkl"
CSV_PATH     = "nifty_5m_data.csv"

INITIAL_CAP        = 100_000
MAX_CAPITAL_SIZING = 100_000
RISK_PCT           = 1.0
CONF_THRESHOLD     = 0.62
RSI_OB = 80.0; RSI_OS = 20.0
ADX_MIN = 10.0; MTF_MIN = 1
SESSION_OPEN_BLK = 3; SESSION_CLOSE_BLK = 3
COOLDOWN_CANDLES = 1
FROM_DATE = "2026-03-18"

SLIPPAGE_PTS = 2.0; BROKERAGE = 40.0
STT_PCT = 0.0001; EXCHANGE_PCT = 0.0000053
GST_RATE = 0.18; STAMP_PCT = 0.00003


def _costs(entry, exit_p, qty):
    return (SLIPPAGE_PTS*2*qty + BROKERAGE + BROKERAGE*GST_RATE
            + exit_p*qty*STT_PCT + (entry+exit_p)*qty*EXCHANGE_PCT + entry*qty*STAMP_PCT)


def _check_filters(row, signal):
    rsi = float(row.get("RSI",50)); adx = float(row.get("ADX",0))
    ema5=int(row.get("EMA_alignment",0)); ema15=int(row.get("EMA_align_15m",0))
    ema1h=int(row.get("EMA_align_1h",0))
    hour=int(row.get("Hour",9)); minute=int(row.get("Minute",15))
    d = 1 if signal=="BUY" else -1
    if signal=="BUY"  and rsi >= RSI_OB: return False
    if signal=="SELL" and rsi <= RSI_OS: return False
    if adx < ADX_MIN: return False
    if 0 <= (hour-9)*60+(minute-15) < SESSION_OPEN_BLK*5: return False
    if 0 <= (15*60+30)-(hour*60+minute) <= SESSION_CLOSE_BLK*5: return False
    if sum([ema5==d, ema15==d, ema1h==d]) < MTF_MIN: return False
    return True


def _sl_tp(row):
    bb=float(row.get("BB_width_pctile",0.5)); atp=float(row.get("ATR_pctile",0.5))
    mr=int(row.get("Market_Regime",0)); adx=float(row.get("ADX",0))
    if bb < 0.15:                     return 1.5, 4.0, "SQUEEZE"
    elif mr==4 or atp > 0.80:         return 1.5, 3.0, "HIGH-VOL"
    elif mr in (1,2) and adx > 30:    return 1.0, 3.5, "TRENDING"
    elif mr==0 and atp < 0.30:        return 0.8, 2.0, "SIDEWAYS"
    else:                             return 1.0, 2.0, "DEFAULT"


with open(MODEL_PATH,"rb") as f: model = pickle.load(f)
features = pickle.load(open(FEATURE_PATH,"rb")) if os.path.exists(FEATURE_PATH) else FEATURES

df = load_and_clean(CSV_PATH)
df = add_features(df)
df.dropna(subset=features+["ATR"], inplace=True)
df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(FROM_DATE, tz="Asia/Kolkata")].reset_index(drop=True)

capital=INITIAL_CAP; in_trade=False; side=None
entry_p=sl_p=tp_p=0.0; qty=0; last_close=-COOLDOWN_CANDLES
trades=[]; entry_meta={}

for i in range(len(df)-1):
    row=df.iloc[i]; nxt=df.iloc[i+1]

    if in_trade:
        hi,lo=float(row["high"]),float(row["low"])
        sl_hit=(lo<=sl_p) if side=="BUY" else (hi>=sl_p)
        tp_hit=(hi>=tp_p) if side=="BUY" else (lo<=tp_p)
        if sl_hit or tp_hit:
            exit_p=sl_p if sl_hit else tp_p
            gross=(exit_p-entry_p)*qty if side=="BUY" else (entry_p-exit_p)*qty
            net=round(gross-_costs(entry_p,exit_p,qty))
            capital+=net
            dt=row["date"]
            trades.append({**entry_meta,
                "exit_date": dt.strftime("%Y-%m-%d"),
                "exit_time": dt.strftime("%H:%M"),
                "exit_price": round(exit_p,2),
                "exit_by": "TP" if tp_hit else "SL",
                "pnl": net,
                "result": "W" if net>0 else "L",
                "capital": round(capital),
            })
            in_trade=False; last_close=i

    if not in_trade and (i-last_close)>=COOLDOWN_CANDLES:
        X=row[features].values.reshape(1,-1)
        prob=model.predict_proba(X)[0]
        p_sell,p_buy=float(prob[0]),float(prob[2])
        rsi=float(row.get("RSI",50)); atr=float(row.get("ATR",1))
        entry=float(nxt["open"]); sl_m,tp_m,rtag=_sl_tp(row)
        sizing=min(capital,MAX_CAPITAL_SIZING)
        sig=None
        if p_buy>=CONF_THRESHOLD and p_buy>p_sell and rsi<RSI_OB:
            if _check_filters(row,"BUY"): sig="BUY"; conf=p_buy
        elif p_sell>=CONF_THRESHOLD and p_sell>p_buy and rsi>RSI_OS:
            if _check_filters(row,"SELL"): sig="SELL"; conf=p_sell
        if sig:
            sl_c=entry-atr*sl_m if sig=="BUY" else entry+atr*sl_m
            tp_c=entry+atr*tp_m if sig=="BUY" else entry-atr*tp_m
            q=int((sizing*RISK_PCT/100)/max(abs(entry-sl_c),1))
            if q>0:
                in_trade=True; side=sig
                entry_p,sl_p,tp_p,qty=entry,sl_c,tp_c,q
                dt=row["date"]
                entry_meta={
                    "entry_date": dt.strftime("%Y-%m-%d"),
                    "entry_time": nxt["date"].strftime("%H:%M") if hasattr(nxt["date"],"strftime") else "",
                    "side": sig,
                    "entry_price": round(entry,2),
                    "sl_price": round(sl_c,2),
                    "tp_price": round(tp_c,2),
                    "conf": round(conf*100,1),
                    "regime": rtag,
                }

if in_trade:
    exit_p=float(df.iloc[-1]["close"])
    gross=(exit_p-entry_p)*qty if side=="BUY" else (entry_p-exit_p)*qty
    net=round(gross-_costs(entry_p,exit_p,qty))
    capital+=net
    trades.append({**entry_meta,
        "exit_date":df.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "exit_time":"EOD","exit_price":round(exit_p,2),"exit_by":"EOD",
        "pnl":net,"result":"W" if net>0 else "L","capital":round(capital),
    })

tr=pd.DataFrame(trades)
if tr.empty:
    print("No trades."); exit()

wins=(tr["result"]=="W").sum(); total=len(tr)
wr=wins/total*100
net_pnl=tr["pnl"].sum()
gross_pnl=tr["pnl"].sum()+tr["pnl"].apply(lambda x:0).sum()
avg_win=tr[tr["result"]=="W"]["pnl"].mean()
avg_loss=tr[tr["result"]=="L"]["pnl"].mean()
max_win=tr["pnl"].max(); max_loss=tr["pnl"].min()

# Max drawdown
cum=INITIAL_CAP+tr["pnl"].cumsum().values
peak=np.maximum.accumulate(np.concatenate([[INITIAL_CAP],cum]))
max_dd=((cum-peak[1:])/peak[1:]*100).min()

sep="="*70
print(f"\n{sep}")
print(f"  NIFTY 50 — LAST 1 MONTH BACKTEST  ({FROM_DATE} to 2026-05-18)")
print(f"{sep}")

# Print day by day
print(f"\n  {'No':<5} {'Date':<12} {'Time':<7} {'Side':<6} {'Entry':>8} {'Exit':>8}  {'Exit By':<8} {'PnL':>8}  Result")
print(f"  {'-'*66}")

prev_date=""
for idx, t in tr.iterrows():
    if t["entry_date"] != prev_date:
        if prev_date:
            day_tr=tr[tr["entry_date"]==prev_date]
            day_pnl=day_tr["pnl"].sum()
            day_w=(day_tr["result"]=="W").sum()
            day_l=(day_tr["result"]=="L").sum()
            print(f"  {'':5} {'':12} {'':7} {'':6} {'':>8} {'':>8}  {'Day P&L':8} {day_pnl:>+8,}  [{day_w}W/{day_l}L]")
            print(f"  {'-'*66}")
        prev_date=t["entry_date"]

    no=idx+1
    pnl_str=f"{t['pnl']:>+8,}"
    res=f"[W]" if t["result"]=="W" else "[L]"
    print(f"  {no:<5} {t['entry_date']:<12} {t['entry_time']:<7} {t['side']:<6} "
          f"{t['entry_price']:>8.0f} {t['exit_price']:>8.0f}  {t['exit_by']:<8} {pnl_str}  {res}  "
          f"({t['conf']}% {t['regime']})")

# Last day
if prev_date:
    day_tr=tr[tr["entry_date"]==prev_date]
    day_pnl=day_tr["pnl"].sum()
    day_w=(day_tr["result"]=="W").sum()
    day_l=(day_tr["result"]=="L").sum()
    print(f"  {'':5} {'':12} {'':7} {'':6} {'':>8} {'':>8}  {'Day P&L':8} {day_pnl:>+8,}  [{day_w}W/{day_l}L]")

print(f"\n  {sep}")
print(f"  SUMMARY")
print(f"  {sep}")
print(f"  Period       : {FROM_DATE}  ->  2026-05-18")
print(f"  Total Trades : {total}  ({wins}W / {total-wins}L)")
print(f"  Win Rate     : {wr:.1f}%")
print(f"  Net P&L      : Rs{net_pnl:>+,.0f}")
print(f"  Final Capital: Rs{capital:,.0f}  ({(capital-INITIAL_CAP)/INITIAL_CAP*100:+.2f}%)")
print(f"  Avg Win      : Rs{avg_win:>+,.0f}")
print(f"  Avg Loss     : Rs{avg_loss:>+,.0f}")
print(f"  Best Trade   : Rs{max_win:>+,.0f}")
print(f"  Worst Trade  : Rs{max_loss:>+,.0f}")
print(f"  Max Drawdown : {max_dd:.2f}%")
print(f"  {sep}\n")
