"""
NIFTY 50 — LAST 1 WEEK LIVE SIMULATION
----------------------------------------
Fetches fresh 5-min NIFTY data from Angel One (falls back to yfinance),
runs all live filters + model, and shows every trade taken last week.

Run:
    python recent_week_sim.py
"""

import os, sys, pickle, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH   = "trading_model.pkl"
FEATURE_PATH = "feature_list.pkl"

INITIAL_CAP        = 100_000
MAX_CAPITAL_SIZING = 100_000
RISK_PCT           = 1.0
CONF_THRESHOLD     = 0.62
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

NIFTY_TOKEN = "99926000"
BNF_TOKEN   = "26009"

FETCH_DAYS  = 30    # days to fetch for feature warmup
SIM_DAYS    = 7     # days to actually simulate (last week)


# ── Angel One fetch ────────────────────────────────────────────────
_angel_api = None

def _angel_login():
    global _angel_api
    cid  = os.getenv("ANGEL_DATA_CLIENT_ID") or os.getenv("ANGEL_CLIENT_ID")
    akey = os.getenv("ANGEL_DATA_API_KEY")   or os.getenv("ANGEL_API_KEY")
    pwd  = os.getenv("ANGEL_DATA_PASSWORD")  or os.getenv("ANGEL_PASSWORD")
    totp = os.getenv("ANGEL_DATA_TOTP_KEY")  or os.getenv("ANGEL_TOTP_KEY")
    if not all([cid, akey, pwd, totp]):
        return False
    try:
        from SmartApi import SmartConnect
        import pyotp
        _angel_api = SmartConnect(api_key=akey)
        data = _angel_api.generateSession(cid, pwd, pyotp.TOTP(totp).now())
        if data and data.get("status"):
            print(f"[ANGEL] Login OK ({cid})")
            return True
        print(f"[ANGEL] Login failed — using yfinance")
        return False
    except Exception as e:
        print(f"[ANGEL] Login error: {e} — using yfinance")
        return False


def _angel_fetch_df(interval, days_back, token=None):
    now = datetime.now()
    frm = (now - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M")
    to  = now.strftime("%Y-%m-%d %H:%M")
    params = {
        "exchange":    "NSE",
        "symboltoken": token or NIFTY_TOKEN,
        "interval":    interval,
        "fromdate":    frm,
        "todate":      to,
    }
    time.sleep(1.0)
    resp = _angel_api.getCandleData(params)
    if not resp or not resp.get("status") or not resp.get("data"):
        return pd.DataFrame()
    df = pd.DataFrame(resp["data"], columns=["Datetime","Open","High","Low","Close","Volume"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def _yfinance_fetch(days_back):
    try:
        import yfinance as yf
        period = f"{min(days_back, 59)}d"
        df = yf.download("^NSEI", period=period, interval="5m", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "Datetime"
        return df
    except Exception as e:
        print(f"[yfinance] Error: {e}")
        return pd.DataFrame()


# ── Feature computation (reuse model_trainer logic) ───────────────
from model_trainer import add_features, FEATURES, CalibratedXGB

def build_feature_df(raw_5m: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Angel/yfinance OHLCV → feature DataFrame via model_trainer."""
    df = raw_5m.copy().reset_index()
    # Normalize column names
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if "datetime" in lc or "date" in lc or "time" in lc:
            col_map[c] = "date"
        elif lc == "open":   col_map[c] = "open"
        elif lc == "high":   col_map[c] = "high"
        elif lc == "low":    col_map[c] = "low"
        elif lc == "close":  col_map[c] = "close"
        elif lc == "volume": col_map[c] = "volume"
        elif lc == "adj close" or lc == "adj_close": col_map[c] = "close"
    df = df.rename(columns=col_map)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})

    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("Asia/Kolkata")
    else:
        df["date"] = df["date"].dt.tz_convert("Asia/Kolkata")

    df = df.sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df = add_features(df)
    return df


# ── Live filter checks ─────────────────────────────────────────────
def _check_filters(row, signal):
    rsi   = float(row.get("RSI",    50))
    adx   = float(row.get("ADX",    0))
    ema5  = int(row.get("EMA_alignment", 0))
    ema15 = int(row.get("EMA_align_15m", 0))
    ema1h = int(row.get("EMA_align_1h",  0))
    hour  = int(row.get("Hour",   9))
    minute= int(row.get("Minute", 15))
    d     = 1 if signal == "BUY" else -1

    if signal == "BUY"  and rsi >= RSI_OB: return False, f"RSI={rsi:.1f}"
    if signal == "SELL" and rsi <= RSI_OS: return False, f"RSI={rsi:.1f}"
    if adx < ADX_MIN: return False, f"ADX={adx:.1f}"

    mins_open  = (hour - 9) * 60 + (minute - 15)
    if 0 <= mins_open < SESSION_OPEN_BLK * 5: return False, "Open-block"
    mins_close = (15 * 60 + 30) - (hour * 60 + minute)
    if 0 <= mins_close <= SESSION_CLOSE_BLK * 5: return False, "Close-block"

    score = sum([ema5 == d, ema15 == d, ema1h == d])
    if score < MTF_MIN: return False, f"MTF {score}/3"
    return True, ""


def _sl_tp(row):
    bb  = float(row.get("BB_width_pctile", 0.5))
    atp = float(row.get("ATR_pctile",      0.5))
    mr  = int(row.get("Market_Regime",    0))
    adx = float(row.get("ADX", 0))
    if bb < 0.15:                       return 1.5, 4.0, "SQUEEZE"
    elif mr == 4 or atp > 0.80:         return 1.5, 3.0, "HIGH-VOL"
    elif mr in (1, 2) and adx > 30:     return 1.0, 3.5, "TRENDING"
    elif mr == 0 and atp < 0.30:        return 0.8, 2.0, "SIDEWAYS"
    else:                               return 1.0, 2.0, "DEFAULT"


def _costs(entry, exit_p, qty):
    slip     = SLIPPAGE_PTS * 2 * qty
    gst      = BROKERAGE * GST_RATE
    stt      = exit_p * qty * STT_PCT
    exchange = (entry + exit_p) * qty * EXCHANGE_PCT
    stamp    = entry * qty * STAMP_PCT
    return slip + BROKERAGE + gst + stt + exchange + stamp


# ── Main ──────────────────────────────────────────────────────────
def run():
    sep = "=" * 64

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    features = pickle.load(open(FEATURE_PATH, "rb")) if os.path.exists(FEATURE_PATH) else FEATURES
    print(f"[MODEL] Loaded ({len(features)} features)\n")

    # Fetch data
    use_angel = _angel_login()
    raw = pd.DataFrame()

    if use_angel:
        print(f"[FETCH] Fetching {FETCH_DAYS}d of 5-min NIFTY from Angel One...")
        raw = _angel_fetch_df("FIVE_MINUTE", FETCH_DAYS)
        if raw.empty:
            print("[FETCH] Angel returned empty — falling back to yfinance")
            use_angel = False

    if not use_angel or raw.empty:
        print(f"[FETCH] Fetching from yfinance (^NSEI, {FETCH_DAYS}d)...")
        raw = _yfinance_fetch(FETCH_DAYS)

    if raw.empty:
        print("[ERR] No data fetched. Check credentials / internet.")
        return

    print(f"[FETCH] Got {len(raw):,} candles | "
          f"{raw.index[0] if hasattr(raw,'index') else ''} -> "
          f"{raw.index[-1] if hasattr(raw,'index') else ''}")

    # Build features
    print("[FEAT] Computing features...")
    df = build_feature_df(raw)
    df.dropna(subset=features + ["ATR"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        print("[ERR] Feature computation produced no rows."); return

    # Filter to last SIM_DAYS trading days
    cutoff = pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(days=SIM_DAYS)
    df_sim = df[df["date"] >= cutoff].reset_index(drop=True)

    if df_sim.empty:
        print(f"[ERR] No data in last {SIM_DAYS} days after feature computation."); return

    date_start = df_sim["date"].iloc[0].strftime("%Y-%m-%d %H:%M")
    date_end   = df_sim["date"].iloc[-1].strftime("%Y-%m-%d %H:%M")

    print(f"\n{sep}")
    print(f"  LAST 1 WEEK SIMULATION")
    print(f"  Period  : {date_start}  ->  {date_end}")
    print(f"  Candles : {len(df_sim):,}  |  Capital: Rs{INITIAL_CAP:,}")
    print(f"  Conf>={CONF_THRESHOLD*100:.0f}%  ADX>={ADX_MIN}  RSI OB/OS={RSI_OB}/{RSI_OS}  MTF>={MTF_MIN}/3")
    print(f"{sep}\n")

    # Simulation
    capital          = INITIAL_CAP
    in_trade         = False
    side             = None
    entry_p = sl_p = tp_p = 0.0
    qty              = 0
    last_close       = -COOLDOWN_CANDLES
    entry_row_idx    = 0
    entry_date_str   = ""
    trades           = []

    for i in range(len(df_sim) - 1):
        row      = df_sim.iloc[i]
        next_row = df_sim.iloc[i + 1]

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
                dur_candles = i - entry_row_idx
                trades.append({
                    "entry_time": entry_date_str,
                    "exit_time":  row["date"].strftime("%Y-%m-%d %H:%M") if hasattr(row["date"], "strftime") else str(row["date"]),
                    "side":       side,
                    "entry":      round(entry_p, 2),
                    "exit":       round(exit_p,  2),
                    "sl":         round(sl_p,    2),
                    "tp":         round(tp_p,    2),
                    "qty":        qty,
                    "gross":      round(gross, 2),
                    "costs":      round(costs, 2),
                    "net":        round(net,   2),
                    "result":     "TP" if tp_hit else "SL",
                    "duration":   f"{dur_candles*5}min",
                    "capital":    round(capital, 2),
                })
                in_trade   = False
                last_close = i

        if not in_trade and (i - last_close) >= COOLDOWN_CANDLES:
            X    = row[features].values.reshape(1, -1)
            prob = model.predict_proba(X)[0]
            p_sell, p_buy = float(prob[0]), float(prob[2])
            rsi   = float(row.get("RSI",  50))
            atr   = float(row.get("ATR",   1))
            entry = float(next_row["open"])
            sl_m, tp_m, regime_tag = _sl_tp(row)
            sizing = min(capital, MAX_CAPITAL_SIZING)

            sig = None
            conf = 0.0
            if p_buy >= CONF_THRESHOLD and p_buy > p_sell and rsi < RSI_OB:
                ok, _ = _check_filters(row, "BUY")
                if ok:
                    sig, conf = "BUY", p_buy
            elif p_sell >= CONF_THRESHOLD and p_sell > p_buy and rsi > RSI_OS:
                ok, _ = _check_filters(row, "SELL")
                if ok:
                    sig, conf = "SELL", p_sell

            if sig:
                sl_c = entry - atr * sl_m if sig == "BUY" else entry + atr * sl_m
                tp_c = entry + atr * tp_m if sig == "BUY" else entry - atr * tp_m
                q    = int((sizing * RISK_PCT / 100) / max(abs(entry - sl_c), 1))
                if q > 0:
                    in_trade      = True
                    side          = sig
                    entry_p, sl_p, tp_p, qty = entry, sl_c, tp_c, q
                    entry_row_idx = i
                    entry_date_str = (row["date"].strftime("%Y-%m-%d %H:%M")
                                      if hasattr(row["date"], "strftime") else str(row["date"]))
                    print(f"  TRADE  {sig:<4}  @ {entry:.2f}  "
                          f"SL={sl_c:.2f}  TP={tp_c:.2f}  "
                          f"Qty={q}  Conf={conf*100:.1f}%  [{regime_tag}]")

    # Close any open trade
    if in_trade:
        exit_p  = float(df_sim.iloc[-1]["close"])
        gross   = (exit_p - entry_p) * qty if side == "BUY" else (entry_p - exit_p) * qty
        costs   = _costs(entry_p, exit_p, qty)
        net     = gross - costs
        capital += net
        trades.append({
            "entry_time": entry_date_str,
            "exit_time":  "EOD",
            "side": side, "entry": round(entry_p,2), "exit": round(exit_p,2),
            "sl": round(sl_p,2), "tp": round(tp_p,2), "qty": qty,
            "gross": round(gross,2), "costs": round(costs,2), "net": round(net,2),
            "result": "EOD", "duration": "open", "capital": round(capital,2),
        })

    # Results
    print(f"\n{sep}")
    if not trades:
        print("  No trades taken in this period.")
        print(f"  (Try running on Monday when market is open)")
        print(f"{sep}\n"); return

    tr = pd.DataFrame(trades)
    total = len(tr)
    wins  = (tr["net"] > 0).sum()
    losses= (tr["net"] <= 0).sum()
    wr    = wins / total * 100
    net_pnl = tr["net"].sum()

    print(f"  RESULT SUMMARY  ({date_start} -> {date_end})")
    print(f"{sep}")
    print(f"  Trades     : {total}  ({wins}W / {losses}L)")
    print(f"  Win Rate   : {wr:.1f}%")
    print(f"  Gross P&L  : Rs{tr['gross'].sum():+,.0f}")
    print(f"  Costs      : Rs{tr['costs'].sum():,.0f}  (avg Rs{tr['costs'].mean():.0f}/trade)")
    print(f"  NET P&L    : Rs{net_pnl:+,.0f}")
    print(f"  Capital    : Rs{INITIAL_CAP:,} -> Rs{capital:,.0f}  ({(capital-INITIAL_CAP)/INITIAL_CAP*100:+.2f}%)")
    print(f"{sep}\n")

    print(f"  {'#':<3}  {'Entry Time':<17}  {'Exit Time':<17}  {'Side':<5}  "
          f"{'Entry':>8}  {'Exit':>8}  {'Qty':>5}  {'Net':>10}  {'Result':<5}  Dur")
    print(f"  {'-'*3}  {'-'*17}  {'-'*17}  {'-'*5}  "
          f"{'-'*8}  {'-'*8}  {'-'*5}  {'-'*10}  {'-'*5}  {'-'*8}")

    for idx, t in tr.iterrows():
        arrow = "+" if t["net"] > 0 else "-"
        win_  = "WIN " if t["net"] > 0 else "LOSS"
        print(f"  {idx+1:<3}  {t['entry_time']:<17}  {t['exit_time']:<17}  "
              f"{t['side']:<5}  {t['entry']:>8.2f}  {t['exit']:>8.2f}  "
              f"{t['qty']:>5}  Rs{t['net']:>+9,.0f}  {win_:<5}  {t['duration']}")

    print(f"\n  Final capital: Rs{capital:,.0f}\n")


if __name__ == "__main__":
    run()
