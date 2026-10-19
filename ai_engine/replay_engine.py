"""
NIFTY 50 -- HISTORICAL REPLAY ENGINE
-------------------------------------
Off-market simulator. Replays historical 5-min candles into Redis as if they
were live ticks. Lets you test the FULL pipeline (AI engine + frontend +
monitor + trades) any time of day -- no live yfinance, no market-hours wait.

NOTHING ELSE IN THE PROJECT IS MODIFIED.
Replaces ONLY two roles for the duration of the replay:
  - indian_quant_engine.py  (writes quant:Nifty50)
  - indian_execution_manager.py (writes active_trade:Nifty50)

The AI engine, backend, frontend, and monitor all keep running normally.

Usage:
  python replay_engine.py
  REPLAY_CANDLES=200 REPLAY_SPEED=2 python replay_engine.py

Pipeline you should run alongside:
  1. python indian_ai_engine.py     (signal generator)
  2. python replay_engine.py        (this file, replaces quant + exec)
  3. (optional) npm run dev in backend_manager + frontend_manager
"""

import os
import csv
import time
import json
import redis
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from model_trainer import add_features, FEATURES, load_and_clean

load_dotenv()

# ─── config ─────────────────────────────────────────────────────
CSV_PATH       = "nifty_5m_data.csv"
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379")
REPLAY_CANDLES = int(os.getenv("REPLAY_CANDLES", "300"))
REPLAY_SPEED   = float(os.getenv("REPLAY_SPEED", "1.5"))   # min seconds between candles
SIGNAL_TIMEOUT = float(os.getenv("SIGNAL_TIMEOUT", "5.0"))  # max wait for AI engine
SYMBOL_KEY     = "Nifty50"

CAPITAL              = float(os.getenv("TRADING_CAPITAL", "100000"))
RISK_PER_TRADE_PCT   = 1.0
CONF_THRESHOLD       = 40.0

# Auto-trade pause feature -- mirrors indian_execution_manager.py
DAILY_PROFIT_TARGET    = 0   # no profit pause
MANUAL_APPROVE_TTL_SEC = 60

# Mirror execution manager schema so /trades endpoint reads replay trades too
FEEDBACK_CSV   = "live_feedback.csv"
FEEDBACK_COLS  = [
    "timestamp", "signal", "outcome",
    "rsi", "rsi_delta", "rsi_lag1", "rsi_lag2",
    "macd_hist", "macd_hist_delta",
    "ema_dist_20", "ema_dist_50", "ema_dist_200", "ema_alignment",
    "bb_pct", "bb_width", "atr_pct",
    "candle_body", "upper_wick", "lower_wick",
    "hour", "minute", "day_of_week",
    "vol_ratio", "daily_regime",
    "entry_price", "exit_price", "pnl",
]
# ────────────────────────────────────────────────────────────────

r = redis.from_url(REDIS_URL, decode_responses=True)

portfolio = {
    "position": None, "entry_price": 0.0,
    "sl": 0.0, "tp": 0.0, "qty": 0,
    "open_time": None,
}
session = {
    "trades":      0,
    "wins":        0,
    "losses":      0,
    "pnl":         0.0,
    "win_pnl":     0.0,    # gross winning amount
    "loss_pnl":    0.0,    # gross losing amount (negative)
    "biggest_win": 0.0,
    "biggest_loss": 0.0,
    "buy_trades":   0,
    "sell_trades":  0,
    "sl_hits":     0,
    "tp_hits":     0,
    "eod_closes":  0,
}

# Snapshot of indicator features at trade entry -- written to feedback CSV on close
_entry_features = {}


def _clean(v, p=4):
    try:
        v = float(v)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return round(v, p)
    except (TypeError, ValueError):
        return 0.0


# ── Auto-trade state machine (matches indian_execution_manager.py) ──────
def get_auto_state() -> str:
    return (r.get("auto_state:Nifty50") or "RUNNING").upper()


def set_auto_state(state: str, reason: str = ""):
    prev = get_auto_state()
    if prev == state:
        return
    r.set("auto_state:Nifty50", state)
    tag = "[AUTO]" if state == "RUNNING" else "[PAUSED]"
    print(f"\n{tag} State: {prev} -> {state}" + (f"  ({reason})" if reason else ""))
    if reason:
        r.set("auto_state_event:Nifty50", json.dumps({
            "prev":   prev,
            "next":   state,
            "reason": reason,
            "ts":     int(time.time() * 1000),
        }), ex=30)


def has_manual_approval() -> bool:
    raw = r.get("manual_approve:Nifty50")
    if not raw:
        return False
    try:
        when = float(raw)
    except (TypeError, ValueError):
        r.delete("manual_approve:Nifty50")
        return False
    if time.time() - when > MANUAL_APPROVE_TTL_SEC:
        r.delete("manual_approve:Nifty50")
        return False
    return True


def consume_manual_approval():
    r.delete("manual_approve:Nifty50")


def check_profit_pause() -> bool:
    """Auto-switch to PAUSED when daily profit crosses target. Returns True if switched."""
    if DAILY_PROFIT_TARGET <= 0:
        return False
    if session["pnl"] < DAILY_PROFIT_TARGET:
        return False
    if get_auto_state() == "PAUSED":
        return False
    set_auto_state("PAUSED",
                   reason=f"Daily profit target Rs{DAILY_PROFIT_TARGET:,.0f} reached "
                          f"(P&L Rs{session['pnl']:+,.0f})")
    return True


def write_quant_data(row):
    """Mirror indian_quant_engine.py exactly -- AI engine reads this key."""
    indicators = {f.lower(): _clean(row[f]) for f in FEATURES}
    # display-only fields the AI engine and monitor expect
    indicators["atr"]       = _clean(row["ATR"])
    indicators["close"]     = _clean(row["close"], 2)
    indicators["daily_rsi"] = _clean(row["RSI"], 2)

    payload = {
        "symbol":     SYMBOL_KEY,
        "price":      float(row["close"]),
        "indicators": indicators,
        "trend":      "BULLISH" if row["close"] > row["EMA_200"] else "BEARISH",
        "session":    f"{int(row['Hour']):02d}:{int(row['Minute']):02d} IST | "
                      f"Day {int(row['Day_of_week'])}",
        "timestamp":  int(time.time() * 1000),
    }
    r.set(f"quant:{SYMBOL_KEY}", json.dumps(payload))


def write_active_trade(status: str, upnl: float = 0.0):
    r.set(f"active_trade:{SYMBOL_KEY}", json.dumps({
        "status":         status,
        "position":       portfolio["position"],
        "entry":          portfolio["entry_price"],
        "sl":             portfolio["sl"],
        "tp":             portfolio["tp"],
        "qty":            portfolio["qty"],
        "unrealized_pnl": round(upnl, 2),
        "realized_pnl":   round(session["pnl"], 2),
        "trades_today":   session["trades"],
        "mode":           "PAPER",          # replay is always virtual
        "capital":        CAPITAL,
        "auto_state":     get_auto_state(),
        "daily_profit_target": DAILY_PROFIT_TARGET,
        "timestamp":      time.time(),
    }))


def wait_for_signal(ref_time: float):
    """Wait until AI engine writes a signal newer than ref_time (max SIGNAL_TIMEOUT)."""
    deadline = time.time() + SIGNAL_TIMEOUT
    while time.time() < deadline:
        raw = r.get(f"signal:{SYMBOL_KEY}")
        if raw:
            try:
                sig = json.loads(raw)
                if float(sig.get("timestamp", 0)) >= ref_time:
                    return sig
            except (json.JSONDecodeError, ValueError):
                pass
        time.sleep(0.15)
    return None


def _snapshot_features(row):
    """Capture indicator features at entry -- same keys as live execution manager."""
    _entry_features.clear()
    for col in FEEDBACK_COLS[3:21]:   # rsi ... daily_regime (lowercase keys)
        try:
            v = float(row[col.upper()] if col.upper() in row.index else row[col])
            if np.isnan(v) or np.isinf(v):
                v = 0.0
            _entry_features[col] = round(v, 4)
        except (KeyError, TypeError, ValueError):
            _entry_features[col] = 0.0
    _entry_features["entry_price"] = float(row["close"])


def _log_feedback(signal: str, outcome: str, exit_price: float, pnl: float):
    """Write closed trade to live_feedback.csv -- same schema as execution manager."""
    try:
        row = {
            "timestamp":   datetime.now().isoformat(),
            "signal":      signal,
            "outcome":     outcome,
            "entry_price": _entry_features.get("entry_price", 0),
            "exit_price":  round(exit_price, 2),
            "pnl":         round(pnl, 2),
        }
        for col in FEEDBACK_COLS[3:21]:
            row[col] = _entry_features.get(col, 0)

        write_header = not os.path.exists(FEEDBACK_CSV)
        with open(FEEDBACK_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"  [FEEDBACK] Could not log trade: {e}")
    finally:
        _entry_features.clear()


def open_trade(signal: str, price: float, sl: float, tp: float, row=None):
    risk_per_unit = abs(price - sl)
    if risk_per_unit <= 0:
        return
    max_risk = CAPITAL * (RISK_PER_TRADE_PCT / 100)
    qty = max(1, int(max_risk / risk_per_unit))

    portfolio.update({
        "position":    signal,
        "entry_price": price,
        "sl":          sl,
        "tp":          tp,
        "qty":         qty,
        "open_time":   datetime.now().isoformat(),
    })
    session["trades"] += 1
    if signal == "BUY":
        session["buy_trades"] += 1
    else:
        session["sell_trades"] += 1
    if row is not None:
        _snapshot_features(row)
    print(f"  [OPEN]  {signal:<4} @ Rs{price:.2f} | SL={sl:.2f} TP={tp:.2f} | Qty={qty}")
    write_active_trade("OPEN")


def close_trade(reason: str, exit_price: float, pnl: float):
    session["pnl"] += pnl
    if pnl > 0:
        session["wins"] += 1
        session["win_pnl"] += pnl
        session["biggest_win"] = max(session["biggest_win"], pnl)
        tag = "[WIN] "
        outcome = "WIN"
    else:
        session["losses"] += 1
        session["loss_pnl"] += pnl
        session["biggest_loss"] = min(session["biggest_loss"], pnl)
        tag = "[LOSS]"
        outcome = "LOSS"
    # Track close reason
    if reason == "STOP LOSS":
        session["sl_hits"] += 1
    elif reason == "TAKE PROFIT":
        session["tp_hits"] += 1
    elif reason in ("REPLAY END", "MARKET CLOSE"):
        session["eod_closes"] += 1
    print(f"  [CLOSE] {reason:<11} @ Rs{exit_price:.2f} | {tag} Rs{pnl:+.2f} "
          f"| Total Rs{session['pnl']:+.2f}")

    closed_signal = portfolio["position"]
    portfolio.update({
        "position":    None,
        "entry_price": 0.0,
        "sl":          0.0,
        "tp":          0.0,
        "qty":         0,
        "open_time":   None,
    })
    write_active_trade("CLOSED")
    # Log to live_feedback.csv so the frontend trade history table picks it up
    if closed_signal:
        _log_feedback(closed_signal, outcome, exit_price, pnl)
    # Auto-pause once daily profit target is crossed
    check_profit_pause()


def check_position(row):
    """Simulate the live monitor: did SL or TP hit during this candle?"""
    if portfolio["position"] is None:
        return
    pos   = portfolio["position"]
    entry = portfolio["entry_price"]
    sl    = portfolio["sl"]
    tp    = portfolio["tp"]
    qty   = portfolio["qty"]
    high, low, close = float(row["high"]), float(row["low"]), float(row["close"])

    if pos == "BUY":
        sl_hit, tp_hit = (low <= sl), (high >= tp)
        upnl = (close - entry) * qty
    else:
        sl_hit, tp_hit = (high >= sl), (low <= tp)
        upnl = (entry - close) * qty

    # Conservative: assume SL checked first within the candle
    if sl_hit:
        pnl = (sl - entry) * qty if pos == "BUY" else (entry - sl) * qty
        close_trade("STOP LOSS", sl, pnl)
    elif tp_hit:
        pnl = (tp - entry) * qty if pos == "BUY" else (entry - tp) * qty
        close_trade("TAKE PROFIT", tp, pnl)
    else:
        write_active_trade("OPEN", upnl)


def force_close_eod(price):
    """Last replay candle -- close any open position at market price."""
    if portfolio["position"] is None:
        return
    pos   = portfolio["position"]
    entry = portfolio["entry_price"]
    qty   = portfolio["qty"]
    pnl   = (price - entry) * qty if pos == "BUY" else (entry - price) * qty
    close_trade("REPLAY END", price, pnl)


def main():
    sep = "=" * 64
    print(f"\n{sep}")
    print("  NIFTY 50 -- HISTORICAL REPLAY ENGINE")
    print(f"  Candles      : last {REPLAY_CANDLES} from {CSV_PATH}")
    print(f"  Min interval : {REPLAY_SPEED:.1f}s / candle "
          f"(AI engine sets the actual pace)")
    print(f"  Sig timeout  : {SIGNAL_TIMEOUT:.1f}s before skipping a candle")
    print(f"  Capital      : Rs{CAPITAL:,.0f} | Risk {RISK_PER_TRADE_PCT}% / trade")
    print(f"  Conf min     : {CONF_THRESHOLD}%")
    print(f"{sep}\n")

    # 1. Load + compute every feature on the full history (cached MTF + VIX)
    print("[LOAD] Reading CSV and computing 28-feature pipeline...")
    df = load_and_clean(CSV_PATH)
    df = add_features(df)
    df.dropna(subset=FEATURES + ["ATR"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[LOAD] Ready: {len(df):,} candles with full feature set\n")

    # 2. Last N candles
    if len(df) < REPLAY_CANDLES:
        print(f"[WARN] Only {len(df)} candles available, replaying all of them.")
        replay = df.copy()
    else:
        replay = df.iloc[-REPLAY_CANDLES:].reset_index(drop=True)

    first_ts = pd.Timestamp(replay["date"].iloc[0]).strftime("%Y-%m-%d %H:%M")
    last_ts  = pd.Timestamp(replay["date"].iloc[-1]).strftime("%Y-%m-%d %H:%M")
    print(f"[REPLAY] {first_ts}  ->  {last_ts}  ({len(replay)} candles)\n")

    # 3. Reset Redis so frontend starts clean
    r.delete(f"signal:{SYMBOL_KEY}")
    write_active_trade("CLOSED")

    # 4. Replay loop
    for i, row in replay.iterrows():
        ts_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d %H:%M")
        ref    = time.time()

        # a) Quant update -> Redis (AI engine picks this up)
        write_quant_data(row)

        # b) Check existing position against this candle's high/low first
        check_position(row)

        # c) Wait for AI engine to react to the new quant data
        sig = wait_for_signal(ref)

        # d) If flat and confident signal received -> open trade
        if portfolio["position"] is None and sig is not None:
            signal_type = sig.get("signal", "WAIT")
            try:
                conf = float(str(sig.get("confidence", "0%")).replace("%", ""))
            except ValueError:
                conf = 0.0
            sl = sig.get("stop_loss")
            tp = sig.get("take_profit")
            if (signal_type in ("BUY", "SELL")
                    and conf >= CONF_THRESHOLD
                    and sl is not None and tp is not None):
                # Gate auto-execution on auto_state -- mirrors live engine
                auto = get_auto_state()
                if auto == "RUNNING":
                    open_trade(signal_type, float(row["close"]),
                               float(sl), float(tp), row=row)
                elif has_manual_approval():
                    print(f"  [MANUAL-APPROVED] User approved this {signal_type} trade.")
                    open_trade(signal_type, float(row["close"]),
                               float(sl), float(tp), row=row)
                    consume_manual_approval()
                # else: PAUSED, no approval -- skip silently (signal stays visible)

        # e) Status line
        sig_disp  = sig.get("signal", "—") if sig else "(no-sig)"
        conf_disp = sig.get("confidence", "—") if sig else "—"
        pos_disp  = portfolio["position"] or "FLAT"
        print(f"  [{i+1:>4}/{len(replay)}] {ts_str} | "
              f"Px={row['close']:>9.2f} | AI={sig_disp:<4} {conf_disp:<7} | "
              f"Pos={pos_disp:<4} | PnL=Rs{session['pnl']:+,.2f}")

        # f) Pace the replay (subtract time we already spent waiting on AI)
        elapsed = time.time() - ref
        if elapsed < REPLAY_SPEED:
            time.sleep(REPLAY_SPEED - elapsed)

    # 5. Close any open position at the final price
    force_close_eod(float(replay["close"].iloc[-1]))

    # 6. Detailed summary
    _print_summary(replay)


def _print_summary(replay):
    """Detailed end-of-replay report."""
    sep = "=" * 64
    t   = session["trades"]
    w   = session["wins"]
    l   = session["losses"]
    pnl = session["pnl"]

    print(f"\n{sep}")
    print("  REPLAY COMPLETE -- FINAL REPORT")
    print(f"{sep}")
    print(f"  Period       : {pd.Timestamp(replay['date'].iloc[0]).strftime('%Y-%m-%d %H:%M')}"
          f"  ->  {pd.Timestamp(replay['date'].iloc[-1]).strftime('%Y-%m-%d %H:%M')}")
    print(f"  Candles      : {len(replay):,}")
    print()
    print("  TRADES")
    print(f"    Total      : {t}")
    print(f"    Successful : {w}  (WIN -- price hit TP or closed in profit)")
    print(f"    Failed     : {l}  (LOSS -- price hit SL or closed in loss)")
    if t > 0:
        wr = w / t * 100
        print(f"    Win Rate   : {wr:.1f}%")
    print(f"    BUY  trades: {session['buy_trades']}")
    print(f"    SELL trades: {session['sell_trades']}")
    print()
    print("  CLOSE REASONS")
    print(f"    Take Profit hit : {session['tp_hits']}")
    print(f"    Stop Loss hit   : {session['sl_hits']}")
    print(f"    Forced (EOD)    : {session['eod_closes']}")
    print()
    print("  PROFIT & LOSS")
    print(f"    Gross Profit   : Rs{session['win_pnl']:>+12,.2f}")
    print(f"    Gross Loss     : Rs{session['loss_pnl']:>+12,.2f}")
    print(f"    Net P&L        : Rs{pnl:>+12,.2f}")
    print(f"    % of Capital   : {pnl / CAPITAL * 100:>+12.2f}%")
    if w > 0:
        avg_win = session['win_pnl'] / w
        print(f"    Avg Win        : Rs{avg_win:>+12,.2f}")
    if l > 0:
        avg_loss = session['loss_pnl'] / l
        print(f"    Avg Loss       : Rs{avg_loss:>+12,.2f}")
    print(f"    Biggest Win    : Rs{session['biggest_win']:>+12,.2f}")
    print(f"    Biggest Loss   : Rs{session['biggest_loss']:>+12,.2f}")
    # Profit factor: gross profit / abs(gross loss). >1 = profitable.
    if session['loss_pnl'] < 0:
        pf = session['win_pnl'] / abs(session['loss_pnl'])
        verdict = "PROFITABLE" if pf > 1 else "UNPROFITABLE"
        print(f"    Profit Factor  : {pf:>12.2f}  [{verdict}]")
    print()
    print("  NEXT STEPS")
    print("    -- Subah live data se testing chalu karne ke liye:")
    print("       1. Yeh replay terminal Ctrl+C karke band karo")
    print("       2. Wahi 4 terminals open karo, replay ki jagah live engines:")
    print("            python indian_quant_engine.py     (replaces replay)")
    print("            python indian_execution_manager.py (replaces replay execution)")
    print("       3. AI engine, backend, frontend wahi continue rahenge")
    print("    -- Fresh slate ke liye (replay trades hatane ke liye):")
    print("            del live_feedback.csv")
    print(f"{sep}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Replay interrupted by user.")
        force_close_eod(portfolio["entry_price"] or 0.0)
        # Show summary even if stopped mid-replay
        try:
            df_snap = load_and_clean(CSV_PATH).tail(REPLAY_CANDLES)
            _print_summary(df_snap.reset_index(drop=True))
        except Exception:
            print(f"  Trades: {session['trades']} | Wins: {session['wins']} | "
                  f"Losses: {session['losses']} | Net P&L: Rs{session['pnl']:+,.2f}")
