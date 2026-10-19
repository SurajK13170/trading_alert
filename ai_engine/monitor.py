"""
LIVE PAPER TRADING MONITOR
---------------------------
Real-time dashboard that shows:
  - Current AI signal + confidence
  - Open position + live P&L
  - Key indicators (RSI, BB%, ATR, Regime)
  - Today's trade stats

Run AFTER starting the 3 engines:
  Terminal 1: python indian_quant_engine.py
  Terminal 2: python indian_ai_engine.py
  Terminal 3: python indian_execution_manager.py

Then in a 4th terminal:
  python monitor.py
"""

import os, json, time, redis
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

SYMBOL = "Nifty50"

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def safe(d, *keys, default="--"):
    try:
        v = d
        for k in keys:
            v = v[k]
        return v
    except (KeyError, TypeError):
        return default

def age_str(ts_sec):
    if ts_sec == "--":
        return "--"
    try:
        secs = int(time.time() - float(ts_sec))
        if secs < 60:
            return f"{secs}s ago"
        return f"{secs//60}m {secs%60}s ago"
    except Exception:
        return "--"

def bar(val, lo, hi, width=20, fill="#", empty="."):
    try:
        frac = max(0.0, min(1.0, (float(val) - lo) / (hi - lo)))
        filled = int(frac * width)
        return fill * filled + empty * (width - filled)
    except Exception:
        return empty * width

def render():
    now = datetime.now().strftime("%H:%M:%S")

    # Read Redis keys
    quant_raw  = r.get(f"quant:{SYMBOL}")
    signal_raw = r.get(f"signal:{SYMBOL}")
    trade_raw  = r.get(f"active_trade:{SYMBOL}")

    quant  = json.loads(quant_raw)  if quant_raw  else {}
    signal = json.loads(signal_raw) if signal_raw else {}
    trade  = json.loads(trade_raw)  if trade_raw  else {}

    inds   = quant.get("indicators", {})
    price  = quant.get("price", 0)

    # Signal fields
    sig      = signal.get("signal", "NO DATA")
    conf     = signal.get("confidence", "0%")
    prob     = signal.get("probabilities", {})
    regime   = signal.get("regime", "--")
    rsi_v    = signal.get("rsi", "--")
    src      = signal.get("source", "--")
    sig_age  = age_str(signal.get("timestamp", "--"))

    # Indicator fields
    rsi      = inds.get("rsi", "--")
    bb_pct   = inds.get("bb_pct", "--")
    atr_pct  = inds.get("atr_pct", "--")
    macd_h   = inds.get("macd_hist", "--")
    ema_al   = inds.get("ema_alignment", "--")
    hour_v   = inds.get("hour", "--")
    q_ts     = quant.get("timestamp", 0)
    q_age    = age_str(q_ts / 1000) if q_ts else "--"

    # Trade fields
    pos      = trade.get("position", None)
    entry    = trade.get("entry", "--")
    sl_v     = trade.get("sl", "--")
    tp_v     = trade.get("tp", "--")
    qty      = trade.get("qty", "--")
    upnl     = trade.get("unrealized_pnl", 0)
    rpnl     = trade.get("realized_pnl", 0)
    trades_t = trade.get("trades_today", 0)
    t_status = trade.get("status", "--")

    # Signal color markers (ASCII)
    if sig == "BUY":
        sig_tag = "[BUY ]"
    elif sig == "SELL":
        sig_tag = "[SELL]"
    else:
        sig_tag = "[WAIT]"

    regime_tag = "[BULL]" if regime == "BULL" else "[BEAR]"

    # EMA alignment text
    if ema_al == 1:
        ema_str = "BULL (+1)"
    elif ema_al == -1:
        ema_str = "BEAR (-1)"
    elif ema_al == 0:
        ema_str = "MIXED (0)"
    else:
        ema_str = str(ema_al)

    # P&L color marker
    if isinstance(upnl, (int, float)):
        upnl_str = f"+{upnl:.0f}" if upnl >= 0 else f"{upnl:.0f}"
    else:
        upnl_str = "--"
    if isinstance(rpnl, (int, float)):
        rpnl_str = f"+{rpnl:.0f}" if rpnl >= 0 else f"{rpnl:.0f}"
    else:
        rpnl_str = "--"

    sep  = "=" * 60
    sep2 = "-" * 60

    lines = [
        sep,
        f"  NIFTY 50 PAPER TRADING MONITOR          {now}",
        sep,
        "",
        f"  PRICE   : {price:>10.2f}                  Data: {q_age}",
        f"  REGIME  : {regime_tag}  EMA-align: {ema_str}",
        "",
        sep2,
        "  SIGNAL",
        sep2,
        f"  {sig_tag}  Confidence: {conf:<8}  Source: {src}",
        f"  BUY  {prob.get('buy', '--'):>6}%  {bar(prob.get('buy', 0), 0, 100)}",
        f"  SELL {prob.get('sell', '--'):>6}%  {bar(prob.get('sell', 0), 0, 100)}",
        f"  WAIT {prob.get('wait', '--'):>6}%  {bar(prob.get('wait', 0), 0, 100)}",
        f"  Updated: {sig_age}",
        "",
        sep2,
        "  INDICATORS",
        sep2,
    ]

    # RSI bar
    try:
        rsi_f = float(rsi)
        rsi_zone = "OVERSOLD" if rsi_f < 30 else ("OVERBOUGHT" if rsi_f > 70 else "NEUTRAL")
        lines.append(f"  RSI     : {rsi_f:>6.1f}  [{bar(rsi_f, 0, 100)}]  {rsi_zone}")
    except Exception:
        lines.append(f"  RSI     : {rsi}")

    # BB%
    try:
        bb_f = float(bb_pct)
        bb_zone = "LOWER" if bb_f < 0.2 else ("UPPER" if bb_f > 0.8 else "MID")
        lines.append(f"  BB%     : {bb_f:>6.3f}  [{bar(bb_f, 0, 1)}]  {bb_zone}")
    except Exception:
        lines.append(f"  BB%     : {bb_pct}")

    lines += [
        f"  ATR%    : {atr_pct}",
        f"  MACD_H  : {macd_h}",
        f"  Hour    : {hour_v}:xx IST",
        "",
        sep2,
        "  POSITION",
        sep2,
    ]

    if pos:
        lines += [
            f"  Status  : {t_status}  |  Side: {pos}",
            f"  Entry   : {entry:.2f}  |  Qty: {qty}",
            f"  SL      : {sl_v:.2f}  |  TP: {tp_v:.2f}",
            f"  Unreal P&L : Rs {upnl_str}",
        ]
    else:
        lines.append("  FLAT  (no open position)")

    lines += [
        "",
        sep2,
        "  SESSION",
        sep2,
        f"  Trades today : {trades_t}",
        f"  Realized P&L : Rs {rpnl_str}",
        "",
        sep,
        "  Ctrl+C to stop  |  Refreshes every 2s",
        sep,
    ]

    clear()
    print("\n".join(lines))


def main():
    print("Connecting to Redis...")
    try:
        r.ping()
        print("Connected. Starting monitor...\n")
        time.sleep(1)
    except Exception as e:
        print(f"[ERR] Redis not reachable: {e}")
        print("      Start Redis first, then run the 3 engines.")
        return

    while True:
        try:
            render()
        except Exception as e:
            clear()
            print(f"[RENDER ERR] {e}")
        time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
