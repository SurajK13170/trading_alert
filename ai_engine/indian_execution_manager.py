"""
NIFTY 50 -- EXECUTION MANAGER v3.0
------------------------------------
PAPER mode : works WITHOUT Angel One credentials.
             Price fetched via yfinance (free, no auth).
LIVE mode  : uses Angel One SmartAPI for price + order placement.

Risk Rules:
  - Max risk per trade  : RISK_PER_TRADE_PCT % of capital
  - Max daily loss      : MAX_DAILY_LOSS_PCT  % of capital
  - Max concurrent pos  : 1 (one trade at a time)
  - Confidence threshold: CONF_THRESHOLD %
"""

import os
import time
import json
import csv
import subprocess
import redis
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

from slack_alerts import send_trade_closed_alert, maybe_send_alert as maybe_send_slack_alert
from telegram_alerts import maybe_send_alert as maybe_send_telegram_alert, send_startup_message as send_telegram_startup_message

# ----------------------------------------------------------------
# Startup default (used until Redis tells us otherwise). The runtime MODE is
# read from Redis on every cycle so the frontend toggle takes effect without
# restarting this process.
MODE                 = os.getenv("EXECUTION_MODE", "PAPER")
PAPER_CAPITAL        = float(os.getenv("TRADING_CAPITAL", "100000"))
LIVE_CAPITAL         = float(os.getenv("LIVE_CAPITAL",    PAPER_CAPITAL))
CAPITAL              = PAPER_CAPITAL  # updated at runtime when mode flips
RISK_PER_TRADE_PCT   = 1.0
MAX_DAILY_LOSS_PCT   = 100.0  # no daily loss halt -- let model trade and learn freely
DAILY_PROFIT_TARGET  = 0      # 0 = no profit pause -- trade without any target restriction
MAX_LOTS_PER_TRADE   = int(os.getenv("MAX_LOTS_PER_TRADE",  "2"))        # default 2 lots
MANUAL_APPROVE_TTL_SEC = 60   # how long a manual approval stays valid
CONF_THRESHOLD       = 62.0
MONITOR_INTERVAL_SEC  = 5
FEEDBACK_CSV          = "live_feedback.csv"
AUTO_RETRAIN_EVERY    = 50    # retrain XGBoost after every N closed trades
CONTRA_EXIT_THRESHOLD = 52.0  # opposite signal confidence needed to auto-exit a trade
TRADE_COOLDOWN_SEC    = 5 * 60    # 5 min wait after any trade closes (matches backtester)
# ----------------------------------------------------------------

redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
)

# Angel One creds (loaded lazily when LIVE mode is activated)
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
API_KEY   = os.getenv("ANGEL_API_KEY")
PASSWORD  = os.getenv("ANGEL_PASSWORD")
TOTP_KEY  = os.getenv("ANGEL_TOTP_KEY")
smartApi  = None        # populated on first LIVE switch
_authed   = False       # have we successfully logged in this session?

def _ensure_live_session() -> bool:
    """Lazy: create + authenticate Angel One SmartConnect when LIVE mode is needed."""
    global smartApi, _authed
    if _authed and smartApi is not None:
        return True
    if not all([CLIENT_ID, API_KEY, PASSWORD, TOTP_KEY]):
        print("[LIVE-AUTH] Angel One creds missing in .env -- cannot enable LIVE.")
        return False
    try:
        from SmartApi import SmartConnect
        import pyotp
        smartApi = SmartConnect(api_key=API_KEY)
        totp = pyotp.TOTP(TOTP_KEY).now()
        data = smartApi.generateSession(CLIENT_ID, PASSWORD, totp)
        if data and data.get("status"):
            print("[LIVE-AUTH] Angel One login successful.")
            _authed = True
            return True
        print(f"[LIVE-AUTH] Login failed: {data}")
        return False
    except ImportError:
        print("[LIVE-AUTH] SmartApi not installed: pip install smartapi-python")
        return False
    except Exception as e:
        print(f"[LIVE-AUTH] Error: {e}")
        return False


# Cache the account balance so we don't hit Angel's RMS endpoint every cycle
_account_balance_cache = {"value": None, "ts": 0}

def fetch_account_balance() -> float | None:
    """Pull available cash from Angel One RMS. Cached for 60 sec."""
    if not _authed or smartApi is None:
        return None
    now = time.time()
    if _account_balance_cache["value"] is not None and (now - _account_balance_cache["ts"] < 60):
        return _account_balance_cache["value"]
    try:
        rms = smartApi.rmsLimit()
        if rms and rms.get("status") and rms.get("data"):
            d = rms["data"]
            # Try common balance keys in priority order. Take the FIRST one
            # that parses as a number -- including 0.0 (a real empty account).
            for key in ("availablecash", "net", "availableintradaypayin", "availablelimitmargin"):
                v = d.get(key)
                if v is None or v == "":
                    continue
                try:
                    bal = float(v)
                except (TypeError, ValueError):
                    continue
                # bal can legitimately be 0.0 -- that's an empty account, not a fetch failure
                _account_balance_cache["value"] = bal
                _account_balance_cache["ts"]    = now
                return bal
            # All numeric keys missing -- log what came back so we can debug
            print(f"[BALANCE] rmsLimit returned no parseable balance. Raw data keys: {list(d.keys())}")
        else:
            print(f"[BALANCE] rmsLimit unsuccessful: {rms}")
    except Exception as e:
        print(f"[BALANCE] Could not fetch from Angel One: {e}")
    return None


# Lazy-import options helper so PAPER-only sessions don't pay the cost
def _options_helper():
    import options_helper as _oh
    return _oh

CASH_SYMBOL      = "NIFTYBEES-EQ"
CASH_TOKEN       = "10576"
CASH_EXCHANGE    = "NSE"

FUTURES_SYMBOL   = os.getenv("NIFTY_FUTURES_SYMBOL", "NIFTY29MAY25FUT")
FUTURES_TOKEN    = os.getenv("NIFTY_FUTURES_TOKEN",  "58009")
FUTURES_EXCHANGE = "NFO"
FUTURES_LOT_SIZE = 75

portfolio = {
    "position":    None,
    "entry_price": 0.0,    # underlying Nifty spot at entry
    "sl":          0.0,    # spot-based SL (ATR)
    "tp":          0.0,    # spot-based TP (ATR)
    "qty":         0,      # total qty (lots * lot_size for options)
    "instrument":  None,
    "open_time":   None,
    # Options-specific fields (None when not in LIVE option mode)
    "opt_symbol":  None,   # e.g. NIFTY19MAY2623550CE
    "opt_token":   None,   # Angel One token
    "opt_type":    None,   # CE or PE
    "opt_strike":  0,
    "opt_lots":    0,      # number of lots
    "opt_entry_premium": 0.0,
}

session = {
    "date":         date.today(),
    "trades":       0,
    "realized_pnl": 0.0,
    "halted":       False,
    "consecutive_losses": 0,   # circuit breaker: pause after too many in a row
}

# Stop trading for the day after this many losing trades in a row (capital guard).
MAX_CONSECUTIVE_LOSSES = 3

_last_trade_close_ts = 0.0   # epoch time of last trade close (for cooldown)


# ══════════════════════════════════════════════════════════════
#  MODE SYNC (runtime toggle from frontend)
# ══════════════════════════════════════════════════════════════
def _compute_live_capital() -> float:
    """
    LIVE mode position sizing uses the SMALLER of:
      - LIVE_CAPITAL  (user-set safety cap from .env)
      - actual available cash in the Angel One account

    Behavior:
      - bal == 0.0     -> real empty account, CAPITAL = 0 (trades will skip)
      - bal == None    -> API fetch failed (network/auth) -> fall back to LIVE_CAPITAL
                          with a warning so the user knows the number is stale
      - bal  > 0       -> use the smaller of the two
    """
    bal = fetch_account_balance()
    if bal is None:
        print("[WARN] Angel balance fetch failed -- using LIVE_CAPITAL safety cap as fallback. "
              "Number on dashboard may not reflect real Angel balance.")
        return LIVE_CAPITAL
    return min(LIVE_CAPITAL, bal)


def sync_mode_from_redis():
    """
    Read `execution_mode:Nifty50` from Redis and update process-wide MODE +
    CAPITAL. Lazy-auths Angel One on first LIVE switch. Called every cycle.
    """
    global MODE, CAPITAL
    desired = (redis_client.get("execution_mode:Nifty50") or MODE).upper()
    if desired not in ("PAPER", "LIVE"):
        return

    if desired == MODE:
        # Already in this mode -- keep CAPITAL fresh (LIVE balance can change as P&L moves)
        if MODE == "LIVE":
            CAPITAL = _compute_live_capital()
        else:
            CAPITAL = PAPER_CAPITAL
        return

    # Mode is changing
    if desired == "LIVE":
        if not _ensure_live_session():
            print("[MODE] LIVE switch denied: Angel One auth failed. Forcing PAPER.")
            redis_client.set("execution_mode:Nifty50", "PAPER")
            MODE    = "PAPER"
            CAPITAL = PAPER_CAPITAL
            return
        MODE    = "LIVE"
        CAPITAL = _compute_live_capital()
        bal     = fetch_account_balance()
        print(f"\n[MODE] *** Switched to LIVE -- REAL MONEY ***")
        print(f"       Angel balance : Rs{bal:,.0f}" if bal is not None else "       Angel balance : (fetch failed)")
        print(f"       Safety cap    : Rs{LIVE_CAPITAL:,.0f}")
        print(f"       Effective cap : Rs{CAPITAL:,.0f}  (used for position sizing)\n")
    else:
        MODE    = "PAPER"
        CAPITAL = PAPER_CAPITAL
        print(f"\n[MODE] Switched to PAPER (virtual) | Capital Rs{CAPITAL:,.0f}\n")


# Kept for backward compat -- now a no-op since lazy auth handles LIVE on demand
def authenticate() -> bool:
    return True


# ══════════════════════════════════════════════════════════════
#  PRICE FETCH
#  PAPER: yfinance (free, no auth)
#  LIVE:  Angel One SmartAPI
# ══════════════════════════════════════════════════════════════
def fetch_live_price() -> float | None:
    if MODE == "PAPER":
        return _price_yfinance()
    return _price_angel()


def _price_yfinance() -> float | None:
    # PRIMARY in PAPER mode: read price from Redis (Angel One quant engine, real-time).
    # yfinance has ~15 min delay for Indian intraday — bad for SL/TP decisions.
    try:
        raw = redis_client.get("quant:Nifty50")
        if raw:
            data = json.loads(raw)
            # Only trust quant data if it's fresh (<60s old)
            age = time.time() - data.get("timestamp", 0) / 1000
            if age < 60:
                price = float(data.get("price", 0))
                if price > 0:
                    return price
    except Exception:
        pass
    # Fallback: yfinance (delayed but available)
    try:
        import yfinance as yf
        data = yf.Ticker("^NSEI").history(period="1d", interval="1m")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception as e:
        print(f"[WARN] yfinance price error: {e}")
    return None


def _price_angel() -> float | None:
    for attempt in range(3):
        try:
            data = smartApi.ltpData(CASH_EXCHANGE, CASH_SYMBOL, CASH_TOKEN)
            if data and data.get("status") and data.get("data"):
                return float(data["data"]["ltp"])
        except Exception as e:
            print(f"[WARN] Angel price attempt {attempt+1}/3: {e}")
            time.sleep(2)
    return None


def _live_option_premium() -> float | None:
    """
    Latest option premium streamed by tick_collector.py into opt_tick:Nifty50.
    Returns None if missing or stale (>30s) so callers fall back to spot estimate.
    """
    try:
        raw = redis_client.get("opt_tick:Nifty50")
        if not raw:
            return None
        d = json.loads(raw)
        if time.time() - float(d.get("ts", 0)) > 30:
            return None
        ltp = float(d.get("ltp", 0))
        return ltp if ltp > 0 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
#  POSITION SIZING
# ══════════════════════════════════════════════════════════════
def calculate_qty(entry: float, sl: float) -> int:
    risk_per_unit = abs(entry - sl)
    if risk_per_unit <= 0:
        return 1
    max_risk_inr = CAPITAL * (RISK_PER_TRADE_PCT / 100)
    return max(1, int(max_risk_inr / risk_per_unit))


# ══════════════════════════════════════════════════════════════
#  SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════
def check_session_reset():
    today = date.today()
    if session["date"] != today:
        print(f"\n[DAY] New trading day: {today}. Resetting session...")
        session["date"]         = today
        session["trades"]       = 0
        session["realized_pnl"] = 0.0
        session["halted"]       = False
        session["consecutive_losses"] = 0


def check_daily_loss_limit() -> bool:
    # Once halted for ANY reason (daily loss OR consecutive-loss breaker), stay halted.
    if session.get("halted"):
        return True
    max_loss = CAPITAL * (MAX_DAILY_LOSS_PCT / 100)
    if session["realized_pnl"] <= -max_loss:
        if not session["halted"]:
            print(f"\n[HALT] Daily loss limit hit! "
                  f"P&L: Rs{session['realized_pnl']:.2f} | Limit: Rs{-max_loss:.2f}")
            session["halted"] = True
        return True
    return False


def get_auto_state() -> str:
    """Read auto-trade state from Redis. Default RUNNING."""
    return (redis_client.get("auto_state:Nifty50") or "RUNNING").upper()


def set_auto_state(state: str, reason: str = ""):
    """Switch auto-trade state. Logs the transition."""
    prev = get_auto_state()
    if prev == state:
        return
    redis_client.set("auto_state:Nifty50", state)
    tag = "[AUTO-TRADE]" if state == "RUNNING" else "[PAUSED]"
    print(f"\n{tag} State: {prev} -> {state}" + (f"  ({reason})" if reason else ""))
    # Surface a one-shot "transition event" so the frontend can pop a modal
    # exactly once for the threshold-crossing case.
    if reason:
        redis_client.set("auto_state_event:Nifty50", json.dumps({
            "prev":   prev,
            "next":   state,
            "reason": reason,
            "ts":     int(time.time() * 1000),
        }), ex=30)   # event expires in 30s


def check_profit_pause() -> bool:
    """
    When daily profit crosses DAILY_PROFIT_TARGET, AUTOMATICALLY switch to
    PAUSED state -- but do NOT halt forever. AI engine still produces signals;
    user can choose to manually approve trades or resume auto.
    Returns True if state changed to PAUSED on this call.
    """
    if DAILY_PROFIT_TARGET <= 0:
        return False
    if session["realized_pnl"] < DAILY_PROFIT_TARGET:
        return False
    if get_auto_state() == "PAUSED":
        return False  # already paused, nothing to do
    set_auto_state("PAUSED",
                   reason=f"Daily profit target Rs{DAILY_PROFIT_TARGET:,.0f} reached "
                          f"(P&L Rs{session['realized_pnl']:+,.0f})")
    return True


def has_manual_approval() -> bool:
    """
    Check if the dashboard has posted a one-shot trade approval.
    Approval auto-expires after MANUAL_APPROVE_TTL_SEC. Consumed on success.
    """
    raw = redis_client.get("manual_approve:Nifty50")
    if not raw:
        return False
    try:
        when = float(raw)
    except (TypeError, ValueError):
        redis_client.delete("manual_approve:Nifty50")
        return False
    if time.time() - when > MANUAL_APPROVE_TTL_SEC:
        redis_client.delete("manual_approve:Nifty50")
        return False
    return True


def consume_manual_approval():
    """Delete the approval token so it only fires once."""
    redis_client.delete("manual_approve:Nifty50")


# ══════════════════════════════════════════════════════════════
#  FEEDBACK LOOP  -- saves real trade outcomes for retraining
# ══════════════════════════════════════════════════════════════
_pending_feedback = {}   # stores entry features until trade closes

FEEDBACK_COLS = [
    "timestamp", "signal", "outcome", "confidence",
    # base features
    "rsi", "rsi_delta", "rsi_lag1", "rsi_lag2",
    "macd_hist", "macd_hist_delta",
    "ema_dist_20", "ema_dist_50", "ema_dist_200", "ema_alignment",
    "bb_pct", "bb_width", "atr_pct",
    # NEW: volatility percentile ranks
    "atr_pctile", "bb_width_pctile",
    "candle_body", "upper_wick", "lower_wick",
    "hour", "minute", "day_of_week",
    "session_bucket",
    "vol_ratio", "daily_regime",
    # NEW: trend + regime
    "adx", "market_regime",
    # NEW: multi-timeframe
    "rsi_15m", "rsi_1h", "ema_align_15m", "ema_align_1h", "bb_pct_1h",
    # VIX
    "vix", "vix_trend",
    # NEW: relative strength
    "rs_ratio", "rs_slope",
    # NEW: VWAP + previous-day levels + opening range
    "vwap_dist", "pdh_dist", "pdl_dist", "pdc_dist",
    "or_size", "or_position",
    # NEW: NIFTY expiry features
    "days_to_expiry", "is_expiry_day", "dist_round_50", "dist_round_100",
    # trade meta
    "entry_price", "exit_price", "pnl",
    # how the trade closed (STOP LOSS / TAKE PROFIT / MANUAL EXIT / CONTRA / EOD).
    # Stored for history; the trainer skips MANUAL rows so manual exits never
    # influence the model.
    "exit_reason",
]

def _save_entry_features(confidence: float = 0.0):
    """Snapshot indicator features + signal confidence from Redis at trade entry time."""
    try:
        raw = redis_client.get("quant:Nifty50")
        if raw:
            data = json.loads(raw)
            inds = data.get("indicators", {})
            _pending_feedback.update(inds)
            _pending_feedback["entry_price"] = data.get("price", 0)
            _pending_feedback["confidence"]  = round(confidence, 2)
    except Exception as e:
        print(f"[FEEDBACK] Could not snapshot features: {e}")


def _log_feedback(signal: str, outcome: str, exit_price: float, pnl: float, reason: str = ""):
    """
    Write completed trade + outcome to live_feedback.csv (model learns from this).

    Manual exits are SKIPPED on purpose: a manual exit is NOT the model's natural
    SL/TP outcome (the user cut the trade early), so feeding it into the learning
    loop would teach the model wrong lessons. This keeps manual exits from changing
    future trade behaviour at all.
    """
    try:
        row = {
            "timestamp":   datetime.now().isoformat(),
            "signal":      signal,
            "outcome":     outcome,   # "WIN" or "LOSS"
            "confidence":  _pending_feedback.get("confidence", 0),
            "entry_price": _pending_feedback.get("entry_price", 0),
            "exit_price":  round(exit_price, 2),
            "pnl":         round(pnl, 2),
            "exit_reason": reason or "AUTO",
        }
        # Copy all indicator columns (skip 4 leading meta + 4 trailing trade-meta)
        for col in FEEDBACK_COLS[4:-4]:
            row[col] = _pending_feedback.get(col, 0)

        write_header = not os.path.exists(FEEDBACK_CSV)
        with open(FEEDBACK_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # Count total rows
        with open(FEEDBACK_CSV) as f:
            total = sum(1 for _ in f) - 1   # minus header

        print(f"[LEARN] Trade logged to {FEEDBACK_CSV} "
              f"({total} total) | {outcome}")

        # Auto-retrain XGBoost when enough data accumulates
        if total > 0 and total % AUTO_RETRAIN_EVERY == 0:
            _trigger_retrain(total)

    except Exception as e:
        print(f"[FEEDBACK] Log error: {e}")
    finally:
        _pending_feedback.clear()


def _trigger_retrain(n_trades: int):
    """Kick off XGBoost retrain in the background using live feedback."""
    print(f"\n[RETRAIN] {n_trades} paper trades accumulated — "
          f"retraining XGBoost in background...")
    try:
        subprocess.Popen(
            ["python", "model_trainer.py", "--with-feedback"],
            stdout=open("retrain.log", "w"),
            stderr=subprocess.STDOUT,
        )
        print("[RETRAIN] Running... check retrain.log for progress.")
    except Exception as e:
        print(f"[RETRAIN] Failed to start: {e}")


# ══════════════════════════════════════════════════════════════
#  TRADE EXECUTION
# ══════════════════════════════════════════════════════════════
def execute_trade(signal: str, price: float, sl: float, tp: float, confidence: float):
    global portfolio

    if portfolio["position"] is not None:
        return

    qty      = calculate_qty(price, sl)
    risk_inr = abs(price - sl) * qty
    rr       = abs(tp - price) / abs(price - sl)

    print(f"\n{'='*54}")
    print(f"  [TRADE] {signal}  |  Mode: {MODE}")
    print(f"  Entry  : Rs{price:.2f}  |  Qty: {qty}")
    print(f"  SL     : Rs{sl:.2f}    |  TP: Rs{tp:.2f}")
    print(f"  Risk   : Rs{risk_inr:.2f}  |  RR: {rr:.2f}:1")
    print(f"  Conf   : {confidence:.1f}%")
    print(f"{'='*54}\n")

    # Default option fields (overwritten if LIVE option order succeeds)
    opt_symbol = None
    opt_token  = None
    opt_type   = None
    opt_strike = 0
    opt_lots   = 0
    opt_premium = 0.0
    opt_qty    = qty

    # ── Resolve the REAL ATM option for this signal (PAPER *and* LIVE) ──
    # BUY signal -> ATM CE | SELL signal -> ATM PE. Both are LONG option buys.
    # PAPER: we still pick the real tradable option + its live premium so the
    #        dashboard shows EXACTLY what would be traded — just no real order.
    # LIVE : same, plus a real Angel One order is placed.
    have_session = _ensure_live_session() and smartApi is not None
    if have_session:
        oh = _options_helper()
        info = oh.resolve_option_for_signal(signal, price, smart_api=smartApi)
        if info is None or info.get("premium") in (None, 0):
            if MODE == "LIVE":
                print(f"[ERR] Could not resolve option for {signal} at spot {price:.2f}. "
                      f"Skipping LIVE trade.")
                return
            print(f"[WARN] Option premium unavailable — paper-trading on spot only.")
        else:
            opt_symbol  = info["symbol"]
            opt_token   = info["token"]
            opt_type    = info["option_type"]
            opt_strike  = info["strike"]
            opt_premium = info["premium"]
            opt_lots    = oh.calculate_lots(CAPITAL, opt_premium, risk_pct=RISK_PER_TRADE_PCT,
                                            max_lots=MAX_LOTS_PER_TRADE, lot_size=oh.NIFTY_LOT_SIZE)
            if opt_lots <= 0:
                min_capital_needed = opt_premium * oh.NIFTY_LOT_SIZE
                if MODE == "LIVE":
                    print(f"[SKIP] Cannot afford even 1 lot of {opt_symbol}.")
                    print(f"       Premium Rs{opt_premium:.2f}/share x 75 = "
                          f"Rs{min_capital_needed:,.0f} per lot | Capital Rs{CAPITAL:,.0f}")
                    return
                opt_lots = 1   # PAPER preview: show 1 lot even if capital is low
            opt_qty = opt_lots * oh.NIFTY_LOT_SIZE
            cost    = opt_premium * opt_qty

            print(f"  [OPTION] {opt_symbol}")
            print(f"  Premium: Rs{opt_premium:.2f}  |  Lots: {opt_lots}  |  Qty: {opt_qty}  "
                  f"|  Cost: Rs{cost:,.2f}")

            if MODE == "LIVE":
                try:
                    order_id = smartApi.placeOrder({
                        "variety":         "NORMAL",
                        "tradingsymbol":   opt_symbol,
                        "symboltoken":     opt_token,
                        "transactiontype": "BUY",      # CE for bullish, PE for bearish
                        "exchange":        "NFO",
                        "ordertype":       "MARKET",
                        "producttype":     "INTRADAY",
                        "duration":        "DAY",
                        "quantity":        str(opt_qty),
                    })
                    print(f"[ORDER] Placed! ID: {order_id}  ({opt_symbol})")
                except Exception as e:
                    print(f"[ERR] Option order failed: {e}")
                    return
    elif MODE == "LIVE":
        print("[ERR] LIVE mode needs an Angel One session but auth failed. Skipping trade.")
        return

    portfolio.update({
        "position":    signal,
        "entry_price": price,           # underlying Nifty spot
        "sl":          sl,
        "tp":          tp,
        "qty":         opt_qty,
        "open_time":   datetime.now().isoformat(),
        "opt_symbol":  opt_symbol,
        "opt_token":   opt_token,
        "opt_type":    opt_type,
        "opt_strike":  opt_strike,
        "opt_lots":    opt_lots,
        "opt_entry_premium": opt_premium,
    })
    session["trades"] += 1
    _save_entry_features(confidence)   # snapshot indicators + confidence for feedback loop

    # Tell tick_collector.py which option to stream the live premium for (or clear it)
    if opt_token:
        redis_client.set("opt_subscribe:Nifty50",
                         json.dumps({"token": opt_token, "symbol": opt_symbol}))
    else:
        redis_client.delete("opt_subscribe:Nifty50")

    _sync_redis("OPEN")

    # Emit a BUY/SELL alert as soon as the trade is actually live. This keeps the
    # alert feed aligned with the real execution decision instead of only logging to stdout.
    alert_payload = {
        "signal": signal,
        "entry_price": float(price),
        "stop_loss": float(sl),
        "take_profit": float(tp),
        "confidence": f"{confidence:.0f}%",
        "source": "Execution Manager",
        "rsi": 0,
        "regime": "N/A",
        "probabilities": {"buy": 50.0, "sell": 50.0, "wait": 0.0},
        "timestamp": time.time(),
    }
    for channel, send_alert in (
        ("SLACK", maybe_send_slack_alert),
        ("TELEGRAM", maybe_send_telegram_alert),
    ):
        try:
            send_alert(alert_payload, allow_open_position=True)
        except Exception as error:
            print(f"[{channel}] Trade-entry alert failed: {error}")

    print(f"[OPEN] Trade #{session['trades']} opened. "
          f"Daily P&L so far: Rs{session['realized_pnl']:.2f}")


# ══════════════════════════════════════════════════════════════
#  POSITION MONITOR
# ══════════════════════════════════════════════════════════════
def process_manual_command(current_price: float) -> bool:
    """
    Frontend can queue a manual EXIT for the current open trade via the
    `trade_command:Nifty50` Redis key. Returns True if the trade was closed.

    Always consumes the command if present -- even when there is no open trade --
    so stale commands cannot accumulate and trigger unintended exits later.
    """
    raw = redis_client.get("trade_command:Nifty50")
    if not raw:
        return False

    # Consume immediately -- a command should only ever fire once
    redis_client.delete("trade_command:Nifty50")

    try:
        cmd = json.loads(raw)
    except Exception:
        return False

    action = cmd.get("action")
    if action != "EXIT":
        return False

    if portfolio["position"] is None:
        print("[MANUAL] Exit command ignored -- no open position to close.")
        return False

    pos   = portfolio["position"]
    entry = portfolio["entry_price"]
    qty   = portfolio["qty"]
    pnl   = (current_price - entry) * qty if pos == "BUY" else (entry - current_price) * qty
    print(f"\n[MANUAL] Exit command received -- closing {pos} at Rs{current_price:.2f}")
    _close_trade("MANUAL EXIT", current_price, pnl)
    return True


def check_contra_signal(current_price: float) -> bool:
    """
    Auto-exit if the model reverses direction with high confidence.
    Example: BUY trade open, model now says SELL at 52%+ → market turning, exit now.
    Returns True if trade was closed.
    """
    if portfolio["position"] is None:
        return False

    raw = redis_client.get("signal:Nifty50")
    if not raw:
        return False

    try:
        sig      = json.loads(raw)
        signal   = sig.get("signal", "WAIT")
        conf_val = float(str(sig.get("confidence", "0%")).replace("%", ""))
        sig_age  = time.time() - sig.get("timestamp", 0)
    except Exception:
        return False

    # Ignore stale signals (older than 2 minutes)
    if sig_age > 120:
        return False

    pos = portfolio["position"]
    is_contra = (pos == "BUY" and signal == "SELL") or \
                (pos == "SELL" and signal == "BUY")

    if not is_contra or conf_val < CONTRA_EXIT_THRESHOLD:
        return False

    entry = portfolio["entry_price"]
    qty   = portfolio["qty"]
    pnl   = (current_price - entry) * qty if pos == "BUY" \
            else (entry - current_price) * qty

    print(f"\n[CONTRA-EXIT] Model reversed: {pos} → {signal} @ {conf_val:.1f}%")
    print(f"              Market turning — exiting {pos} @ Rs{current_price:.2f} | P&L: Rs{pnl:+.2f}")
    _close_trade(f"CONTRA-SIGNAL ({signal} {conf_val:.1f}%)", current_price, pnl)
    return True


def monitor_position(current_price: float):
    global portfolio

    if portfolio["position"] is None:
        return

    # Apply any frontend command first (may exit the trade)
    if process_manual_command(current_price):
        return

    pos   = portfolio["position"]
    entry = portfolio["entry_price"]
    sl    = portfolio["sl"]
    tp    = portfolio["tp"]
    qty   = portfolio["qty"]

    if pos == "BUY":
        sl_hit = current_price <= sl
        tp_hit = current_price >= tp
    else:
        sl_hit = current_price >= sl
        tp_hit = current_price <= tp

    # P&L display: prefer the REAL option premium move (tick_collector). Both CE
    # and PE are long buys, so profit = premium rising. Fall back to a spot-based
    # estimate when no live premium tick is available. (SL/TP still trigger on spot.)
    entry_prem = portfolio.get("opt_entry_premium", 0) or 0
    live_prem  = _live_option_premium() if portfolio.get("opt_symbol") else None
    if live_prem is not None and entry_prem > 0:
        upnl = (live_prem - entry_prem) * qty
    else:
        upnl = (current_price - entry) * qty if pos == "BUY" else (entry - current_price) * qty

    _sync_redis("OPEN", upnl=upnl)

    if sl_hit:
        pnl = (sl - entry) * qty if pos == "BUY" else (entry - sl) * qty
        _close_trade("STOP LOSS", sl, pnl)
    elif tp_hit:
        pnl = (tp - entry) * qty if pos == "BUY" else (entry - tp) * qty
        _close_trade("TAKE PROFIT", tp, pnl)


def _close_trade(reason: str, exit_price: float, pnl: float):
    global portfolio

    # ── If a LIVE option position is open, place a SELL order to close it ──
    if MODE == "LIVE" and smartApi and portfolio.get("opt_symbol"):
        oh = _options_helper()
        opt_symbol = portfolio["opt_symbol"]
        opt_token  = portfolio["opt_token"]
        opt_qty    = portfolio["qty"]
        entry_prem = portfolio["opt_entry_premium"]
        exit_prem  = oh.fetch_option_ltp(smartApi, opt_token, opt_symbol)
        try:
            order_id = smartApi.placeOrder({
                "variety":         "NORMAL",
                "tradingsymbol":   opt_symbol,
                "symboltoken":     opt_token,
                "transactiontype": "SELL",
                "exchange":        "NFO",
                "ordertype":       "MARKET",
                "producttype":     "INTRADAY",
                "duration":        "DAY",
                "quantity":        str(opt_qty),
            })
            print(f"[ORDER] Close SELL placed! ID: {order_id}  ({opt_symbol})")
        except Exception as e:
            print(f"[ERR] Option close order failed: {e} -- position may still be open!")
        # Recompute real P&L from actual option premium move (overrides spot-based pnl)
        if exit_prem is not None and entry_prem > 0:
            pnl = (exit_prem - entry_prem) * opt_qty
            print(f"  [OPTION P&L] Entry prem Rs{entry_prem:.2f} -> Exit prem Rs{exit_prem:.2f}")
            print(f"               Realized: Rs{pnl:+,.2f}  (qty {opt_qty})")

    # PAPER preview: recompute realized P&L from the last live option premium tick
    elif portfolio.get("opt_symbol") and portfolio.get("opt_entry_premium", 0) > 0:
        last_prem = _live_option_premium()
        if last_prem is not None:
            pnl = (last_prem - portfolio["opt_entry_premium"]) * portfolio["qty"]
            print(f"  [PAPER OPTION P&L] Entry prem Rs{portfolio['opt_entry_premium']:.2f} "
                  f"-> Last prem Rs{last_prem:.2f} | P&L Rs{pnl:+,.2f}")

    global _last_trade_close_ts
    _last_trade_close_ts = time.time()   # start cooldown timer
    # Publish cooldown end time so the AI engine / Slack can suppress alerts that
    # cannot be acted on (no trade is taken during cooldown -> no point alerting).
    redis_client.set("cooldown_until:Nifty50", str(time.time() + TRADE_COOLDOWN_SEC),
                     ex=TRADE_COOLDOWN_SEC + 5)
    session["realized_pnl"] += pnl
    outcome = "WIN" if pnl > 0 else "LOSS"
    tag     = "[WIN] " if pnl > 0 else "[LOSS]"

    # ── Consecutive-loss circuit breaker (capital guard) ──
    # Reset on any win. Count only strategy losses (manual exits are the user's
    # own call, so they don't trip the breaker). Halt the day after MAX in a row.
    if outcome == "WIN":
        session["consecutive_losses"] = 0
    elif not reason.upper().startswith("MANUAL"):
        session["consecutive_losses"] = session.get("consecutive_losses", 0) + 1
        if session["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
            session["halted"] = True
            print(f"[HALT] {session['consecutive_losses']} losses in a row — trading PAUSED "
                  f"for today (capital guard). Restart the manager to resume.")

    print(f"\n{'='*54}")
    print(f"  [CLOSED] {reason}")
    print(f"  Exit   : Rs{exit_price:.2f}  (spot)")
    print(f"  P&L    : {tag} Rs{pnl:+,.2f}")
    print(f"  Daily  : Rs{session['realized_pnl']:+,.2f}")
    print(f"{'='*54}\n")

    closed_signal = portfolio["position"]
    closed_entry  = portfolio["entry_price"]   # save before reset
    portfolio = {
        "position": None, "entry_price": 0.0,
        "sl": 0.0, "tp": 0.0, "qty": 0,
        "instrument": None, "open_time": None,
        "opt_symbol": None, "opt_token": None, "opt_type": None,
        "opt_strike": 0, "opt_lots": 0, "opt_entry_premium": 0.0,
    }
    redis_client.delete("opt_subscribe:Nifty50")   # stop tick_collector streaming premium
    _sync_redis("CLOSED")
    _log_feedback(closed_signal, outcome, exit_price, pnl, reason)  # learn (skips manual exits)
    send_trade_closed_alert(closed_signal, outcome, closed_entry, exit_price, pnl)


def _sync_redis(status: str, upnl: float = 0.0):
    # In LIVE mode also publish the real Angel One balance for the dashboard
    angel_balance = fetch_account_balance() if MODE == "LIVE" else None

    # ── Option premium view: live LTP + estimated TP/SL in premium terms ──
    # The strategy's SL/TP are SPOT levels. An ATM option's delta is ~0.5, so its
    # premium moves roughly half of the spot move. Both CE and PE are long buys:
    # a favourable spot move (toward TP) RAISES the premium, an adverse one (toward
    # SL) lowers it. These are ESTIMATES for display — the real exit triggers on spot.
    OPT_DELTA  = 0.5
    opt_entry  = portfolio.get("opt_entry_premium", 0) or 0
    opt_ltp    = _live_option_premium() if portfolio.get("opt_symbol") else None
    opt_tp_prem = opt_sl_prem = None
    if opt_entry > 0:
        entry_spot = portfolio.get("entry_price", 0) or 0
        tp_spot    = portfolio.get("tp", 0) or 0
        sl_spot    = portfolio.get("sl", 0) or 0
        opt_tp_prem = round(opt_entry + OPT_DELTA * abs(tp_spot - entry_spot), 2)
        opt_sl_prem = round(max(0.0, opt_entry - OPT_DELTA * abs(entry_spot - sl_spot)), 2)
    redis_client.set("active_trade:Nifty50", json.dumps({
        "status":         status,
        "position":       portfolio["position"],
        "entry":          portfolio["entry_price"],
        "sl":             portfolio["sl"],
        "tp":             portfolio["tp"],
        "qty":            portfolio["qty"],
        "unrealized_pnl": round(upnl, 2),
        "realized_pnl":   round(session["realized_pnl"], 2),
        "trades_today":   session["trades"],
        "mode":           MODE,
        "capital":        round(CAPITAL, 2),
        "angel_balance":  round(angel_balance, 2) if angel_balance is not None else None,
        "daily_profit_target": DAILY_PROFIT_TARGET,
        "max_lots_per_trade":  MAX_LOTS_PER_TRADE,
        "halted":         session.get("halted", False),
        "auto_state":     get_auto_state(),
        # Options metadata (None in PAPER mode)
        "opt_symbol":     portfolio.get("opt_symbol"),
        "opt_type":       portfolio.get("opt_type"),
        "opt_strike":     portfolio.get("opt_strike", 0),
        "opt_lots":       portfolio.get("opt_lots", 0),
        "opt_entry_premium": round(opt_entry, 2),
        "opt_ltp":           round(opt_ltp, 2) if opt_ltp is not None else None,
        "opt_tp_premium":    opt_tp_prem,
        "opt_sl_premium":    opt_sl_prem,
        "timestamp":      time.time(),
    }))


# ══════════════════════════════════════════════════════════════
#  MARKET HOURS CHECK
# ══════════════════════════════════════════════════════════════
def is_market_open() -> bool:
    from datetime import datetime, timedelta, timezone
    # IST is UTC+05:30 year-round, so no external timezone package is needed.
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════
def main():
    if not authenticate():
        print("[ERR] Cannot start without Angel One authentication.")
        return

    send_telegram_startup_message()

    print(f"\n[START] Execution Manager v3.0")
    print(f"   Mode        : {MODE}")
    print(f"   Capital     : Rs{CAPITAL:,.0f}")
    print(f"   Risk/Trade  : {RISK_PER_TRADE_PCT}%  (max Rs{CAPITAL * RISK_PER_TRADE_PCT/100:,.0f})")
    print(f"   Daily limit : {MAX_DAILY_LOSS_PCT}%  (halt at -Rs{CAPITAL * MAX_DAILY_LOSS_PCT/100:,.0f})")
    print(f"   Min conf    : {CONF_THRESHOLD}%")
    print(f"   Price source: {'yfinance (free)' if MODE == 'PAPER' else 'Angel One'}\n")

    while True:
        try:
            check_session_reset()
            sync_mode_from_redis()   # pick up any toggle from the dashboard

            # Loss limit is the only HARD halt
            if check_daily_loss_limit():
                time.sleep(60)
                continue

            # Profit target is a SOFT pause -- transitions auto_state but loop keeps going
            check_profit_pause()

            if not is_market_open():
                # Force-close any open position at end of day
                if portfolio["position"] is not None:
                    eod_price = fetch_live_price() or portfolio["entry_price"]
                    pos   = portfolio["position"]
                    entry = portfolio["entry_price"]
                    qty   = portfolio["qty"]
                    pnl   = (eod_price - entry) * qty if pos == "BUY" else (entry - eod_price) * qty
                    print(f"\n[EOD] Market closed with open {pos}! "
                          f"Force-closing at Rs{eod_price:.2f} | P&L: Rs{pnl:.2f}")
                    _close_trade("MARKET CLOSE", eod_price, pnl)
                else:
                    # Drain any stale manual commands so they don't pile up in Redis
                    # while we sit idle off-market.
                    if redis_client.get("trade_command:Nifty50"):
                        process_manual_command(0.0)
                    print(f"[WAIT] Market closed. Checking again in 60s...")
                time.sleep(60)
                continue

            # 1. Live price
            current_price = fetch_live_price()
            if current_price is None:
                print("[ERR] Price unavailable. Retrying in 10s...")
                time.sleep(10)
                continue

            # 2a. Drain any manual command from the frontend (always, even when flat
            #     -- otherwise stale commands sit in Redis after the trade auto-closed)
            process_manual_command(current_price)

            # 2b. Contra-signal exit — model reversed direction with high confidence
            if portfolio["position"] is not None:
                if check_contra_signal(current_price):
                    time.sleep(MONITOR_INTERVAL_SEC)
                    continue

            # 2c. Monitor open position (SL/TP check)
            if portfolio["position"] is not None:
                monitor_position(current_price)

            # 3. Check for new signal (only if flat)
            if portfolio["position"] is None:
                raw = redis_client.get("signal:Nifty50")
                if raw:
                    sig_data   = json.loads(raw)
                    signal     = sig_data.get("signal", "WAIT")
                    conf_str   = sig_data.get("confidence", "0%")
                    conf_val   = float(conf_str.replace("%", ""))
                    sl         = sig_data.get("stop_loss")
                    tp         = sig_data.get("take_profit")
                    sig_age    = time.time() - sig_data.get("timestamp", 0)

                    # Fix 1: Skip signals from previous session (stale Redis data)
                    sig_date = datetime.fromtimestamp(
                        sig_data.get("timestamp", 0)
                    ).date()
                    if sig_date < date.today():
                        print(f"[SKIP] Signal is from {sig_date} (previous session) "
                              f"— waiting for today's fresh data")
                    # Ignore stale signals (older than 2 minutes)
                    elif sig_age > 120:
                        print(f"[SKIP] Signal is {sig_age:.0f}s old — waiting for fresh signal")
                    elif signal in ("BUY", "SELL") and conf_val >= CONF_THRESHOLD:
                        if sl is None or tp is None:
                            print(f"[SKIP] {signal} at {conf_val:.1f}% but SL/TP missing.")
                        else:
                            # Fix 2: Cooldown — wait 15 min after any trade closes
                            cooldown_left = TRADE_COOLDOWN_SEC - (time.time() - _last_trade_close_ts)
                            if _last_trade_close_ts > 0 and cooldown_left > 0:
                                print(f"[COOLDOWN] {cooldown_left/60:.1f} min remaining "
                                      f"before next trade entry")
                            else:
                                # Use the signal's own entry price (from Angel One quant engine)
                                # so entry, SL and TP are all computed from the same data source.
                                # In LIVE mode the broker fills at real market price anyway.
                                signal_entry = float(sig_data.get("entry_price", current_price))
                                trade_entry  = signal_entry if MODE == "PAPER" else current_price

                                auto_state = get_auto_state()
                                if auto_state == "RUNNING":
                                    execute_trade(signal, trade_entry,
                                                  float(sl), float(tp), conf_val)
                                elif has_manual_approval():
                                    print(f"[MANUAL-APPROVED] User approved this {signal} trade.")
                                    execute_trade(signal, trade_entry,
                                                  float(sl), float(tp), conf_val)
                                    consume_manual_approval()
                                else:
                                    print(f"[PAUSED] {signal} {conf_val:.1f}% -- auto-trade off, "
                                          f"waiting for user approval")
                    else:
                        print(f"[WAIT] {signal} {conf_val:.1f}% | "
                              f"Price: Rs{current_price:.2f} | "
                              f"Need >={CONF_THRESHOLD}% to trade")

            time.sleep(MONITOR_INTERVAL_SEC)

        except KeyboardInterrupt:
            print("\n[STOP] Manual stop.")
            break
        except Exception as e:
            print(f"[ERR] Main loop: {e}")
            import traceback; traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    main()
