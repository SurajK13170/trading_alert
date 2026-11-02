"""
NIFTY 50 — LIVE TICK COLLECTOR  (Angel One SmartWebSocketV2)
============================================================
Runs as its own process. Streams live ticks over WebSocket and publishes them
to Redis so the rest of the system gets sub-second prices (instead of the 30s
quant-engine poll) AND a live OPTION premium for accurate profit display.

What it publishes to Redis:
  tick:Nifty50        -> {"price", "ts"}            (NIFTY 50 index spot, every tick)
  opt_tick:Nifty50    -> {"symbol","ltp","ts"}      (premium of the option currently held)
  quant:Nifty50.price -> overwritten with live spot (read-modify-write, keeps freshness)

How the option subscription works (dynamic):
  The execution manager writes  opt_subscribe:Nifty50 = {"token","symbol"}  when a
  trade opens, and DELETES that key when the trade closes. This collector watches
  that key and subscribes / unsubscribes the option token on the fly. So you always
  get the live premium of whatever option is currently open — and nothing when flat.

Run:  python tick_collector.py     (keep it running alongside the other engines)

Requires:  pip install logzero websocket-client   (already handled)
"""

import os
import json
import time
import threading

import redis
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

load_dotenv()

# ── Config ───────────────────────────────────────────────────────
REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
CLIENT_ID  = os.getenv("ANGEL_CLIENT_ID")
API_KEY    = os.getenv("ANGEL_API_KEY")
PASSWORD   = os.getenv("ANGEL_PASSWORD")
TOTP_KEY   = os.getenv("ANGEL_TOTP_KEY")

NIFTY_TOKEN = "99926000"     # NIFTY 50 index
EXCH_NSE_CM = 1              # NSE cash / index segment
EXCH_NSE_FO = 2              # NSE futures & options segment (where options live)
MODE_LTP    = 1              # lightest tick mode (last traded price only)
CORRELATION_ID = "nifty_tick_stream"

OPT_SUBSCRIBE_KEY = "opt_subscribe:Nifty50"   # execution manager writes/clears this
TICK_TTL_SEC      = 30                          # ticks expire so stale data is obvious
CANDLE_INTERVAL_SEC = 5 * 60
CANDLE_TTL_SEC      = 60
# Finished tick-built candles, published the moment a 5-min candle closes so the
# signal engine can check for a B without waiting 1-3 min for Angel's history API.
TICK_CANDLES_KEY    = "tick_candles:Nifty50"
TICK_CANDLES_KEEP   = 3
PARTIAL_GRACE_SEC   = 15   # first tick later than this after the candle start -> partial candle

# redis-py 4.6 defaults to RESP2, which is compatible with local Redis 5.
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Shared state across threads
_state = {
    "sws":         None,   # current SmartWebSocketV2 instance
    "opt_token":   None,   # option token we are currently subscribed to (NFO)
    "opt_symbol":  None,
    "connected":   False,
    "live_candle": None,
    "last_finished_ts": 0,   # timestamp (ms) of the last candle published as finished
}
_lock = threading.Lock()


def _publish_finished_candle(candle):
    """Append a finished 5-min candle to tick_candles:Nifty50 (keeps the last few)."""
    try:
        raw = redis_client.get(TICK_CANDLES_KEY)
        recent = [c for c in (json.loads(raw) if raw else []) if c["timestamp"] != candle["timestamp"]]
        recent.append(candle)
        redis_client.set(TICK_CANDLES_KEY, json.dumps(recent[-TICK_CANDLES_KEEP:]),
                         ex=CANDLE_INTERVAL_SEC * 3)
    except Exception as e:
        print(f"[TICK] Could not publish finished candle: {e}")


def _finish_candle_if_closed(now):
    """
    Publish the current candle once its 5-min window has passed. Called on every
    tick and every second from the watcher thread, so a candle is finished on
    time even when no new tick arrives (e.g. the 15:25 candle after the close).
    """
    with _lock:
        candle = _state["live_candle"]
        if (not candle
                or candle["timestamp"] // 1000 + CANDLE_INTERVAL_SEC > now
                or candle["timestamp"] <= _state["last_finished_ts"]):
            return
        _state["last_finished_ts"] = candle["timestamp"]
        finished = dict(candle)
    _publish_finished_candle(finished)


# ── Angel login (REST) → jwt + feed token for the socket ─────────
def login():
    if not all([CLIENT_ID, API_KEY, PASSWORD, TOTP_KEY]):
        raise RuntimeError("Angel One creds missing in .env "
                           "(ANGEL_CLIENT_ID/API_KEY/PASSWORD/TOTP_KEY)")
    api  = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_KEY).now()
    data = api.generateSession(CLIENT_ID, PASSWORD, totp)
    if not (data and data.get("status")):
        raise RuntimeError(f"Angel login failed: {data}")
    jwt     = data["data"]["jwtToken"]
    refresh = data["data"].get("refreshToken", "")
    feed    = api.getfeedToken()
    # Publish the session so other processes (quant engine) can REUSE it instead
    # of a second Angel login. Two logins with the same 30s TOTP get rejected by
    # Angel — which is what forced the quant engine onto delayed yfinance data.
    try:
        redis_client.set("angel_session:Nifty50", json.dumps({
            "jwt": jwt, "feed_token": feed, "refresh": refresh,
            "client_id": CLIENT_ID, "api_key": API_KEY, "ts": time.time(),
        }))
        print("[TICK] Angel login OK — session published to Redis for sharing.")
    except Exception as e:
        print(f"[TICK] Angel login OK (could not publish session: {e})")
    return jwt, feed


# ── Option subscription watcher (dynamic) ────────────────────────
def _option_watcher():
    """
    Poll opt_subscribe:Nifty50 every second. Subscribe / switch / unsubscribe the
    option token on the live socket so we always stream the held option's premium.
    Runs as a daemon thread for the life of one socket connection.
    """
    while _state["connected"]:
        _finish_candle_if_closed(time.time())
        try:
            raw = redis_client.get(OPT_SUBSCRIBE_KEY)
            want_token = want_symbol = None
            if raw:
                info = json.loads(raw)
                want_token  = str(info.get("token") or "") or None
                want_symbol = info.get("symbol")

            cur = _state["opt_token"]
            sws = _state["sws"]
            if sws is None:
                time.sleep(1); continue

            if want_token != cur:
                # Unsubscribe the old option (if any)
                if cur:
                    try:
                        sws.unsubscribe(CORRELATION_ID, MODE_LTP,
                                        [{"exchangeType": EXCH_NSE_FO, "tokens": [cur]}])
                        print(f"[TICK] Unsubscribed old option {_state['opt_symbol']} ({cur})")
                    except Exception as e:
                        print(f"[TICK] Unsubscribe error: {e}")
                # Subscribe the new option (if any)
                if want_token:
                    try:
                        sws.subscribe(CORRELATION_ID, MODE_LTP,
                                      [{"exchangeType": EXCH_NSE_FO, "tokens": [want_token]}])
                        print(f"[TICK] Subscribed option {want_symbol} ({want_token})")
                    except Exception as e:
                        print(f"[TICK] Subscribe error: {e}")
                with _lock:
                    _state["opt_token"]  = want_token
                    _state["opt_symbol"] = want_symbol
                # When the option clears, drop the stale premium tick
                if not want_token:
                    redis_client.delete("opt_tick:Nifty50")
        except Exception as e:
            print(f"[TICK] Watcher loop error: {e}")
        time.sleep(1)


# ── WebSocket callbacks ──────────────────────────────────────────
def on_open(wsapp):
    print("[TICK] WebSocket open — subscribing NIFTY 50 index.")
    with _lock:
        _state["connected"] = True
    # subscribe/unsubscribe live on the SmartWebSocketV2 instance, NOT the raw
    # WebSocketApp that the callback hands us — so use the stored sws object.
    sws = _state["sws"]
    try:
        sws.subscribe(CORRELATION_ID, MODE_LTP,
                      [{"exchangeType": EXCH_NSE_CM, "tokens": [NIFTY_TOKEN]}])
    except Exception as e:
        print(f"[TICK] Index subscribe error: {e}")
    # (Re)start the dynamic option watcher for this connection
    threading.Thread(target=_option_watcher, daemon=True).start()


def on_data(wsapp, message):
    try:
        if not isinstance(message, dict):
            return
        token = str(message.get("token", "")).strip('"')
        ltp_paise = message.get("last_traded_price")
        if ltp_paise is None:
            return
        ltp = float(ltp_paise) / 100.0     # Angel sends price in paise
        now = time.time()

        if token == NIFTY_TOKEN:
            # Index spot tick
            redis_client.set("tick:Nifty50",
                             json.dumps({"price": ltp, "ts": now}), ex=TICK_TTL_SEC)
            candle_start = int(now // CANDLE_INTERVAL_SEC) * CANDLE_INTERVAL_SEC
            _finish_candle_if_closed(now)
            with _lock:
                candle = _state["live_candle"]
                if not candle or candle["timestamp"] // 1000 != candle_start:
                    candle = {
                        "timestamp": candle_start * 1000,
                        "open": ltp,
                        "high": ltp,
                        "low": ltp,
                        "close": ltp,
                        # Started mid-candle (late start / reconnect): the open
                        # and range are incomplete, so the signal engine skips it.
                        "partial": now - candle_start > PARTIAL_GRACE_SEC,
                    }
                    _state["live_candle"] = candle
                else:
                    candle["high"] = max(candle["high"], ltp)
                    candle["low"] = min(candle["low"], ltp)
                    candle["close"] = ltp
                redis_client.set("live_candle:Nifty50", json.dumps(candle), ex=CANDLE_TTL_SEC)
            # Throttled heartbeat so you can SEE ticks flowing (max once / 5s)
            if now - _state.get("last_print", 0) >= 5:
                _state["last_print"] = now
                opt = f" | opt({_state['opt_symbol']})=Rs{_state.get('last_opt', '—')}" \
                      if _state.get("opt_token") else ""
                print(f"[TICK] NIFTY = {ltp:,.2f}{opt}")
            # Keep quant:Nifty50 price fresh (read-modify-write, don't clobber indicators)
            try:
                raw = redis_client.get("quant:Nifty50")
                if raw:
                    q = json.loads(raw)
                    q["price"] = ltp
                    q["timestamp"] = int(now * 1000)
                    redis_client.set("quant:Nifty50", json.dumps(q))
            except Exception:
                pass

        elif token and token == _state["opt_token"]:
            # Premium tick for the option we currently hold
            _state["last_opt"] = round(ltp, 2)
            redis_client.set("opt_tick:Nifty50",
                             json.dumps({"symbol": _state["opt_symbol"],
                                         "ltp": ltp, "ts": now}), ex=TICK_TTL_SEC)
    except Exception as e:
        print(f"[TICK] on_data error: {e}")


def on_error(wsapp, error):
    print(f"[TICK] WebSocket error: {error}")


def on_close(wsapp):
    print("[TICK] WebSocket closed.")
    with _lock:
        _state["connected"] = False
        _state["opt_token"] = None


# ── Main loop with exponential-backoff reconnect ─────────────────
def run():
    print("[START] NIFTY Tick Collector (Angel WebSocket V2)")
    backoff = 2
    while True:
        try:
            jwt, feed = login()
            sws = SmartWebSocketV2(jwt, API_KEY, CLIENT_ID, feed)
            sws.on_open  = on_open
            sws.on_data  = on_data
            sws.on_error = on_error
            sws.on_close = on_close
            with _lock:
                _state["sws"] = sws
            backoff = 2                       # reset after a clean connect attempt
            sws.connect()                     # blocking until socket drops
        except KeyboardInterrupt:
            print("\n[TICK] Stopped by user.")
            return
        except Exception as e:
            print(f"[TICK] Connection failed: {e}")
        with _lock:
            _state["connected"] = False
            _state["opt_token"] = None
        print(f"[TICK] Reconnecting in {backoff}s...")
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)        # exponential backoff, cap 60s


if __name__ == "__main__":
    run()
