"""Publish LuxAlgo Trendlines with Breaks signals from live closed candles."""

import json
import os
import time

import redis
from dotenv import load_dotenv

from telegram_alerts import maybe_send_alert as maybe_send_telegram_alert, send_startup_message as send_telegram_startup_message
from trendline_breaks import indicator_data

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
LOOP_SEC = 1
CANDLE_MS = 5 * 60 * 1000
IST_OFFSET_MS = 5.5 * 60 * 60 * 1000
MARKET_OPEN_MS = (9 * 60 + 15) * 60 * 1000   # 09:15 IST, as ms since IST midnight
# Alert every B whose candle closed within this window, not only a B on the
# latest candle -- so a B is not lost when two candles arrive at once, and a B
# that only shows up once Angel's candle replaces the tick candle is still sent.
# Older B markers (e.g. before a late start) are not alerted.
# Must stay below TELEGRAM_ALERT_DEDUP_TTL_SEC (900s) so a B is never re-sent.
ALERT_MAX_AGE_SEC = 10 * 60
# A tick-built candle's close can differ from Angel's official close by a point
# or two, so a B on a tick candle is trusted only when the close clears the
# trendline by this fraction of ATR. Borderline breaks wait for Angel's candle.
TICK_BREAKOUT_BUFFER_ATR = 0.10
ATR_LENGTH = 14
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
_borderline_logged = set()   # candle timestamps already reported as borderline


def _atr(candles, index, length=ATR_LENGTH):
    """Simple average true range of the `length` candles ending at `index`."""
    ranges = []
    for j in range(max(1, index - length + 1), index + 1):
        high, low, prev_close = candles[j]["high"], candles[j]["low"], candles[j - 1]["close"]
        ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return sum(ranges) / len(ranges) if ranges else 0.0


def _clear_breakouts(candles, angel_count):
    """
    B markers as (index, direction). Markers on Angel candles are kept as-is;
    markers on tick-built candles (index >= angel_count) only when the close is
    beyond the trendline by TICK_BREAKOUT_BUFFER_ATR x ATR.
    """
    data = indicator_data(candles)
    lines = {"BUY": dict(data["upper"]), "SELL": dict(data["lower"])}
    events = []
    for index, direction in data["events"]:
        if index >= angel_count:
            line = lines[direction].get(index)
            if line is None:
                continue
            close = candles[index]["close"]
            margin = close - line if direction == "BUY" else line - close
            buffer = TICK_BREAKOUT_BUFFER_ATR * _atr(candles, index)
            if margin < buffer:
                stamp = candles[index]["timestamp"]
                if stamp not in _borderline_logged:   # log once, not every loop
                    _borderline_logged.add(stamp)
                    print(f"[{direction}] Borderline break on tick candle (margin {margin:.2f} < "
                          f"{buffer:.2f}) -- waiting for Angel's candle")
                continue
        events.append((index, direction))
    return events


def _with_tick_candles(candles, redis_key):
    """
    Append the finished tick-built candles (tick_collector) that Angel's history
    API has not delivered yet, so a B is detected seconds after the candle close
    instead of 1-3 min later. Only contiguous, complete candles are appended;
    Angel's own candle replaces the tick one as soon as it arrives.
    """
    raw = redis_client.get(f"tick_candles:{redis_key}")
    if not candles or not raw:
        return candles
    extra = []
    for candle in sorted(json.loads(raw), key=lambda c: c["timestamp"]):
        previous = extra[-1]["timestamp"] if extra else candles[-1]["timestamp"]
        if candle["timestamp"] <= previous:
            continue
        ist_ms = candle["timestamp"] + IST_OFFSET_MS
        new_day_open = (ist_ms // 86_400_000 != (previous + IST_OFFSET_MS) // 86_400_000
                        and ist_ms % 86_400_000 == MARKET_OPEN_MS)
        if candle.get("partial") or not (candle["timestamp"] == previous + CANDLE_MS or new_day_open):
            break
        extra.append({key: candle[key] for key in ("timestamp", "open", "high", "low", "close")})
    return candles + extra


def generate_signal(redis_key="Nifty50"):
    try:
        raw_quant = redis_client.get(f"quant:{redis_key}")
        quant_data = json.loads(raw_quant) if raw_quant else {}
        age_sec = (time.time() * 1000 - quant_data.get("timestamp", 0)) / 1000
        alert_events = []   # (candle_timestamp, direction) of recent B markers

        if age_sec > 90:
            signal = "WAIT"
            reason = "Live candle data is stale"
        elif quant_data.get("data_source") != "Angel One":
            signal = "WAIT"
            reason = "Waiting for Angel One live candles"
        else:
            raw_candles = redis_client.get(f"candles:{redis_key}")
            angel_candles = json.loads(raw_candles) if raw_candles else []
            candles = _with_tick_candles(angel_candles, redis_key)
            events = _clear_breakouts(candles, len(angel_candles))
            breakout = events[-1][1] if events and events[-1][0] == len(candles) - 1 else None
            now_ms = time.time() * 1000
            alert_events = [
                (candles[index]["timestamp"], direction)
                for index, direction in events
                if now_ms - (candles[index]["timestamp"] + CANDLE_MS) <= ALERT_MAX_AGE_SEC * 1000
            ]
            signal = breakout or "WAIT"
            reason = "LuxAlgo trendline breakout" if breakout else "No breakout on latest closed candle"

        payload = {
            "symbol": redis_key,
            "signal": signal,
            "confidence": "0%",
            "entry_price": quant_data.get("price", 0),
            "stop_loss": None,
            "take_profit": None,
            "source": "LuxAlgo - Trendlines with Breaks",
            "reason": reason,
            "timestamp": time.time(),
        }
        redis_client.set(f"signal:{redis_key}", json.dumps(payload), ex=120)

        # One Telegram alert per B candle: the dedupe key includes the candle
        # timestamp, so repeated loop iterations (and Angel's candle replacing
        # the tick candle) never re-send, but every new B -- even two BUYs in a
        # row -- is alerted. Slack remains confirmation-only in the execution manager.
        for b_timestamp, direction in alert_events:
            try:
                maybe_send_telegram_alert({**payload, "signal": direction, "candle_timestamp": b_timestamp})
            except Exception as error:
                print(f"[TELEGRAM] Alert failed: {error}")

        print(f"[{signal}] {reason}")
    except Exception as error:
        print(f"[ERR] Trendline signal engine: {error}")


if __name__ == "__main__":
    send_telegram_startup_message()
    print("[START] LuxAlgo Trendlines with Breaks signal engine")
    while True:
        generate_signal("Nifty50")
        time.sleep(LOOP_SEC)
