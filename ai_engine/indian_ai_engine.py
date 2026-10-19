"""Publish LuxAlgo Trendlines with Breaks signals from live closed candles."""

import json
import os
import time

import redis
from dotenv import load_dotenv

from telegram_alerts import maybe_send_alert as maybe_send_telegram_alert, send_startup_message as send_telegram_startup_message
from trendline_breaks import latest_breakout

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
LOOP_SEC = 5
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


def generate_signal(redis_key="Nifty50"):
    try:
        raw_quant = redis_client.get(f"quant:{redis_key}")
        quant_data = json.loads(raw_quant) if raw_quant else {}
        age_sec = (time.time() * 1000 - quant_data.get("timestamp", 0)) / 1000

        if age_sec > 90:
            signal = "WAIT"
            reason = "Live candle data is stale"
        elif quant_data.get("data_source") != "Angel One":
            signal = "WAIT"
            reason = "Waiting for Angel One live candles"
        else:
            raw_candles = redis_client.get(f"candles:{redis_key}")
            candles = json.loads(raw_candles) if raw_candles else []
            breakout = latest_breakout(candles)
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

        # Telegram publishes the first actionable model signal.  Its Redis-backed
        # dedupe guard ignores the following loop iterations (even when price
        # changes).  Slack remains confirmation-only in the execution manager.
        if signal in ("BUY", "SELL"):
            try:
                maybe_send_telegram_alert(payload)
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
