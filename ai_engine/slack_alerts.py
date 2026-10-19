"""
Slack Alert System for NIFTY 50 AI Trader
Sends BUY/SELL signals + trade-closed updates to a Slack channel via an
Incoming Webhook. WAIT signals are silently ignored.

Setup (one-time):
  1. https://api.slack.com/apps -> Create New App -> From scratch
  2. Incoming Webhooks -> On -> Add New Webhook to Workspace -> pick a channel
  3. Copy the webhook URL into .env:
        SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../XXXX

Drop-in replacement for telegram_alerts.py — same 3 public functions:
  maybe_send_alert(signal_data), send_startup_message(), send_trade_closed_alert(...)
"""

import os
import json
import time
import urllib.request
import urllib.error
import redis
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# Who to ping on each alert. Slack needs special tokens (plain "@channel" does NOT notify):
#   channel -> <!channel> (everyone in the channel)   here -> <!here> (only active members)
#   off/empty -> no ping        or paste a user id like U0123ABCD to ping one person
SLACK_MENTION = os.getenv("SLACK_MENTION", "channel").strip()
_redis      = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

_last_alert_key = ""   # fallback when Redis is temporarily unavailable
POSITION_STATE_MAX_AGE_SEC = 120
ALERT_DEDUP_TTL_SEC = max(1, int(os.getenv("SLACK_ALERT_DEDUP_TTL_SEC", "900")))


def _mention_prefix() -> str:
    """Return the Slack mention token (with trailing space) for alerts, or ''."""
    m = SLACK_MENTION.lower()
    if m in ("channel", "@channel", "<!channel>"):
        return "<!channel> "
    if m in ("here", "@here", "<!here>"):
        return "<!here> "
    if m in ("", "off", "none"):
        return ""
    # Treat anything else as a raw Slack user/group ID, e.g. U0123ABCD
    return f"<@{SLACK_MENTION}> " if not SLACK_MENTION.startswith("<") else f"{SLACK_MENTION} "


def _has_open_position() -> bool:
    """Returns True if execution manager already has an open trade."""
    try:
        raw = _redis.get("active_trade:Nifty50")
        if not raw:
            return False
        data = json.loads(raw)
        timestamp = data.get("timestamp")
        if timestamp is not None and time.time() - float(timestamp) > POSITION_STATE_MAX_AGE_SEC:
            return False
        return data.get("position") is not None
    except Exception:
        return False


def _in_cooldown() -> bool:
    """True during the post-trade cooldown — no trade can be taken, so don't alert."""
    try:
        raw = _redis.get("cooldown_until:Nifty50")
        if not raw:
            return False
        return time.time() < float(raw)
    except Exception:
        return False


def _claim_alert(alert_key: str) -> bool:
    """Atomically reserve an alert so separate processes cannot both post it."""
    try:
        return bool(_redis.set(
            f"slack:alert_sent:Nifty50:{alert_key}",
            "1",
            nx=True,
            ex=ALERT_DEDUP_TTL_SEC,
        ))
    except Exception as error:
        # Redis may be down during startup; retain the in-process safeguard.
        print(f"[SLACK] Redis dedupe unavailable: {error}")
        return alert_key != _last_alert_key


def _send_message(text: str) -> bool:
    """Post a message to the configured Slack channel via Incoming Webhook."""
    if not WEBHOOK_URL:
        print("[SLACK] SLACK_WEBHOOK_URL not set in .env — alert skipped.")
        return False
    try:
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", "ignore")
            if body.strip() != "ok":
                print(f"[SLACK] Unexpected response: {body}")
                return False
            return True
    except urllib.error.URLError as e:
        print(f"[SLACK] Network error: {e}")
    except Exception as e:
        print(f"[SLACK] Error: {e}")
    return False


def _format_signal(signal_data: dict) -> str:
    signal = signal_data.get("signal", "WAIT")
    price  = signal_data.get("entry_price", 0)
    sl     = signal_data.get("stop_loss")
    tp     = signal_data.get("take_profit")
    conf   = signal_data.get("confidence", "0%")
    rsi    = signal_data.get("rsi", 0)
    regime = signal_data.get("regime", "N/A")
    probs  = signal_data.get("probabilities", {})
    source = signal_data.get("source", "XGB")

    if signal == "BUY":
        emoji   = ":large_green_circle:"
        action  = "BUY SIGNAL"
        sl_diff = f"-₹{abs(price - sl):.0f}" if sl else "N/A"
        tp_diff = f"+₹{abs(tp - price):.0f}" if tp else "N/A"
    else:
        emoji   = ":red_circle:"
        action  = "SELL SIGNAL"
        sl_diff = f"-₹{abs(sl - price):.0f}" if sl else "N/A"
        tp_diff = f"+₹{abs(price - tp):.0f}" if tp else "N/A"

    rr_ratio = ""
    if sl and tp and abs(price - sl) > 0:
        rr = abs(tp - price) / abs(price - sl)
        rr_ratio = f"\n⚖️ *RR Ratio:* {rr:.1f}:1"

    regime_emoji = ":ox:" if regime == "BULL" else ":bear:"
    sl_display = f"₹{sl:,.2f}  ({sl_diff})" if sl is not None else "N/A"
    tp_display = f"₹{tp:,.2f}  ({tp_diff})" if tp is not None else "N/A"

    # What option to actually buy: BUY -> ATM Call (CE), SELL -> ATM Put (PE)
    strike   = int(round(price / 50.0) * 50)
    opt_kind = "CE (Call)" if signal == "BUY" else "PE (Put)"

    # Slack uses mrkdwn: *bold*, line breaks with \n (no HTML)
    msg = (
        f"{emoji} *NIFTY 50 — {action}*\n"
        f"📍 *Buy option:* NIFTY {strike} {opt_kind}\n"
        f"{'─' * 22}\n"
        f"💰 *Spot Entry:* ₹{price:,.2f}\n"
        f"🛑 *Stop Loss:* {sl_display}\n"
        f"🎯 *Target:* {tp_display}"
        f"{rr_ratio}\n"
        f"{'─' * 22}\n"
        f"📊 *Confidence:* {conf}\n"
        f"📉 *RSI:* {rsi:.1f}\n"
        f"{regime_emoji} *Regime:* {regime}\n"
        f"🤖 *Model:* {source}\n"
        f"{'─' * 22}\n"
        f"📈 B={probs.get('buy',0):.0f}%   "
        f"📉 S={probs.get('sell',0):.0f}%   "
        f"⏸ W={probs.get('wait',0):.0f}%\n"
        f"⏱ {time.strftime('%H:%M:%S IST', time.localtime())}"
    )
    return msg


def maybe_send_alert(signal_data: dict, *, allow_open_position=False):
    """
    Call on every new signal.
    - Fires only on BUY/SELL (WAIT ignored)
    - Skipped for a fresh open position, except confirmed trade-entry notifications
    - Deduplicates: won't re-alert the same signal at the same price
    """
    global _last_alert_key

    signal = signal_data.get("signal", "WAIT")
    if signal not in ("BUY", "SELL"):
        return
    if not allow_open_position and _has_open_position():
        return
    if not allow_open_position and _in_cooldown():
        return   # 15-min post-trade cooldown — no trade will fire, so no alert

    price     = signal_data.get("entry_price", 0)
    alert_key = f"{signal}_{price:.0f}"
    if alert_key == _last_alert_key:
        return
    if not _claim_alert(alert_key):
        return

    if _send_message(_mention_prefix() + _format_signal(signal_data)):
        _last_alert_key = alert_key
        print(f"[SLACK] Alert sent: {signal} @ ₹{price:.2f}")


def send_startup_message():
    """One-time message when the AI engine starts."""
    _send_message(
        ":rocket: *NIFTY 50 AI Trader — ONLINE*\n"
        f"{'─' * 22}\n"
        "Bot is running. You'll get an alert for every BUY and SELL signal.\n"
        f"⏱ Started at {time.strftime('%H:%M:%S IST', time.localtime())}"
    )


def send_trade_closed_alert(signal: str, outcome: str, entry: float,
                            exit_price: float, pnl: float):
    """Alert when a trade closes (WIN/LOSS)."""
    emoji  = ":white_check_mark:" if outcome == "WIN" else ":x:"
    action = "BUY" if signal == "BUY" else "SELL"
    pnl_emoji = ":large_green_circle:" if pnl > 0 else ":red_circle:"
    msg = (
        f"{emoji} *NIFTY 50 — TRADE CLOSED ({outcome})*\n"
        f"{'─' * 22}\n"
        f"📌 *Direction:* {action}\n"
        f"💰 *Entry:* ₹{entry:,.2f}\n"
        f"🚪 *Exit:* ₹{exit_price:,.2f}\n"
        f"💵 *P&L:* {pnl_emoji} ₹{pnl:+,.2f}\n"
        f"⏱ {time.strftime('%H:%M:%S IST', time.localtime())}"
    )
    _send_message(_mention_prefix() + msg)


# ── Quick manual test:  python slack_alerts.py ───────────────────
if __name__ == "__main__":
    if not WEBHOOK_URL:
        print("Set SLACK_WEBHOOK_URL in .env first, then run again.")
    else:
        print("Sending a test message to Slack...")
        _send_message(":wrench: *Test alert from NIFTY 50 AI Trader* — Slack is wired up correctly!")
        print("Done. Check your Slack channel.")
