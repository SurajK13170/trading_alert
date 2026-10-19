"""
Telegram Alert System for NIFTY 50 AI Trader
Sends BUY/SELL signals to Telegram bot instantly.
WAIT signals are silently ignored.
"""

import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from html import escape
import redis
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_IDS     = [cid.strip() for cid in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if cid.strip()]
_redis       = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

_last_alert_key = ""   # fallback when Redis is temporarily unavailable
_last_alert_recipients = set()
POSITION_STATE_MAX_AGE_SEC = 120
ALERT_DEDUP_TTL_SEC = max(1, int(os.getenv("TELEGRAM_ALERT_DEDUP_TTL_SEC", "900")))


def _refresh_config():
    """Reload Telegram credentials from the current environment before sending."""
    global BOT_TOKEN, CHAT_IDS
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    CHAT_IDS = [cid.strip() for cid in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if cid.strip()]


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


def _claim_alert(alert_key: str) -> bool:
    """Atomically reserve an alert so restarts cannot send it again."""
    try:
        return bool(_redis.set(
            f"telegram:alert_sent:Nifty50:{alert_key}",
            "1",
            nx=True,
            ex=ALERT_DEDUP_TTL_SEC,
        ))
    except Exception as error:
        print(f"[TELEGRAM] Redis dedupe unavailable: {error}")
        return alert_key != _last_alert_key


def _send_message(text: str, chat_ids=None) -> set:
    """Return the chat IDs whose messages Telegram accepted."""
    _refresh_config()
    if not BOT_TOKEN or not CHAT_IDS:
        print("[TELEGRAM] BOT_TOKEN or TELEGRAM_CHAT_IDS not set in .env — alert skipped.")
        return set()

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    delivered = set()

    for chat_id in dict.fromkeys(CHAT_IDS if chat_ids is None else chat_ids):
        chat_id = chat_id.strip()
        if not chat_id:
            continue
        try:
            payload = json.dumps({
                "chat_id":    chat_id,
                "text":       text,
                "parse_mode": "HTML",
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                if result.get("ok"):
                    delivered.add(chat_id)
                else:
                    description = str(result.get("description", "Unknown error")).replace(BOT_TOKEN, "[REDACTED]")
                    print(f"[TELEGRAM] Send failed for {chat_id}: {description}")
        except urllib.error.HTTPError as e:
            try:
                description = json.loads(e.read()).get("description", str(e.reason))
            except (ValueError, OSError):
                description = str(e.reason)
            description = str(description).replace(BOT_TOKEN, "[REDACTED]")
            print(f"[TELEGRAM] HTTP {e.code} for {chat_id}: {description}")
        except urllib.error.URLError as e:
            print(f"[TELEGRAM] Network error for {chat_id}: {str(e.reason).replace(BOT_TOKEN, '[REDACTED]')}")
        except Exception as e:
            print(f"[TELEGRAM] Error for {chat_id}: {str(e).replace(BOT_TOKEN, '[REDACTED]')}")
    return delivered


def _format_signal(signal_data: dict) -> str:
    signal     = signal_data.get("signal", "WAIT")
    price      = signal_data.get("entry_price", 0)
    sl         = signal_data.get("stop_loss")
    tp         = signal_data.get("take_profit")
    conf       = escape(str(signal_data.get("confidence", "0%")))
    rsi        = signal_data.get("rsi", 0)
    regime     = escape(str(signal_data.get("regime", "N/A")))
    probs      = signal_data.get("probabilities", {})
    source     = escape(str(signal_data.get("source", "XGB")))

    if signal == "BUY":
        emoji   = "🟢"
        action  = "BUY SIGNAL"
        sl_diff = f"-₹{abs(price - sl):.0f}" if sl else "N/A"
        tp_diff = f"+₹{abs(tp - price):.0f}" if tp else "N/A"
    else:
        emoji   = "🔴"
        action  = "SELL SIGNAL"
        sl_diff = f"-₹{abs(sl - price):.0f}" if sl else "N/A"
        tp_diff = f"+₹{abs(price - tp):.0f}" if tp else "N/A"

    rr_ratio = ""
    if sl and tp and abs(price - sl) > 0:
        rr = abs(tp - price) / abs(price - sl)
        rr_ratio = f"\n⚖️ <b>RR Ratio :</b> {rr:.1f}:1"

    regime_emoji = "🐂" if regime == "BULL" else "🐻"
    sl_display = f"₹{sl:,.2f}  ({sl_diff})" if sl is not None else "N/A"
    tp_display = f"₹{tp:,.2f}  ({tp_diff})" if tp is not None else "N/A"

    msg = (
        f"{emoji} <b>{action} — NIFTY 50</b>\n"
        f"{'━' * 26}\n"
        f"💰 <b>Entry    :</b> ₹{price:,.2f}\n"
        f"🛑 <b>Stop Loss:</b> {sl_display}\n"
        f"🎯 <b>Target   :</b> {tp_display}\n"
        f"{rr_ratio}\n"
        f"{'━' * 26}\n"
        f"📊 <b>Confidence:</b> {conf}\n"
        f"📉 <b>RSI       :</b> {rsi:.1f}\n"
        f"{regime_emoji} <b>Regime    :</b> {regime}\n"
        f"🤖 <b>Model     :</b> {source}\n"
        f"{'━' * 26}\n"
        f"📈 B={probs.get('buy',0):.0f}%  "
        f"📉 S={probs.get('sell',0):.0f}%  "
        f"⏸ W={probs.get('wait',0):.0f}%\n"
        f"⏱ {time.strftime('%H:%M:%S IST', time.localtime())}"
    )
    return msg


def maybe_send_alert(signal_data: dict, *, allow_open_position=False):
    """
    Call this every time a new signal is generated.
    - Fires only on BUY/SELL (WAIT is ignored)
    - Skipped for a fresh open position, except confirmed trade-entry notifications
    - Deduplicates: sends a directional signal only once per dedupe window
    """
    global _last_alert_key, _last_alert_recipients

    signal = signal_data.get("signal", "WAIT")
    if signal not in ("BUY", "SELL"):
        return

    # Skip if execution manager already has an open position
    if not allow_open_position and _has_open_position():
        print("[TELEGRAM] Alert skipped: an open position is active.")
        return

    # Dedupe by direction, not price.  Live spot prices move on every polling
    # cycle, so including price here turned one BUY/SELL signal into many alerts.
    price      = signal_data.get("entry_price", 0)
    alert_key  = signal
    _refresh_config()
    if not BOT_TOKEN or not CHAT_IDS:
        return
    sent_to = _last_alert_recipients if alert_key == _last_alert_key else set()
    pending = [chat_id for chat_id in CHAT_IDS if chat_id not in sent_to]
    if not pending:
        return
    if not _claim_alert(alert_key):
        return

    msg = _format_signal(signal_data)
    delivered = _send_message(msg, chat_ids=pending)
    if not delivered:
        return
    if alert_key != _last_alert_key:
        _last_alert_recipients = set()
    _last_alert_key = alert_key
    _last_alert_recipients.update(delivered)
    print(f"[TELEGRAM] Alert sent: {signal} @ ₹{price:.2f} ({len(delivered)} chat(s))")


def send_startup_message():
    """Send a one-time message when the AI engine starts."""
    _send_message(
        "🚀 <b>NIFTY 50 AI Trader — ONLINE</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Bot is running. You will receive alerts\n"
        "for every BUY and SELL signal.\n"
        f"⏱ Started at {time.strftime('%H:%M:%S IST', time.localtime())}"
    )


def send_trade_closed_alert(signal: str, outcome: str, entry: float,
                             exit_price: float, pnl: float):
    """Send alert when a trade closes (WIN/LOSS)."""
    emoji  = "✅" if outcome == "WIN" else "❌"
    action = "BUY" if signal == "BUY" else "SELL"
    msg = (
        f"{emoji} <b>TRADE CLOSED — {outcome}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Direction :</b> {action}\n"
        f"💰 <b>Entry     :</b> ₹{entry:,.2f}\n"
        f"🚪 <b>Exit      :</b> ₹{exit_price:,.2f}\n"
        f"💵 <b>P&L       :</b> {'🟢' if pnl > 0 else '🔴'} ₹{pnl:+,.2f}\n"
        f"⏱ {time.strftime('%H:%M:%S IST', time.localtime())}"
    )
    _send_message(msg)
