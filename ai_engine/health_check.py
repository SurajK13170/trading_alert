"""
LIVE SYSTEM HEALTH CHECK  —  run: python health_check.py
Reads Redis and shows whether each engine is alive + how fresh its data is.
Green = data is recent. If something is stale/missing, that engine is down.
"""
import os, json, time
import redis
from dotenv import load_dotenv

load_dotenv()
r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
now = time.time()


def age(ts):
    return f"{round(now - ts, 1)}s ago" if ts else "—"


def get(key):
    v = r.get(key)
    return json.loads(v) if v else None


print("=" * 54)
print("  LIVE SYSTEM HEALTH CHECK")
print("=" * 54)

t = get("tick:Nifty50")
if t:
    print(f"[1] tick_collector : OK  | spot={t['price']:.2f} | {age(t['ts'])}")
else:
    print("[1] tick_collector : NO DATA (is tick_collector.py running?)")

q = get("quant:Nifty50")
if q:
    i = q.get("indicators", {})
    print(f"[2] quant_engine   : OK  | RSI={i.get('rsi')} ADX={i.get('adx')} | {age(q.get('timestamp', 0) / 1000)}")
else:
    print("[2] quant_engine   : NO DATA (is indian_quant_engine.py running?)")

s = get("signal:Nifty50")
if s:
    print(f"[3] ai_engine      : OK  | {s.get('signal')} {s.get('confidence')} | {age(s.get('timestamp', 0))}")
else:
    print("[3] ai_engine      : NO DATA (is indian_ai_engine.py running?)")

a = get("active_trade:Nifty50")
if a:
    print(f"[4] exec_manager   : OK  | mode={a.get('mode')} status={a.get('status')} pos={a.get('position')}")
    if a.get("status") == "OPEN":
        print(f"                       option={a.get('opt_symbol')} entry={a.get('opt_entry_premium')} "
              f"ltp={a.get('opt_ltp')} uPNL={a.get('unrealized_pnl')}")
    print(f"                       daily P&L=Rs{a.get('realized_pnl')} | trades today={a.get('trades_today')}")
else:
    print("[4] exec_manager   : NO DATA (is indian_execution_manager.py running?)")

print("=" * 54)
