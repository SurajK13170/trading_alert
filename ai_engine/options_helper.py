"""
NIFTY OPTIONS HELPER -- authenticated Angel One lookups
--------------------------------------------------------
Pure logic + Angel One authenticated lookups (smartApi.searchScrip + ltpData).
No more static scrip-master file -- Angel removed that endpoint, so all token
resolution now goes through the authenticated session.

Used by indian_execution_manager.py when MODE == LIVE.

Responsibilities:
  - Round Nifty spot to nearest valid strike (50-point grid)
  - Find nearest weekly expiry (Tuesday for Nifty)
  - Build the Angel One option symbol string
  - Look up symbol -> token via authenticated searchScrip (cached in-memory)
  - Fetch live option LTP via SmartConnect
  - Calculate lot count for a given capital + premium + risk %
"""

import time
from datetime import datetime, timedelta
from typing import Optional

# ─── config ─────────────────────────────────────────────────
NIFTY_STRIKE_INTERVAL = 50
NIFTY_LOT_SIZE        = 75
# ────────────────────────────────────────────────────────────

# In-memory token cache. Lives for one execution_manager session -- restart clears it.
# Keeps Angel API hits to a minimum (same option queried multiple times during one trade).
_TOKEN_CACHE: dict = {}


def round_to_strike(price: float, interval: int = NIFTY_STRIKE_INTERVAL) -> int:
    """Round any price to the nearest valid Nifty strike (50-pt grid)."""
    return int(round(price / interval) * interval)


def get_nearest_weekly_expiry(today: Optional[datetime] = None) -> datetime:
    """
    Nifty weekly options expire on Tuesday (post Apr-2024 SEBI rule).
    Returns the next upcoming Tuesday. If today is Tuesday after 15:30,
    skip to next week's Tuesday.
    """
    if today is None:
        today = datetime.now()
    # Monday=0, Tuesday=1, ..., Sunday=6
    days_to_tuesday = (1 - today.weekday()) % 7
    if days_to_tuesday == 0 and today.hour >= 15 and today.minute >= 30:
        days_to_tuesday = 7
    return today + timedelta(days=days_to_tuesday)


def make_option_symbol(expiry: datetime, strike: int, option_type: str) -> str:
    """
    Build the Angel One option symbol string.

    Format: NIFTY{DD}{MMM}{YY}{STRIKE}{CE|PE}
    Example: NIFTY19MAY2623550CE

    Angel uses 2-digit year, all uppercase month abbreviation.
    """
    if option_type not in ("CE", "PE"):
        raise ValueError(f"option_type must be CE or PE, got {option_type!r}")
    expiry_str = expiry.strftime("%d%b%y").upper()
    return f"NIFTY{expiry_str}{strike}{option_type}"


# ══════════════════════════════════════════════════════════════
#  SYMBOL -> TOKEN LOOKUP (authenticated)
# ══════════════════════════════════════════════════════════════
def lookup_token(smart_api, symbol: str) -> Optional[str]:
    """
    Find the Angel token for an option symbol using authenticated searchScrip.
    Cached in-memory for the duration of the session.
    Returns None on failure (caller should skip the trade).
    """
    if smart_api is None or not symbol:
        return None
    if symbol in _TOKEN_CACHE:
        return _TOKEN_CACHE[symbol]

    try:
        resp = smart_api.searchScrip(exchange="NFO", searchscrip=symbol)
    except Exception as e:
        print(f"[OPT-LOOKUP] searchScrip failed for {symbol}: {e}")
        return None

    if not resp or not resp.get("status") or not resp.get("data"):
        print(f"[OPT-LOOKUP] No match for {symbol} in Angel scrip search.")
        return None

    # Look for an exact tradingsymbol match first
    for item in resp["data"]:
        if item.get("tradingsymbol") == symbol:
            token = str(item.get("symboltoken"))
            _TOKEN_CACHE[symbol] = token
            return token

    # Fallback: take first result if it contains our symbol
    first = resp["data"][0]
    first_sym = str(first.get("tradingsymbol", ""))
    if symbol in first_sym or first_sym in symbol:
        token = str(first.get("symboltoken"))
        _TOKEN_CACHE[symbol] = token
        print(f"[OPT-LOOKUP] Using closest match: {first_sym} (token {token})")
        return token

    print(f"[OPT-LOOKUP] {symbol} not found. searchScrip returned {len(resp['data'])} other matches.")
    return None


# ══════════════════════════════════════════════════════════════
#  LIVE PREMIUM FETCH
# ══════════════════════════════════════════════════════════════
def fetch_option_ltp(smart_api, token: str, symbol: str) -> Optional[float]:
    """Current premium (last traded price) for an option."""
    if smart_api is None or not token:
        return None
    try:
        resp = smart_api.ltpData("NFO", symbol, token)
        if resp and resp.get("status") and resp.get("data"):
            return float(resp["data"].get("ltp", 0))
    except Exception as e:
        print(f"[OPT-LTP] Fetch error for {symbol}: {e}")
    return None


# ══════════════════════════════════════════════════════════════
#  POSITION SIZING
# ══════════════════════════════════════════════════════════════
def calculate_lots(
    capital: float,
    premium: float,
    risk_pct: float = 1.0,
    max_lots: int = 2,
    lot_size: int = NIFTY_LOT_SIZE,
) -> int:
    """
    How many lots to buy given user's capital and option premium.

    Returns 0 if you cannot afford even one lot -- caller should skip.

    Risk model: treat premium-paid as worst-case loss (buying options can lose
    100% of premium). Cap at risk_pct of capital and at max_lots.
    """
    if premium <= 0 or capital <= 0:
        return 0
    cost_per_lot = premium * lot_size
    if cost_per_lot > capital:
        return 0   # cannot afford one lot, skip
    max_risk_inr = capital * (risk_pct / 100.0)
    lots_by_risk = max(1, int(max_risk_inr / cost_per_lot))
    return min(lots_by_risk, max_lots)


# ══════════════════════════════════════════════════════════════
#  ALL-IN-ONE: resolve everything for a given signal
# ══════════════════════════════════════════════════════════════
def resolve_option_for_signal(
    signal: str,
    nifty_spot: float,
    smart_api=None,
) -> Optional[dict]:
    """
    Given AI signal (BUY/SELL) + current Nifty spot, return everything needed
    to place an Angel One option order:
        { symbol, token, strike, expiry, option_type, premium, lot_size }

    BUY  -> buy ATM CE (call)
    SELL -> buy ATM PE (put)

    Returns None if:
      - signal isn't BUY/SELL
      - smart_api missing (we need authenticated session for token lookup)
      - token lookup or premium fetch fails
    """
    if signal not in ("BUY", "SELL"):
        return None
    if smart_api is None:
        print("[OPT] resolve_option_for_signal needs an authenticated smart_api.")
        return None

    opt_type = "CE" if signal == "BUY" else "PE"
    strike   = round_to_strike(nifty_spot)
    expiry   = get_nearest_weekly_expiry()
    symbol   = make_option_symbol(expiry, strike, opt_type)

    token = lookup_token(smart_api, symbol)
    if token is None:
        return None

    premium = fetch_option_ltp(smart_api, token, symbol)
    if premium in (None, 0):
        print(f"[OPT] No premium available for {symbol}. May be illiquid / outside hours.")
        return None

    return {
        "symbol":      symbol,
        "token":       token,
        "strike":      strike,
        "expiry":      expiry.strftime("%Y-%m-%d"),
        "option_type": opt_type,
        "premium":     premium,
        "lot_size":    NIFTY_LOT_SIZE,
    }


# ── quick CLI test (no Angel One auth -- only pure logic) ─────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  OPTIONS HELPER -- PURE LOGIC SELF-TEST (no API calls)")
    print("="*60 + "\n")

    spot = 23550
    print(f"Spot Nifty           : {spot}")
    print(f"Rounded strike       : {round_to_strike(spot)}")
    print(f"Nearest expiry       : {get_nearest_weekly_expiry().strftime('%Y-%m-%d (%A)')}")
    print(f"BUY  -> symbol       : {make_option_symbol(get_nearest_weekly_expiry(), round_to_strike(spot), 'CE')}")
    print(f"SELL -> symbol       : {make_option_symbol(get_nearest_weekly_expiry(), round_to_strike(spot), 'PE')}")
    print()
    print("Position sizing tests:")
    for cap, prem in [(1000, 120), (10000, 120), (50000, 120), (50000, 800)]:
        lots = calculate_lots(cap, prem, risk_pct=1.0, max_lots=2)
        if lots == 0:
            print(f"  Cap Rs{cap:>6,}  prem Rs{prem:>3}  -> SKIP (need Rs {prem*75:,} per lot)")
        else:
            cost = lots * prem * 75
            print(f"  Cap Rs{cap:>6,}  prem Rs{prem:>3}  -> {lots} lot(s) = Rs {cost:,} cost")
    print()
    print("Token lookup test: requires authenticated SmartConnect session.")
    print("  Run when LIVE mode is active in indian_execution_manager.py.")
