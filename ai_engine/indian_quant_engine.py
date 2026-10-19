import pandas as pd
import numpy as np
import redis
import json
from dotenv import load_dotenv
import os
import time
import warnings
from trendline_breaks import indicator_data

# Silence the harmless "DataFrame is highly fragmented" notice — we add ~30 feature
# columns one-by-one to a tiny (~1k row) frame, so the perf impact is nil. It only
# clutters the terminal.
try:
    from pandas.errors import PerformanceWarning
    warnings.simplefilter("ignore", category=PerformanceWarning)
except Exception:
    pass

# ── Shared helpers (match model_trainer.py exactly) ─────────────
def _compute_adx(df, n=14):
    high, low, close = df['high'], df['low'], df['close']
    up   = high.diff()
    down = -low.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    alpha    = 1.0 / n
    atr_w    = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(alpha=alpha, adjust=False).mean().fillna(0)


def _compute_regime(adx, ema_align, atr_pctile, bb_width_pctile, vix):
    regime = 0
    if adx > 25:
        regime = 2 if ema_align > 0 else 1
    if bb_width_pctile < 0.15:
        regime = 3
    if atr_pctile > 0.80 or vix > 20:
        regime = 4
    return regime

load_dotenv()
redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)


def _drop_forming_candle(d, interval_minutes):
    """
    Drop the last row if it is the still-forming candle.

    A candle is stamped with its START time and covers [ts, ts + interval).
    If 'now' is still inside that window, the candle is incomplete — its OHLC
    will keep changing, so its indicators do not match the completed-candle
    distribution the model was trained on. We drop it and use the last CLOSED
    candle. Defensive about timezone: indices may be tz-aware (IST/UTC) or naive.
    """
    if d is None or d.empty:
        return d
    try:
        ts = pd.Timestamp(d.index[-1])
        if ts.tzinfo is None:
            ts = ts.tz_localize('Asia/Kolkata')
        else:
            ts = ts.tz_convert('Asia/Kolkata')
        now = pd.Timestamp.now(tz='Asia/Kolkata')
        if now < ts + pd.Timedelta(minutes=interval_minutes):
            return d.iloc[:-1]
    except Exception:
        # If anything is off with timestamps, be safe and drop the last bar.
        return d.iloc[:-1]
    return d

def normalize_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip().lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = df.index.name.lower() if df.index.name else df.index.name
    return df

def calculate_advanced_indicators(df):
    c = df['close']

    # ── EMAs + Distance % ─────────────────────────────────────
    df['EMA_20']  = c.ewm(span=20,  adjust=False).mean()
    df['EMA_50']  = c.ewm(span=50,  adjust=False).mean()
    df['EMA_200'] = c.ewm(span=200, adjust=False).mean()

    df['EMA_dist_20']  = (c - df['EMA_20'])  / df['EMA_20']  * 100
    df['EMA_dist_50']  = (c - df['EMA_50'])  / df['EMA_50']  * 100
    df['EMA_dist_200'] = (c - df['EMA_200']) / df['EMA_200'] * 100

    # EMA alignment: 1=full bull, -1=full bear, 0=mixed
    bull = (df['EMA_20'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_200'])
    bear = (df['EMA_20'] < df['EMA_50']) & (df['EMA_50'] < df['EMA_200'])
    df['EMA_alignment'] = np.where(bull, 1, np.where(bear, -1, 0))

    # ── RSI ───────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI']       = 100 - (100 / (1 + rs))
    df['RSI_Delta'] = df['RSI'].diff()
    df['RSI_Lag1']  = df['RSI'].shift(1)
    df['RSI_Lag2']  = df['RSI'].shift(2)

    # ── MACD Histogram + Acceleration ─────────────────────────
    macd_line             = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig              = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD']            = macd_line
    df['MACD_Signal']     = macd_sig
    df['MACD_Hist']       = macd_line - macd_sig
    df['MACD_Hist_Delta'] = df['MACD_Hist'].diff()

    # ── ATR ───────────────────────────────────────────────────
    high_low   = df['high'] - df['low']
    high_close = np.abs(df['high'] - c.shift())
    low_close  = np.abs(df['low']  - c.shift())
    df['ATR']     = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()
    df['ATR_pct'] = df['ATR'] / c * 100

    # ── Bollinger Bands + %B + Width ──────────────────────────
    df['MA20']     = c.rolling(window=20).mean()
    df['STD20']    = c.rolling(window=20).std()
    df['BBU']      = df['MA20'] + (df['STD20'] * 2)
    df['BBL']      = df['MA20'] - (df['STD20'] * 2)
    df['BB_pct']   = (c - df['BBL']) / (df['BBU'] - df['BBL'] + 1e-9)
    df['BB_width'] = (df['BBU'] - df['BBL']) / df['MA20'] * 100

    # ── Candle Structure ──────────────────────────────────────
    candle_range       = (df['high'] - df['low']).replace(0, np.nan)
    body               = (df['close'] - df['open']).abs()
    df['Candle_Body']  = body / candle_range
    df['Upper_Wick']   = (df['high'] - df[['close', 'open']].max(axis=1)) / candle_range
    df['Lower_Wick']   = (df[['close', 'open']].min(axis=1) - df['low'])  / candle_range

    # ── Volume ────────────────────────────────────────────────
    if 'volume' in df.columns and not df['volume'].isna().all() and df['volume'].sum() > 0:
        vol_ema         = df['volume'].ewm(span=20, adjust=False).mean()
        df['Vol_Ratio'] = (df['volume'] / vol_ema.replace(0, np.nan)).fillna(1.0)
    else:
        df['Vol_Ratio'] = df['ATR_pct']

    # ── ADX ───────────────────────────────────────────────────
    df['ADX'] = _compute_adx(df)

    # ── Volatility percentile ranks (100-bar rolling) ─────────
    df['ATR_pctile']      = df['ATR_pct'].rolling(100, min_periods=20).rank(pct=True).fillna(0.5)
    df['BB_width_pctile'] = df['BB_width'].rolling(100, min_periods=20).rank(pct=True).fillna(0.5)

    return df


# ── Angel One helpers ───────────────────────────────────────────
_angel_api    = None
_angel_authed = False
# Daily + BANKNIFTY candles change once a day — cache them (refresh every 10 min)
# so we don't re-fetch every 30s and trip Angel's historical rate limit.
_daily_cache  = {"5m": None, "ts_5m": 0, "daily": None, "bnf": None, "ts": 0,
                 "15m": None, "ts_15m": 0, "1h": None, "ts_1h": 0}

NIFTY_TOKEN = "99926000"   # NIFTY 50 index token (NSE)
BNF_TOKEN   = "26009"      # BANKNIFTY index token (NSE)
ANGEL_5M_REFRESH_SEC = 60

def _ensure_angel():
    """Lazy-authenticate Angel One. Returns True if session is live."""
    global _angel_api, _angel_authed
    if _angel_authed and _angel_api is not None:
        return True
    # Prefer the separate DATA account to avoid session conflicts with trading.
    # Fall back to trading credentials only when dedicated data credentials are absent.
    cid  = os.getenv("ANGEL_DATA_CLIENT_ID") or os.getenv("ANGEL_CLIENT_ID")
    akey = os.getenv("ANGEL_DATA_API_KEY")   or os.getenv("ANGEL_API_KEY")
    pwd  = os.getenv("ANGEL_DATA_PASSWORD")  or os.getenv("ANGEL_PASSWORD")
    totp = os.getenv("ANGEL_DATA_TOTP_KEY")  or os.getenv("ANGEL_TOTP_KEY")
    if not all([cid, akey, pwd, totp]):
        return False

    try:
        from SmartApi import SmartConnect
        import pyotp
        _angel_api = SmartConnect(api_key=akey)
        data = _angel_api.generateSession(cid, pwd, pyotp.TOTP(totp).now())
        if data and data.get("status"):
            _angel_authed = True
            print(f"[QUANT] Angel data-account login OK ({cid}) — LIVE candle data.")
            return True
        print(f"[QUANT] Angel One login failed: {data}")
        _angel_authed = False
        return False
    except Exception as e:
        print(f"[QUANT] Angel One auth error: {e}")
        _angel_authed = False
        return False


# ── Angel rate-limit handling (same proven approach as BankNifty) ──────
# Min gap between ANY two calls + a hard 60s cooldown after a rate-limit hit, so
# we stop making Angel requests until the cooldown expires.
_last_angel_call_ts = 0.0
_rate_limit_until   = 0.0
ANGEL_MIN_GAP       = 1.0
ANGEL_RETRY_WAIT    = 20.0
ANGEL_COOLDOWN_SEC  = 60.0


def _throttle_angel():
    global _last_angel_call_ts
    elapsed = time.time() - _last_angel_call_ts
    if elapsed < ANGEL_MIN_GAP:
        time.sleep(ANGEL_MIN_GAP - elapsed)
    _last_angel_call_ts = time.time()


def _angel_in_cooldown() -> bool:
    return time.time() < _rate_limit_until


def _angel_fetch(interval, days_back, token=None, exchange="NSE"):
    """
    Fetch OHLCV candles from Angel One → DataFrame with datetime index.
    Throttled (1s min gap) + retries once on rate-limit, then sets a 60s global
    cooldown so we stop hammering Angel until it resets.
    """
    from datetime import datetime, timedelta
    global _rate_limit_until
    now = datetime.now()
    frm = (now - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M")
    to  = now.strftime("%Y-%m-%d %H:%M")
    params = {
        "exchange":    exchange,
        "symboltoken": token or NIFTY_TOKEN,
        "interval":    interval,
        "fromdate":    frm,
        "todate":      to,
    }
    for attempt in range(2):
        _throttle_angel()
        try:
            resp = _angel_api.getCandleData(params)
        except Exception as e:
            msg = str(e).lower()
            if "exceeding access rate" in msg or "access denied" in msg or "rate" in msg:
                if attempt == 0:
                    print(f"[ANGEL] Rate-limited on {interval} -- waiting {ANGEL_RETRY_WAIT:.0f}s, retry once...")
                    time.sleep(ANGEL_RETRY_WAIT)
                    continue
                _rate_limit_until = time.time() + ANGEL_COOLDOWN_SEC
                print(f"[ANGEL] Rate limit persists -- pausing requests for {ANGEL_COOLDOWN_SEC:.0f}s.")
                raise
            raise
        if not resp or not resp.get("status") or not resp.get("data"):
            return pd.DataFrame()
        df = pd.DataFrame(
            resp["data"],
            columns=["Datetime", "Open", "High", "Low", "Close", "Volume"],
        )
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime").sort_index()
        return df   # normalize_df will lowercase columns
    return pd.DataFrame()


def main_engine(redis_key="Nifty50"):
    print("[START] Quant Engine v4.0 | Data source: Angel One only")

    while True:
        try:
            def clean_val(val, precision=4):
                try:
                    v = float(val)
                    if np.isnan(v) or np.isinf(v):
                        return 0.0
                    return round(v, precision)
                except (TypeError, ValueError):
                    return 0.0

            print("[1/3] Fetching Angel One candles...")
            if not _ensure_angel():
                print("[WAIT] Angel One unavailable; skipping candle update.")
                time.sleep(60)
                continue
            if _angel_in_cooldown():
                remaining = max(1, int(_rate_limit_until - time.time()))
                print(f"[WAIT] Angel One rate-limit cooldown: {remaining}s remaining.")
                time.sleep(min(remaining, 60))
                continue

            try:
                now_t = time.time()
                if (_daily_cache["5m"] is None
                        or now_t - _daily_cache["ts_5m"] >= ANGEL_5M_REFRESH_SEC):
                    _daily_cache["5m"] = _angel_fetch("FIVE_MINUTE", 7)
                    _daily_cache["ts_5m"] = now_t
                df_hourly_raw = _daily_cache["5m"]
                if df_hourly_raw is None or df_hourly_raw.empty:
                    raise ValueError("Angel One returned no 5-minute candles")

                chart_frame = _drop_forming_candle(normalize_df(df_hourly_raw.copy()), 5)
                if chart_frame.empty:
                    raise ValueError("Angel One returned no closed 5-minute candles")
                chart_df = chart_frame[['open', 'high', 'low', 'close']].copy()
                chart_df['timestamp'] = [int(ts.timestamp() * 1000) for ts in chart_frame.index]
                chart_candles = chart_df.to_dict('records')
                overlay = indicator_data(chart_candles)
                breakouts = [
                    {"timestamp": chart_candles[index]["timestamp"], "signal": direction}
                    for index, direction in overlay["events"]
                ]
                trendlines = {
                    name: [
                        {"timestamp": chart_candles[index]["timestamp"], "value": value}
                        for index, value in overlay[name]
                    ]
                    for name in ("upper", "lower")
                }
                chart_quant = {
                    "symbol": redis_key,
                    "price": float(chart_candles[-1]["close"]),
                    "data_source": "Angel One",
                    "breakouts": breakouts,
                    "trendlines": trendlines,
                    "timestamp": int(time.time() * 1000),
                }
                redis_client.set(f"quant:{redis_key}", json.dumps(chart_quant), ex=120)
                redis_client.set(f"candles:{redis_key}", json.dumps(chart_candles))

                if _daily_cache["15m"] is None or now_t - _daily_cache["ts_15m"] > 180:
                    _daily_cache["15m"] = _angel_fetch("FIFTEEN_MINUTE", 20)
                    _daily_cache["ts_15m"] = now_t
                if _daily_cache["1h"] is None or now_t - _daily_cache["ts_1h"] > 600:
                    _daily_cache["1h"] = _angel_fetch("ONE_HOUR", 50)
                    _daily_cache["ts_1h"] = now_t
                if _daily_cache["daily"] is None or now_t - _daily_cache["ts"] > 600:
                    _daily_cache["daily"] = _angel_fetch("ONE_DAY", 250)
                    _daily_cache["bnf"] = _angel_fetch("ONE_DAY", 60, token=BNF_TOKEN)
                    _daily_cache["ts"] = now_t
                df_15m_raw = _daily_cache["15m"]
                df_1h_raw = _daily_cache["1h"]
                df_daily_raw = _daily_cache["daily"]
                _bnf_raw = _daily_cache["bnf"]
                if any(frame is None or frame.empty for frame in (
                    df_hourly_raw, df_15m_raw, df_1h_raw, df_daily_raw, _bnf_raw
                )):
                    raise ValueError("Angel One returned empty candle data")
            except Exception as error:
                print(f"[WARN] Angel One candle fetch failed; no fallback data used: {error}")
                time.sleep(10)
                continue

            if df_daily_raw.empty:
                print("[ERR] Daily data empty!"); time.sleep(10); continue

            # ── Process daily ─────────────────────────────────────
            df_daily     = normalize_df(df_daily_raw)
            df_daily     = calculate_advanced_indicators(df_daily)
            latest_daily = df_daily.iloc[-1]
            daily_rsi    = latest_daily['RSI']
            # NOTE: daily_regime is computed below on the 5-MIN EMA200 (not the
            # daily-candle EMA200). model_trainer.add_features() defines
            # Daily_Regime as (5m close > 5m EMA200), so live MUST match that or
            # the model receives a feature with a different meaning than it learned.
            print("[2/3] Processing multi-timeframe data...")
            if df_hourly_raw.empty:
                print("[ERR] 5-min data empty!"); time.sleep(10); continue

            df_hourly = normalize_df(df_hourly_raw)
            df_hourly = calculate_advanced_indicators(df_hourly)
            # ── Drop the still-forming candle ─────────────────────
            # The latest 5-min candle is INCOMPLETE (it is still building until the
            # 5-min boundary). The model was trained ONLY on completed candles, so
            # acting on a half-formed candle feeds it a feature distribution it has
            # never seen and makes the signal flip every 30s. Use the last CLOSED
            # candle instead.
            df_hourly = _drop_forming_candle(df_hourly, 5)
            if df_hourly.empty:
                print("[ERR] No closed 5-min candle yet!"); time.sleep(10); continue
            latest    = df_hourly.iloc[-1]

            # daily_regime — match training: 5-min close vs 5-min EMA200 (NOT daily)
            daily_regime = 1 if latest['close'] > latest['EMA_200'] else 0
            print(f"[OK]  Daily  | Regime: {'BULL' if daily_regime else 'BEAR'} | src: Angel One")
            print(f"[OK]  5-min | Candles: {len(df_hourly)} | Price: {latest['close']:.2f}")

            # ── 15-min indicators (drop forming bar too) ──────────
            df_15m  = normalize_df(df_15m_raw)
            df_15m  = calculate_advanced_indicators(df_15m)
            df_15m  = _drop_forming_candle(df_15m, 15)
            lat_15m = df_15m.iloc[-1]

            # ── 1-hour indicators (drop forming bar too) ──────────
            df_1h  = normalize_df(df_1h_raw)
            df_1h  = calculate_advanced_indicators(df_1h)
            df_1h  = _drop_forming_candle(df_1h, 60)
            lat_1h = df_1h.iloc[-1]

            # India VIX is not fetched; keep neutral values without another data source.
            vix_val, vix_trend = 15.0, 0

            # ── BANKNIFTY Relative Strength ────────────────────────
            # Align NIFTY and BANKNIFTY by DATE (not by row position). If either
            # index has a missing/holiday day, position-alignment silently pairs
            # the wrong days and produces a RS_ratio the model never saw in
            # training (model_trainer merges on Date). Match that exactly.
            try:
                bnf_df  = normalize_df(_bnf_raw)
                n_ser = df_daily['close'].copy()
                b_ser = bnf_df['close'].copy()
                n_ser.index = pd.DatetimeIndex(n_ser.index).normalize().tz_localize(None)
                b_ser.index = pd.DatetimeIndex(b_ser.index).normalize().tz_localize(None)
                rs_df = pd.DataFrame({'nifty': n_ser, 'bnf': b_ser}).dropna()
                raw_ratio = rs_df['nifty'] / rs_df['bnf'].replace(0, np.nan)
                rs_norm   = raw_ratio / raw_ratio.rolling(20, min_periods=5).mean()
                rs_ratio  = round(float(rs_norm.iloc[-1]) if not rs_norm.empty else 1.0, 4)
                rs_slope  = round(float(rs_norm.diff(5).iloc[-1]) if len(rs_norm) >= 6 else 0.0, 4)
                if np.isnan(rs_ratio): rs_ratio = 1.0
                if np.isnan(rs_slope): rs_slope = 0.0
            except Exception:
                rs_ratio, rs_slope = 1.0, 0.0

            # ── Session timing ─────────────────────────────────────
            latest_ts = pd.Timestamp(df_hourly.index[-1])
            if latest_ts.tzinfo is None:
                latest_ts = latest_ts.tz_localize('UTC')
            latest_ts_ist = latest_ts.tz_convert('Asia/Kolkata')
            hour        = int(latest_ts_ist.hour)
            minute_val  = int(latest_ts_ist.minute)
            day_of_week = int(latest_ts_ist.dayofweek)

            # ── VWAP (daily, resets each session; volume-less fallback) ──
            try:
                h_ist = df_hourly.copy()
                if h_ist.index.tz is None:
                    h_ist.index = h_ist.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
                else:
                    h_ist.index = h_ist.index.tz_convert('Asia/Kolkata')
                today_date = latest_ts_ist.date()
                today_5m   = h_ist[h_ist.index.date == today_date]
                if not today_5m.empty:
                    typ = (today_5m['high'] + today_5m['low'] + today_5m['close']) / 3
                    if 'volume' in today_5m.columns and today_5m['volume'].sum() > 0:
                        vwap_val = float((typ * today_5m['volume']).cumsum().iloc[-1]
                                         / today_5m['volume'].cumsum().iloc[-1])
                    else:
                        vwap_val = float(typ.cumsum().iloc[-1] / len(typ))  # volume-less proxy
                    vwap_dist = round((float(latest['close']) - vwap_val)
                                      / vwap_val * 100, 4)
                else:
                    vwap_dist = 0.0
            except Exception:
                vwap_dist = 0.0

            # ── Previous day levels ────────────────────────────────
            try:
                if len(df_daily) >= 2:
                    prev    = df_daily.iloc[-2]
                    cur     = float(latest['close'])
                    pdh_dist = round((cur - float(prev['high']))  / float(prev['high'])  * 100, 4)
                    pdl_dist = round((cur - float(prev['low']))   / float(prev['low'])   * 100, 4)
                    pdc_dist = round((cur - float(prev['close'])) / float(prev['close']) * 100, 4)
                else:
                    pdh_dist = pdl_dist = pdc_dist = 0.0
            except Exception:
                pdh_dist = pdl_dist = pdc_dist = 0.0

            # ── Opening range (9:15–9:45) ──────────────────────────
            try:
                t_idx    = today_5m.index.hour * 60 + today_5m.index.minute
                or_bars  = today_5m[(t_idx >= 9*60+15) & (t_idx < 9*60+45)]
                if not or_bars.empty:
                    or_high = float(or_bars['high'].max())
                    or_low  = float(or_bars['low'].min())
                    or_rng  = or_high - or_low
                    cur     = float(latest['close'])
                    or_size     = round(or_rng / cur * 100, 4) if cur > 0 else 0.0
                    or_position = round(max(0, (cur - or_low) / or_rng)
                                        if or_rng > 0 else 0.5, 4)
                else:
                    or_size = or_position = 0.0
            except Exception:
                or_size = or_position = 0.0

            # ── NIFTY weekly expiry features ───────────────────────
            days_to_expiry  = (3 - day_of_week) % 7    # Thursday=3
            is_expiry_day   = 1 if day_of_week == 3 else 0
            cur_price       = float(latest['close'])
            mod50           = cur_price % 50
            mod100          = cur_price % 100
            dist_round_50   = round(min(mod50,  50  - mod50),  4)
            dist_round_100  = round(min(mod100, 100 - mod100), 4)

            # Session bucket (matches model_trainer.py encoding)
            t = hour * 60 + minute_val
            if   t < 9*60+15:  session_bucket = 0
            elif t < 10*60:    session_bucket = 1   # opening volatility
            elif t < 12*60:    session_bucket = 2   # mid-morning trend
            elif t < 13*60+30: session_bucket = 3   # lunch chop
            elif t < 14*60+30: session_bucket = 4   # afternoon
            else:              session_bucket = 5   # power hour

            # Market Regime (needs ADX + pctiles + vix, all computed above)
            market_regime = _compute_regime(
                clean_val(latest['ADX']),
                int(latest['EMA_alignment']),
                clean_val(latest['ATR_pctile']),
                clean_val(latest['BB_width_pctile']),
                vix_val,
            )

            # 35-Feature Vector -- must match model_trainer.py FEATURES list exactly
            # Keys are lowercase of the FEATURES list (ai engine does feat.lower())
            indicators = {
                # RSI group
                "rsi":       clean_val(latest['RSI']),
                "rsi_delta": clean_val(latest['RSI_Delta']),
                "rsi_lag1":  clean_val(latest['RSI_Lag1']),
                "rsi_lag2":  clean_val(latest['RSI_Lag2']),
                # MACD group
                "macd_hist":       clean_val(latest['MACD_Hist']),
                "macd_hist_delta": clean_val(latest['MACD_Hist_Delta']),
                # EMA group
                "ema_dist_20":   clean_val(latest['EMA_dist_20']),
                "ema_dist_50":   clean_val(latest['EMA_dist_50']),
                "ema_dist_200":  clean_val(latest['EMA_dist_200']),
                "ema_alignment": int(latest['EMA_alignment']),
                # Volatility
                "bb_pct":   clean_val(latest['BB_pct']),
                "bb_width": clean_val(latest['BB_width']),
                "atr_pct":  clean_val(latest['ATR_pct']),
                # Volatility percentile ranks
                "atr_pctile":      clean_val(latest['ATR_pctile']),
                "bb_width_pctile": clean_val(latest['BB_width_pctile']),
                # Candle structure
                "candle_body": clean_val(latest['Candle_Body']),
                "upper_wick":  clean_val(latest['Upper_Wick']),
                "lower_wick":  clean_val(latest['Lower_Wick']),
                # Session timing
                "hour":           hour,
                "minute":         minute_val,
                "day_of_week":    day_of_week,
                "session_bucket": session_bucket,
                # Volume
                "vol_ratio": clean_val(latest['Vol_Ratio']),
                # Regime
                "daily_regime":  daily_regime,
                # Trend strength
                "adx":           clean_val(latest['ADX']),
                # Market regime classifier
                "market_regime": market_regime,
                # Multi-timeframe confluence
                "rsi_15m":       clean_val(lat_15m['RSI']),
                "rsi_1h":        clean_val(lat_1h['RSI']),
                "ema_align_15m": int(lat_15m['EMA_alignment']),
                "ema_align_1h":  int(lat_1h['EMA_alignment']),
                "bb_pct_1h":     clean_val(lat_1h['BB_pct']),
                # VIX
                "vix":       vix_val,
                "vix_trend": vix_trend,
                # Relative strength NIFTY vs BANKNIFTY
                "rs_ratio": rs_ratio,
                "rs_slope": rs_slope,
                # VWAP
                "vwap_dist": vwap_dist,
                # Previous day levels
                "pdh_dist": pdh_dist,
                "pdl_dist": pdl_dist,
                "pdc_dist": pdc_dist,
                # Opening range
                "or_size":     or_size,
                "or_position": or_position,
                # NIFTY weekly expiry features
                "days_to_expiry":  days_to_expiry,
                "is_expiry_day":   is_expiry_day,
                "dist_round_50":   dist_round_50,
                "dist_round_100":  dist_round_100,
                # Display-only (not used by model)
                "atr":       clean_val(latest['ATR']),
                "close":     clean_val(latest['close'], 2),
                "daily_rsi": clean_val(daily_rsi, 2),
            }

            quant_data = {
                "symbol":    redis_key,
                "price":     float(latest['close']),
                "data_source": "Angel One",
                "breakouts": breakouts,
                "trendlines": trendlines,
                "indicators": indicators,
                "trend":     "BULLISH" if latest['close'] > latest['EMA_200'] else "BEARISH",
                "session":   f"{hour:02d}:{minute_val:02d} IST | Day {day_of_week}",
                "timestamp": int(time.time() * 1000),
            }

            print(f"[3/3] Writing to Redis (quant:{redis_key})...")
            # TTL=120s: if this engine crashes, key auto-expires so ai_engine
            # stops trading on stale data rather than acting on an hours-old snapshot.
            redis_client.set(f"quant:{redis_key}", json.dumps(quant_data), ex=120)

            # Chart candles
            redis_client.set(f"candles:{redis_key}", json.dumps(chart_candles))

            print(f"[DONE] RSI={clean_val(latest['RSI'],1)} "
                  f"| ADX={clean_val(latest['ADX'],1)} "
                  f"| Regime={market_regime} "
                  f"| BB%={clean_val(latest['BB_pct'],2)} "
                  f"| ATR_pctile={clean_val(latest['ATR_pctile'],2)} "
                  f"| RS={rs_ratio:.3f}")
            time.sleep(10)

        except Exception as e:
            print(f"[ERR] ENGINE ERROR: {e}")
            import traceback; traceback.print_exc()
            time.sleep(10)


if __name__ == "__main__":
    main_engine()
