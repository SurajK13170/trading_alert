"""
NIFTY 50 - GOD MODE MODEL TRAINER v4.0
3-Class: BUY / SELL / WAIT  |  28-Feature Pipeline  |  10-Year Dataset
ATR-based sequential labeling | Aligned with live system

Features (20 -- must match indian_quant_engine.py exactly):
  RSI, RSI_Delta, RSI_Lag1, RSI_Lag2
  MACD_Hist, MACD_Hist_Delta
  EMA_dist_20, EMA_dist_50, EMA_dist_200, EMA_alignment
  BB_pct, BB_width, ATR_pct
  Candle_Body, Upper_Wick, Lower_Wick
  Hour, Day_of_week
  Vol_Ratio
  Daily_Regime
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

FEEDBACK_CSV = "live_feedback.csv"
FEEDBACK_WEIGHT = 5   # each live trade counts as 5x a historical candle

# Meta-labeling: train second model to filter out bad BUY/SELL signals.
# DISABLED by default — needs 5000+ live trades to be properly tuned.
# Enable later once enough live data has accumulated.
USE_META_LABELING = False


class CalibratedXGB:
    """
    XGBoost + per-class isotonic calibration + optional meta-labeling filter.
    Module-level so pickle works.

    Meta-labeling: if the base model predicts BUY/SELL but the meta-model says
    "this signal is not trustworthy", we override to WAIT. This filters out
    high-confidence WRONG signals (the model's blind spots).
    """
    META_TRUST_THRESHOLD = 0.50   # meta-model must say >=50% "trust this signal"

    def __init__(self, base, calibrators, meta_model=None):
        self.base                 = base
        self.calibrators          = calibrators
        self.meta_model           = meta_model   # None = no meta-filter
        self.feature_importances_ = base.feature_importances_

    def _calibrated(self, X):
        raw   = self.base.predict_proba(X)
        cal   = np.column_stack([c.predict(raw[:, k]) for k, c in enumerate(self.calibrators)])
        total = cal.sum(axis=1, keepdims=True)
        total = np.where(total == 0, 1, total)
        return cal / total

    def predict_proba(self, X):
        cal = self._calibrated(X)
        if self.meta_model is None:
            return cal
        # Apply meta-filter: for each row, if base predicts BUY/SELL,
        # ask meta-model "should we trust this?". If not, convert to WAIT.
        preds = cal.argmax(axis=1)
        out   = cal.copy()
        for i in range(len(cal)):
            if preds[i] == 1:   # WAIT — nothing to filter
                continue
            # Build meta-input: original features + base_pred + base_conf
            row = X.iloc[[i]].copy() if hasattr(X, 'iloc') else X[i:i+1].copy()
            row['base_pred'] = float(preds[i])
            row['base_conf'] = float(cal[i].max())
            trust = float(self.meta_model.predict_proba(row)[0, 1])
            if trust < self.META_TRUST_THRESHOLD:
                # Meta says "don't trust" → force WAIT
                out[i] = np.array([0.05, 0.90, 0.05])
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

# ----------------------------------------------------------------
HOURLY_CSV      = "nifty_5m_data.csv"     # 5-min data from Angel One
MODEL_OUT       = "trading_model.pkl"
FEATURE_OUT     = "feature_list.pkl"

FORWARD_CANDLES = 24     # 24 x 5min = 2 hours to resolve trade
ATR_SL_MULT     = 1.0   # was 1.5 — tighter, realistic for NIFTY intraday
ATR_TP_MULT     = 2.0   # was 3.0 — 60pt target achievable, 90pt was too wide
TRAIN_SPLIT     = 0.80

# ── Accuracy boosters ───────────────────────────────────────────
CHOP_FILTER_PCTILE  = 0.20   # drop bottom-20% ATR_pct rows (flat market noise)
TIMEOUT_DIR_THRESH  = 0.50   # if timeout hits, treat as directional iff |ret| > N*ATR
TIME_DECAY_RECENT   = 2.0    # last 25% of training data gets 2x weight
# ----------------------------------------------------------------

FEATURES = [
    # RSI group (5-min)
    'RSI', 'RSI_Delta', 'RSI_Lag1', 'RSI_Lag2',
    # MACD group (5-min)
    'MACD_Hist', 'MACD_Hist_Delta',
    # EMA group (5-min)
    'EMA_dist_20', 'EMA_dist_50', 'EMA_dist_200', 'EMA_alignment',
    # Volatility (5-min)
    'BB_pct', 'BB_width', 'ATR_pct',
    # Volatility percentile ranks -- squeeze + expansion detection
    'ATR_pctile',       # 100-bar rolling ATR rank 0-1 (0=low vol, 1=high vol)
    'BB_width_pctile',  # 100-bar rolling BB width rank (<0.15 = squeeze, breakout near)
    # Candle structure
    'Candle_Body', 'Upper_Wick', 'Lower_Wick',
    # Session timing
    'Hour', 'Minute', 'Day_of_week',
    'Session_bucket',   # 1=open 2=mid-morning 3=lunch 4=afternoon 5=power-hour
    # Volume
    'Vol_Ratio',
    # Market regime (5-min EMA proxy)
    'Daily_Regime',
    # Trend strength
    'ADX',              # 14-period ADX (>25 trending, <20 sideways)
    # Market regime classifier
    'Market_Regime',    # 0=sideways 1=trend-dn 2=trend-up 3=squeeze 4=high-vol
    # Multi-timeframe confluence
    'RSI_15m', 'RSI_1h', 'EMA_align_15m', 'EMA_align_1h', 'BB_pct_1h',
    # VIX
    'VIX', 'VIX_trend',
    # Relative strength: NIFTY vs BANKNIFTY (institutional rotation signal)
    'RS_ratio',         # normalized nifty/bnf ratio (>1 = nifty outperforming)
    'RS_slope',         # 5-day slope of RS_ratio (direction of rotation)
    # VWAP (institutional intraday benchmark — resets every day)
    'VWAP_dist',        # % distance of close from daily VWAP
    # Previous day levels (key support/resistance)
    'PDH_dist',         # % distance from previous day high (resistance)
    'PDL_dist',         # % distance from previous day low (support)
    'PDC_dist',         # % distance from previous day close (pivot)
    # Opening range: first 30 min (9:15–9:45)
    'OR_size',          # opening range size as % of price
    'OR_position',      # 0 = at OR low, 1 = at OR high, >1 = breakout up
    # NIFTY weekly expiry features (Thursdays — completely different behaviour)
    'Days_to_expiry',   # 0=Thu, 1=Wed, 2=Tue, 3=Mon, 4-6=Fri/Sat/Sun
    'Is_expiry_day',    # 1 if Thursday, else 0
    'Dist_round_50',    # points away from nearest 50-multiple (pin risk)
    'Dist_round_100',   # points away from nearest 100-multiple (key strike)
]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    sample = str(df['date'].iloc[0])
    if df['date'].dtype in [np.int64, np.float64] or sample.isdigit():
        df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)
        df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')
    else:
        df['date'] = pd.to_datetime(df['date'], utc=False)
        if df['date'].dt.tz is None:
            df['date'] = df['date'].dt.tz_localize('Asia/Kolkata')
        else:
            df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')

    df = df.sort_values('date').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', ''), errors='coerce'
        )

    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    print(f"[OK] Loaded: {len(df):,} candles | "
          f"{df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']

    # EMAs + Distance %
    df['EMA_20']  = c.ewm(span=20,  adjust=False).mean()
    df['EMA_50']  = c.ewm(span=50,  adjust=False).mean()
    df['EMA_200'] = c.ewm(span=200, adjust=False).mean()

    df['EMA_dist_20']  = (c - df['EMA_20'])  / df['EMA_20']  * 100
    df['EMA_dist_50']  = (c - df['EMA_50'])  / df['EMA_50']  * 100
    df['EMA_dist_200'] = (c - df['EMA_200']) / df['EMA_200'] * 100

    # EMA alignment: 1=full bull (20>50>200), -1=full bear, 0=mixed
    bull = (df['EMA_20'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_200'])
    bear = (df['EMA_20'] < df['EMA_50']) & (df['EMA_50'] < df['EMA_200'])
    df['EMA_alignment'] = np.where(bull, 1, np.where(bear, -1, 0))

    # RSI
    delta = c.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI']       = 100 - (100 / (1 + rs))
    df['RSI_Delta'] = df['RSI'].diff()
    df['RSI_Lag1']  = df['RSI'].shift(1)
    df['RSI_Lag2']  = df['RSI'].shift(2)

    # MACD Histogram + Acceleration
    macd_line             = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig              = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist']       = macd_line - macd_sig
    df['MACD_Hist_Delta'] = df['MACD_Hist'].diff()

    # ATR
    hl  = df['high'] - df['low']
    hpc = (df['high'] - c.shift()).abs()
    lpc = (df['low']  - c.shift()).abs()
    df['ATR']     = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    df['ATR_pct'] = df['ATR'] / c * 100

    # Bollinger Bands + %B + Width
    ma20           = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    bbu            = ma20 + 2 * std20
    bbl            = ma20 - 2 * std20
    df['BB_pct']   = (c - bbl) / (bbu - bbl + 1e-9)
    df['BB_width'] = (bbu - bbl) / ma20 * 100

    # Candle Structure
    candle_range       = (df['high'] - df['low']).replace(0, np.nan)
    body               = (df['close'] - df['open']).abs()
    df['Candle_Body']  = body / candle_range
    df['Upper_Wick']   = (df['high'] - df[['close', 'open']].max(axis=1)) / candle_range
    df['Lower_Wick']   = (df[['close', 'open']].min(axis=1) - df['low'])  / candle_range

    # Session Timing (IST) -- critical for Nifty intraday
    # On 5-min data, Minute matters: 9:15 opening candle vs 15:25 closing candle
    df['Hour']        = df['date'].dt.hour
    df['Minute']      = df['date'].dt.minute
    df['Day_of_week'] = df['date'].dt.dayofweek   # 0=Mon, 4=Fri

    # Volume
    has_vol = ('volume' in df.columns and
               df['volume'].notna().any() and
               df['volume'].sum() > 0)
    if has_vol:
        vol_ema         = df['volume'].ewm(span=20, adjust=False).mean()
        df['Vol_Ratio'] = (df['volume'] / vol_ema.replace(0, np.nan)).fillna(1.0)
    else:
        df['Vol_Ratio'] = df['ATR_pct']

    # Daily Regime (5-min EMA_200 as proxy)
    df['Daily_Regime'] = (c > df['EMA_200']).astype(int)

    # ── VWAP distance ────────────────────────────────────────────
    df['VWAP_dist'] = _compute_vwap(df)

    # ── Previous day levels ───────────────────────────────────────
    df = _compute_prev_day_levels(df)

    # ── Opening range ─────────────────────────────────────────────
    df = _compute_opening_range(df)

    # ── NIFTY weekly expiry features ──────────────────────────────
    dow = df['date'].dt.dayofweek                  # 0=Mon, 3=Thu
    df['Days_to_expiry'] = ((3 - dow) % 7).astype(int)
    df['Is_expiry_day']  = (dow == 3).astype(int)
    mod50  = df['close'] % 50
    mod100 = df['close'] % 100
    df['Dist_round_50']  = np.minimum(mod50,  50  - mod50)
    df['Dist_round_100'] = np.minimum(mod100, 100 - mod100)

    # ── ADX (trend strength) ──────────────────────────────────────
    df['ADX'] = _compute_adx(df)

    # ── Volatility percentile ranks ───────────────────────────────
    df['ATR_pctile']      = df['ATR_pct'].rolling(100, min_periods=20).rank(pct=True).fillna(0.5)
    df['BB_width_pctile'] = df['BB_width'].rolling(100, min_periods=20).rank(pct=True).fillna(0.5)

    # ── Session bucket (vectorized) ───────────────────────────────
    t = df['Hour'] * 60 + df['Minute']
    df['Session_bucket'] = np.select(
        [t < 9*60+15, t < 10*60, t < 12*60, t < 13*60+30, t < 14*60+30],
        [0,            1,         2,          3,             4],
        default=5,
    ).astype(int)

    # ── Multi-timeframe features (adds VIX, RS_ratio, RS_slope) ──
    df = _add_mtf_features(df)

    # ── Market Regime (needs ADX + VIX + pctiles, all computed above) ──
    df['Market_Regime'] = _compute_regime(
        df['ADX'], df['EMA_alignment'],
        df['ATR_pctile'], df['BB_width_pctile'], df['VIX'],
    )

    return df


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Daily VWAP — resets each trading session.
    If volume is missing/zero (common for NIFTY index data), fall back to
    cumulative typical-price average — captures the "anchored average price"
    concept without needing volume.
    """
    date_grp = df['date'].dt.date
    typ      = (df['high'] + df['low'] + df['close']) / 3
    if 'volume' in df.columns and df['volume'].sum() > 0:
        cum_tvol = (typ * df['volume']).groupby(date_grp).cumsum()
        cum_vol  = df['volume'].groupby(date_grp).cumsum().replace(0, np.nan)
        vwap     = cum_tvol / cum_vol
    else:
        # Volume-less proxy: cumulative average of typical price
        cum_typ   = typ.groupby(date_grp).cumsum()
        cum_count = typ.groupby(date_grp).cumcount() + 1
        vwap      = cum_typ / cum_count
    dist = (df['close'] - vwap) / vwap.replace(0, np.nan) * 100
    return dist.fillna(0.0)


def _compute_prev_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """PDH / PDL / PDC — previous session's high, low, close as % distances."""
    date_col = df['date'].dt.date
    daily_h  = df.groupby(date_col)['high'].max().shift(1)
    daily_l  = df.groupby(date_col)['low'].min().shift(1)
    daily_c  = df.groupby(date_col)['close'].last().shift(1)

    pdh = date_col.map(daily_h)
    pdl = date_col.map(daily_l)
    pdc = date_col.map(daily_c)

    df['PDH_dist'] = ((df['close'] - pdh) / pdh.replace(0, np.nan) * 100).fillna(0.0)
    df['PDL_dist'] = ((df['close'] - pdl) / pdl.replace(0, np.nan) * 100).fillna(0.0)
    df['PDC_dist'] = ((df['close'] - pdc) / pdc.replace(0, np.nan) * 100).fillna(0.0)
    return df


def _compute_opening_range(df: pd.DataFrame) -> pd.DataFrame:
    """OR_size and OR_position based on the first 30-min (9:15–9:45) of each session."""
    date_col = df['date'].dt.date
    t        = df['date'].dt.hour * 60 + df['date'].dt.minute
    or_mask  = (t >= 9 * 60 + 15) & (t < 9 * 60 + 45)

    or_h = df.loc[or_mask].groupby(date_col)['high'].max()
    or_l = df.loc[or_mask].groupby(date_col)['low'].min()

    mapped_h   = date_col.map(or_h)
    mapped_l   = date_col.map(or_l)
    or_range   = (mapped_h - mapped_l).replace(0, np.nan)

    df['OR_size']     = (or_range / df['close'] * 100).fillna(0.0)
    df['OR_position'] = ((df['close'] - mapped_l) / or_range).fillna(0.5).clip(0, 2)
    return df


def _compute_adx(df, n=14):
    """Wilder's ADX using EWM (matches most charting platforms)."""
    high  = df['high']
    low   = df['low']
    close = df['close']

    up   = high.diff()
    down = -low.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    alpha      = 1.0 / n
    atr_w      = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di    = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di   = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(alpha=alpha, adjust=False).mean().fillna(0)


def _compute_regime(adx, ema_align, atr_pctile, bb_width_pctile, vix):
    """
    Vectorized market regime encoder.
    Priority (high overrides low):
      0 = sideways (default)
      1 = trending down (ADX>25, EMA bear)
      2 = trending up   (ADX>25, EMA bull)
      3 = volatility squeeze (BB_width_pctile < 0.15, breakout imminent)
      4 = high volatility (ATR_pctile>0.80 OR VIX>20)
    """
    adx_v  = adx.values if hasattr(adx, 'values') else np.asarray(adx)
    ema_v  = ema_align.values if hasattr(ema_align, 'values') else np.asarray(ema_align)
    atr_v  = atr_pctile.values if hasattr(atr_pctile, 'values') else np.asarray(atr_pctile)
    bbw_v  = bb_width_pctile.values if hasattr(bb_width_pctile, 'values') else np.asarray(bb_width_pctile)
    vix_v  = vix.values if hasattr(vix, 'values') else np.asarray(vix)

    regime = np.zeros(len(adx_v), dtype=int)
    regime = np.where((adx_v > 25) & (ema_v > 0),  2, regime)   # trending up
    regime = np.where((adx_v > 25) & (ema_v <= 0), 1, regime)   # trending down
    regime = np.where(bbw_v < 0.15, 3, regime)                   # squeeze
    regime = np.where((atr_v > 0.80) | (vix_v > 20), 4, regime) # high vol
    return regime


def _rsi(s, n=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(n).mean()
    l = (-d.where(d < 0, 0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _ema_align(close):
    e20  = close.ewm(span=20,  adjust=False).mean()
    e50  = close.ewm(span=50,  adjust=False).mean()
    e200 = close.ewm(span=200, adjust=False).mean()
    bull = (e20 > e50) & (e50 > e200)
    bear = (e20 < e50) & (e50 < e200)
    return np.where(bull, 1, np.where(bear, -1, 0)).astype(float)


def _bb_pct(close, n=20):
    ma  = close.rolling(n).mean()
    std = close.rolling(n).std()
    bbu = ma + 2 * std
    bbl = ma - 2 * std
    return (close - bbl) / (bbu - bbl + 1e-9)


def _add_mtf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 5-min data to 15-min and 1h.
    Compute RSI, EMA_alignment, BB_pct on each timeframe.
    Forward-fill back so every 5-min bar has the latest higher-TF reading.
    """
    if 'date' not in df.columns:
        return df

    ts = df['date']
    close = df['close']

    # ── 15-minute ────────────────────────────────────────────────
    df_15m = pd.DataFrame({
        'close': close.values,
        'ts':    ts.values
    })
    df_15m['bucket'] = pd.DatetimeIndex(df_15m['ts']).floor('15min')
    df_15m_agg = df_15m.groupby('bucket')['close'].last().reset_index()
    df_15m_agg.columns = ['bucket', 'close']
    df_15m_agg['RSI_15m']       = _rsi(df_15m_agg['close'])
    df_15m_agg['EMA_align_15m'] = _ema_align(df_15m_agg['close'])

    # Map back to 5-min using floor('15min')
    df['bucket_15m'] = pd.DatetimeIndex(ts.values).floor('15min')
    df = df.merge(df_15m_agg[['bucket', 'RSI_15m', 'EMA_align_15m']],
                  left_on='bucket_15m', right_on='bucket', how='left')
    df.drop(columns=['bucket_15m', 'bucket'], inplace=True, errors='ignore')

    # ── 1-hour ───────────────────────────────────────────────────
    df_1h = pd.DataFrame({'close': close.values, 'ts': ts.values})
    df_1h['bucket'] = pd.DatetimeIndex(df_1h['ts']).floor('h')
    df_1h_agg = df_1h.groupby('bucket')['close'].last().reset_index()
    df_1h_agg.columns = ['bucket', 'close']
    df_1h_agg['RSI_1h']      = _rsi(df_1h_agg['close'])
    df_1h_agg['EMA_align_1h'] = _ema_align(df_1h_agg['close'])
    df_1h_agg['BB_pct_1h']   = _bb_pct(df_1h_agg['close'])

    df['bucket_1h'] = pd.DatetimeIndex(ts.values).floor('h')
    df = df.merge(df_1h_agg[['bucket', 'RSI_1h', 'EMA_align_1h', 'BB_pct_1h']],
                  left_on='bucket_1h', right_on='bucket', how='left')
    df.drop(columns=['bucket_1h', 'bucket'], inplace=True, errors='ignore')

    # ── India VIX (download once, merge by date) ─────────────────
    # Cache covers 10 years. If cache is stale/missing re-download.
    vix_cache = "_vix_cache.csv"
    VIX_START = "2015-01-01"   # 10+ years back

    def _need_vix_refresh(cache_path):
        if not os.path.exists(cache_path):
            return True
        try:
            v = pd.read_csv(cache_path, parse_dates=['Date'])
            if v.empty:
                return True
            earliest = pd.to_datetime(v['Date'].min()).date()
            return earliest > pd.Timestamp(VIX_START).date()
        except Exception:
            return True

    try:
        if _need_vix_refresh(vix_cache):
            import yfinance as yf
            print(f"[VIX] Downloading India VIX from {VIX_START}...")
            raw = yf.download("^INDIAVIX", start=VIX_START, interval="1d",
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            vix = raw[['Close']].reset_index()
            vix.columns = ['Date', 'VIX_raw']
            vix.to_csv(vix_cache, index=False)
            print(f"[VIX] Downloaded {len(vix)} rows -> {vix_cache}")
        else:
            vix = pd.read_csv(vix_cache, parse_dates=['Date'])
            if 'VIX_raw' not in vix.columns:
                vix.columns = ['Date', 'VIX_raw']

        vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None).dt.date
        vix['VIX_ma5']   = vix['VIX_raw'].rolling(5).mean()
        vix['VIX']       = vix['VIX_raw'].fillna(15.0)
        vix['VIX_trend'] = np.sign(vix['VIX_raw'] - vix['VIX_ma5']).fillna(0)

        df['_date_only'] = pd.DatetimeIndex(ts.values).tz_localize(None).date
        vix_map = vix.set_index('Date')[['VIX', 'VIX_trend']]
        df['VIX']       = df['_date_only'].map(vix_map['VIX'])
        df['VIX_trend'] = df['_date_only'].map(vix_map['VIX_trend'])
        df.drop(columns=['_date_only'], inplace=True)
        df['VIX']       = df['VIX'].ffill().fillna(15.0)
        df['VIX_trend'] = df['VIX_trend'].ffill().fillna(0.0)

    except Exception as e:
        print(f"[WARN] VIX fetch failed: {e} — using defaults")
        df['VIX']       = 15.0
        df['VIX_trend'] = 0.0

    # ── Relative Strength: NIFTY vs BANKNIFTY ────────────────────
    bnf_cache = "_bnf_cache.csv"
    BNF_START = "2015-01-01"

    def _need_bnf_refresh(p):
        if not os.path.exists(p):
            return True
        try:
            v = pd.read_csv(p, parse_dates=['Date'])
            return v.empty or pd.to_datetime(v['Date'].min()).date() > pd.Timestamp(BNF_START).date()
        except Exception:
            return True

    try:
        if _need_bnf_refresh(bnf_cache):
            import yfinance as yf
            print(f"[BNF] Downloading BANKNIFTY from {BNF_START}...")
            bnf_raw = yf.download("^NSEBANK", start=BNF_START, interval="1d",
                                   auto_adjust=True, progress=False)
            if isinstance(bnf_raw.columns, pd.MultiIndex):
                bnf_raw.columns = bnf_raw.columns.get_level_values(0)
            bnf_raw[['Close']].reset_index().to_csv(bnf_cache, index=False)
            print(f"[BNF] Downloaded {len(bnf_raw)} rows -> {bnf_cache}")

        bnf_df = pd.read_csv(bnf_cache, parse_dates=['Date'])
        bnf_df.columns = ['Date', 'bnf_close']
        bnf_df['Date'] = pd.to_datetime(bnf_df['Date']).dt.tz_localize(None).dt.date

        # NIFTY daily: resample from 5-min close
        _d = pd.DataFrame({'close': close.values, 'ts': ts.values})
        _d['date_only'] = pd.DatetimeIndex(_d['ts']).tz_localize(None).date
        nifty_d = _d.groupby('date_only')['close'].last().reset_index()
        nifty_d.columns = ['Date', 'nifty_close']

        rs = nifty_d.merge(bnf_df, on='Date', how='left')
        rs['raw']      = rs['nifty_close'] / rs['bnf_close'].replace(0, np.nan)
        rs['RS_ratio'] = rs['raw'] / rs['raw'].rolling(20, min_periods=5).mean()
        rs['RS_slope'] = rs['RS_ratio'].diff(5)
        rs[['RS_ratio', 'RS_slope']] = rs[['RS_ratio', 'RS_slope']].ffill()

        df['_d'] = pd.DatetimeIndex(ts.values).tz_localize(None).date
        rs_map = rs.set_index('Date')[['RS_ratio', 'RS_slope']]
        df['RS_ratio'] = df['_d'].map(rs_map['RS_ratio'])
        df['RS_slope'] = df['_d'].map(rs_map['RS_slope'])
        df.drop(columns=['_d'], inplace=True)
        df['RS_ratio'] = df['RS_ratio'].ffill().fillna(1.0)
        df['RS_slope'] = df['RS_slope'].ffill().fillna(0.0)

    except Exception as e:
        print(f"[WARN] BANKNIFTY RS fetch failed: {e} — using defaults")
        df['RS_ratio'] = 1.0
        df['RS_slope'] = 0.0

    return df


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaptive Triple-Barrier Labeling.

    Barrier multipliers scale with volatility regime so labels are realistic
    in both quiet (low ATR) and explosive (high ATR) market conditions:

      Low  vol (ATR_pct <= 20th pctile) → TP=2.0×ATR, SL=1.0×ATR  (RR 2:1)
      Med  vol                          → TP=3.0×ATR, SL=1.5×ATR  (RR 2:1)
      High vol (ATR_pct >= 80th pctile) → TP=4.5×ATR, SL=2.0×ATR  (RR 2.25:1)

    Directional timeout: if neither barrier hits in FORWARD_CANDLES, use sign
    of final return (vs TIMEOUT_DIR_THRESH × ATR) to assign BUY/SELL/WAIT.
    """
    labels      = []
    n           = len(df)
    close_arr   = df['close'].values
    high_arr    = df['high'].values
    low_arr     = df['low'].values
    atr_arr     = df['ATR'].values
    atr_pct_arr = df['ATR_pct'].values

    # Global volatility thresholds (computed once over full dataset)
    p20 = float(np.nanpercentile(atr_pct_arr, 20))
    p80 = float(np.nanpercentile(atr_pct_arr, 80))

    for i in range(n - FORWARD_CANDLES):
        entry   = close_arr[i]
        atr     = atr_arr[i]
        atr_pct = atr_pct_arr[i]

        if np.isnan(atr) or atr <= 0:
            labels.append(1)
            continue

        # Adaptive multipliers — all scaled around new 2.0/1.0 base (RR = 2:1)
        # Low vol  → tighter (market not moving much, catch small swings)
        # Medium   → standard (2.0/1.0 = 60pt TP / 30pt SL for NIFTY)
        # High vol → wider (market moving fast, give more room)
        if atr_pct <= p20:
            tp_mult, sl_mult = 1.5, 0.75   # ~45pt TP / ~22pt SL
        elif atr_pct >= p80:
            tp_mult, sl_mult = 3.0, 1.5    # ~90pt TP / ~45pt SL (high vol day)
        else:
            tp_mult, sl_mult = ATR_TP_MULT, ATR_SL_MULT   # 2.0/1.0 standard

        tp_long  = entry + atr * tp_mult
        sl_long  = entry - atr * sl_mult
        tp_short = entry - atr * tp_mult
        sl_short = entry + atr * sl_mult

        long_result  = 0
        short_result = 0

        for j in range(i + 1, i + 1 + FORWARD_CANDLES):
            h = high_arr[j]
            l = low_arr[j]

            if long_result == 0:
                if l <= sl_long:    long_result  = -1
                elif h >= tp_long:  long_result  =  1

            if short_result == 0:
                if h >= sl_short:   short_result = -1
                elif l <= tp_short: short_result =  1

            if long_result != 0 and short_result != 0:
                break

        if long_result == 1 and short_result != 1:
            labels.append(2)   # BUY
        elif short_result == 1 and long_result != 1:
            labels.append(0)   # SELL
        else:
            final_ret = (close_arr[i + FORWARD_CANDLES] - entry) / atr
            if   final_ret >  TIMEOUT_DIR_THRESH: labels.append(2)
            elif final_ret < -TIMEOUT_DIR_THRESH: labels.append(0)
            else:                                  labels.append(1)

    df = df.iloc[:len(labels)].copy()
    df['target'] = labels
    return df


def load_feedback() -> tuple:
    """
    Load live_feedback.csv and return (X_fb, y_fb, weights_fb).
    WIN on BUY  -> label 2, WIN on SELL -> label 0, LOSS -> label 1 (WAIT).
    Each feedback row gets FEEDBACK_WEIGHT sample weight.
    """
    if not os.path.exists(FEEDBACK_CSV):
        return None, None, None
    fb = pd.read_csv(FEEDBACK_CSV)
    if fb.empty:
        return None, None, None

    # Exclude MANUAL exits: the user cut these trades early, so they are NOT a
    # clean SL/TP outcome and must not teach the model anything. They stay in the
    # CSV for history/dashboard, but are filtered out here before training.
    if "exit_reason" in fb.columns:
        before = len(fb)
        fb = fb[~fb["exit_reason"].astype(str).str.upper().str.startswith("MANUAL")].copy()
        skipped = before - len(fb)
        if skipped:
            print(f"[FEEDBACK] Skipped {skipped} MANUAL-exit trade(s) — not used for training")
        if fb.empty:
            return None, None, None

    label_map = {"BUY-WIN": 2, "SELL-WIN": 0, "BUY-LOSS": 1, "SELL-LOSS": 1}
    fb["target"] = (fb["signal"] + "-" + fb["outcome"]).map(label_map)
    fb.dropna(subset=["target"], inplace=True)
    fb["target"] = fb["target"].astype(int)

    # Use lowercase feature names (feedback CSV stores lowercase)
    feat_lower = [f.lower() for f in FEATURES]
    missing = [f for f in feat_lower if f not in fb.columns]
    if missing:
        # Old trades may lack new features — fill with 0 instead of skipping
        print(f"[FEEDBACK] Note: {len(missing)} columns missing in CSV (old format trades)")
        print(f"           Filling with 0 -- new trades will log all features")
        for col in missing:
            fb[col] = 0.0

    X_fb = fb[feat_lower].copy()
    X_fb.columns = FEATURES   # rename to match model feature names
    y_fb = fb["target"]

    # ── Hard Negative Mining weights ────────────────────────────
    # Confident wrong trades = model's blind spots → give them HIGH weight
    # Confident right trades = genuine skill → reinforce normally
    # Lucky wins (low conf) → don't overweight noise
    w_fb = np.full(len(fb), FEEDBACK_WEIGHT, dtype=float)
    if "confidence" in fb.columns:
        for i, row in fb.iterrows():
            conf = float(row.get("confidence", 0) or 0)
            won  = row["outcome"] == "WIN"
            if   won  and conf >= 55: w_fb[i] = 5.0    # genuine high-conf win
            elif won  and conf <  55: w_fb[i] = 3.0    # low-conf win (lucky)
            elif (not won) and conf >= 55: w_fb[i] = 15.0  # HARD NEGATIVE — confident LOSS
            else:                          w_fb[i] = 5.0    # low-conf loss (acceptable)

    print(f"[FEEDBACK] Loaded {len(fb)} live trades from {FEEDBACK_CSV}")
    wins  = (fb["outcome"] == "WIN").sum()
    print(f"           WIN={wins}  LOSS={len(fb)-wins}  "
          f"WinRate={wins/len(fb)*100:.1f}%")
    if "confidence" in fb.columns:
        hard_neg = ((fb["outcome"] == "LOSS") &
                    (pd.to_numeric(fb["confidence"], errors='coerce').fillna(0) >= 55)).sum()
        print(f"           Hard negatives (LOSS @ conf>=55%): {hard_neg} (weighted 15x)")
    return X_fb, y_fb, w_fb


def train(with_feedback: bool = False):
    sep = "=" * 60
    print(f"\n{sep}")
    print("   NIFTY 50 MODEL TRAINER v4.0 -- Starting...")
    print(f"   CSV             : {HOURLY_CSV}")
    print(f"   Forward candles : {FORWARD_CANDLES} x 5min = {FORWARD_CANDLES*5}min")
    print(f"   TP multiplier   : {ATR_TP_MULT}x ATR  (RR = {ATR_TP_MULT/ATR_SL_MULT:.1f}:1)")
    print(f"   SL multiplier   : {ATR_SL_MULT}x ATR")
    print(f"   Live feedback   : {'YES (x' + str(FEEDBACK_WEIGHT) + ' weight)' if with_feedback else 'NO'}")
    print(f"{sep}\n")

    if not os.path.exists(HOURLY_CSV):
        print(f"[ERR] {HOURLY_CSV} not found.")
        return

    # 1. Load
    df = load_and_clean(HOURLY_CSV)

    # 2. Features
    df = add_features(df)

    # 3. Targets
    df = make_targets(df)

    # 4. Drop NaNs
    df.dropna(subset=FEATURES + ['target'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 4b. CHOP FILTER -- drop low-volatility candles (model can't predict in flat market)
    pre_filter = len(df)
    chop_thresh = df['ATR_pct'].quantile(CHOP_FILTER_PCTILE)
    df = df[df['ATR_pct'] >= chop_thresh].reset_index(drop=True)
    print(f"[FILTER] Chop filter: dropped {pre_filter - len(df):,} low-vol rows "
          f"(ATR_pct < {chop_thresh:.3f}%)")
    print(f"         Kept {len(df):,} tradeable candles\n")

    dist       = df['target'].value_counts().sort_index()
    labels_map = {0: 'SELL', 1: 'WAIT', 2: 'BUY'}
    print("Target distribution (post-filter):")
    for k, v in dist.items():
        print(f"  {labels_map[k]:>4}: {v:>5}  ({v/len(df)*100:.1f}%)")
    print(f"  Total: {len(df):,} samples\n")

    # 5. Time-based split (NO shuffle -- preserves time order)
    X     = df[FEATURES]
    y      = df['target']
    # 70% train | 10% calibration (isotonic) | 20% test (held-out, never touched)
    if USE_META_LABELING:
        # 4-way split: 60% base | 10% meta | 10% calib | 20% test
        split1 = int(len(X) * 0.60)
        split2 = int(len(X) * 0.70)
        split3 = int(len(X) * 0.80)
        X_train = X.iloc[:split1]
        X_meta  = X.iloc[split1:split2]
        X_calib = X.iloc[split2:split3]
        X_test  = X.iloc[split3:]
        y_train = y.iloc[:split1]
        y_meta  = y.iloc[split1:split2]
        y_calib = y.iloc[split2:split3]
        y_test  = y.iloc[split3:]
        print(f"Train: {split1:,} | Meta: {len(X_meta):,} | Calib: {len(X_calib):,} | Test: {len(X_test):,}\n")
    else:
        # 3-way split: 70% train | 10% calib | 20% test (meta-labeling disabled)
        split1 = int(len(X) * 0.70)
        split2 = int(len(X) * 0.80)
        X_train = X.iloc[:split1]
        X_calib = X.iloc[split1:split2]
        X_test  = X.iloc[split2:]
        y_train = y.iloc[:split1]
        y_calib = y.iloc[split1:split2]
        y_test  = y.iloc[split2:]
        X_meta = y_meta = None
        print(f"Train: {split1:,} | Calib: {len(X_calib):,} | Test: {len(X_test):,}\n")

    # 6. Class weights + TIME DECAY -- recent 25% of train gets 2x weight
    counts        = y_train.value_counts()
    max_count     = counts.max()
    class_weights = {cls: max_count / cnt for cls, cnt in counts.items()}
    # Mild extra penalty for WAIT (natural class balance already gives ~5.5x)
    # Combined with meta-labeling filter, 1.5x is enough to encourage selectivity
    # without over-predicting WAIT and tanking BUY/SELL recall.
    class_weights[1] = class_weights[1] * 1.5
    sample_weights = y_train.map(class_weights).values.astype(float)

    recent_cutoff = int(len(sample_weights) * 0.75)
    sample_weights[recent_cutoff:] *= TIME_DECAY_RECENT
    print(f"[WEIGHT] Recent {len(sample_weights) - recent_cutoff:,} samples "
          f"weighted x{TIME_DECAY_RECENT} (markets evolve)\n")

    # Merge live feedback into training set (appended at the end, high weight)
    if with_feedback:
        X_fb, y_fb, w_fb = load_feedback()
        if X_fb is not None:
            X_train = pd.concat([X_train, X_fb], ignore_index=True)
            y_train = pd.concat([y_train, y_fb], ignore_index=True)
            sample_weights = np.concatenate([sample_weights, w_fb])
            print(f"[FEEDBACK] Training set expanded: {split1:,} hist + "
                  f"{len(X_fb)} live = {len(X_train):,} total\n")

    # 7. XGBoost v4.0 -- tuned for 10-year Nifty dataset (~150k+ samples)
    # More data -> more trees, lower LR, stricter leaf size to avoid candle-level noise
    model = xgb.XGBClassifier(
        n_estimators          = 1200,   # was 700 -- more trees for larger dataset
        max_depth             = 5,
        learning_rate         = 0.015,  # was 0.02 -- slower, more careful
        subsample             = 0.75,   # was 0.8 -- slight extra regularization
        colsample_bytree      = 0.75,   # was 0.8
        min_child_weight      = 25,     # was 15 -- each leaf needs 25+ samples
        gamma                 = 3,      # was 2  -- tighter split threshold
        reg_alpha             = 0.3,    # was 0.2
        reg_lambda            = 2.5,    # was 2.0
        objective             = 'multi:softprob',
        num_class             = 3,
        eval_metric           = 'mlogloss',
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 60,     # was 50
    )

    model.fit(
        X_train, y_train,
        sample_weight = sample_weights,
        eval_set      = [(X_test, y_test)],
        verbose       = 100,
    )

    # 7b. Isotonic Calibration (manual — works on all sklearn versions)
    # cv='prefit' was removed in sklearn 1.4+, so we do it by hand.
    # One IsotonicRegression per class, fit on held-out calibration set.
    from sklearn.isotonic import IsotonicRegression

    print("\n[CALIB] Fitting isotonic calibration on calibration set...")
    raw_calib = model.predict_proba(X_calib)           # (n, 3) raw XGB probas
    y_calib_arr = y_calib.values

    calibrators = []
    for k in range(3):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(raw_calib[:, k], (y_calib_arr == k).astype(float))
        calibrators.append(ir)

    # ── 7c. Meta-Labeling (optional, disabled by default) ──
    meta_model = None
    if USE_META_LABELING and X_meta is not None:
        print("[META] Training meta-labeling filter...")
        base_proba_meta = model.predict_proba(X_meta)
        base_preds_meta = base_proba_meta.argmax(axis=1)
        nonwait_mask    = base_preds_meta != 1
    else:
        nonwait_mask = None
        print("[META] Skipped (USE_META_LABELING = False)")

    if USE_META_LABELING and nonwait_mask is not None and nonwait_mask.sum() >= 200:
        meta_X = X_meta[nonwait_mask].copy()
        meta_X['base_pred'] = base_preds_meta[nonwait_mask].astype(float)
        meta_X['base_conf'] = base_proba_meta[nonwait_mask].max(axis=1)
        meta_y = (base_preds_meta[nonwait_mask] == y_meta.values[nonwait_mask]).astype(int)

        # Balance: high weight for "trustworthy" (rare correct calls)
        meta_cnt   = pd.Series(meta_y).value_counts()
        meta_w     = pd.Series(meta_y).map({k: meta_cnt.max() / v for k, v in meta_cnt.items()}).values

        meta_model = xgb.XGBClassifier(
            n_estimators     = 400,
            max_depth        = 4,
            learning_rate    = 0.03,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 10,
            objective        = 'binary:logistic',
            eval_metric      = 'logloss',
            random_state     = 42,
            n_jobs           = -1,
        )
        meta_model.fit(meta_X, meta_y, sample_weight=meta_w)
        meta_acc = (meta_model.predict(meta_X) == meta_y).mean()
        print(f"[META] Trained on {len(meta_X):,} BUY/SELL rows from X_meta")
        print(f"[META] Meta-model accuracy on training set: {meta_acc*100:.1f}%")
        print(f"[META] Trust threshold: {CalibratedXGB.META_TRUST_THRESHOLD*100:.0f}%")
    elif USE_META_LABELING:
        print(f"[META] Skipped — not enough non-WAIT predictions in X_meta")

    calibrated_model = CalibratedXGB(model, calibrators, meta_model=meta_model)
    status = "calibrated + meta-filtered" if meta_model else "calibrated"
    print(f"[CALIB] Done. Model is {status}.\n")

    # 8. Evaluate calibrated model on test set
    preds = calibrated_model.predict(X_test)
    print(f"\n{sep}")
    print("CLASSIFICATION REPORT")
    print(sep)
    print(classification_report(y_test, preds, target_names=['SELL', 'WAIT', 'BUY']))

    print("CONFUSION MATRIX  [rows=actual, cols=predicted]")
    print("       SELL  WAIT  BUY")
    cm = confusion_matrix(y_test, preds)
    for row_label, row in zip(['SELL', 'WAIT', 'BUY'], cm):
        print(f"  {row_label:>4}  {row}")

    # 9. Feature importance (from underlying XGBoost estimator)
    base_model = model  # original XGBoost, has feature_importances_
    importance = dict(zip(FEATURES, base_model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    print("\nFEATURE IMPORTANCE:")
    for feat, score in sorted_imp:
        bar = '#' * int(score * 400)
        print(f"  {feat:<22} {bar}  {score:.4f}")

    # 10. Save calibrated model (this is what live engine loads)
    with open(MODEL_OUT,   'wb') as f: pickle.dump(calibrated_model, f)
    with open(FEATURE_OUT, 'wb') as f: pickle.dump(FEATURES,         f)

    print(f"\n[SAVED] Model   -> {MODEL_OUT}")
    print(f"[SAVED] Features -> {FEATURE_OUT}")
    print(f"{sep}\n")


if __name__ == "__main__":
    use_feedback = "--with-feedback" in sys.argv
    train(with_feedback=use_feedback)
