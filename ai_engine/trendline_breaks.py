"""LuxAlgo Trendlines with Breaks signal calculation for closed OHLC candles."""

import numpy as np


def _calculate_slope(high, low, close, length, mult, method):
    count = len(close)
    slopes = np.full(count, np.nan, dtype=float)
    method = method.lower()

    if method == "atr":
        true_range = np.empty(count, dtype=float)
        true_range[0] = high[0] - low[0]
        true_range[1:] = np.maximum.reduce((
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ))
        if count >= length:
            slopes[length - 1] = np.mean(true_range[:length]) / length * mult
            for index in range(length, count):
                atr = (slopes[index - 1] * length * (length - 1) + true_range[index]) / length
                slopes[index] = atr / length * mult
    elif method == "stdev":
        for index in range(length - 1, count):
            window = close[index - length + 1:index + 1]
            slopes[index] = np.std(window, ddof=0) / length * mult
    elif method == "linreg":
        bar_index = np.arange(count, dtype=float)
        for index in range(length - 1, count):
            window = slice(index - length + 1, index + 1)
            source = close[window]
            bars = bar_index[window]
            variance = np.mean((bars - np.mean(bars)) ** 2)
            if variance:
                covariance = np.mean(source * bars) - np.mean(source) * np.mean(bars)
                slopes[index] = abs(covariance) / variance / 2 * mult
    else:
        raise ValueError("method must be Atr, Stdev, or Linreg")

    return slopes


def indicator_data(candles, length=14, mult=1.0, method="Atr"):
    """Return LuxAlgo B markers and non-backpainted upper/lower plot levels."""
    if length < 1 or mult < 0 or not candles:
        return {"events": [], "upper": [], "lower": []}

    try:
        high = np.asarray([float(candle["high"]) for candle in candles], dtype=float)
        low = np.asarray([float(candle["low"]) for candle in candles], dtype=float)
        close = np.asarray([float(candle["close"]) for candle in candles], dtype=float)
    except (KeyError, TypeError, ValueError):
        return {"events": [], "upper": [], "lower": []}

    count = len(close)
    if count < length * 2 + 2 or not (
        np.isfinite(high).all() and np.isfinite(low).all() and np.isfinite(close).all()
    ):
        return {"events": [], "upper": [], "lower": []}

    slopes = _calculate_slope(high, low, close, length, mult, method)
    upper = lower = slope_ph = slope_pl = 0.0
    upos = dnos = 0
    events = []
    upper_line = []
    lower_line = []
    upper_ready = lower_ready = False

    for index in range(count):
        previous_upos, previous_dnos = upos, dnos
        pivot_index = index - length
        ph = pl = None

        if pivot_index >= length:
            left = slice(pivot_index - length, pivot_index)
            right = slice(pivot_index + 1, pivot_index + length + 1)
            pivot_high = high[pivot_index]
            pivot_low = low[pivot_index]
            if pivot_high > np.max(high[left]) and pivot_high >= np.max(high[right]):
                ph = pivot_high
            if pivot_low < np.min(low[left]) and pivot_low <= np.min(low[right]):
                pl = pivot_low

        if ph is not None:
            slope_ph = slopes[index]
            upper = ph
            upper_ready = True
        else:
            upper -= slope_ph

        if pl is not None:
            slope_pl = slopes[index]
            lower = pl
            lower_ready = True
        else:
            lower += slope_pl

        upper_line.append((
            index,
            None if ph is not None or not upper_ready else float(upper - slope_ph * length),
        ))
        lower_line.append((
            index,
            None if pl is not None or not lower_ready else float(lower + slope_pl * length),
        ))

        upos = 0 if ph is not None else (
            1 if close[index] > upper - slope_ph * length else upos
        )
        dnos = 0 if pl is not None else (
            1 if close[index] < lower + slope_pl * length else dnos
        )

        if upos > previous_upos:
            events.append((index, "BUY"))
        elif dnos > previous_dnos:
            events.append((index, "SELL"))

    return {"events": events, "upper": upper_line, "lower": lower_line}


def breakout_events(candles, length=14, mult=1.0, method="Atr"):
    """Return (candle_index, direction) for each LuxAlgo B marker."""
    return indicator_data(candles, length, mult, method)["events"]


def latest_breakout(candles, length=14, mult=1.0, method="Atr"):
    """Return BUY/SELL only when the latest candle creates a new B marker."""
    events = breakout_events(candles, length, mult, method)
    if events and events[-1][0] == len(candles) - 1:
        return events[-1][1]
    return None