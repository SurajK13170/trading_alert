import { useEffect, useMemo, useRef, useState } from "react";

const RANGES = { "1D": 24 * 60 * 60, "5D": 5 * 24 * 60 * 60, "1W": 7 * 24 * 60 * 60 };
const IST_OFFSET_MS = 5.5 * 60 * 60 * 1000;
const COLORS = {
  background: "#0b0f14",
  grid: "rgba(255,255,255,0.05)",
  daySeparator: "rgba(255,255,255,0.14)",
  text: "#8b98a8",
  green: "#19b89a",
  red: "#ff5364",
  upper: "#008f78",
  lower: "#e14355",
  crosshair: "rgba(240,180,41,0.6)",
  tagBg: "#24313a",
  tagText: "#e6edf3",
};
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const formatPrice = (value) => Number(value).toLocaleString("en-IN", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

// IST is a fixed UTC+05:30, so shifting the epoch and reading UTC fields is
// exact and far cheaper than Intl on every bar of every redraw.
const istParts = (ms) => {
  const d = new Date(Number(ms) + IST_OFFSET_MS);
  return { day: d.getUTCDate(), month: d.getUTCMonth(), hour: d.getUTCHours(), minute: d.getUTCMinutes() };
};
const pad = (n) => String(n).padStart(2, "0");
const formatDay = (ms) => { const p = istParts(ms); return `${pad(p.day)} ${MONTHS[p.month]}`; };
const formatClock = (ms) => { const p = istParts(ms); return `${pad(p.hour)}:${pad(p.minute)}`; };

// Rounded rectangle path (falls back to a plain rect on old canvases).
const roundRectPath = (ctx, x, y, w, h, r) => {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
};

// Axis tag: filled box with text, used for last price, crosshair price and time.
const drawTag = (ctx, text, x, y, bg, fg, align = "left") => {
  ctx.font = "600 10px 'IBM Plex Mono', monospace";
  const w = ctx.measureText(text).width + 12;
  const h = 18;
  const left = align === "center" ? x - w / 2 : x;
  ctx.fillStyle = bg;
  roundRectPath(ctx, left, y - h / 2, w, h, 2);
  ctx.fill();
  ctx.fillStyle = fg;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, left + w / 2, y + 0.5);
};

export default function ChartComponent({ candles = [], breakouts = [], trendlines = {}, liveCandle, feedLive }) {
  const canvasRef = useRef(null);
  const interactionRef = useRef(null);
  const [canvasWidth, setCanvasWidth] = useState(0);
  const [range, setRange] = useState("5D");
  const [hover, setHover] = useState(null); // { index, y }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      const width = Math.floor(entry.contentRect.width);
      setCanvasWidth((current) => current === width ? current : width);
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  // Closed candles + the live (tick-built) candle merged into one sorted series.
  const allBars = useMemo(() => {
    const candleMap = new Map();
    for (const bar of candles) {
      if (Number.isFinite(Number(bar.timestamp))) candleMap.set(Number(bar.timestamp), { ...bar });
    }
    if (liveCandle && Number.isFinite(Number(liveCandle.timestamp))) {
      const stamp = Number(liveCandle.timestamp);
      const prior = candleMap.get(stamp);
      candleMap.set(stamp, prior ? {
        ...liveCandle,
        open: Number(prior.open),
        high: Math.max(Number(prior.high), Number(liveCandle.high)),
        low: Math.min(Number(prior.low), Number(liveCandle.low)),
        live: true,
      } : { ...liveCandle, live: true });
    }
    return [...candleMap.values()]
      .filter((bar) => [bar.open, bar.high, bar.low, bar.close].every((value) => Number.isFinite(Number(value))))
      .sort((left, right) => Number(left.timestamp) - Number(right.timestamp));
  }, [candles, liveCandle]);

  const visibleBars = useMemo(() => {
    if (!allBars.length) return [];
    const cutoff = Number(allBars.at(-1).timestamp) - RANGES[range] * 1000;
    return allBars.filter((bar) => Number(bar.timestamp) >= cutoff);
  }, [allBars, range]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = COLORS.background;
    context.fillRect(0, 0, width, height);

    if (!visibleBars.length) {
      interactionRef.current = null;
      context.fillStyle = COLORS.text;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.font = "12px 'IBM Plex Mono', monospace";
      context.fillText("Waiting for Angel One candles", width / 2, height / 2);
      return;
    }

    const plot = { left: 8, right: width - 84, top: 16, bottom: height - 30 };
    plot.width = plot.right - plot.left;
    plot.height = plot.bottom - plot.top;
    if (plot.width <= 0 || plot.height <= 0) return;

    // ── Price scale (candles + visible trendline points) ──
    let minPrice = Math.min(...visibleBars.map((bar) => Number(bar.low)));
    let maxPrice = Math.max(...visibleBars.map((bar) => Number(bar.high)));
    const firstTime = Number(visibleBars[0].timestamp);
    const lastTime = Number(visibleBars.at(-1).timestamp);
    const linePoints = [...(trendlines.upper || []), ...(trendlines.lower || [])]
      .filter((point) => point.value !== null && Number.isFinite(Number(point.value))
        && Number(point.timestamp) >= firstTime && Number(point.timestamp) <= lastTime);
    if (linePoints.length) {
      minPrice = Math.min(minPrice, ...linePoints.map((point) => Number(point.value)));
      maxPrice = Math.max(maxPrice, ...linePoints.map((point) => Number(point.value)));
    }
    const padding = Math.max((maxPrice - minPrice) * 0.07, maxPrice * 0.0005, 1);
    minPrice -= padding;
    maxPrice += padding;
    const yForPrice = (price) => plot.bottom - ((price - minPrice) / (maxPrice - minPrice)) * plot.height;
    const priceForY = (y) => minPrice + ((plot.bottom - y) / plot.height) * (maxPrice - minPrice);
    const step = plot.width / visibleBars.length;
    const xForIndex = (index) => plot.left + step * (index + 0.5);
    const indexByTime = new Map(visibleBars.map((bar, index) => [Number(bar.timestamp), index]));

    // ── Horizontal grid + price labels ──
    context.font = "10px 'IBM Plex Mono', monospace";
    context.lineWidth = 1;
    const gridLines = 6;
    const lastPriceY = yForPrice(Number(visibleBars.at(-1).close));
    for (let index = 0; index <= gridLines; index += 1) {
      const y = Math.round(plot.top + (plot.height * index) / gridLines) + 0.5;
      const price = maxPrice - ((maxPrice - minPrice) * index) / gridLines;
      context.strokeStyle = COLORS.grid;
      context.beginPath();
      context.moveTo(plot.left, y);
      context.lineTo(plot.right, y);
      context.stroke();
      if (Math.abs(y - lastPriceY) < 14) continue;   // last-price tag sits here
      context.fillStyle = COLORS.text;
      context.textAlign = "left";
      context.textBaseline = "middle";
      context.fillText(formatPrice(price), plot.right + 10, y);
    }

    // ── Day separators + time axis ──
    // Day start: brighter line + date. Short ranges also get hourly labels.
    const hourly = visibleBars.length <= 160;
    let lastLabelX = -Infinity;
    let previousDay = null;
    context.textBaseline = "top";
    visibleBars.forEach((bar, index) => {
      const parts = istParts(bar.timestamp);
      const dayKey = `${parts.month}-${parts.day}`;
      const x = Math.round(xForIndex(index)) + 0.5;
      const newDay = dayKey !== previousDay;
      previousDay = dayKey;
      let label = null;
      if (newDay) {
        if (index > 0) {
          const sx = Math.round(plot.left + step * index) + 0.5;
          context.strokeStyle = COLORS.daySeparator;
          context.setLineDash([2, 4]);
          context.beginPath();
          context.moveTo(sx, plot.top);
          context.lineTo(sx, plot.bottom);
          context.stroke();
          context.setLineDash([]);
        }
        label = formatDay(bar.timestamp);
      } else if (hourly && parts.minute === 0) {
        context.strokeStyle = COLORS.grid;
        context.beginPath();
        context.moveTo(x, plot.top);
        context.lineTo(x, plot.bottom);
        context.stroke();
        label = formatClock(bar.timestamp);
      }
      if (label) {
        context.font = newDay ? "600 10px 'IBM Plex Mono', monospace" : "10px 'IBM Plex Mono', monospace";
        const half = context.measureText(label).width / 2;
        const labelX = Math.max(x, plot.left + half);   // keep the first label inside the chart
        if (labelX - lastLabelX >= 56) {
          context.fillStyle = newDay ? "#c3ccd6" : COLORS.text;
          context.textAlign = "center";
          context.fillText(label, labelX, plot.bottom + 9);
          lastLabelX = labelX;
        }
      }
    });

    // ── Candles ──
    const candleWidth = Math.max(1, Math.min(11, step * 0.66));
    visibleBars.forEach((bar, index) => {
      const x = Math.round(xForIndex(index)) + 0.5;
      const open = Number(bar.open);
      const close = Number(bar.close);
      const color = close >= open ? COLORS.green : COLORS.red;
      context.globalAlpha = bar.live ? 0.75 : 1;
      context.strokeStyle = color;
      context.fillStyle = color;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(x, yForPrice(Number(bar.high)));
      context.lineTo(x, yForPrice(Number(bar.low)));
      context.stroke();
      const top = yForPrice(Math.max(open, close));
      const bodyHeight = Math.max(1, Math.abs(yForPrice(open) - yForPrice(close)));
      context.fillRect(x - candleWidth / 2, top, candleWidth, bodyHeight);
      context.globalAlpha = 1;
    });

    // ── Trendlines ──
    for (const [name, color] of [["upper", COLORS.upper], ["lower", COLORS.lower]]) {
      const points = (trendlines[name] || []).map((point) => ({
        index: indexByTime.get(Number(point.timestamp)),
        value: point.value === null ? null : Number(point.value),
      }));
      if (!points.length) continue;
      context.strokeStyle = color;
      context.lineWidth = 1.6;
      context.lineJoin = "round";
      context.beginPath();
      let drawingSegment = false;
      for (const point of points) {
        if (point.index === undefined || point.value === null || !Number.isFinite(point.value)) {
          drawingSegment = false;
          continue;
        }
        const x = xForIndex(point.index);
        const y = yForPrice(point.value);
        if (!drawingSegment) context.moveTo(x, y);
        else context.lineTo(x, y);
        drawingSegment = true;
      }
      context.stroke();
    }

    // ── B markers (LuxAlgo style: BUY box below the bar, SELL box above) ──
    for (const item of breakouts) {
      const index = indexByTime.get(Number(item.timestamp));
      if (index === undefined) continue;
      const bar = visibleBars[index];
      const buy = item.signal === "BUY";
      const x = xForIndex(index);
      const anchor = yForPrice(buy ? Number(bar.low) : Number(bar.high));
      const size = 16;
      const gap = 6;
      const boxTop = buy ? anchor + gap + 4 : anchor - gap - 4 - size;
      context.fillStyle = buy ? COLORS.upper : COLORS.lower;
      roundRectPath(context, x - size / 2, boxTop, size, size, 3);
      context.fill();
      // pointer toward the candle
      context.beginPath();
      if (buy) {
        context.moveTo(x, anchor + gap);
        context.lineTo(x - 4, boxTop);
        context.lineTo(x + 4, boxTop);
      } else {
        context.moveTo(x, anchor - gap);
        context.lineTo(x - 4, boxTop + size);
        context.lineTo(x + 4, boxTop + size);
      }
      context.closePath();
      context.fill();
      context.fillStyle = "#ffffff";
      context.font = "bold 10px 'IBM Plex Mono', monospace";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText("B", x, boxTop + size / 2 + 0.5);
    }

    // ── Last price line + tag ──
    const lastBar = visibleBars.at(-1);
    const lastClose = Number(lastBar.close);
    const lastColor = lastClose >= Number(lastBar.open) ? COLORS.green : COLORS.red;
    const lastY = Math.round(lastPriceY) + 0.5;
    context.strokeStyle = lastColor;
    context.globalAlpha = 0.7;
    context.setLineDash([4, 4]);
    context.beginPath();
    context.moveTo(plot.left, lastY);
    context.lineTo(plot.right, lastY);
    context.stroke();
    context.setLineDash([]);
    context.globalAlpha = 1;
    drawTag(context, formatPrice(lastClose), plot.right + 4, lastY, lastColor, "#ffffff");

    // ── Crosshair ──
    if (hover && hover.index < visibleBars.length) {
      const x = Math.round(xForIndex(hover.index)) + 0.5;
      const y = Math.round(Math.min(Math.max(hover.y, plot.top), plot.bottom)) + 0.5;
      context.strokeStyle = COLORS.crosshair;
      context.lineWidth = 1;
      context.setLineDash([3, 4]);
      context.beginPath();
      context.moveTo(x, plot.top);
      context.lineTo(x, plot.bottom);
      context.moveTo(plot.left, y);
      context.lineTo(plot.right, y);
      context.stroke();
      context.setLineDash([]);
      drawTag(context, formatPrice(priceForY(y)), plot.right + 4, y, COLORS.tagBg, COLORS.tagText);
      const bar = visibleBars[hover.index];
      drawTag(context, `${formatDay(bar.timestamp)} ${formatClock(bar.timestamp)}`, x, plot.bottom + 14,
        COLORS.tagBg, COLORS.tagText, "center");
    }

    interactionRef.current = { plot, step, count: visibleBars.length };
  }, [visibleBars, breakouts, trendlines, hover, canvasWidth]);

  const hoveredBar = hover && hover.index < visibleBars.length ? visibleBars[hover.index] : null;
  const currentBar = hoveredBar || allBars.at(-1);
  const barChange = currentBar ? Number(currentBar.close) - Number(currentBar.open) : 0;
  const barChangePct = currentBar && Number(currentBar.open) ? (barChange / Number(currentBar.open)) * 100 : 0;

  const handlePointerMove = (event) => {
    const interaction = interactionRef.current;
    if (!interaction) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const index = Math.floor((x - interaction.plot.left) / interaction.step);
    const inside = index >= 0 && index < interaction.count && x <= interaction.plot.right;
    setHover(inside ? { index, y } : null);
  };

  return (
    <section style={{ background: COLORS.background, border: "1px solid rgba(255,255,255,0.09)", borderRadius: 4, overflow: "hidden" }}>
      <header style={{ minHeight: 60, padding: "12px 16px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div>
            <div style={{ color: "#e6edf3", fontFamily: "var(--mono)", fontSize: 13, fontWeight: 600 }}>NIFTY 50 <span style={{ color: "#667382", fontWeight: 400 }}>/ NSE</span></div>
            <div style={{ marginTop: 4, color: "#788596", fontFamily: "var(--mono)", fontSize: 10 }}>5 MIN · LUXALGO TRENDLINES WITH BREAKS</div>
          </div>
          <span style={{ color: feedLive ? COLORS.green : "#f0b429", fontFamily: "var(--mono)", fontSize: 10, whiteSpace: "nowrap" }}>{feedLive ? "● LIVE TICK" : "○ TICK UNAVAILABLE"}</span>
        </div>
        <div role="group" aria-label="Chart range" style={{ display: "flex", gap: 4, padding: 3, background: "#111820", border: "1px solid rgba(255,255,255,0.08)" }}>
          {Object.keys(RANGES).map((item) => (
            <button key={item} type="button" onClick={() => { setRange(item); setHover(null); }} aria-pressed={range === item} style={{ border: 0, padding: "6px 9px", background: range === item ? "#24313a" : "transparent", color: range === item ? "#f0b429" : COLORS.text, fontFamily: "var(--mono)", fontSize: 10, cursor: "pointer" }}>{item}</button>
          ))}
        </div>
      </header>
      <div style={{ padding: "7px 16px 0", display: "flex", gap: 14, flexWrap: "wrap", color: COLORS.text, fontFamily: "var(--mono)", fontSize: 9 }}>
        <span style={{ color: COLORS.upper }}>━ UPPER TRENDLINE</span>
        <span style={{ color: COLORS.lower }}>━ LOWER TRENDLINE</span>
        <span><span style={{ color: COLORS.upper }}>■</span> B = BUY</span>
        <span><span style={{ color: COLORS.lower }}>■</span> B = SELL</span>
      </div>
      <div style={{ minHeight: 30, padding: "7px 16px", display: "flex", gap: 14, flexWrap: "wrap", color: COLORS.text, fontFamily: "var(--mono)", fontSize: 10, borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
        {currentBar ? <>
          <span style={{ color: "#cbd4dd" }}>{formatDay(currentBar.timestamp)} {formatClock(currentBar.timestamp)}{currentBar.live ? " · LIVE" : ""}</span>
          <span>O <b style={{ color: "#cbd4dd" }}>{formatPrice(currentBar.open)}</b></span>
          <span>H <b style={{ color: COLORS.green }}>{formatPrice(currentBar.high)}</b></span>
          <span>L <b style={{ color: COLORS.red }}>{formatPrice(currentBar.low)}</b></span>
          <span>C <b style={{ color: "#e6edf3" }}>{formatPrice(currentBar.close)}</b></span>
          <span style={{ color: barChange >= 0 ? COLORS.green : COLORS.red }}>
            {barChange >= 0 ? "+" : ""}{barChange.toFixed(2)} ({barChange >= 0 ? "+" : ""}{barChangePct.toFixed(2)}%)
          </span>
        </> : <span>Waiting for Angel One candles</span>}
      </div>
      <canvas
        ref={canvasRef}
        onPointerMove={handlePointerMove}
        onPointerLeave={() => setHover(null)}
        role="img"
        aria-label="NIFTY 50 five-minute candlestick chart with live Angel One candles and LuxAlgo breakout markers"
        style={{ display: "block", width: "100%", height: "clamp(340px, 58vh, 540px)", touchAction: "pan-y", cursor: "crosshair" }}
      />
    </section>
  );
}
