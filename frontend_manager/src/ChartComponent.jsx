import { useEffect, useRef, useState } from "react";

const RANGES = { "1D": 24 * 60 * 60, "5D": 5 * 24 * 60 * 60, "1W": 7 * 24 * 60 * 60 };
const COLORS = {
  background: "#0b0f14",
  grid: "rgba(255,255,255,0.08)",
  text: "#8b98a8",
  green: "#19b89a",
  red: "#ff5364",
  upper: "#008f78",
  lower: "#e14355",
  crosshair: "rgba(240,180,41,0.72)",
};

const formatPrice = (value) => Number(value).toLocaleString("en-IN", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const formatMarketTime = (seconds, includeDate = false) => new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata",
  ...(includeDate ? { day: "2-digit", month: "short" } : {}),
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
}).format(new Date(seconds * 1000));

export default function ChartComponent({ candles = [], breakouts = [], trendlines = {}, liveCandle, feedLive }) {
  const canvasRef = useRef(null);
  const interactionRef = useRef(null);
  const hoverIndexRef = useRef(null);
  const [canvasWidth, setCanvasWidth] = useState(0);
  const [range, setRange] = useState("5D");
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [hoveredBar, setHoveredBar] = useState(null);

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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

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
      } : { ...liveCandle });
    }

    const allBars = [...candleMap.values()]
      .filter((bar) => [bar.open, bar.high, bar.low, bar.close].every((value) => Number.isFinite(Number(value))))
      .sort((left, right) => Number(left.timestamp) - Number(right.timestamp));
    if (!allBars.length) {
      const context = canvas.getContext("2d");
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = COLORS.background;
      context.fillRect(0, 0, width, height);
      context.fillStyle = COLORS.text;
      context.textAlign = "center";
      context.font = "12px 'IBM Plex Mono', monospace";
      context.fillText("Waiting for Angel One candles", width / 2, height / 2);
      return;
    }

    const latestSeconds = Math.floor(Number(allBars.at(-1).timestamp) / 1000);
    const cutoff = latestSeconds - RANGES[range];
    const visibleBars = allBars.filter((bar) => Math.floor(Number(bar.timestamp) / 1000) >= cutoff);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = COLORS.background;
    context.fillRect(0, 0, width, height);

    const plot = { left: 12, right: width - 82, top: 18, bottom: height - 34 };
    plot.width = plot.right - plot.left;
    plot.height = plot.bottom - plot.top;
    if (plot.width <= 0 || plot.height <= 0) return;

    let minPrice = Math.min(...visibleBars.map((bar) => Number(bar.low)));
    let maxPrice = Math.max(...visibleBars.map((bar) => Number(bar.high)));
    const firstTime = Math.floor(Number(visibleBars[0].timestamp) / 1000);
    const lastTime = Math.floor(Number(visibleBars.at(-1).timestamp) / 1000);
    const linePoints = [...(trendlines.upper || []), ...(trendlines.lower || [])]
      .filter((point) => point.value !== null && Number.isFinite(Number(point.value))
        && Number(point.timestamp) / 1000 >= firstTime && Number(point.timestamp) / 1000 <= lastTime);
    if (linePoints.length) {
      minPrice = Math.min(minPrice, ...linePoints.map((point) => Number(point.value)));
      maxPrice = Math.max(maxPrice, ...linePoints.map((point) => Number(point.value)));
    }
    const padding = Math.max((maxPrice - minPrice) * 0.06, maxPrice * 0.0005, 1);
    minPrice -= padding;
    maxPrice += padding;
    const yForPrice = (price) => plot.bottom - ((price - minPrice) / (maxPrice - minPrice)) * plot.height;
    const step = plot.width / visibleBars.length;
    const xForIndex = (index) => plot.left + step * (index + 0.5);
    const indexByTime = new Map(visibleBars.map((bar, index) => [Math.floor(Number(bar.timestamp) / 1000), index]));

    context.font = "10px 'IBM Plex Mono', monospace";
    context.textBaseline = "middle";
    context.lineWidth = 1;
    for (let index = 0; index <= 4; index += 1) {
      const y = plot.top + (plot.height * index) / 4;
      const price = maxPrice - ((maxPrice - minPrice) * index) / 4;
      context.strokeStyle = COLORS.grid;
      context.beginPath();
      context.moveTo(plot.left, y);
      context.lineTo(plot.right, y);
      context.stroke();
      context.fillStyle = COLORS.text;
      context.textAlign = "left";
      context.fillText(formatPrice(price), plot.right + 9, y);
    }

    const candleWidth = Math.max(1, Math.min(12, step * 0.62));
    visibleBars.forEach((bar, index) => {
      const x = xForIndex(index);
      const open = Number(bar.open);
      const close = Number(bar.close);
      const high = Number(bar.high);
      const low = Number(bar.low);
      const color = close >= open ? COLORS.green : COLORS.red;
      context.strokeStyle = color;
      context.fillStyle = color;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(x, yForPrice(high));
      context.lineTo(x, yForPrice(low));
      context.stroke();
      const top = yForPrice(Math.max(open, close));
      const bodyHeight = Math.max(1, Math.abs(yForPrice(open) - yForPrice(close)));
      context.fillRect(x - candleWidth / 2, top, candleWidth, bodyHeight);
    });

    for (const [name, color] of [["upper", COLORS.upper], ["lower", COLORS.lower]]) {
      const points = (trendlines[name] || [])
        .map((point) => ({
          index: indexByTime.get(Math.floor(Number(point.timestamp) / 1000)),
          value: point.value === null ? null : Number(point.value),
        }));
      if (!points.length) continue;
      context.strokeStyle = color;
      context.lineWidth = 1.6;
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

    for (const item of breakouts) {
      const index = indexByTime.get(Math.floor(Number(item.timestamp) / 1000));
      if (index === undefined) continue;
      const bar = visibleBars[index];
      const buy = item.signal === "BUY";
      const x = xForIndex(index);
      const tipY = yForPrice(buy ? Number(bar.low) : Number(bar.high)) + (buy ? 13 : -13);
      context.fillStyle = buy ? COLORS.upper : COLORS.lower;
      context.beginPath();
      context.moveTo(x, tipY + (buy ? -7 : 7));
      context.lineTo(x - 5, tipY + (buy ? 1 : -1));
      context.lineTo(x + 5, tipY + (buy ? 1 : -1));
      context.closePath();
      context.fill();
      context.font = "bold 10px 'IBM Plex Mono', monospace";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText("B", x, tipY + (buy ? 7 : -7));
    }

    const labelCount = Math.min(6, visibleBars.length);
    context.fillStyle = COLORS.text;
    context.font = "9px 'IBM Plex Mono', monospace";
    context.textAlign = "center";
    context.textBaseline = "top";
    for (let index = 0; index < labelCount; index += 1) {
      const barIndex = Math.round((index * (visibleBars.length - 1)) / Math.max(labelCount - 1, 1));
      const x = xForIndex(barIndex);
      context.fillText(formatMarketTime(Math.floor(Number(visibleBars[barIndex].timestamp) / 1000), true), x, plot.bottom + 9);
    }

    const selectedIndex = hoveredIndex ?? visibleBars.length - 1;
    if (hoveredIndex !== null && hoveredIndex < visibleBars.length) {
      const x = xForIndex(hoveredIndex);
      context.strokeStyle = COLORS.crosshair;
      context.setLineDash([3, 4]);
      context.beginPath();
      context.moveTo(x, plot.top);
      context.lineTo(x, plot.bottom);
      context.stroke();
      context.setLineDash([]);
    }
    interactionRef.current = { visibleBars, plot, step };
    hoverIndexRef.current = selectedIndex;
  }, [candles, breakouts, trendlines, liveCandle, feedLive, range, hoveredIndex, canvasWidth]);

  const currentBar = hoveredBar || liveCandle || candles.at(-1);

  const handlePointerMove = (event) => {
    const interaction = interactionRef.current;
    if (!interaction) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const index = Math.floor((x - interaction.plot.left) / interaction.step);
    const next = index >= 0 && index < interaction.visibleBars.length ? index : null;
    if (next !== hoverIndexRef.current) {
      hoverIndexRef.current = next;
      setHoveredIndex(next);
      setHoveredBar(next === null ? null : interaction.visibleBars[next]);
    }
  };

  const handlePointerLeave = () => {
    hoverIndexRef.current = null;
    setHoveredIndex(null);
    setHoveredBar(null);
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
            <button key={item} type="button" onClick={() => { setRange(item); setHoveredIndex(null); setHoveredBar(null); }} aria-pressed={range === item} style={{ border: 0, padding: "6px 9px", background: range === item ? "#24313a" : "transparent", color: range === item ? "#f0b429" : COLORS.text, fontFamily: "var(--mono)", fontSize: 10, cursor: "pointer" }}>{item}</button>
          ))}
        </div>
      </header>
      <div style={{ padding: "7px 16px 0", display: "flex", gap: 14, flexWrap: "wrap", color: COLORS.text, fontFamily: "var(--mono)", fontSize: 9 }}>
        <span style={{ color: COLORS.upper }}>━ UPPER TRENDLINE</span>
        <span style={{ color: COLORS.lower }}>━ LOWER TRENDLINE</span>
        <span>▲ B = BUY</span><span>▼ B = SELL</span>
      </div>
      <div style={{ height: 30, padding: "7px 16px", display: "flex", gap: 14, color: COLORS.text, fontFamily: "var(--mono)", fontSize: 10, borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
        {currentBar ? <>
          <span>O <b style={{ color: "#cbd4dd" }}>{formatPrice(currentBar.open)}</b></span>
          <span>H <b style={{ color: COLORS.green }}>{formatPrice(currentBar.high)}</b></span>
          <span>L <b style={{ color: COLORS.red }}>{formatPrice(currentBar.low)}</b></span>
          <span>C <b style={{ color: "#e6edf3" }}>{formatPrice(currentBar.close)}</b></span>
        </> : <span>Waiting for Angel One candles</span>}
      </div>
      <canvas
        ref={canvasRef}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        role="img"
        aria-label="NIFTY 50 five-minute candlestick chart with live Angel One candles and LuxAlgo breakout markers"
        style={{ display: "block", width: "100%", height: 420, touchAction: "pan-y", cursor: "crosshair" }}
      />
    </section>
  );
}