import { useEffect, useState } from "react";
import axios from "axios";
import {
  Activity,
  ShieldAlert,
  Target,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Wallet,
  TrendingUp,
  Zap,
  X,
} from "lucide-react";
import ChartComponent from "./ChartComponent.jsx";

/* ─── CSS-in-JS keyframes injected once ─── */
const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Bebas+Neue&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #06080d;
    --surface:   #0c1018;
    --surface2:  #111722;
    --border:    rgba(255,255,255,0.07);
    --border2:   rgba(255,255,255,0.12);
    --gold:      #f0b429;
    --gold2:     #ffd166;
    --cyan:      #00d4ff;
    --green:     #00e676;
    --red:       #ff3d57;
    --dim:       #4a5568;
    --muted:     #718096;
    --text:      #e2e8f0;
    --mono:      'IBM Plex Mono', monospace;
    --display:   'Bebas Neue', sans-serif;
    --sans:      'IBM Plex Sans', sans-serif;
  }

  body { background: var(--bg); }

  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes ping  { 75%,100% { transform: scale(2); opacity: 0; } }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .4; } }
  @keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
  }
  @keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes glow-gold {
    0%,100% { box-shadow: 0 0 20px rgba(240,180,41,0.15); }
    50%     { box-shadow: 0 0 40px rgba(240,180,41,0.35); }
  }
  @keyframes glow-green {
    0%,100% { box-shadow: 0 0 20px rgba(0,230,118,0.15); }
    50%     { box-shadow: 0 0 40px rgba(0,230,118,0.35); }
  }
  @keyframes glow-red {
    0%,100% { box-shadow: 0 0 20px rgba(255,61,87,0.15); }
    50%     { box-shadow: 0 0 40px rgba(255,61,87,0.35); }
  }
  @keyframes bar-fill {
    from { width: 0%; }
    to   { width: var(--target-width); }
  }

  .scanline-overlay {
    pointer-events: none;
    position: fixed; inset: 0; z-index: 999;
    background: repeating-linear-gradient(
      to bottom,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.07) 2px,
      rgba(0,0,0,0.07) 4px
    );
  }
  .scan-beam {
    pointer-events: none;
    position: fixed; left: 0; right: 0; height: 2px; z-index: 1000;
    background: linear-gradient(to right, transparent, rgba(0,212,255,0.06), transparent);
    animation: scanline 6s linear infinite;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.5s ease both;
  }
  .card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(to right, transparent, rgba(240,180,41,0.4), transparent);
  }
  .card-inner {
    padding: 28px;
  }

  .label {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .val-huge {
    font-family: var(--display);
    font-size: 72px;
    line-height: 1;
    letter-spacing: 0.02em;
    color: var(--text);
  }

  .val-large {
    font-family: var(--display);
    font-size: 52px;
    line-height: 1;
    letter-spacing: 0.02em;
  }

  .signal-text {
    font-family: var(--display);
    font-size: 96px;
    line-height: 1;
    letter-spacing: 0.04em;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 2px;
    border: 1px solid;
  }

  .dot-live {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    position: relative;
  }
  .dot-live::after {
    content: '';
    position: absolute; inset: -3px;
    border-radius: 50%;
    background: var(--green);
    animation: ping 1.5s ease-in-out infinite;
    opacity: 0;
  }

  .progress-track {
    height: 4px;
    background: var(--surface2);
    border-radius: 0;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(to right, var(--gold), var(--cyan));
    transition: width 1s ease;
    position: relative;
  }
  .progress-fill::after {
    content: '';
    position: absolute; right: 0; top: -2px; bottom: -2px; width: 3px;
    background: white;
    box-shadow: 0 0 8px 2px white;
  }

  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
  }

  .data-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 12px;
  }
  .data-row:last-child { border-bottom: none; }

  .corner-mark {
    position: absolute;
    width: 12px; height: 12px;
    border-color: var(--gold);
    border-style: solid;
    opacity: 0.5;
  }
  .corner-tl { top: 8px; left: 8px; border-width: 1px 0 0 1px; }
  .corner-tr { top: 8px; right: 8px; border-width: 1px 1px 0 0; }
  .corner-bl { bottom: 8px; left: 8px; border-width: 0 0 1px 1px; }
  .corner-br { bottom: 8px; right: 8px; border-width: 0 1px 1px 0; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }

  @media (max-width: 640px) {
    .grid-2, .grid-3 { grid-template-columns: 1fr; }
    .signal-text { font-size: 64px; }
    .val-huge { font-size: 52px; }
    .val-large { font-size: 38px; }
  }

  .ticker-wrap {
    overflow: hidden;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    padding: 8px 0;
  }
  .ticker-inner {
    display: flex;
    white-space: nowrap;
    animation: ticker 30s linear infinite;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    gap: 0;
  }
  .ticker-item {
    padding: 0 32px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-right: 1px solid var(--border);
  }
  .ticker-item span.up   { color: var(--green); }
  .ticker-item span.down { color: var(--red); }
  .ticker-item span.name { color: var(--text); font-weight: 600; letter-spacing: 0.08em; }

  .main-layout {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 20px;
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px 24px 40px;
  }
  @media (max-width: 1024px) {
    .main-layout { grid-template-columns: 1fr; }
  }

  .left-col { display: flex; flex-direction: column; gap: 20px; }
  .right-col { display: flex; flex-direction: column; gap: 20px; }

  .pnl-positive { color: var(--green); animation: glow-green 2s ease-in-out infinite; border-color: rgba(0,230,118,0.2) !important; background: rgba(0,230,118,0.05) !important; }
  .pnl-negative { color: var(--red);   animation: glow-red   2s ease-in-out infinite; border-color: rgba(255,61,87,0.2) !important;  background: rgba(255,61,87,0.05) !important; }

  .signal-buy  { color: var(--green); text-shadow: 0 0 40px rgba(0,230,118,0.6); }
  .signal-sell { color: var(--red);   text-shadow: 0 0 40px rgba(255,61,87,0.6); }
  .signal-wait { color: var(--muted); }

  .signal-card-buy  { border-color: rgba(0,230,118,0.25) !important; animation: glow-green 3s ease-in-out infinite; }
  .signal-card-sell { border-color: rgba(255,61,87,0.25)  !important; animation: glow-red   3s ease-in-out infinite; }

  .icon-box {
    width: 40px; height: 40px;
    display: flex; align-items: center; justify-content: center;
    border: 1px solid var(--border2);
    background: var(--surface2);
    border-radius: 2px;
    flex-shrink: 0;
  }

  .navbar {
    position: sticky; top: 0; z-index: 50;
    background: rgba(6,8,13,0.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    padding: 0 24px;
    height: 64px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .navbar-logo {
    font-family: var(--display);
    font-size: 28px;
    letter-spacing: 0.08em;
    color: var(--text);
    display: flex; align-items: center; gap: 10px;
  }
  .navbar-logo span { color: var(--gold); }
  .navbar-right {
    display: flex; align-items: center; gap: 24px;
  }
  .nav-clock {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    display: flex; align-items: center; gap: 8px;
  }
  .nav-status {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    display: flex; align-items: center; gap: 6px;
    color: var(--green);
    padding: 5px 12px;
    border: 1px solid rgba(0,230,118,0.25);
    border-radius: 2px;
    background: rgba(0,230,118,0.05);
  }

  .section-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px;
  }
  .section-title {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    display: flex; align-items: center; gap: 10px;
  }
  .section-title::before {
    content: '';
    display: inline-block;
    width: 16px; height: 1px;
    background: var(--gold);
  }

  .mini-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 16px 20px;
  }

  .loading-screen {
    min-height: 100vh;
    background: var(--bg);
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
    gap: 24px;
    font-family: var(--mono);
  }
  .spinner {
    width: 48px; height: 48px;
    border: 2px solid var(--surface2);
    border-top-color: var(--gold);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .trades-table {
    width: 100%;
    border-collapse: collapse;
  }
  .trades-table th {
    text-align: left;
    padding: 8px 14px;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    font-weight: 600;
  }
  .trades-table td {
    padding: 10px 14px;
    font-family: var(--mono);
    font-size: 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }
  .trades-table tr:last-child td { border-bottom: none; }
  .trade-row-open  { background: rgba(240,180,41,0.04); }
  .trade-row-win   { background: rgba(0,230,118,0.05); }
  .trade-row-loss  { background: rgba(255,61,87,0.05); }
  .trade-row-open:hover  { background: rgba(240,180,41,0.09); }
  .trade-row-win:hover   { background: rgba(0,230,118,0.09); }
  .trade-row-loss:hover  { background: rgba(255,61,87,0.09); }
  .outcome-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border: 1px solid;
  }
`;

function StyleInjector() {
  useEffect(() => {
    const tag = document.createElement("style");
    tag.textContent = STYLES;
    document.head.appendChild(tag);
    return () => document.head.removeChild(tag);
  }, []);
  return null;
}

function Ticker({ price, change, feedLive }) {
  return (
    <div className="ticker-wrap">
      <div className="ticker-item" style={{ justifyContent: "space-between", maxWidth: 1400, margin: "0 auto" }}>
        <span className="name">NIFTY 50</span>
        <span style={{ color: "var(--text)" }}>₹{Number(price || 0).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
        <span className={change >= 0 ? "up" : "down"}>{change >= 0 ? "+" : ""}{Number(change || 0).toFixed(2)} pts</span>
        <span className="name">{feedLive ? "ANGEL ONE · LIVE TICK" : "TICK FEED UNAVAILABLE"}</span>
      </div>
    </div>
  );
}

function CornerMarks() {
  return (
    <>
      <div className="corner-mark corner-tl" />
      <div className="corner-mark corner-tr" />
      <div className="corner-mark corner-bl" />
      <div className="corner-mark corner-br" />
    </>
  );
}

export default function App() {
  const [data, setData] = useState(null);
  const [tick, setTick] = useState(null);
  const [activeTrade, setActiveTrade] = useState(null);
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [time, setTime] = useState(new Date());
  // Manual exit
  const [busy, setBusy] = useState(false);
  const [cmdMsg, setCmdMsg] = useState(null); // { text, ok }
  // Execution mode (PAPER vs LIVE)
  const [mode, setMode] = useState("PAPER");
  const [hasAngelCreds, setHasAngelCreds] = useState(false);
  const [modeBusy, setModeBusy] = useState(false);
  const [showLiveConfirm, setShowLiveConfirm] = useState(false);
  const [liveTypeInput, setLiveTypeInput] = useState("");
  // Auto-trade state
  const [autoState, setAutoState] = useState("RUNNING");
  const [pendingApproval, setPendingApproval] = useState(false);

  const symbol = "Nifty50";
  const API_BASE = "http://localhost:3000";

  const fetchData = async () => {
    try {
      const results = await Promise.allSettled([
        axios.get(`${API_BASE}/analysis/${symbol}`),
        axios.get(`${API_BASE}/active-trade/${symbol}`),
        axios.get(`${API_BASE}/trades/${symbol}`),
        axios.get(`${API_BASE}/mode/${symbol}`),
        axios.get(`${API_BASE}/control/${symbol}`),
        axios.get(`${API_BASE}/market-tick/${symbol}`),
      ]);
      const [analysisRes, tradeRes, tradesRes, modeRes, controlRes, tickRes] = results;

      if (analysisRes.status === "fulfilled") setData(analysisRes.value.data);
      if (tradeRes.status === "fulfilled") setActiveTrade(tradeRes.value.data);
      if (tradesRes.status === "fulfilled") setTrades(tradesRes.value.data || []);
      else setTrades([]); // backend may not have /trades endpoint yet
      if (modeRes.status === "fulfilled") {
        setMode(modeRes.value.data?.mode || "PAPER");
        setHasAngelCreds(!!modeRes.value.data?.hasAngelCreds);
      }
      if (controlRes.status === "fulfilled") {
        const c = controlRes.value.data;
        setAutoState(c?.auto_state || "RUNNING");
        setPendingApproval(!!c?.pending_approval);
      }
      setTick(tickRes.status === "fulfilled" && !tickRes.value.data?.error ? tickRes.value.data : null);

      // only show error screen if BOTH critical endpoints failed
      if (analysisRes.status === "rejected" && tradeRes.status === "rejected") {
        setError("Server is offline. Check if all engines are running.");
      } else {
        setError(null);
      }
    } catch {
      setError("Server is offline. Check if all engines are running.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const initialFetch = setTimeout(fetchData, 0);
    const interval = setInterval(fetchData, 3000);
    const clock = setInterval(() => setTime(new Date()), 1000);
    return () => { clearTimeout(initialFetch); clearInterval(interval); clearInterval(clock); };
  }, []);

  const setExecutionMode = async (target) => {
    // target: "PAPER" or "LIVE"
    setModeBusy(true);
    try {
      const r = await axios.post(`${API_BASE}/mode/${symbol}`, { mode: target });
      setMode(r.data.mode);
      setCmdMsg({ text: `Mode set to ${r.data.mode}. Execution manager will pick up next cycle.`, ok: true });
    } catch (e) {
      setCmdMsg({ text: e.response?.data?.error || "Mode change failed", ok: false });
    } finally {
      setModeBusy(false);
      setShowLiveConfirm(false);
      setLiveTypeInput("");
      setTimeout(() => setCmdMsg(null), 5000);
    }
  };

  const handleToggleMode = () => {
    if (activeTrade?.status === "OPEN") {
      setCmdMsg({ text: "Close the open trade before switching mode.", ok: false });
      setTimeout(() => setCmdMsg(null), 4000);
      return;
    }
    if (mode === "LIVE") {
      // Switching back to PAPER -- safe, just confirm
      if (!window.confirm("Switch back to PAPER (virtual money) mode?")) return;
      setExecutionMode("PAPER");
    } else {
      // Switching to LIVE -- show type-LIVE confirmation modal
      setLiveTypeInput("");
      setShowLiveConfirm(true);
    }
  };

  const setAutoTrade = async (target) => {
    // target: "RUNNING" | "PAUSED"
    try {
      const r = await axios.post(`${API_BASE}/control/${symbol}/auto`, { state: target });
      setAutoState(r.data.auto_state);
      setCmdMsg({ text: `Auto-trade ${r.data.auto_state}`, ok: true });
    } catch (e) {
      setCmdMsg({ text: e.response?.data?.error || "Auto-trade toggle failed", ok: false });
    } finally {
      setTimeout(() => setCmdMsg(null), 4000);
    }
  };

  const approveNextTrade = async () => {
    try {
      await axios.post(`${API_BASE}/control/${symbol}/approve`);
      setPendingApproval(true);
      setCmdMsg({ text: "Approved -- next valid signal within 60s will execute", ok: true });
    } catch (e) {
      setCmdMsg({ text: e.response?.data?.error || "Approve failed", ok: false });
    } finally {
      setTimeout(() => setCmdMsg(null), 4000);
    }
  };

  const handleExit = async () => {
    if (!window.confirm("Close this trade at market price right now?")) return;
    setBusy(true);
    try {
      await axios.post(`${API_BASE}/trade/exit/${symbol}`);
      setCmdMsg({ text: "Exit command sent. Execution manager will close on next cycle.", ok: true });
    } catch (e) {
      setCmdMsg({ text: e.response?.data?.error || "Exit failed", ok: false });
    } finally {
      setBusy(false);
      setTimeout(() => setCmdMsg(null), 4000);
    }
  };

  if (loading)
    return (
      <>
        <StyleInjector />
        <div className="loading-screen" style={{ color: "var(--text)" }}>
          <div className="spinner" />
          <div style={{ textAlign: "center" }}>
            <div style={{ fontFamily: "var(--display)", fontSize: 32, letterSpacing: "0.1em", color: "var(--gold)" }}>QUANT AI</div>
            <div style={{ fontSize: 11, letterSpacing: "0.18em", color: "var(--muted)", marginTop: 8 }}>INITIALIZING MARKET ENGINES</div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {[0,1,2].map(i => (
              <div key={i} style={{
                width: 6, height: 6, borderRadius: "50%", background: "var(--gold)",
                animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`
              }} />
            ))}
          </div>
        </div>
      </>
    );

  if (error)
    return (
      <>
        <StyleInjector />
        <div className="loading-screen" style={{ color: "var(--text)" }}>
          <div className="card" style={{ maxWidth: 400, width: "100%", margin: "0 24px" }}>
            <CornerMarks />
            <div className="card-inner" style={{ textAlign: "center" }}>
              <ShieldAlert size={48} style={{ color: "var(--red)", margin: "0 auto 20px" }} />
              <div style={{ fontFamily: "var(--display)", fontSize: 36, letterSpacing: "0.06em", marginBottom: 12 }}>
                LINK SEVERED
              </div>
              <p style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--muted)", lineHeight: 1.7, marginBottom: 28 }}>
                {error}
              </p>
              <button
                onClick={() => window.location.reload()}
                style={{
                  fontFamily: "var(--mono)", fontSize: 11, letterSpacing: "0.16em",
                  textTransform: "uppercase", fontWeight: 600,
                  background: "var(--gold)", color: "#06080d",
                  border: "none", borderRadius: 2, padding: "12px 32px",
                  cursor: "pointer", width: "100%",
                }}
              >
                Reconnect
              </button>
            </div>
          </div>
        </div>
      </>
    );

  // Price lives in quant payload (quant engine + replay engine both write it there).
  // data.market is a legacy path that nobody writes to anymore.
  const tickAgeSec = tick?.ts ? time.getTime() / 1000 - Number(tick.ts) : Infinity;
  const feedLive = tickAgeSec >= 0 && tickAgeSec <= 20;
  const candles = Array.isArray(data?.candles) ? data.candles : [];
  const breakouts = Array.isArray(data?.quant?.breakouts) ? data.quant.breakouts : [];
  const candleChange = candles.length > 1
    ? Number(candles.at(-1).close) - Number(candles.at(-2).close)
    : 0;
  const price = Number((feedLive ? tick.price : null) ?? data?.quant?.price ?? data?.market?.last_price ?? 0);
  const confidence = data?.signal?.confidence || "0%";
  const signal = data?.signal?.signal || "WAIT";
  const target = data?.signal?.take_profit || "N/A";
  const sl = data?.signal?.stop_loss || "N/A";

  let pnl = 0;
  if (activeTrade?.status === "OPEN") {
    // Prefer the execution manager's REAL premium-based unrealized P&L (in ₹,
    // streamed via tick_collector). Fall back to a spot-point estimate only when
    // the live premium tick isn't available yet.
    if (activeTrade.unrealized_pnl !== undefined && activeTrade.unrealized_pnl !== null) {
      pnl = Number(activeTrade.unrealized_pnl);
    } else {
      pnl = activeTrade.position === "BUY"
        ? price - activeTrade.entry
        : activeTrade.entry - price;
    }
  }

  const sigClass = signal === "BUY" ? "signal-buy" : signal === "SELL" ? "signal-sell" : "signal-wait";
  const cardSigClass = signal === "BUY" ? "signal-card-buy" : signal === "SELL" ? "signal-card-sell" : "";

  const timeStr = time.toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const dateStr = time.toLocaleDateString("en-IN", { timeZone: "Asia/Kolkata", day: "2-digit", month: "short", year: "numeric" });

  return (
    <>
      <StyleInjector />
      <div className="scanline-overlay" />
      <div className="scan-beam" />

      {/* Navbar */}
      <nav className="navbar">
        <div className="navbar-logo">
          <div className="icon-box">
            <Activity size={18} color="var(--gold)" />
          </div>
          QUANT <span>AI</span>
        </div>
        <div className="navbar-right">
          <div className="nav-clock">
            <Clock size={13} color="var(--dim)" />
            <span>{dateStr} &nbsp;{timeStr}</span>
          </div>

          {/* PAPER/LIVE toggle -- replaces the old static "LIVE" badge */}
          <button
            onClick={handleToggleMode}
            disabled={modeBusy || activeTrade?.status === "OPEN"}
            title={activeTrade?.status === "OPEN" ? "Close current trade to switch mode" : "Toggle PAPER / LIVE"}
            style={{
              fontFamily: "var(--mono)",
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              padding: "6px 14px",
              borderRadius: 2,
              cursor: (modeBusy || activeTrade?.status === "OPEN") ? "not-allowed" : "pointer",
              border: `1px solid ${mode === "LIVE" ? "rgba(255,61,87,0.5)" : "rgba(0,212,255,0.4)"}`,
              background: mode === "LIVE" ? "rgba(255,61,87,0.1)" : "rgba(0,212,255,0.06)",
              color: mode === "LIVE" ? "var(--red)" : "var(--cyan)",
              display: "flex",
              alignItems: "center",
              gap: 8,
              opacity: (modeBusy || activeTrade?.status === "OPEN") ? 0.5 : 1,
              ...(mode === "LIVE" ? { animation: "glow-red 2.5s ease-in-out infinite" } : {}),
            }}
          >
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: mode === "LIVE" ? "var(--red)" : "var(--cyan)",
            }} />
            {mode === "LIVE" ? "LIVE -- REAL MONEY" : "PAPER MODE"}
          </button>

          {/* Auto-trade RUNNING/PAUSED toggle */}
          <button
            onClick={() => setAutoTrade(autoState === "RUNNING" ? "PAUSED" : "RUNNING")}
            title={autoState === "RUNNING" ? "Pause auto-trading (signals keep flowing)" : "Resume auto-trading"}
            style={{
              fontFamily: "var(--mono)",
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              padding: "6px 12px",
              borderRadius: 2,
              cursor: "pointer",
              border: `1px solid ${autoState === "RUNNING" ? "rgba(0,230,118,0.4)" : "rgba(240,180,41,0.4)"}`,
              background: autoState === "RUNNING" ? "rgba(0,230,118,0.06)" : "rgba(240,180,41,0.08)",
              color:  autoState === "RUNNING" ? "var(--green)" : "var(--gold)",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: autoState === "RUNNING" ? "var(--green)" : "var(--gold)",
            }} />
            {autoState === "RUNNING" ? "AUTO ON" : "AUTO PAUSED"}
          </button>
        </div>
      </nav>

      {/* LIVE mode banner -- spans across page top under navbar */}
      {mode === "LIVE" && (
        <div style={{
          background: "rgba(255,61,87,0.12)",
          borderBottom: "1px solid rgba(255,61,87,0.4)",
          padding: "10px 24px",
          fontFamily: "var(--mono)",
          fontSize: 12,
          letterSpacing: "0.16em",
          color: "var(--red)",
          fontWeight: 700,
          textTransform: "uppercase",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 20,
          flexWrap: "wrap",
        }}>
          <span>
            [WARNING] LIVE TRADING MODE -- REAL MONEY
            {Number(activeTrade?.angel_balance) === 0 && (
              <span style={{ marginLeft: 12, padding: "2px 8px", background: "rgba(255,255,255,0.08)", borderRadius: 2, fontSize: 10 }}>
                INSUFFICIENT FUNDS -- no trades will fire
              </span>
            )}
          </span>
          <span style={{ display: "flex", gap: 24, fontSize: 11 }}>
            {activeTrade?.angel_balance !== null && activeTrade?.angel_balance !== undefined ? (
              <span>
                <span style={{ color: "rgba(255,61,87,0.6)" }}>Angel Balance:</span>{" "}
                ₹{Number(activeTrade.angel_balance).toLocaleString("en-IN")}
              </span>
            ) : (
              <span style={{ color: "rgba(255,255,255,0.5)" }}>
                Angel Balance: <em>(fetch failed)</em>
              </span>
            )}
            {activeTrade?.capital !== undefined && (
              <span>
                <span style={{ color: "rgba(255,61,87,0.6)" }}>Trading Cap:</span>{" "}
                ₹{Number(activeTrade.capital).toLocaleString("en-IN")}
              </span>
            )}
          </span>
        </div>
      )}

      <Ticker price={price} change={candleChange} feedLive={feedLive} />

      {/* Main grid */}
      <div className="main-layout">
        {/* LEFT COLUMN */}
        <div className="left-col">

          {/* Price Card */}
          <div className="card" style={{ animationDelay: "0.0s" }}>
            <CornerMarks />
            <div className="card-inner">
              <div className="section-header">
                <div className="section-title">Nifty 50 &bull; NSE</div>
                <div className="badge" style={{ borderColor: feedLive ? "rgba(0,230,118,0.3)" : "rgba(240,180,41,0.3)", color: feedLive ? "var(--green)" : "var(--gold)", background: feedLive ? "rgba(0,230,118,0.05)" : "rgba(240,180,41,0.05)" }}>
                  <div className={feedLive ? "dot-live" : ""} style={!feedLive ? { width: 8, height: 8, borderRadius: "50%", background: "var(--gold)" } : undefined} />
                  {feedLive ? "LIVE TICK" : "STALE / WAITING"}
                </div>
              </div>
              <div className="val-huge" style={{ color: "var(--gold2)" }}>₹{price.toLocaleString("en-IN")}</div>
              <div style={{ marginTop: 16, display: "flex", gap: 16 }}>
                <div>
                  <div className="label" style={{ marginBottom: 4 }}>Session</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--text)" }}>NSE CASH</div>
                </div>
                <div>
                  <div className="label" style={{ marginBottom: 4 }}>Symbol</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--text)" }}>^NSEI</div>
                </div>
                <div>
                  <div className="label" style={{ marginBottom: 4 }}>Interval</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--text)" }}>5 MIN + LIVE TICK</div>
                </div>
              </div>
            </div>
          </div>

          {/* Feed status */}
          <div className="card" style={{ animationDelay: "0.1s" }}>
            <CornerMarks />
            <div className="card-inner">
              <div className="section-header">
                <div className="section-title">Market Feed</div>
                <div className="icon-box">
                  <Activity size={16} color={feedLive ? "var(--green)" : "var(--gold)"} />
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 20 }}>
                <div className="val-large" style={{ color: feedLive ? "var(--green)" : "var(--gold)" }}>{feedLive ? "LIVE" : "WAIT"}</div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)" }}>{feedLive ? `tick ${tickAgeSec.toFixed(1)}s ago` : "waiting for tick collector"}</div>
              </div>

              <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", lineHeight: 1.8 }}>
                <div className="data-row">
                  <span>Tick source</span><span style={{ color: "var(--text)" }}>{feedLive ? "Angel One WebSocket" : "Unavailable"}</span>
                </div>
                <div className="data-row">
                  <span>Candle source</span><span style={{ color: "var(--text)" }}>{data?.quant?.data_source || "Unavailable"}</span>
                </div>
                <div className="data-row">
                  <span>Closed candles</span><span style={{ color: "var(--text)" }}>{candles.length}</span>
                </div>
                <div className="data-row">
                  <span>Interval</span><span style={{ color: "var(--text)" }}>5 minutes</span>
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* RIGHT COLUMN */}
        <div className="right-col">

          <ChartComponent candles={candles} breakouts={breakouts} trendlines={data?.quant?.trendlines} liveCandle={data?.live_candle} feedLive={feedLive} />

          {/* Auto-trade PAUSED banner -- visible when state is PAUSED.
              Shows current signal + Approve button for one-shot manual execution. */}
          {autoState === "PAUSED" && (
            <div className="card" style={{
              animationDelay: "0.02s",
              borderColor: "rgba(240,180,41,0.4)",
              animation: "glow-gold 3s ease-in-out infinite",
            }}>
              <CornerMarks />
              <div className="card-inner" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
                <div>
                  <div className="label" style={{ marginBottom: 8, color: "var(--gold)" }}>
                    Auto-trading is PAUSED
                  </div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--text)", lineHeight: 1.6 }}>
                    Signals are still flowing for monitoring.{" "}
                    {signal !== "WAIT" ? (
                      <span>
                        Current signal: <strong style={{ color: signal === "BUY" ? "var(--green)" : "var(--red)" }}>{signal}</strong> @ {confidence}
                      </span>
                    ) : (
                      <span style={{ color: "var(--muted)" }}>No actionable signal right now.</span>
                    )}
                  </div>
                  {pendingApproval && (
                    <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--gold)", marginTop: 6 }}>
                      Approval queued -- waiting for next valid signal (60s window).
                    </div>
                  )}
                </div>
                <div style={{ display: "flex", gap: 10 }}>
                  <button
                    onClick={approveNextTrade}
                    disabled={pendingApproval || signal === "WAIT" || activeTrade?.status === "OPEN"}
                    style={{
                      padding: "10px 18px",
                      border: "1px solid rgba(0,230,118,0.5)",
                      background: "rgba(0,230,118,0.08)",
                      color: "var(--green)",
                      borderRadius: 2,
                      cursor: (pendingApproval || signal === "WAIT" || activeTrade?.status === "OPEN") ? "not-allowed" : "pointer",
                      fontFamily: "var(--mono)", fontSize: 11, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase",
                      opacity: (pendingApproval || signal === "WAIT" || activeTrade?.status === "OPEN") ? 0.5 : 1,
                      whiteSpace: "nowrap",
                    }}
                  >
                    Approve This Trade
                  </button>
                  <button
                    onClick={() => setAutoTrade("RUNNING")}
                    style={{
                      padding: "10px 18px",
                      border: "1px solid var(--border2)",
                      background: "var(--surface2)",
                      color: "var(--text)",
                      borderRadius: 2,
                      cursor: "pointer",
                      fontFamily: "var(--mono)", fontSize: 11, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase",
                      whiteSpace: "nowrap",
                    }}
                  >
                    Resume Auto
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Signal Card */}
          <div className={`card ${cardSigClass}`} style={{ animationDelay: "0.05s" }}>
            <CornerMarks />
            <div className="card-inner">
              <div className="section-header">
                <div className="section-title">LuxAlgo Breakout Signal</div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div className="icon-box">
                    {signal === "BUY" ? (
                      <ArrowUpRight size={18} color="var(--green)" />
                    ) : signal === "SELL" ? (
                      <ArrowDownRight size={18} color="var(--red)" />
                    ) : (
                      <Activity size={18} color="var(--muted)" />
                    )}
                  </div>
                  <div className="icon-box">
                    <Zap size={16} color="var(--gold)" />
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "baseline", gap: 20, margin: "12px 0 8px" }}>
                <div className={`signal-text ${sigClass}`}>{signal}</div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", maxWidth: 260, lineHeight: 1.8 }}>
                  {data?.signal?.reason || "Waiting for the next closed-candle trendline breakout."}
                </div>
              </div>

              <hr className="divider" />

              {/* TP / SL */}
              <div className="grid-2">
                <div className="mini-card">
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                    <Target size={14} color="var(--green)" />
                    <span className="label">Take Profit</span>
                  </div>
                  <div className={`val-large`} style={{ color: "var(--green)" }}>₹{target}</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)", marginTop: 8 }}>TARGET LEVEL</div>
                </div>

                <div className="mini-card">
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                    <ShieldAlert size={14} color="var(--red)" />
                    <span className="label">Stop Loss</span>
                  </div>
                  <div className={`val-large`} style={{ color: "var(--red)" }}>₹{sl}</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)", marginTop: 8 }}>RISK LEVEL</div>
                </div>
              </div>
            </div>
          </div>

          {/* Live Position Card */}
          <div className="card" style={{ animationDelay: "0.15s" }}>
            <CornerMarks />
            <div className="card-inner">
              <div className="section-header">
                <div className="section-title" style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div className="icon-box">
                    <Wallet size={16} color="var(--gold)" />
                  </div>
                  Live Position
                </div>
                {activeTrade?.status === "OPEN" && (
                  <div className="badge" style={{ borderColor: "rgba(240,180,41,0.3)", color: "var(--gold)", background: "rgba(240,180,41,0.05)", animation: "pulse 2s ease-in-out infinite" }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--gold)" }} />
                    ACTIVE
                  </div>
                )}
              </div>

              {activeTrade?.status === "CLOSED" || !activeTrade ? (
                <div style={{ padding: "40px 0", textAlign: "center" }}>
                  <div style={{ fontSize: 48, opacity: 0.12, marginBottom: 16 }}>◈</div>
                  <div style={{ fontFamily: "var(--display)", fontSize: 28, letterSpacing: "0.06em", color: "var(--muted)" }}>
                    NO OPEN POSITION
                  </div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--dim)", marginTop: 8, letterSpacing: "0.12em" }}>
                    AI SCANNING FOR OPTIMAL ENTRY
                  </div>
                </div>
              ) : (
                <div className="grid-3">
                  {/* Position */}
                  <div className="mini-card">
                    <div className="label" style={{ marginBottom: 14 }}>Direction</div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      {activeTrade.position === "BUY"
                        ? <ArrowUpRight size={20} color="var(--green)" />
                        : <ArrowDownRight size={20} color="var(--red)" />}
                      <div className="val-large" style={{ color: activeTrade.position === "BUY" ? "var(--green)" : "var(--red)", fontSize: 36 }}>
                        {activeTrade.position}
                      </div>
                    </div>
                  </div>

                  {/* Entry */}
                  <div className="mini-card">
                    <div className="label" style={{ marginBottom: 14 }}>Entry Price</div>
                    <div className="val-large" style={{ color: "var(--text)", fontSize: 36 }}>₹{activeTrade.entry}</div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)", marginTop: 8 }}>FILLED</div>
                  </div>

                  {/* PNL */}
                  <div className={`mini-card ${pnl >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                    <div className="label" style={{ marginBottom: 14 }}>Live P&amp;L</div>
                    <div className="val-large" style={{ fontSize: 36, color: pnl >= 0 ? "var(--green)" : "var(--red)" }}>
                      {pnl >= 0 ? "+" : ""}₹{pnl.toFixed(2)}
                    </div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)", marginTop: 8 }}>
                      {pnl >= 0 ? "PROFIT" : "LOSS"}
                    </div>
                  </div>
                </div>
              )}

              {/* Option contract info -- only shown when LIVE option trade is open */}
              {activeTrade?.status === "OPEN" && activeTrade?.opt_symbol && (
                <div style={{
                  marginTop: 16, padding: 14,
                  background: "rgba(0,212,255,0.05)",
                  border: "1px solid rgba(0,212,255,0.25)",
                  borderRadius: 2,
                }}>
                  <div className="label" style={{ marginBottom: 10, color: "var(--cyan)" }}>
                    Option Contract (NFO)
                  </div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--text)", marginBottom: 6, letterSpacing: "0.08em" }}>
                    <strong style={{ color: "var(--cyan)" }}>{activeTrade.opt_symbol}</strong>
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 20, fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)" }}>
                    <span>Type: <strong style={{ color: activeTrade.opt_type === "CE" ? "var(--green)" : "var(--red)" }}>{activeTrade.opt_type}</strong></span>
                    <span>Strike: <strong style={{ color: "var(--text)" }}>{activeTrade.opt_strike}</strong></span>
                    <span>Lots: <strong style={{ color: "var(--text)" }}>{activeTrade.opt_lots}</strong></span>
                    <span>Entry premium: <strong style={{ color: "var(--text)" }}>₹{Number(activeTrade.opt_entry_premium).toFixed(2)}</strong></span>
                  </div>

                  {/* Live premium + change (from tick_collector) */}
                  {activeTrade.opt_ltp !== null && activeTrade.opt_ltp !== undefined && (() => {
                    const ltp    = Number(activeTrade.opt_ltp);
                    const entry  = Number(activeTrade.opt_entry_premium) || 0;
                    const diff   = ltp - entry;
                    const pct    = entry > 0 ? (diff / entry) * 100 : 0;
                    const up     = diff >= 0;
                    return (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 20, marginTop: 8, fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)" }}>
                        <span>Live premium: <strong style={{ color: "var(--cyan)" }}>₹{ltp.toFixed(2)}</strong></span>
                        <span>Change: <strong style={{ color: up ? "var(--green)" : "var(--red)" }}>{up ? "+" : ""}₹{diff.toFixed(2)} ({up ? "+" : ""}{pct.toFixed(1)}%)</strong></span>
                      </div>
                    );
                  })()}

                  {/* Premium-based TP / SL (estimated, ATM delta ~0.5) */}
                  {(activeTrade.opt_tp_premium != null || activeTrade.opt_sl_premium != null) && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 20, marginTop: 8, fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)" }}>
                      <span>🎯 TP @ premium: <strong style={{ color: "var(--green)" }}>≈₹{Number(activeTrade.opt_tp_premium).toFixed(2)}</strong></span>
                      <span>🛑 SL @ premium: <strong style={{ color: "var(--red)" }}>≈₹{Number(activeTrade.opt_sl_premium).toFixed(2)}</strong></span>
                      <span style={{ color: "var(--dim)" }}>(est.)</span>
                    </div>
                  )}
                </div>
              )}

              {/* TP / SL display + Manual Exit (only when trade is OPEN) */}
              {activeTrade?.status === "OPEN" && (
                <div style={{ marginTop: 20 }}>
                  <div className="grid-2" style={{ marginBottom: 16 }}>
                    <div className="mini-card" style={{ borderColor: "rgba(0,230,118,0.25)" }}>
                      <div className="label" style={{ marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
                        <Target size={12} color="var(--green)" /> Target (TP)
                      </div>
                      <div className="val-large" style={{ fontSize: 28, color: "var(--green)" }}>
                        ₹{Number(activeTrade.tp).toFixed(2)}
                      </div>
                    </div>
                    <div className="mini-card" style={{ borderColor: "rgba(255,61,87,0.25)" }}>
                      <div className="label" style={{ marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
                        <ShieldAlert size={12} color="var(--red)" /> Stop Loss (SL)
                      </div>
                      <div className="val-large" style={{ fontSize: 28, color: "var(--red)" }}>
                        ₹{Number(activeTrade.sl).toFixed(2)}
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={handleExit}
                    disabled={busy}
                    style={{
                      width: "100%",
                      padding: "14px 16px",
                      background: "rgba(255,61,87,0.08)",
                      border: "1px solid rgba(255,61,87,0.5)",
                      color: "var(--red)",
                      borderRadius: 2,
                      cursor: busy ? "wait" : "pointer",
                      fontFamily: "var(--mono)",
                      fontSize: 13,
                      fontWeight: 700,
                      letterSpacing: "0.16em",
                      textTransform: "uppercase",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 10,
                    }}
                  >
                    <X size={16} /> {busy ? "Closing..." : "Exit Trade Now"}
                  </button>

                  {cmdMsg && (
                    <div style={{
                      marginTop: 12,
                      padding: "8px 12px",
                      background: cmdMsg.ok ? "rgba(0,230,118,0.08)" : "rgba(255,61,87,0.08)",
                      border: `1px solid ${cmdMsg.ok ? "rgba(0,230,118,0.3)" : "rgba(255,61,87,0.3)"}`,
                      borderRadius: 2,
                      fontFamily: "var(--mono)",
                      fontSize: 11,
                      color: cmdMsg.ok ? "var(--green)" : "var(--red)"
                    }}>
                      {cmdMsg.text}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Market Stats Row */}
          <div className="card" style={{ animationDelay: "0.2s" }}>
            <div className="card-inner" style={{ padding: "20px 28px" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                {[
                  { label: "Refresh Rate", val: "3s" },
                  { label: "Data Source", val: "NSE LIVE" },
                  { label: "Strategy", val: "MOMENTUM" },
                  { label: "Engine", val: "QUANT v4" },
                  { label: "Mode", val: "AUTO" },
                ].map((item, i) => (
                  <div key={i} style={{ textAlign: "center" }}>
                    <div className="label" style={{ marginBottom: 6 }}>{item.label}</div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: 13, fontWeight: 600, color: "var(--gold)" }}>
                      {item.val}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Trade History Card */}
          <div className="card" style={{ animationDelay: "0.25s" }}>
            <CornerMarks />
            <div className="card-inner">
              <div className="section-header">
                <div className="section-title">
                  <TrendingUp size={13} color="var(--gold)" style={{ marginRight: 6 }} />
                  Trade History &bull; Today
                </div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)", letterSpacing: "0.12em" }}>
                  {trades.length} TRADE{trades.length !== 1 ? "S" : ""}
                </div>
              </div>

              {trades.length === 0 && activeTrade?.status !== "OPEN" ? (
                <div style={{ padding: "32px 0", textAlign: "center" }}>
                  <div style={{ fontSize: 36, opacity: 0.1, marginBottom: 12 }}>◈</div>
                  <div style={{ fontFamily: "var(--display)", fontSize: 22, letterSpacing: "0.06em", color: "var(--muted)" }}>
                    NO TRADES YET
                  </div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--dim)", marginTop: 6, letterSpacing: "0.1em" }}>
                    WAITING FOR SIGNAL
                  </div>
                </div>
              ) : (
                <div style={{ overflowX: "auto" }}>
                  <table className="trades-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Direction</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>P&amp;L (₹)</th>
                        <th>Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {/* Open position as first row */}
                      {activeTrade?.status === "OPEN" && (
                        <tr className="trade-row-open">
                          <td style={{ color: "var(--muted)" }}>
                            {activeTrade.open_time
                              ? new Date(activeTrade.open_time).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" })
                              : "—"}
                          </td>
                          <td>
                            <span style={{ color: activeTrade.position === "BUY" ? "var(--green)" : "var(--red)", fontWeight: 700 }}>
                              {activeTrade.position === "BUY" ? "▲ BUY" : "▼ SELL"}
                            </span>
                          </td>
                          <td>₹{activeTrade.entry?.toFixed(2) ?? "—"}</td>
                          <td style={{ color: "var(--dim)" }}>—</td>
                          <td style={{ color: pnl >= 0 ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                            {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}
                          </td>
                          <td>
                            <span className="outcome-badge" style={{ color: "var(--gold)", borderColor: "rgba(240,180,41,0.35)", background: "rgba(240,180,41,0.08)" }}>
                              OPEN
                            </span>
                          </td>
                        </tr>
                      )}
                      {/* Closed trades */}
                      {trades.map((t, i) => {
                        const isWin = t.outcome === "WIN";
                        const timeStr = t.timestamp
                          ? new Date(t.timestamp).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" })
                          : "—";
                        return (
                          <tr key={i} className={isWin ? "trade-row-win" : "trade-row-loss"}>
                            <td style={{ color: "var(--muted)" }}>{timeStr}</td>
                            <td>
                              <span style={{ color: t.signal === "BUY" ? "var(--green)" : "var(--red)", fontWeight: 700 }}>
                                {t.signal === "BUY" ? "▲ BUY" : "▼ SELL"}
                              </span>
                            </td>
                            <td>₹{t.entry_price?.toFixed(2) ?? "—"}</td>
                            <td>₹{t.exit_price?.toFixed(2) ?? "—"}</td>
                            <td style={{ color: isWin ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                              {t.pnl >= 0 ? "+" : ""}{t.pnl?.toFixed(2) ?? "0.00"}
                            </td>
                            <td>
                              {isWin ? (
                                <span className="outcome-badge" style={{ color: "var(--green)", borderColor: "rgba(0,230,118,0.35)", background: "rgba(0,230,118,0.08)" }}>
                                  WIN
                                </span>
                              ) : (
                                <span className="outcome-badge" style={{ color: "var(--red)", borderColor: "rgba(255,61,87,0.35)", background: "rgba(255,61,87,0.08)" }}>
                                  LOSS
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>

        </div>
      </div>

      {/* LIVE-mode confirmation modal -- requires typing "LIVE" to confirm */}
      {showLiveConfirm && (
        <div
          onClick={() => setShowLiveConfirm(false)}
          style={{
            position: "fixed", inset: 0, zIndex: 2000,
            background: "rgba(6,8,13,0.85)", backdropFilter: "blur(4px)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="card"
            style={{
              maxWidth: 480, width: "calc(100% - 32px)", padding: 0,
              borderColor: "rgba(255,61,87,0.5)",
              animation: "glow-red 2.5s ease-in-out infinite",
            }}
          >
            <CornerMarks />
            <div className="card-inner">
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
                <div className="icon-box" style={{ borderColor: "rgba(255,61,87,0.4)", background: "rgba(255,61,87,0.08)" }}>
                  <ShieldAlert size={18} color="var(--red)" />
                </div>
                <div style={{ fontFamily: "var(--display)", fontSize: 28, letterSpacing: "0.06em", color: "var(--red)" }}>
                  CONFIRM LIVE MODE
                </div>
              </div>
              <p style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--text)", lineHeight: 1.7, marginBottom: 16 }}>
                You are about to enable <strong style={{ color: "var(--red)" }}>REAL MONEY</strong> trading.
                Every signal the AI generates will place an order through your Angel One account.
                Daily loss limit and per-trade risk caps still apply.
              </p>
              <p style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", marginBottom: 10, letterSpacing: "0.08em" }}>
                Type <strong style={{ color: "var(--red)" }}>LIVE</strong> below to confirm:
              </p>
              <input
                type="text"
                autoFocus
                value={liveTypeInput}
                onChange={(e) => setLiveTypeInput(e.target.value)}
                disabled={modeBusy || !hasAngelCreds}
                placeholder='Type "LIVE" exactly'
                style={{
                  width: "100%", padding: "10px 12px", marginBottom: 12,
                  background: "var(--surface2)", border: "1px solid var(--border2)",
                  borderRadius: 2, color: "var(--text)",
                  fontFamily: "var(--mono)", fontSize: 14, letterSpacing: "0.16em",
                  outline: "none",
                }}
              />
              {!hasAngelCreds && (
                <div style={{
                  padding: "8px 12px", marginBottom: 12,
                  background: "rgba(255,61,87,0.08)", border: "1px solid rgba(255,61,87,0.3)", borderRadius: 2,
                  fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)",
                }}>
                  Angel One credentials missing in .env -- cannot enable LIVE.
                </div>
              )}
              <div style={{ display: "flex", gap: 10 }}>
                <button
                  onClick={() => { setShowLiveConfirm(false); setLiveTypeInput(""); }}
                  disabled={modeBusy}
                  style={{
                    flex: 1, padding: "12px", border: "1px solid var(--border2)", background: "var(--surface2)",
                    color: "var(--muted)", borderRadius: 2, cursor: "pointer",
                    fontFamily: "var(--mono)", fontSize: 12, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase",
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={() => setExecutionMode("LIVE")}
                  disabled={modeBusy || liveTypeInput !== "LIVE" || !hasAngelCreds}
                  style={{
                    flex: 1, padding: "12px",
                    border: "1px solid rgba(255,61,87,0.6)",
                    background: liveTypeInput === "LIVE" && hasAngelCreds ? "rgba(255,61,87,0.18)" : "rgba(255,61,87,0.05)",
                    color: "var(--red)", borderRadius: 2,
                    cursor: (modeBusy || liveTypeInput !== "LIVE" || !hasAngelCreds) ? "not-allowed" : "pointer",
                    fontFamily: "var(--mono)", fontSize: 12, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase",
                    opacity: (liveTypeInput === "LIVE" && hasAngelCreds) ? 1 : 0.4,
                  }}
                >
                  {modeBusy ? "Switching..." : "Activate LIVE"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
