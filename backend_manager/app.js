const express = require('express');
const axios = require('axios');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { connectDB, redisClient } = require('./db_config');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';

// --- Helper Function: Safe JSON Parsing ---
// यह फंक्शन सुनिश्चित करता है कि अगर डेटा खराब है, तो सर्वर क्रैश न हो


app.get('/status', async (req, res) => {
    try {
        const aiResponse = await axios.get(`${AI_ENGINE_URL}/health`);
        res.json({ manager_status: 'Active', ai_engine_health: aiResponse.data });
    } catch (error) {
        res.status(500).json({ error: 'AI Engine unreachable' });
    }
});

app.get('/market-data/:symbol/:pair', async (req, res) => {
    const { symbol, pair } = req.params;
    try {
        const data = await redisClient.get(`market:${symbol}/${pair}`);
        res.json(safeParse(data) || { error: 'Data not found' });
    } catch (error) {
        res.status(500).json({ error: 'Redis Error' });
    }
});

app.get('/quant-data/:symbol/:pair', async (req, res) => {
    const { symbol, pair } = req.params;
    try {
        const data = await redisClient.get(`quant:${symbol}/${pair}`);
        res.json(safeParse(data) || { error: 'Quant data not found' });
    } catch (error) {
        res.status(500).json({ error: 'Redis Error' });
    }
});

app.get('/ai-signal/:symbol/:pair', async (req, res) => {
    const { symbol, pair } = req.params;
    try {
        const data = await redisClient.get(`signal:${symbol}/${pair}`);
        res.json(safeParse(data) || { error: 'AI Signal not yet generated' });
    } catch (error) {
        res.status(500).json({ error: 'Redis Error' });
    }
});
app.get('/active-trade', async (req, res) => {
    const data = await redisClient.get("active_trade");
    res.json(data ? JSON.parse(data) : { status: "CLOSED" });
});

app.get('/market-tick/:symbol', async (req, res) => {
    try {
        const data = await redisClient.get(`tick:${req.params.symbol}`);
        res.json(safeParse(data) || { error: 'Live tick unavailable' });
    } catch (error) {
        res.status(500).json({ error: 'Redis Error' });
    }
});

// Helper function: JSON parse करने के लिए (ताकि सर्वर क्रैश न हो)
const safeParse = (data) => {
    if (!data) return null;
    try {
        // अगर डेटा Buffer है, तो उसे String में बदलें
        const stringData = Buffer.isBuffer(data) ? data.toString() : data;
        return JSON.parse(stringData);
    } catch (e) {
        console.error("❌ JSON Parse Error! Raw Data was:", data);
        console.error("❌ Error details:", e.message);
        return null;
    }
};


app.get('/analysis/:symbol', async (req, res) => {
    const { symbol } = req.params;
    try {
        const allKeys = await redisClient.keys('*');
        
        // डेटा फेच करें
        const rawQuant = await redisClient.get(`quant:${symbol}`);
        const rawSignal = await redisClient.get(`signal:${symbol}`);
        const rawMarket = await redisClient.get(`market:${symbol}`);
        const rawCandles = await redisClient.get(`candles:${symbol}`);
        const rawLiveCandle = await redisClient.get(`live_candle:${symbol}`);

        // Parse करें
        const quant = safeParse(rawQuant);
        const signal = safeParse(rawSignal) || {
            symbol,
            signal: 'WAIT',
            confidence: '0%',
            source: 'LuxAlgo - Trendlines with Breaks',
            reason: 'Signal engine is waiting for a closed-candle breakout.',
            timestamp: Date.now() / 1000,
        };
        const market = safeParse(rawMarket) || { last_price: "N/A" };
        const candles = safeParse(rawCandles) || [];
        const liveCandle = safeParse(rawLiveCandle);

        // 🔍 LOGGING: यह हमें टर्मिनल में बताएगा कि पार्सिंग के बाद क्या बचा
        console.log(`--- 📊 Analysis Request for ${symbol} ---`);
        console.log(`Raw Quant exists: ${!!rawQuant} | Parsed: ${!!quant}`);
        console.log(`Raw Signal exists: ${!!rawSignal} | Parsed: ${!!signal}`);

        const response = {
            symbol: symbol,
            market: market,
            quant: quant,
            signal: signal,
            candles: candles,
            live_candle: liveCandle,
            timestamp: Date.now()
        };

        if (!quant) {
            return res.status(404).json({ 
            error: "Quant candle data is missing or corrupted in cache", 
            details: "Quant: FAIL",
                debug_keys_found: allKeys 
            });
        }

        res.json(response);
    } catch (error) {
        console.error("CRITICAL API ERROR:", error);
        res.status(500).json({ error: "Unified API Error", details: error.message });
    }
});



app.get('/active-trade/:symbol', async (req, res) => {
    const symbol = req.params.symbol;
    try {
        const data = await redisClient.get(`active_trade:${symbol}`);
        if (data) {
            res.json(JSON.parse(data));
        } else {
            res.json({ status: "CLOSED" });
        }
    } catch (error) {
        res.status(500).json({ error: "Redis Error" });
    }
});

// Execution mode (PAPER vs LIVE) -- backed by Redis so the execution manager
// can pick up changes without a restart.
app.get('/mode/:symbol', async (req, res) => {
    const { symbol } = req.params;
    try {
        const mode = (await redisClient.get(`execution_mode:${symbol}`)) || 'PAPER';
        // Surface whether Angel One creds are wired up (frontend can disable LIVE button)
        const hasAngelCreds = !!(process.env.ANGEL_CLIENT_ID && process.env.ANGEL_API_KEY
                                 && process.env.ANGEL_PASSWORD && process.env.ANGEL_TOTP_KEY);
        res.json({ mode, hasAngelCreds });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/mode/:symbol', async (req, res) => {
    const { symbol } = req.params;
    const newMode = String(req.body?.mode || '').toUpperCase();

    if (newMode !== 'PAPER' && newMode !== 'LIVE') {
        return res.status(400).json({ error: 'mode must be "PAPER" or "LIVE"' });
    }

    try {
        // Reject mid-trade switches -- a paper position has no real-money equivalent
        const at = await redisClient.get(`active_trade:${symbol}`);
        if (at) {
            const active = JSON.parse(at);
            if (active.status === 'OPEN') {
                return res.status(400).json({
                    error: 'Close the current open trade before switching mode.',
                });
            }
        }

        // For LIVE mode, require Angel One credentials to exist
        if (newMode === 'LIVE') {
            const ok = process.env.ANGEL_CLIENT_ID && process.env.ANGEL_API_KEY
                       && process.env.ANGEL_PASSWORD && process.env.ANGEL_TOTP_KEY;
            if (!ok) {
                return res.status(400).json({
                    error: 'Angel One credentials missing in .env (ANGEL_CLIENT_ID, ANGEL_API_KEY, ANGEL_PASSWORD, ANGEL_TOTP_KEY)',
                });
            }
        }

        await redisClient.set(`execution_mode:${symbol}`, newMode);
        console.log(`[MODE] Execution mode set to ${newMode} for ${symbol}`);
        res.json({ ok: true, mode: newMode });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ── Auto-trade control ───────────────────────────────────────
// auto_state: "RUNNING" (auto-execute signals) | "PAUSED" (only signal, no exec)
// When PAUSED, the user can manually approve a single trade via /control/:symbol/approve.
app.get('/control/:symbol', async (req, res) => {
    const { symbol } = req.params;
    try {
        const autoState = (await redisClient.get(`auto_state:${symbol}`)) || 'RUNNING';
        const eventRaw  = await redisClient.get(`auto_state_event:${symbol}`);
        const approvalRaw = await redisClient.get(`manual_approve:${symbol}`);
        res.json({
            auto_state: autoState,
            event:      eventRaw ? JSON.parse(eventRaw) : null,
            pending_approval: !!approvalRaw,
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/control/:symbol/auto', async (req, res) => {
    const { symbol } = req.params;
    const state = String(req.body?.state || '').toUpperCase();
    if (state !== 'RUNNING' && state !== 'PAUSED') {
        return res.status(400).json({ error: 'state must be "RUNNING" or "PAUSED"' });
    }
    try {
        await redisClient.set(`auto_state:${symbol}`, state);
        // Clear any pending state-transition event so frontend doesn't re-pop modal
        await redisClient.del(`auto_state_event:${symbol}`);
        res.json({ ok: true, auto_state: state });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// One-shot approval to fire a single trade while in PAUSED state.
// Valid for 60 seconds, consumed by execution manager on first matching signal.
app.post('/control/:symbol/approve', async (req, res) => {
    const { symbol } = req.params;
    try {
        await redisClient.set(`manual_approve:${symbol}`, String(Date.now() / 1000), { EX: 60 });
        res.json({ ok: true, message: 'Approved -- next valid signal within 60s will execute' });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Manual exit: queue a command for the execution manager to close current trade at market.
app.post('/trade/exit/:symbol', async (req, res) => {
    const { symbol } = req.params;
    try {
        const raw = await redisClient.get(`active_trade:${symbol}`);
        const active = raw ? JSON.parse(raw) : null;
        if (!active || active.status !== 'OPEN') {
            return res.status(400).json({ error: 'No open trade to exit' });
        }
        await redisClient.set(`trade_command:${symbol}`, JSON.stringify({
            action:    'EXIT',
            timestamp: Date.now(),
        }));
        res.json({ ok: true, message: 'Exit command queued. Execution manager will close on next cycle.' });
    } catch (err) {
        console.error('Exit command error:', err);
        res.status(500).json({ error: err.message });
    }
});

app.get('/trades/:symbol', (_req, res) => {
    try {
        const csvPath = path.join(__dirname, '../ai_engine/live_feedback.csv');
        if (!fs.existsSync(csvPath)) {
            return res.json([]);
        }
        const content = fs.readFileSync(csvPath, 'utf8');
        const lines = content.trim().split('\n').filter(l => l.trim());
        if (lines.length < 2) return res.json([]);

        const headers = lines[0].split(',').map(h => h.trim());
        const trades = lines.slice(1).map(line => {
            const vals = line.split(',');
            const obj = {};
            headers.forEach((h, i) => { obj[h] = (vals[i] || '').trim(); });
            return {
                timestamp:   obj.timestamp || '',
                signal:      obj.signal    || '',
                outcome:     obj.outcome   || '',
                entry_price: parseFloat(obj.entry_price) || 0,
                exit_price:  parseFloat(obj.exit_price)  || 0,
                pnl:         parseFloat(obj.pnl)         || 0,
            };
        });

        res.json(trades.reverse()); // newest first
    } catch (err) {
        console.error('Trades read error:', err.message);
        res.json([]);
    }
});

const PORT = process.env.PORT || 3000;

const startServer = async () => {
    await connectDB();
    app.listen(PORT, () => console.log(`🚀 Backend Manager running on port ${PORT}`));
};

if (require.main === module) {
    startServer();
}

module.exports = app;
