# pro_dashboard.py
"""
Pro Trading Suite (Plotly + Auth + Redis/SQLite push fallback + Order events + Journal + OCO)
- Safe for Render deployment
- Requires: auth_utils.py (authenticate, get_role)
- Optional: Redis (provide REDIS_URL). If not present, uses SQLite fallback for ticks and events.
"""
import os
import json
import sqlite3
from datetime import datetime
import time

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

from alpaca.trading.client import TradingClient

# local auth helpers (must exist in repo)
from auth_utils import authenticate  # returns {"username":..., "role":...} or None

# -------------------------
# Configuration
# -------------------------
st.set_page_config(page_title="Pro Trading Suite", layout="wide")
LOGFILE = os.getenv("JOURNAL_FILE", "journal_trades.csv")
TICKS_DB = os.getenv("WS_SQLITE", "ws_store.sqlite")
ORDER_DB = os.getenv("ORDER_SQLITE", "orders_store.sqlite")
REDIS_URL = os.getenv("REDIS_URL", None)

# UI refresh interval note (we don't do blocking sleeps; use manual refresh)
DEFAULT_POLL_SECONDS = int(os.getenv("UI_POLL", "10"))

# -------------------------
# Utilities: Redis or SQLite fallback
# -------------------------
try:
    import redis
except Exception:
    redis = None

def get_redis():
    if not REDIS_URL or redis is None:
        return None
    try:
        return redis.from_url(REDIS_url := REDIS_URL)
    except Exception:
        try:
            return redis.from_url(REDIS_URL)  # fallback attempt
        except Exception:
            return None

def sqlite_set_ticks(latest_dict):
    conn = sqlite3.connect(TICKS_DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS ticks (symbol TEXT PRIMARY KEY, payload TEXT, updated TIMESTAMP)
    """)
    for sym, payload in latest_dict.items():
        c.execute("INSERT OR REPLACE INTO ticks(symbol,payload,updated) VALUES(?,?,?)",
                  (sym, json.dumps(payload), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def sqlite_get_ticks():
    if not os.path.exists(TICKS_DB):
        return {}
    conn = sqlite3.connect(TICKS_DB)
    c = conn.cursor()
    try:
        c.execute("SELECT symbol, payload FROM ticks")
        rows = c.fetchall()
        data = {r[0]: json.loads(r[1]) for r in rows}
    except Exception:
        data = {}
    conn.close()
    return data

def get_latest_tick(symbol):
    r = get_redis()
    if r:
        try:
            raw = r.hget("market_ticks", symbol)
            if raw:
                # redis returns bytes; decode if needed
                if isinstance(raw, bytes):
                    raw = raw.decode()
                return json.loads(raw)
        except Exception:
            pass
    # fallback to sqlite
    ticks = sqlite_get_ticks()
    return ticks.get(symbol)

# -------------------------
# Indicators / plotting / backtest helpers
# -------------------------
def compute_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["MID"] = df["Close"].rolling(20).mean()
    df["STD"] = df["Close"].rolling(20).std()
    df["Upper"] = df["MID"] + 2 * df["STD"]
    df["Lower"] = df["MID"] - 2 * df["STD"]
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def format_currency(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

# -------------------------
# Logging trades & journal
# -------------------------
def log_trade(record: dict):
    df = pd.DataFrame([record])
    header = not os.path.exists(LOGFILE)
    df.to_csv(LOGFILE, mode="a", header=header, index=False)

def load_journal():
    if os.path.exists(LOGFILE):
        try:
            return pd.read_csv(LOGFILE, parse_dates=["time"])
        except Exception:
            return pd.read_csv(LOGFILE)
    return pd.DataFrame()

# -------------------------
# Alpaca helper (safe)
# -------------------------
def get_alpaca_client():
    API_KEY = os.getenv("API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")
    if not API_KEY or not SECRET_KEY:
        return None
    try:
        return TradingClient(API_KEY, SECRET_KEY, paper=True)
    except Exception:
        return None

def submit_bracket_alpaca(client, symbol, qty, side, take_profit_price, stop_loss_price):
    """
    Best-effort bracket order via alpaca-py. If bracket class not available, fallback to market + children.
    """
    try:
        # attempt bracket order (some versions accept this)
        order = client.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(take_profit_price)},
            stop_loss={"stop_price": str(stop_loss_price)}
        )
        return {"status": "ok", "order": str(order)}
    except Exception:
        # fallback: market then try to create TP and SL
        try:
            market = client.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="day")
            opp = "sell" if side == "buy" else "buy"
            tp = None
            sl = None
            try:
                tp = client.submit_order(symbol=symbol, qty=qty, side=opp, type="limit", time_in_force="gtc", limit_price=str(take_profit_price))
            except Exception:
                tp = None
            try:
                sl = client.submit_order(symbol=symbol, qty=qty, side=opp, type="stop", time_in_force="gtc", stop_price=str(stop_loss_price))
            except Exception:
                sl = None
            return {"status": "ok", "market": str(market), "tp": str(tp), "sl": str(sl)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# -------------------------
# Streamlit: authentication
# -------------------------
if "auth" not in st.session_state:
    st.session_state.auth = None

if st.session_state.auth is None:
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        user = authenticate(username, password)
        if user:
            # ensure the returned object contains username and role
            if "username" not in user:
                user["username"] = username
            if "role" not in user:
                user["role"] = "Reviewer"
            st.session_state.auth = user
            st.sidebar.success(f"Welcome {user['username']} ({user['role']})")
            # rerun to refresh UI
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")
    # Prevent unauthenticated access
    st.stop()

# Now we are authenticated
user = st.session_state.auth
role = user.get("role", "Reviewer")

# -------------------------
# Layout: Left controls, center chart, right account/journal
# -------------------------
left, center, right = st.columns([0.18, 0.56, 0.26])

with left:
    st.header("Controls")
    symbol = st.text_input("Symbol", value="AAPL").upper()
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=2)
    mode = st.selectbox("Mode", ["Dry-run", "Alpaca Paper"])
    risk_pct = st.slider("Risk % of equity", 1, 5, 2)
    tp_pct = st.number_input("Take Profit %", min_value=0.5, max_value=50.0, value=3.0, step=0.5)
    sl_pct = st.number_input("Stop Loss %", min_value=0.2, max_value=50.0, value=1.5, step=0.1)
    st.markdown("---")
    st.subheader("Screener / Watchlist")
    watchlist_input = st.text_area("Tickers (comma separated)", value="AAPL,MSFT,SPY,QQQ", height=80)
    screener_rsi = st.slider("RSI <= ", 10, 50, 30)
    screener_vol_min = st.number_input("Min Avg Volume (30d)", min_value=1000, value=100000, step=1000)
    run_screener = st.button("Run Screener")
    st.markdown("---")
    st.subheader("Trading Engine")
    run_check = st.button("Run Live Check")
    confirm_execute = st.checkbox("Confirm & Execute (Admin only)")
    st.markdown("---")
    st.caption(f"Signed in as: {user['username']} ({role})")

with center:
    st.header(f"{symbol} Chart")
    # Download historical data
    period = "60d" if interval in ["1m", "5m", "15m"] else "365d"
    with st.spinner("Downloading market data..."):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
        except Exception as e:
            st.error(f"Market data error: {e}")
            st.stop()
    if df.empty:
        st.error("No market data returned for this symbol/interval.")
        st.stop()

    df = compute_indicators(df)

    # Build Plotly candlestick
    def build_figure(df):
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="green", decreasing_line_color="red", name="Candles"
        )])
        if "SMA20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], mode="lines", name="SMA20"))
        if "SMA50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], mode="lines", name="SMA50"))
        if "Upper" in df.columns and "Lower" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], mode="lines", name="BB Upper", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], mode="lines", name="BB Lower", line=dict(dash="dash")))
            # Shaded band
            fig.add_traces([
                go.Scatter(x=df.index, y=df["Upper"], mode='lines', line=dict(width=0), showlegend=False),
                go.Scatter(x=df.index, y=df["Lower"], mode='lines', fill='tonexty', fillcolor='rgba(173,216,230,0.1)', line=dict(width=0), showlegend=False)
            ])
        fig.update_layout(xaxis_rangeslider_visible=False, height=600)
        return fig

    fig = build_figure(df)
    st.plotly_chart(fig, use_container_width=True)

    # RSI plot
    if "RSI" in df.columns:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(height=200)
        st.plotly_chart(rsi_fig, use_container_width=True)

    # Latest tick from stream (if available)
    tick = get_latest_tick(symbol)
    if tick:
        st.markdown(f"**Stream tick:** {tick.get('symbol')} — {format_currency(tick.get('close'))} @ {tick.get('ts')}")
    else:
        st.markdown("**Stream tick:** not available (Redis or sqlite fallback empty)")

with right:
    st.header("Account & Journal")
    client = get_alpaca_client() if mode == "Alpaca Paper" else None
    if client:
        try:
            acct = client.get_account()
            st.metric("Equity", format_currency(acct.equity))
            st.metric("Cash", format_currency(acct.cash))
            st.metric("Buying Power", format_currency(acct.buying_power))
        except Exception as e:
            st.error(f"Failed to fetch Alpaca account: {e}")
    else:
        st.info("Dry-run mode (simulated account: $100,000)")

    st.markdown("---")
    st.subheader("Journal (recent)")
    journal = load_journal()
    if not journal.empty:
        try:
            st.dataframe(journal.sort_values("time", ascending=False).head(10))
            csv = journal.to_csv(index=False).encode("utf-8")
            st.download_button("Download Journal CSV", data=csv, file_name=f"journal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        except Exception:
            st.write(journal.head(10))
    else:
        st.info("No trades logged yet.")

    st.markdown("---")
    st.subheader("Recent Order Events")
    # Read order events from Redis list 'order_events' or sqlite fallback
    events = []
    r = get_redis()
    if r:
        try:
            raw = r.lrange("order_events", -100, -1)
            for item in raw:
                try:
                    if isinstance(item, bytes):
                        item = item.decode()
                    events.append(json.loads(item))
                except Exception:
                    events.append({"raw": str(item)})
        except Exception:
            events = []
    else:
        # sqlite fallback
        if os.path.exists(ORDER_DB):
            conn = sqlite3.connect(ORDER_DB)
            c = conn.cursor()
            try:
                c.execute("SELECT payload, ts FROM order_events ORDER BY id DESC LIMIT 50")
                rows = c.fetchall()
                for row in rows:
                    try:
                        events.append(json.loads(row[0]))
                    except Exception:
                        events.append({"raw": row[0]})
            except Exception:
                events = []
            conn.close()
    if events:
        st.write(events[:20])
    else:
        st.info("No recent order events.")

# -------------------------
# Screener logic
# -------------------------
if run_screener:
    tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    screener_results = []
    if not tickers:
        st.warning("Provide tickers for the screener.")
    else:
        with st.spinner("Running screener..."):
            for t in tickers:
                try:
                    hist = yf.download(t, period="60d", interval="1d", progress=False)
                    if hist.empty:
                        continue
                    hist = compute_indicators(hist)
                    last = hist.iloc[-1]
                    avg_vol = hist["Volume"].tail(30).mean() if "Volume" in hist.columns else np.nan
                    screener_results.append({
                        "symbol": t,
                        "close": float(last.Close),
                        "rsi": float(last.RSI) if not pd.isna(last.RSI) else np.nan,
                        "avg_vol_30d": float(avg_vol)
                    })
                except Exception as e:
                    # non-fatal
                    continue
        if screener_results:
            sdf = pd.DataFrame(screener_results)
            filtered = sdf[(sdf["rsi"] <= screener_rsi) & (sdf["avg_vol_30d"] >= screener_vol_min)]
            if filtered.empty:
                st.info("No screener results after filtering.")
            else:
                st.dataframe(filtered.sort_values("avg_vol_30d", ascending=False))
        else:
            st.info("No screener results.")

# -------------------------
# Live check & execution
# -------------------------
signal = "HOLD"
latest_row = df.iloc[-1]
# safe scalar extraction
try:
    price = float(latest_row["Close"])
    lower = float(df["Lower"].iloc[-1]) if not pd.isna(df["Lower"].iloc[-1]) else None
    upper = float(df["Upper"].iloc[-1]) if not pd.isna(df["Upper"].iloc[-1]) else None
except Exception:
    price = None
    lower = None
    upper = None

if run_check:
    if price is None:
        st.error("Price or indicators not available for signal calculation.")
    else:
        if lower is not None and price < lower and not pd.isna(latest_row.get("RSI", np.nan)) and latest_row["RSI"] < 30:
            signal = "BUY"
        elif upper is not None and price > upper and not pd.isna(latest_row.get("RSI", np.nan)) and latest_row["RSI"] > 70:
            signal = "SELL"
        else:
            signal = "HOLD"
        st.info(f"Signal: {signal} | Price: {format_currency(price)} | RSI: {latest_row.get('RSI',np.nan):.1f}")

# Execution (Admin-only if Alpaca)
if run_check and confirm_execute:
    if role != "Admin":
        st.error("Only Admin users may execute trades.")
    else:
        if signal == "HOLD":
            st.warning("No trade signal to execute.")
        else:
            # Compute equity safely
            equity = 100000.0
            if mode == "Alpaca Paper":
                client = get_alpaca_client()
                if client:
                    try:
                        acct = client.get_account()
                        equity = float(acct.equity)
                    except Exception:
                        st.warning("Failed to fetch Alpaca account; using fallback equity.")
                else:
                    st.error("Alpaca client not configured. Check API keys.")
                    st.stop()
            # compute qty
            try:
                qty = int(max(1, np.floor(equity * (risk_pct/100.0) / float(price))))
            except Exception:
                qty = 1
            entry = float(price)
            if signal == "BUY":
                tp_price = round(entry * (1 + tp_pct/100.0), 4)
                sl_price = round(entry * (1 - sl_pct/100.0), 4)
                side = "buy"
            else:
                tp_price = round(entry * (1 - tp_pct/100.0), 4)
                sl_price = round(entry * (1 + sl_pct/100.0), 4)
                side = "sell"

            if mode == "Dry-run":
                rec = {
                    "time": datetime.utcnow().isoformat(),
                    "user": user["username"],
                    "mode": "dry-run",
                    "symbol": symbol,
                    "signal": signal,
                    "qty": qty,
                    "entry": entry,
                    "tp": tp_price,
                    "sl": sl_price,
                    "note": "simulated"
                }
                log_trade(rec)
                st.success(f"[SIMULATED] {signal} {qty} {symbol} @ {entry} | TP {tp_price} | SL {sl_price}")
            else:
                client = get_alpaca_client()
                if not client:
                    st.error("Alpaca client not available.")
                else:
                    res = submit_bracket_alpaca(client, symbol, qty, side, tp_price, sl_price)
                    if res.get("status") == "ok":
                        rec = {
                            "time": datetime.utcnow().isoformat(),
                            "user": user["username"],
                            "mode": "alpaca-paper",
                            "symbol": symbol,
                            "signal": signal,
                            "qty": qty,
                            "entry": entry,
                            "tp": tp_price,
                            "sl": sl_price,
                            "result": res
                        }
                        log_trade(rec)
                        st.success("[ALPACA PAPER] Bracket attempt placed, check paper account.")
                    else:
                        st.error("Failed to place Alpaca bracket: " + str(res.get("error")))

# -------------------------
# Footer / hints
# -------------------------
st.markdown("---")
st.caption("Notes: Use background workers ws_streamer.py and order_monitor.py for real-time updates. Redis recommended; sqlite fallback used if Redis not configured. Always test in Dry-run and Alpaca Paper before using real funds.")
