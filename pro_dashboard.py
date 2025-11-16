# pro_dashboard.py
"""
Pro Trading Suite (Plotly + Auth + Redis/SQLite push fallback + Order events + Journal + OCO)
- Safe for Render deployment
- Requires: auth_utils.py (authenticate)
- Optional: Redis (provide REDIS_URL). If not present, uses SQLite fallback.
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
from auth_utils import authenticate

# -------------------------
# Configuration
# -------------------------
st.set_page_config(page_title="Pro Trading Suite", layout="wide")
LOGFILE = os.getenv("JOURNAL_FILE", "journal_trades.csv")
TICKS_DB = os.getenv("WS_SQLITE", "ws_store.sqlite")
ORDER_DB = os.getenv("ORDER_SQLITE", "orders_store.sqlite")
REDIS_URL = os.getenv("REDIS_URL", None)

DEFAULT_POLL_SECONDS = int(os.getenv("UI_POLL", "10"))

# -------------------------
# Redis / SQLite fallback
# -------------------------
try:
    import redis
except Exception:
    redis = None

def get_redis():
    if not REDIS_URL or redis is None:
        return None
    try:
        return redis.from_url(REDIS_URL)
    except Exception:
        return None

def sqlite_set_ticks(latest_dict):
    conn = sqlite3.connect(TICKS_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            symbol TEXT PRIMARY KEY,
            payload TEXT,
            updated TIMESTAMP
        )
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
                if isinstance(raw, bytes):
                    raw = raw.decode()
                return json.loads(raw)
        except Exception:
            pass

    ticks = sqlite_get_ticks()
    return ticks.get(symbol)

# -------------------------
# Indicators
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
    except:
        return str(x)

# -------------------------
# Journal
# -------------------------
def log_trade(record):
    df = pd.DataFrame([record])
    header = not os.path.exists(LOGFILE)
    df.to_csv(LOGFILE, mode="a", header=header, index=False)

def load_journal():
    if os.path.exists(LOGFILE):
        try:
            return pd.read_csv(LOGFILE, parse_dates=["time"])
        except:
            return pd.read_csv(LOGFILE)
    return pd.DataFrame()

# -------------------------
# Alpaca
# -------------------------
def get_alpaca_client():
    API_KEY = os.getenv("API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")
    if not API_KEY or not SECRET_KEY:
        return None
    try:
        return TradingClient(API_KEY, SECRET_KEY, paper=True)
    except:
        return None

def submit_bracket_alpaca(client, symbol, qty, side, tp_price, sl_price):
    try:
        order = client.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(tp_price)},
            stop_loss={"stop_price": str(sl_price)}
        )
        return {"status": "ok", "order": str(order)}
    except Exception:
        try:
            market = client.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="day")
            opp = "sell" if side == "buy" else "buy"
            tp = None
            sl = None
            try:
                tp = client.submit_order(symbol=symbol, qty=qty, side=opp, type="limit", time_in_force="gtc", limit_price=str(tp_price))
            except:
                pass
            try:
                sl = client.submit_order(symbol=symbol, qty=qty, side=opp, type="stop", time_in_force="gtc", stop_price=str(sl_price))
            except:
                pass
            return {"status": "ok", "market": str(market), "tp": str(tp), "sl": str(sl)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# -------------------------
# Authentication
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
            if "username" not in user:
                user["username"] = username
            if "role" not in user:
                user["role"] = "Reviewer"
            st.session_state.auth = user
            st.sidebar.success(f"Welcome {user['username']} ({user['role']})")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")

    st.stop()

user = st.session_state.auth
role = user.get("role", "Reviewer")

# -------------------------
# Layout
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
    st.subheader("Screener")
    watchlist_input = st.text_area("Tickers (comma separated)", value="AAPL,MSFT,SPY,QQQ")
    screener_rsi = st.slider("RSI <=", 10, 50, 30)
    screener_vol_min = st.number_input("Min Avg Volume (30d)", min_value=1000, value=100000, step=1000)
    run_screener = st.button("Run Screener")

    st.markdown("---")
    st.subheader("Trading Engine")
    run_check = st.button("Run Live Signal Check")
    confirm_execute = st.checkbox("Confirm & Execute (Admin only)")

    st.markdown("---")
    st.caption(f"Signed in as: {user['username']} ({role})")

# ------------------------------------------------------
# END OF PART 1 — Scroll down for PART 2
# ------------------------------------------------------
# -------------------------
# CENTER SECTION (Chart + RSI + Live Tick)
# -------------------------
with center:
    st.header(f"{symbol} Chart")

    # Download historical data safely
    period = "60d" if interval in ["1m", "5m", "15m"] else "365d"

    with st.spinner("Fetching market data..."):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
        except Exception as e:
            st.error(f"Download error: {e}")
            st.stop()

    if df.empty:
        st.error("No market data returned.")
        st.stop()

    df = compute_indicators(df)

    # -------------------------
    # Build Plotly Candlestick
    # -------------------------
    def build_figure(df):
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="green",
            decreasing_line_color="red",
            name="Candle"
        )])

        # Add SMA lines
        if "SMA20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], mode="lines", name="SMA20"))
        if "SMA50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], mode="lines", name="SMA50"))

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], mode="lines", name="Upper BB", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], mode="lines", name="Lower BB", line=dict(dash="dash")))

        # Create shaded area
        fig.add_traces([
            go.Scatter(x=df.index, y=df["Upper"], mode='lines', line=dict(width=0), showlegend=False),
            go.Scatter(x=df.index, y=df["Lower"], mode='lines', fill='tonexty',
                       fillcolor='rgba(173,216,230,0.1)', line=dict(width=0), showlegend=False)
        ])

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=600
        )
        return fig

    fig = build_figure(df)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # RSI Plot
    # -------------------------
    if "RSI" in df.columns:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(height=200)
        st.plotly_chart(rsi_fig, use_container_width=True)

    # -------------------------
    # Live Tick (Redis or SQLite)
    # -------------------------
    tick = get_latest_tick(symbol)
    if tick:
        st.success(f"Live Tick → {tick.get('symbol')} @ {tick.get('close')}   (Time: {tick.get('ts')})")
    else:
        st.info("No live ticks yet (Redis/sqlite empty).")

# -------------------------
# RIGHT COLUMN — Account + Journal + Order Events
# -------------------------
with right:
    st.header("Account")

    client = get_alpaca_client() if mode == "Alpaca Paper" else None

    if client:
        try:
            acct = client.get_account()
            st.metric("Equity", format_currency(acct.equity))
            st.metric("Cash", format_currency(acct.cash))
            st.metric("Buying Power", format_currency(acct.buying_power))
        except Exception as e:
            st.error(f"Alpaca error: {e}")
    else:
        st.info("Dry-run mode (fake equity = $100,000)")

    st.markdown("---")
    st.subheader("Journal (recent)")

    journal = load_journal()
    if not journal.empty:
        try:
            st.dataframe(journal.sort_values("time", ascending=False).head(10))
            csv = journal.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="journal.csv")
        except:
            st.write(journal.head(10))
    else:
        st.info("No trades logged yet.")

    st.markdown("---")
    st.subheader("Order Events")

    # Load events (Redis first)
    events = []
    r = get_redis()

    if r:
        try:
            raw_list = r.lrange("order_events", -50, -1)
            for row in raw_list:
                row = row.decode() if isinstance(row, bytes) else row
                try:
                    events.append(json.loads(row))
                except:
                    events.append({"raw": row})
        except:
            pass
    else:
        # SQLite fallback
        if os.path.exists(ORDER_DB):
            conn = sqlite3.connect(ORDER_DB)
            c = conn.cursor()
            try:
                c.execute("SELECT payload FROM order_events ORDER BY id DESC LIMIT 50")
                rows = c.fetchall()
                for r2 in rows:
                    try:
                        events.append(json.loads(r2[0]))
                    except:
                        events.append({"raw": r2[0]})
            except:
                pass
            conn.close()

    if events:
        st.json(events[:20])
    else:
        st.info("No order events found.")

# -------------------------
# Screener Logic
# -------------------------
if run_screener:
    tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    results = []

    if not tickers:
        st.warning("Enter tickers first.")
    else:
        with st.spinner("Screening..."):
            for t in tickers:
                try:
                    data = yf.download(t, period="60d", interval="1d", progress=False)
                    if data.empty:
                        continue
                    data = compute_indicators(data)
                    last = data.iloc[-1]
                    avg_vol = data["Volume"].tail(30).mean()

                    results.append({
                        "symbol": t,
                        "close": float(last.Close),
                        "rsi": float(last.RSI) if not pd.isna(last.RSI) else np.nan,
                        "avg_vol_30d": float(avg_vol)
                    })
                except:
                    pass

        if results:
            sdf = pd.DataFrame(results)
            filtered = sdf[(sdf["rsi"] <= screener_rsi) &
                           (sdf["avg_vol_30d"] >= screener_vol_min)]
            st.dataframe(filtered if not filtered.empty else sdf)
        else:
            st.info("No screener results.")

# -------------------------
# Live Signal Logic
# -------------------------
signal = "HOLD"
latest_row = df.iloc[-1]

try:
    price = float(latest_row["Close"])
except:
    price = None

try:
    lower = float(df["Lower"].iloc[-1])
except:
    lower = None

try:
    upper = float(df["Upper"].iloc[-1])
except:
    upper = None

if run_check:
    if price is None:
        st.error("Price unavailable.")
    else:
        rsi_val = latest_row.get("RSI", np.nan)
        try:
            rsi_val = float(rsi_val)
        except:
            rsi_val = np.nan

        if lower is not None and price < lower and rsi_val < 30:
            signal = "BUY"
        elif upper is not None and price > upper and rsi_val > 70:
            signal = "SELL"
        else:
            signal = "HOLD"

        st.info(f"Signal: {signal} | Price: {format_currency(price)} | RSI: {rsi_val:.1f}")

# -------------------------
# Trade Execution (Admin only)
# -------------------------
if run_check and confirm_execute:

    if role != "Admin":
        st.error("Only Admin can execute trades.")
        st.stop()

    if signal == "HOLD":
        st.warning("No actionable signal.")
        st.stop()

    # Determine account equity
    equity = 100000.0
    if mode == "Alpaca Paper":
        client = get_alpaca_client()
        if client:
            try:
                acct = client.get_account()
                equity = float(acct.equity)
            except:
                st.warning("Using fallback equity $100,000.")
        else:
            st.error("Alpaca not configured.")
            st.stop()

    # position size
    try:
        qty = int(max(1, np.floor(equity * (risk_pct / 100.0) / price)))
    except:
        qty = 1

    # TP/SL calculation
    entry = price
    if signal == "BUY":
        tp_price = round(entry * (1 + tp_pct / 100.0), 4)
        sl_price = round(entry * (1 - sl_pct / 100.0), 4)
        side = "buy"
    else:
        tp_price = round(entry * (1 - tp_pct / 100.0), 4)
        sl_price = round(entry * (1 + sl_pct / 100.0), 4)
        side = "sell"

    if mode == "Dry-run":
        rec = {
            "time": datetime.utcnow().isoformat(),
            "user": user["username"],
            "symbol": symbol,
            "signal": signal,
            "entry": entry,
            "qty": qty,
            "tp": tp_price,
            "sl": sl_price,
            "mode": "dry-run",
            "note": "simulated"
        }
        log_trade(rec)
        st.success(f"[SIMULATION] {signal} {qty} {symbol} @ {entry} → TP {tp_price}, SL {sl_price}")

    else:
        client = get_alpaca_client()
        if not client:
            st.error("Alpaca unavailable.")
        else:
            res = submit_bracket_alpaca(client, symbol, qty, side, tp_price, sl_price)
            if res.get("status") == "ok":
                rec = {
                    "time": datetime.utcnow().isoformat(),
                    "user": user["username"],
                    "symbol": symbol,
                    "signal": signal,
                    "entry": entry,
                    "qty": qty,
                    "tp": tp_price,
                    "sl": sl_price,
                    "mode": "alpaca-paper",
                    "raw": str(res)
                }
                log_trade(rec)
                st.success("Order sent (check Alpaca Paper).")
            else:
                st.error(f"Order failed: {res.get('error')}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Pro Trading Suite — Powered by Redis/SQLite fallback, Alpaca, Plotly, Streamlit 1.41+")
