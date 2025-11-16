# pro_dashboard.py
import os
import json
import time
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from alpaca.trading.client import TradingClient
from auth_utils import authenticate, get_role
try:
    import redis
except Exception:
    redis = None
import sqlite3

st.set_page_config(page_title="Pro Auto-Trader (Auth + Push)", layout="wide")
LOGFILE = "journal_trades.csv"
REDIS_URL = os.getenv("REDIS_URL", None)
SQLITE_WS = os.getenv("WS_SQLITE", "ws_store.sqlite")
POLL_SECONDS = int(os.getenv("UI_POLL", "10"))

def get_redis():
    if not REDIS_URL or redis is None:
        return None
    return redis.from_url(REDIS_URL)

def sqlite_get_ticks():
    if not os.path.exists(SQLITE_WS):
        return {}
    conn = sqlite3.connect(SQLITE_WS)
    c = conn.cursor()
    try:
        c.execute("SELECT symbol,payload FROM ticks")
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
                return json.loads(raw)
        except Exception:
            pass
    # fallback:
    ticks = sqlite_get_ticks()
    return ticks.get(symbol)

def format_currency(x):
    return f"${x:,.2f}"

# --------------------------
# Auth
# --------------------------
if "auth" not in st.session_state:
    st.session_state.auth = None

if st.session_state.auth is None:
    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        user = authenticate(username, password)
        if user:
            st.session_state.auth = user
            st.sidebar.success(f"Welcome {username} ({user['role']})")
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid credentials")
    st.stop()  # require login

user = st.session_state.auth
role = user.get("role", "Reviewer")

st.sidebar.write(f"Signed in as: {user['username']} ({role})")
if role == "Admin":
    st.sidebar.success("Role: Admin — full access")
else:
    st.sidebar.info("Role: Reviewer — view + simulate")

# --------------------------
# UI layout
# --------------------------
left, center, right = st.columns([0.18, 0.56, 0.26])

with left:
    st.subheader("Controls")
    symbol = st.text_input("Symbol", value="AAPL").upper()
    interval = st.selectbox("Interval", ["1m","5m","15m","1h","1d"], index=2)
    mode = st.selectbox("Mode", ["Dry-run", "Alpaca Paper"])
    risk_pct = st.slider("Risk % of equity", 1, 5, 2)
    tp_pct = st.number_input("TP %", min_value=0.5, max_value=50.0, value=3.0, step=0.5)
    sl_pct = st.number_input("SL %", min_value=0.2, max_value=50.0, value=1.5, step=0.1)
    run_check = st.button("Run Live Check")
    confirm_execute = st.button("Confirm & Execute")

with center:
    st.subheader(f"{symbol} Candles (Plotly)")
    # Attempt to show last ticks from Redis/SQLite
    tick = get_latest_tick(symbol)
    # download price history for chart
    period = "60d" if interval in ["1m","5m","15m"] else "365d"
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        st.error("No market data")
        st.stop()
    # compute indicators
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Upper"] = df["Close"].rolling(20).mean() + 2*df["Close"].rolling(20).std()
    df["Lower"] = df["Close"].rolling(20).mean() - 2*df["Close"].rolling(20).std()

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
    if "SMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
    if "SMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Latest tick (from stream):", tick)

with right:
    st.subheader("Account & Journal")
    if mode == "Alpaca Paper":
        API_KEY = os.getenv("API_KEY"); SECRET_KEY = os.getenv("SECRET_KEY")
        if not API_KEY or not SECRET_KEY:
            st.error("Alpaca keys missing")
            st.stop()
        client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        try:
            acct = client.get_account()
            st.metric("Equity", format_currency(float(acct.equity)))
            st.metric("Cash", format_currency(float(acct.cash)))
        except Exception as e:
            st.error("Failed to load account: " + str(e))
    else:
        st.info("Dry-run mode (simulated account $100k)")
        st.metric("Equity", "$100,000.00")
    # show recent journal
    if os.path.exists(LOGFILE):
        j = pd.read_csv(LOGFILE, parse_dates=["time"])
        st.subheader("Journal (last 10)")
        st.dataframe(j.sort_values("time", ascending=False).head(10))
    # show order events (from Redis or sqlite - published by order_monitor)
    r = get_redis()
    events = []
    if r:
        try:
            raw = r.lrange("order_events", 0, -1)
            for item in raw[-50:]:
                try:
                    events.append(json.loads(item))
                except:
                    events.append({"raw": str(item)})
        except Exception:
            pass
    else:
        # sqlite fallback
        dbfile = os.getenv("ORDER_SQLITE", "orders_store.sqlite")
        if os.path.exists(dbfile):
            conn = sqlite3.connect(dbfile)
            c = conn.cursor()
            try:
                c.execute("SELECT payload, ts FROM order_events ORDER BY id DESC LIMIT 50")
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
        st.subheader("Recent Order Events")
        st.write(events[-10:])

# Execution logic (must be Admin to execute)
if confirm_execute:
    if role != "Admin":
        st.error("Only Admin users can execute live trades.")
    else:
        latest = df.iloc[-1]
        signal = "HOLD"
        if latest["Close"] < df["Lower"].iloc[-1]:
            signal = "BUY"
        elif latest["Close"] > df["Upper"].iloc[-1]:
            signal = "SELL"
        st.info(f"Signal: {signal} @ {latest['Close']:.2f}")
        if signal in ("BUY","SELL"):
            qty = int(max(1, np.floor(100000 * (risk_pct/100) / latest["Close"]))) if mode=="Dry-run" else int(max(1, np.floor(float(TradingClient(os.getenv('API_KEY'),os.getenv('SECRET_KEY'),paper=True).get_account().equity) * (risk_pct/100) / latest["Close"])))
            entry = float(latest["Close"])
            if signal == "BUY":
                tp = round(entry * (1 + tp_pct/100.0), 4)
                sl = round(entry * (1 - sl_pct/100.0), 4)
                side = "buy"
            else:
                tp = round(entry * (1 - tp_pct/100.0), 4)
                sl = round(entry * (1 + sl_pct/100.0), 4)
                side = "sell"
            rec = {"time": datetime.utcnow().isoformat(), "user": st.session_state.auth["username"], "symbol": symbol, "signal": signal, "qty": qty, "entry": entry, "tp": tp, "sl": sl, "mode": mode}
            # log
            pd.DataFrame([rec]).to_csv(LOGFILE, mode="a", header=not os.path.exists(LOGFILE), index=False)
            # execute in Alpaca if requested
            if mode == "Alpaca Paper":
                client = TradingClient(os.getenv("API_KEY"), os.getenv("SECRET_KEY"), paper=True)
                try:
                    # try bracket
                    res = client.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="day",
                                             order_class="bracket", take_profit={"limit_price":str(tp)}, stop_loss={"stop_price":str(sl)})
                    st.success("Alpaca bracket order placed.")
                except Exception as e:
                    st.error("Failed to place bracket: " + str(e))
            else:
                st.success("Simulated trade logged (dry-run).")

st.markdown("---")
st.caption("Notes: Use ws_streamer.py and order_monitor.py as background workers. Use Render Cron or a background service. Admin role required for live executes.")
