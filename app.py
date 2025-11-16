import streamlit as st
import yfinance as yf
import pandas as pd
import time
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient

# === Load secrets from Render Environment Variables ===
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    st.error("❌ API_KEY or SECRET_KEY not found in Render Environment Variables.")
    st.stop()

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

st.set_page_config(page_title="NGN Free Auto-Trader", layout="centered")
st.title("🇳🇬 Your Free Auto-Trader Is LIVE")

st.info("Bot checks market once whenever you press START.")

symbol = st.text_input("Symbol", "QQQ").upper()
risk = st.slider("Risk % per trade", 1, 5, 2)

start = st.button("START TRADING CHECK")

placeholder = st.empty()

if start:
    try:
        df = yf.download(symbol, period="30d", interval="5m")

        df["Mid"] = df.Close.rolling(20).mean()
        df["Std"] = df.Close.rolling(20).std()
        df["Upper"] = df.Mid + 2 * df.Std
        df["Lower"] = df.Mid - 2 * df.Std

        delta = df.Close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        latest = df.iloc[-1]
        price = latest.Close

        # === Signal Logic ===
        signal = "HOLD"
        if latest.Close < latest.Lower and latest.RSI < 30:
            signal = "BUY"
        elif latest.Close > latest.Upper and latest.RSI > 70:
            signal = "SELL"

        # === Account info ===
        account = trading_client.get_account()
        equity = float(account.equity)
        risk_amount = equity * (risk / 100)
        qty = max(1, int(risk_amount / price))

        # === Trading logic ===
        if signal == "BUY":
            order = MarketOrderRequest(
                symbol=symbol, 
                qty=qty, 
                side="buy", 
                time_in_force="day"
            )
            trading_client.submit_order(order)
            placeholder.success(f"BOUGHT {qty} {symbol} @ ${price:.2f}")

        elif signal == "SELL" and float(account.long_market_value) > 10:
            order = MarketOrderRequest(
                symbol=symbol, 
                qty=qty, 
                side="sell", 
                time_in_force="day"
            )
            trading_client.submit_order(order)
            placeholder.success(f"SOLD {qty} {symbol} @ ${price:.2f}")

        else:
            placeholder.info(f"HOLD | RSI {latest.RSI:.1f} | Price ${price:.2f}")

    except Exception as e:
        placeholder.error(str(e))
<<<<<<< HEAD:app.py
        time.sleep(60)
=======
>>>>>>> 50e0e6c (Updated App.py with environment variable authentication fix):App.py
