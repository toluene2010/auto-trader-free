# run_trader.py
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from backtest import compute_returns, vectorized_backtest_signals
from alpaca.trading.client import TradingClient
import json

SYMBOL = os.getenv("SYMBOL", "AAPL")
INTERVAL = os.getenv("INTERVAL", "15m")
MODE = os.getenv("MODE", "dry")  # dry or alpaca
RISK = float(os.getenv("RISK_PCT", "2.0"))
LOGFILE = os.getenv("CRON_LOG", "cron_trade_log.csv")

def compute_indicators(df):
    df = df.copy()
    df["Mid"] = df["Close"].rolling(20).mean()
    df["Std"] = df["Close"].rolling(20).std()
    df["Upper"] = df["Mid"] + 2 * df["Std"]
    df["Lower"] = df["Mid"] - 2 * df["Std"]
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def sized_qty(equity, risk_percent, price):
    risk_amount = equity * (risk_percent / 100.0)
    qty = int(max(1, np.floor(risk_amount / price)))
    return qty

def submit_bracket(client, symbol, qty, side, tp, sl):
    try:
        order = client.submit_order(
            symbol=symbol, qty=qty, side=side, type="market", time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(tp)},
            stop_loss={"stop_price": str(sl)}
        )
        return {"status": "ok", "order": str(order)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    df = yf.download(SYMBOL, period="30d", interval=INTERVAL, progress=False)
    if df.empty:
        print("No data")
        return
    df = compute_indicators(df)
    latest = df.iloc[-1]
    signal = None
    if pd.notna(latest.Lower) and pd.notna(latest.RSI):
        if latest.Close < latest.Lower and latest.RSI < 30:
            signal = "buy"
        elif latest.Close > latest.Upper and latest.RSI > 70:
            signal = "sell"
    if not signal:
        print("No signal")
        return
    if MODE == "alpaca":
        API_KEY = os.getenv("API_KEY")
        SECRET_KEY = os.getenv("SECRET_KEY")
        client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        account = client.get_account()
        equity = float(account.equity)
    else:
        client = None
        equity = 100000.0
    qty = sized_qty(equity, RISK, latest.Close)
    entry = latest.Close
    if signal == "buy":
        tp = round(entry * 1.03, 4)
        sl = round(entry * 0.985, 4)
    else:
        tp = round(entry * 0.97, 4)
        sl = round(entry * 1.015, 4)
    record = {"time": datetime.utcnow().isoformat(), "symbol": SYMBOL, "signal": signal, "qty": qty, "entry": entry, "tp": tp, "sl": sl, "mode": MODE}
    pd.DataFrame([record]).to_csv(LOGFILE, mode="a", header=not os.path.exists(LOGFILE), index=False)
    if MODE == "alpaca":
        res = submit_bracket(client, SYMBOL, qty, signal, tp, sl)
        pd.DataFrame([{"time": datetime.utcnow().isoformat(), "result": str(res)}]).to_csv(LOGFILE, mode="a", header=False, index=False)
        print("Alpaca res:", res)
    else:
        print("Dry-run record saved:", record)

if __name__ == "__main__":
    main()
