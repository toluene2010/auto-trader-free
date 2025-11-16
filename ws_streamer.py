# ws_streamer.py
import os
import time
import json
from datetime import datetime
import yfinance as yf
try:
    import redis
except Exception:
    redis = None
import sqlite3

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "5"))
REDIS_URL = os.getenv("REDIS_URL", None)
SYMBOLS = os.getenv("WS_SYMBOLS", "AAPL,MSFT").split(",")

# Redis helper
def get_redis():
    if not REDIS_URL or redis is None:
        return None
    return redis.from_url(REDIS_URL)

# SQLite fallback
DB_FILE = os.getenv("WS_SQLITE", "ws_store.sqlite")
def sqlite_set(latest_dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS ticks (symbol TEXT PRIMARY KEY, payload TEXT, updated TIMESTAMP)
    """)
    for sym, payload in latest_dict.items():
        c.execute("INSERT OR REPLACE INTO ticks(symbol,payload,updated) VALUES(?,?,?)",
                  (sym, json.dumps(payload), datetime.utcnow()))
    conn.commit()
    conn.close()

def sqlite_get_all():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT symbol, payload FROM ticks")
    rows = c.fetchall()
    conn.close()
    return {r[0]: json.loads(r[1]) for r in rows}

def main():
    r = get_redis()
    while True:
        latest = {}
        for sym in SYMBOLS:
            try:
                df = yf.download(sym, period="2d", interval="1m", progress=False)
                if df.empty:
                    continue
                last = df.iloc[-1]
                payload = {
                    "symbol": sym,
                    "close": float(last.Close),
                    "open": float(last.Open),
                    "high": float(last.High),
                    "low": float(last.Low),
                    "volume": int(last.Volume),
                    "ts": datetime.utcnow().isoformat()
                }
                latest[sym] = payload
            except Exception as e:
                print("poll error", sym, e)
        if r:
            try:
                for s, p in latest.items():
                    r.hset("market_ticks", s, json.dumps(p))
            except Exception as e:
                print("redis write error", e)
        else:
            # sqlite fallback
            sqlite_set(latest)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    print("Starting polling streamer (polling x Alpaca/YFinance).")
    print("If you prefer Alpaca streaming websocket, update this file with alpaca-py stream client.")
    main()
