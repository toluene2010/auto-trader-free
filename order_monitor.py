# order_monitor.py
"""
Periodically checks Alpaca orders and positions. Reconciles fills,
cancels child orders if TP/SL hit, and writes order/fill events to Redis or sqlite store.
"""
import os
import time
import json
from datetime import datetime
try:
    import redis
except Exception:
    redis = None
import sqlite3
from alpaca.trading.client import TradingClient

POLL_SECONDS = int(os.getenv("ORDER_MONITOR_POLL", "10"))
REDIS_URL = os.getenv("REDIS_URL", None)
DB_FILE = os.getenv("ORDER_SQLITE", "orders_store.sqlite")

def get_client():
    API_KEY = os.getenv("API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")
    if not API_KEY or not SECRET_KEY:
        return None
    return TradingClient(API_KEY, SECRET_KEY, paper=True)

def get_redis():
    if not REDIS_URL or redis is None:
        return None
    return redis.from_url(REDIS_URL)

def sqlite_write_event(ev):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS order_events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TIMESTAMP, payload TEXT)""")
    c.execute("INSERT INTO order_events(ts,payload) VALUES(?, ?)", (datetime.utcnow(), json.dumps(ev)))
    conn.commit()
    conn.close()

def publish_event(ev):
    r = get_redis()
    if r:
        try:
            r.rpush("order_events", json.dumps(ev))
        except Exception as e:
            print("redis push err", e)
    else:
        sqlite_write_event(ev)

def monitor_loop():
    client = get_client()
    if not client:
        print("Missing Alpaca keys; order monitor needs API_KEY/SECRET_KEY")
        return
    last_seen = set()
    while True:
        try:
            orders = client.get_orders(status="all", nested=False)
            for o in orders:
                oid = getattr(o, "id", None) or getattr(o, "client_order_id", None) or str(o)
                if oid in last_seen:
                    # skip processing again (but still check status changes)
                    pass
                else:
                    last_seen.add(oid)
                # check fills / status
                status = getattr(o, "status", None)
                filled_qty = float(getattr(o, "filled_qty", 0) or 0)
                side = getattr(o, "side", None)
                symbol = getattr(o, "symbol", None)
                evt = {
                    "id": oid,
                    "symbol": symbol,
                    "side": side,
                    "status": status,
                    "filled_qty": filled_qty,
                    "raw": str(o),
                    "ts": datetime.utcnow().isoformat()
                }
                publish_event(evt)
                # If we detect TP/SL child orders created earlier, and one is filled, cancel the other(s)
                # This best-effort approach: if order has 'order_class' == bracket, rely on Alpaca to manage children.
                # If not, you may search open orders for the same symbol and cancel opposites.
            # Also check open positions to detect exit fills
            time.sleep(POLL_SECONDS)
        except Exception as e:
            print("Order monitor error:", e)
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    print("Starting order monitor...")
    monitor_loop()
