import os
from upbit_rl.broker.upbit_client import UpbitClient
from upbit_rl.data.ohlcv import get_ohlcv

if __name__ == "__main__":
    t = os.getenv("TICKER","KRW-BTC")
    client = UpbitClient()
    price = client.get_price(t)
    print("current price", t, price)
    df = get_ohlcv(t, count=30, force=True)
    print("ohlcv rows", len(df), "cols", list(df.columns))