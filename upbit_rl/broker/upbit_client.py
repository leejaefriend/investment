import os, time
from typing import Optional, Dict

try:
    import pyupbit
except ImportError:
    pyupbit = None

class UpbitClient:
    """Thin wrapper around pyupbit for price, balance, and market orders."""
    def __init__(self, access: Optional[str]=None, secret: Optional[str]=None):
        access = access or os.getenv("UPBIT_ACCESS")
        secret = secret or os.getenv("UPBIT_SECRET")
        self.ticker_price_cache: Dict[str, tuple[float, float]] = {}
        self.upbit = None
        if pyupbit and access and secret:
            self.upbit = pyupbit.Upbit(access, secret)

    def get_price(self, ticker: str = "KRW-BTC") -> float:
        now = time.time()
        if ticker in self.ticker_price_cache and now - self.ticker_price_cache[ticker][0] < 5:
            return float(self.ticker_price_cache[ticker][1])
        if not pyupbit:
            raise RuntimeError("pyupbit not installed. pip install pyupbit")
        price = pyupbit.get_current_price(ticker)
        if price is None:
            raise RuntimeError(f"Failed to fetch price for {ticker}")
        self.ticker_price_cache[ticker] = (now, float(price))
        return float(price)

    def get_balance(self, ticker: str = "KRW-BTC"):
        if not self.upbit:
            raise RuntimeError("Real balance requires UPBIT_ACCESS/UPBIT_SECRET and pyupbit.")
        krw = float(self.upbit.get_balance("KRW") or 0)
        coin = ticker.split("-")[1]
        qty = float(self.upbit.get_balance(coin) or 0)
        price = self.get_price(ticker)
        return {
            "krw": krw,
            "coin_qty": qty,
            "coin_value": qty * price,
            "total_value": krw + qty * price,
        }

    def market_buy(self, ticker: str, krw_amount: float):
        if not self.upbit:
            raise RuntimeError("market_buy requires UPBIT_ACCESS/UPBIT_SECRET.")
        return self.upbit.buy_market_order(ticker, krw_amount)

    def market_sell(self, ticker: str, qty: float):
        if not self.upbit:
            raise RuntimeError("market_sell requires UPBIT_ACCESS/UPBIT_SECRET.")
        return self.upbit.sell_market_order(ticker, qty)