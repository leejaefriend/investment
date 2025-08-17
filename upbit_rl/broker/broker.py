import os, time
from dataclasses import dataclass
from typing import Optional

FEE_RATE = float(os.getenv("UPBIT_FEE", "0.0005"))  # default 0.05%
PAPER = os.getenv("PAPER_TRADE", "1") == "1"

@dataclass
class OrderResult:
    side: str
    price: float
    qty: float
    fee: float
    ts: float

class Broker:
    def __init__(self, client, init_krw: Optional[float]=None):
        self.client = client
        self.paper = PAPER
        self.krw = None
        self.coin = None
        self.init_krw = init_krw or float(os.getenv("PAPER_INIT_KRW", "1000000"))

    def _ensure_paper_state(self, ticker: str):
        if self.krw is None:
            self.krw = float(self.init_krw)
            self.coin = 0.0

    def get_balance(self, ticker: str = "KRW-BTC"):
        if self.paper:
            self._ensure_paper_state(ticker)
            price = self.client.get_price(ticker)
            return {
                "krw": self.krw,
                "coin_qty": self.coin,
                "coin_value": self.coin * price,
                "total_value": self.krw + self.coin * price,
            }
        else:
            return self.client.get_balance(ticker)

    def place_market_buy(self, ticker: str, krw_amount: float) -> OrderResult:
        price = self.client.get_price(ticker)
        fee = krw_amount * FEE_RATE
        qty = (krw_amount - fee) / price
        if self.paper:
            self._ensure_paper_state(ticker)
            self.krw -= krw_amount
            self.coin += qty
        else:
            self.client.market_buy(ticker, krw_amount)
        return OrderResult("buy", price, qty, fee, time.time())

    def place_market_sell(self, ticker: str, qty: float) -> OrderResult:
        price = self.client.get_price(ticker)
        proceeds = price * qty
        fee = proceeds * FEE_RATE
        if self.paper:
            self._ensure_paper_state(ticker)
            self.krw += proceeds - fee
            self.coin -= qty
        else:
            self.client.market_sell(ticker, qty)
        return OrderResult("sell", price, qty, fee, time.time())