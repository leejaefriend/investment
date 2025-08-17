import numpy as np
import pandas as pd

class DailyOHLCVEnv:
    """Simple daily OHLCV environment with target weight action in [0,1]."""
    def __init__(self, ohlcv: pd.DataFrame, fee: float = 0.0005, init_krw: float = 1_000_000, window: int = 30):
        self.df = ohlcv.reset_index(drop=True).copy()
        self.N = len(self.df)
        self.fee = float(fee)
        self.init_krw = float(init_krw)
        self.window = int(window)
        self.reset()

    def reset(self):
        self.t = 0
        self.krw = self.init_krw
        self.qty = 0.0
        self.price = float(self.df.loc[self.t, "close"])
        self._update_total()
        return self._state()

    def step(self, target_weight: float):
        target_weight = float(np.clip(target_weight, 0.0, 1.0))
        self.price = float(self.df.loc[self.t, "close"])
        self._update_total()
        curr_w = 0.0 if self.total <= 0 else (self.qty * self.price) / self.total
        delta_w = target_weight - curr_w

        # trade
        if delta_w > 1e-9:  # buy
            buy_krw = self.total * delta_w
            fee = buy_krw * self.fee
            qty = (buy_krw - fee) / self.price
            self.krw -= buy_krw
            self.qty += qty
        elif delta_w < -1e-9:  # sell
            sell_qty = min(self.qty, self.total * (-delta_w) / self.price)
            proceeds = sell_qty * self.price
            fee = proceeds * self.fee
            self.krw += proceeds - fee
            self.qty -= sell_qty

        # next day
        prev_total = self.total
        self.t = min(self.t + 1, self.N - 1)
        self.price = float(self.df.loc[self.t, "close"])
        self._update_total()
        reward = (self.total - prev_total) / max(1e-9, prev_total)
        done = (self.t == self.N - 1)
        return self._state(), float(reward), bool(done), {"total": self.total, "weight": (self.qty * self.price) / max(self.total, 1e-9)}

    def _state(self):
        # Use [close, high, low, volume, position]
        row = self.df.loc[self.t, ["close","high","low","volume"]].astype(float).values
        pos = 0.0 if self.total <= 0 else (self.qty * self.price) / self.total
        return np.concatenate([row, [pos]]).astype(np.float32)

    def _update_total(self):
        self.total = self.krw + self.qty * self.price