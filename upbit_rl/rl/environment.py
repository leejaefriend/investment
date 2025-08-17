# upbit_rl/rl/environment.py  (교체/추가)
import numpy as np
import pandas as pd

class DailyOHLCVEnv:
    def __init__(self, ohlcv: pd.DataFrame, fee: float = 0.0005, init_krw: float = 1_000_000, window: int = 30, trade_penalty: float = 0.0002):
        df = ohlcv.reset_index(drop=True).copy()
        # 피처 전처리
        df["ret"] = np.log(df["close"]).diff().fillna(0.0)
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df["hl_spread"] = df["hl_spread"].fillna(0.0).clip(0, 0.2)
        df["logv"] = np.log(df["volume"] + 1.0)
        df["v_z"] = (df["logv"] - df["logv"].rolling(60, min_periods=1).mean()) / (df["logv"].rolling(60, min_periods=1).std(ddof=0) + 1e-8)
        self.df = df
        self.N = len(df)
        self.fee = float(fee)
        self.pen = float(trade_penalty)
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

        # 체결(수수료 포함)
        if delta_w > 1e-9:
            buy_krw = self.total * delta_w
            fee = buy_krw * self.fee
            qty = (buy_krw - fee) / self.price
            self.krw -= buy_krw
            self.qty += qty
        elif delta_w < -1e-9:
            sell_qty = min(self.qty, self.total * (-delta_w) / self.price)
            proceeds = sell_qty * self.price
            fee = proceeds * self.fee
            self.krw += proceeds - fee
            self.qty -= sell_qty

        prev_total = self.total
        self.t = min(self.t + 1, self.N - 1)
        self.price = float(self.df.loc[self.t, "close"])
        self._update_total()

        # 로그수익 보상 + 거래패널티
        reward = np.log(max(self.total, 1e-9) / max(prev_total, 1e-9)) - self.pen * abs(delta_w)
        done = (self.t == self.N - 1)
        return self._state(), float(reward), bool(done), {"total": self.total, "weight": (self.qty * self.price) / max(self.total, 1e-9)}

    def _state(self):
        row = self.df.loc[self.t, ["ret", "hl_spread", "v_z"]].astype(float).values
        pos = 0.0 if self.total <= 0 else (self.qty * self.price) / self.total
        return np.concatenate([row, [pos]]).astype(np.float32)

    def _update_total(self):
        self.total = self.krw + self.qty * self.price
