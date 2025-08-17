import os, pathlib
import pandas as pd
from typing import Optional

def get_ohlcv(ticker: str = "KRW-BTC", count: int = 400, cache_dir: str = "data/cache", force: bool = False) -> pd.DataFrame:
    """Fetch day OHLCV using pyupbit and cache to CSV for reproducibility."""
    path = pathlib.Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    f = path / f"{ticker.replace('-', '_')}_day_{count}.csv"
    if f.exists() and not force:
        return pd.read_csv(f)
    import pyupbit
    df = pyupbit.get_ohlcv(ticker, interval="day", count=count)
    if df is None or df.empty:
        raise RuntimeError("Failed to fetch OHLCV")
    df = df.reset_index().rename(columns={"index":"date"})
    df.to_csv(f, index=False)
    return df