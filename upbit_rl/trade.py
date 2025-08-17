import argparse, os, numpy as np
from .broker.upbit_client import UpbitClient
from .broker.broker import Broker
from .data.ohlcv import get_ohlcv
from .rl.environment import DailyOHLCVEnv
from .rl.networks import make_actor
import tensorflow as tf

def main(args):
    client = UpbitClient()
    broker = Broker(client)
    df = get_ohlcv(args.ticker, count=max(120, args.window+1))
    env = DailyOHLCVEnv(df, fee=float(os.getenv("UPBIT_FEE","0.0005")), init_krw=1_000_000, window=args.window)

    actor = make_actor(input_dim=env._state().shape[0])
    actor_path = "models/actor_latest.h5"
    if os.path.exists(actor_path):
        actor = tf.keras.models.load_model(actor_path)
    else:
        print(f"[warn] {actor_path} not found. Using untrained actor (outputs ~0.5).")

    # Build a recent sequence
    s = env.reset()
    states = [s]
    for _ in range(args.window-1):
        ns, _, _, _ = env.step(0.0)
        states.append(ns)
    seq = np.array([np.vstack(states)[-args.window:]])

    target_w = float(actor.predict(seq, verbose=0)[0,0])
    bal = broker.get_balance(args.ticker)
    price = client.get_price(args.ticker)
    curr_w = (bal["coin_qty"] * price) / max(1e-9, bal["total_value"])
    delta_w = target_w - curr_w

    print(f"target_w={target_w:.3f} curr_w={curr_w:.3f} delta={delta_w:.3f}")

    if delta_w > args.thresh:
        amt = bal["total_value"] * delta_w
        print(f"Buying KRW {amt:.0f} @ {price:.0f}")
        broker.place_market_buy(args.ticker, amt)
    elif delta_w < -args.thresh:
        sell_qty = (bal["total_value"] * (-delta_w)) / price
        sell_qty = min(sell_qty, bal["coin_qty"])
        print(f"Selling {sell_qty:.6f} units @ {price:.0f}")
        broker.place_market_sell(args.ticker, sell_qty)
    else:
        print("No trade (within threshold).")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default=os.getenv("TICKER","KRW-BTC"))
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--thresh", type=float, default=0.01, help="min abs delta weight to trade")
    args = p.parse_args()
    main(args)