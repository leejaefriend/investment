# upbit_rl/trade.py (핵심 추가/변경만)
import time, os, numpy as np, tensorflow as tf
from .notify.slack import SlackNotifier
MIN_KRW = int(os.getenv("MIN_ORDER_KRW", "5000"))
COOLDOWN_SEC = int(os.getenv("TRADE_COOLDOWN_SEC", "60"))

last_trade_ts = 0
notifier = SlackNotifier()

# ... actor 로드/target_w 계산 동일 ...

if abs(delta_w) > args.thresh:
    amt = bal["total_value"] * abs(delta_w)
    if amt < MIN_KRW:
        print(f"Skip: below MIN_KRW ({amt:.0f} < {MIN_KRW})")
    elif time.time() - last_trade_ts < COOLDOWN_SEC:
        print("Skip: cooldown")
    else:
        if delta_w > 0:
            res = broker.place_market_buy(args.ticker, amt)
            action_txt = f"BUY KRW {amt:.0f}"
        else:
            sell_qty = min((bal["total_value"] * (-delta_w)) / price, bal["coin_qty"])
            res = broker.place_market_sell(args.ticker, sell_qty)
            action_txt = f"SELL {sell_qty:.6f}"

        last_trade_ts = time.time()
        msg = (f"*Trade* {action_txt}\n"
               f"- target_w: {target_w:.3f}, curr_w: {curr_w:.3f}\n"
               f"- price: {price:.0f}, fee: {res.fee:.0f}\n"
               f"- total: {bal['total_value']:.0f} → (post) ?")
        notifier.send(msg)
else:
    print("No trade (within threshold).")
