import argparse, os
import numpy as np
import tensorflow as tf

from .data.ohlcv import get_ohlcv
from .rl.environment import DailyOHLCVEnv
from .rl.networks import make_actor, make_critic

def discounted(rs, gamma=0.99):
    out=[]; g=0.0
    for r in rs[::-1]:
        g = r + gamma*g
        out.append(g)
    return np.array(out[::-1], dtype=np.float32)

def pad_seq(states, window, dim):
    x = np.vstack(states)[-window:]
    if len(x) < window:
        pad = np.zeros((window-len(x), dim), dtype=np.float32)
        x = np.vstack([pad, x])
    return x[None, ...]

def main(args):
    df = get_ohlcv(args.ticker, count=args.count)
    env = DailyOHLCVEnv(df, fee=float(os.getenv("UPBIT_FEE", "0.0005")), init_krw=1_000_000, window=args.window)

    input_dim = env._state().shape[0]
    actor = make_actor(input_dim)
    critic = make_critic(input_dim)

    opt_a = tf.keras.optimizers.Adam(args.lr)
    opt_c = tf.keras.optimizers.Adam(args.lr)

    for ep in range(args.epochs):
        s = env.reset()
        done = False
        states = [s]
        s_hist=[]; a_hist=[]; r_hist=[]; v_hist=[]

        while not done:
            seq = pad_seq(states, args.window, input_dim)
            v = float(critic.predict(seq, verbose=0)[0,0])
            a = float(actor.predict(seq, verbose=0)[0,0])
            s_next, r, done, info = env.step(a)
            s_hist.append(seq); a_hist.append([a]); r_hist.append(r); v_hist.append(v)
            states.append(s_next)

        returns = discounted(r_hist, gamma=args.gamma)
        adv = returns - np.array(v_hist, dtype=np.float32)

        S = np.concatenate(s_hist, axis=0)
        A = np.array(a_hist, dtype=np.float32)
        R = returns.astype(np.float32)
        ADV = adv.astype(np.float32)

        with tf.GradientTape() as ta, tf.GradientTape() as tc:
            pi = actor(S, training=True)  # (T,1) in [0,1]
            # Bernoulli-like log-prob around action A (continuous [0,1] approx)
            logp = A * tf.math.log(pi + 1e-6) + (1.0 - A) * tf.math.log(1.0 - pi + 1e-6)
            loss_a = -tf.reduce_mean(logp[:,0] * ADV)
            v_pred = critic(S, training=True)[:,0]
            loss_c = tf.keras.losses.MSE(v_pred, R)

        opt_a.apply_gradients(zip(ta.gradient(loss_a, actor.trainable_weights), actor.trainable_weights))
        opt_c.apply_gradients(zip(tc.gradient(loss_c, critic.trainable_weights), critic.trainable_weights))

        ep_ret = float(np.sum(r_hist))
        print(f"[EP {ep}] return={ep_ret:.5f}  loss_a={float(loss_a):.5f}  loss_c={float(loss_c):.5f}")

    os.makedirs("models", exist_ok=True)
    actor.save("models/actor_latest.h5")
    critic.save("models/critic_latest.h5")
    print("Saved models to models/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default=os.getenv("TICKER","KRW-BTC"))
    p.add_argument("--count", type=int, default=400)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    args = p.parse_args()
    main(args)