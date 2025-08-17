# upbit_rl/train_ppo.py
import argparse, os, numpy as np, tensorflow as tf
from tensorflow.keras import backend as K
from .data.ohlcv import get_ohlcv
from .rl.environment import DailyOHLCVEnv
from .rl.networks import make_actor_beta, make_critic

def beta_log_prob(a, alpha, beta):
    # 안정성 위해 epsilon
    eps = 1e-6
    a = K.clip(a, eps, 1-eps)
    logB = tf.math.lgamma(alpha) + tf.math.lgamma(beta) - tf.math.lgamma(alpha+beta)
    return (alpha-1.0)*tf.math.log(a) + (beta-1.0)*tf.math.log(1.0-a) - logB

def discounted(rs, gamma=0.99):
    out=[]; g=0.0
    for r in rs[::-1]:
        g = r + gamma*g; out.append(g)
    return np.array(out[::-1], dtype=np.float32)

def main(args):
    df = get_ohlcv(args.ticker, count=args.count)
    env = DailyOHLCVEnv(df, fee=float(os.getenv("UPBIT_FEE","0.0005")), init_krw=1_000_000, window=args.window)

    input_dim = env._state().shape[0]
    actor = make_actor_beta(input_dim)
    critic = make_critic(input_dim)
    opt_a = tf.keras.optimizers.Adam(args.lr)
    opt_c = tf.keras.optimizers.Adam(args.lr)

    clip_eps = 0.2
    for ep in range(args.epochs):
        s = env.reset(); done=False
        states=[s]; A=[]; R=[]; old_logp=[]; V=[]

        # rollout
        while not done:
            S = np.array([np.vstack(states)[-args.window:]])
            ab = actor.predict(S, verbose=0)[0]
            alpha = ab[0] + 1.0; beta = ab[1] + 1.0
            a = np.random.beta(alpha, beta)
            v = float(critic.predict(S, verbose=0)[0,0])
            s2, r, done, info = env.step(a)
            A.append([a]); R.append(r); V.append(v); states.append(s2)
            # old_logp 저장
            old_logp.append(beta_log_prob(tf.constant([[a]],dtype=tf.float32),
                                          tf.constant([[alpha]],dtype=tf.float32),
                                          tf.constant([[beta]],dtype=tf.float32)).numpy()[0,0])

        # advantage
        returns = discounted(R, gamma=args.gamma)
        adv = returns - np.array(V, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        S_all = np.array([np.vstack(states)[:-1]])  # 마지막 state 제외
        # 여러 epoch로 미니배치 업데이트
        for _ in range(args.update_epochs):
            with tf.GradientTape() as ta, tf.GradientTape() as tc:
                ab = actor(S_all, training=True)
                alpha = ab[:,:,0:1] + 1.0
                beta  = ab[:,:,1:2] + 1.0
                A_tf = tf.constant(A, dtype=tf.float32)  # (T,1)
                old_logp_tf = tf.constant(old_logp, dtype=tf.float32)[:,None]
                logp = beta_log_prob(A_tf, alpha, beta)
                ratio = tf.exp(logp - old_logp_tf)
                adv_tf = tf.constant(adv, dtype=tf.float32)[:,None]
                loss_a = -tf.reduce_mean(tf.minimum(ratio*adv_tf,
                              tf.clip_by_value(ratio, 1.0-clip_eps, 1.0+clip_eps)*adv_tf))
                v_pred = critic(S_all, training=True)[:, :, 0]
                loss_c = tf.reduce_mean(tf.square(v_pred - tf.constant(returns, dtype=tf.float32)[None,:]))
            opt_a.apply_gradients(zip(ta.gradient(loss_a, actor.trainable_weights), actor.trainable_weights))
            opt_c.apply_gradients(zip(tc.gradient(loss_c, critic.trainable_weights), critic.trainable_weights))

        print(f"[EP {ep}] R_sum={np.sum(R):.5f} A_loss={float(loss_a):.5f} C_loss={float(loss_c):.5f}")

    os.makedirs("models", exist_ok=True)
    actor.save("models/actor_latest.h5")
    critic.save("models/critic_latest.h5")
    print("Saved models to models/")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default=os.getenv("TICKER","KRW-BTC"))
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--update_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    args = p.parse_args()
    main(args)
