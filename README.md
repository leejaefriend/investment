# upbit-rl-trader

Crypto trading research bot using **Reinforcement Learning** (actor-critic with Keras) for **Upbit**.  
Includes **paper trading**, **Slack notifications**, and a clean separation between the **broker** (orders) and **RL** (policy).

> Recommended Python: **3.10–3.12** (TensorFlow currently supports Python 3.9–3.12).  
> If you use Python 3.13, TensorFlow may not install. Downgrade to 3.11 for now.

## 1) Setup

```bash
# 1. Create & activate venv (Windows PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install deps
pip install -U pip
pip install -r requirements.txt

# 3. Create .env from template
copy .env .env
# Fill: UPBIT_ACCESS, UPBIT_SECRET (and optionally Slack)
```

**Upbit Keys:** Issue at Upbit Developer Center (add your IP to whitelist).  
**Slack:** Use either Incoming Webhook URL **or** Bot Token (`chat:write`) + channel.

## 2) Project Layout

```
upbit_rl/
  broker/
    upbit_client.py   # pyupbit client wrapper
    broker.py         # unified interface + paper trading
  notify/
    slack.py          # Slack notifier
  data/
    ohlcv.py          # OHLCV fetch/cache
  rl/
    environment.py    # daily OHLCV env with target weight action
    networks.py       # actor/critic (Keras LSTM)
    agent.py          # agent wrapper
train.py              # simple A2C-style training loop (skeleton)
trade.py              # daily rebalance by actor (paper/real)
scripts/
  test_slack.py
  test_upbit.py
```

## 3) Quick Tests

```bash
# Test Upbit market price & OHLCV
python scripts/test_upbit.py

# Test Slack message
python scripts/test_slack.py
```

## 4) Training

```bash
python -m upbit_rl.train --ticker KRW-BTC --window 30 --epochs 5
```

Models are saved under `models/actor_latest.h5` and `models/critic_latest.h5`.

## 5) Paper Trading (Daily Rebalance)

```bash
# Uses PAPER_TRADE=1 in .env by default
python -m upbit_rl.trade --ticker KRW-BTC
```

## 6) GitHub

```bash
git init
git add .
git commit -m "boot: RL Upbit scaffold"
git branch -M main
git remote add origin https://github.com/<yourname>/<repo>.git
git push -u origin main
```

## Notes
- This repo is **educational**. Real money trading is risky. Start with PAPER_TRADE=1.
- For GPU on Windows, prefer WSL2; native GPU is not supported beyond TF 2.10.