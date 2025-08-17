# upbit_rl/rl/agent.py
import numpy as np
from scipy.stats import beta as beta_dist

class Agent:
    def __init__(self, actor_beta, explore=True, min_alpha=1.0, min_beta=1.0):
        self.actor = actor_beta
        self.explore = explore
        self.min_alpha = min_alpha
        self.min_beta = min_beta

    def act(self, seq_state):
        ab = self.actor.predict(seq_state, verbose=0)[0]  # [alpha_raw, beta_raw]
        a = float(ab[0]) + self.min_alpha
        b = float(ab[1]) + self.min_beta
        if self.explore:
            return float(np.clip(beta_dist.rvs(a, b), 0.0, 1.0))  # 샘플
        # 평가/실거래: 평균
        return float(np.clip(a/(a+b), 0.0, 1.0))
