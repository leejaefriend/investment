import numpy as np

class Agent:
    def __init__(self, actor, sigma: float = 0.02):
        self.actor = actor
        self.sigma = float(sigma)  # exploration std

    def act(self, seq_state):
        w = float(self.actor.predict(seq_state, verbose=0)[0,0])
        # add small exploration noise during training
        noise = np.random.normal(0, self.sigma)
        return float(np.clip(w + noise, 0.0, 1.0))