# upbit_rl/rl/networks.py
import tensorflow as tf
from tensorflow.keras import layers, models

def make_actor_beta(input_dim: int):
    x_in = layers.Input(shape=(None, input_dim))
    x = layers.Masking()(x_in)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    # alpha, beta > 1 보장 (과도한 극단 방지) -> softplus + 1.0
    alpha = layers.Dense(1, activation="softplus")(x)
    beta  = layers.Dense(1, activation="softplus")(x)
    out = layers.Concatenate()([alpha, beta])
    return models.Model(x_in, out)

def make_critic(input_dim: int):
    x_in = layers.Input(shape=(None, input_dim))
    x = layers.Masking()(x_in)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    v = layers.Dense(1, activation="linear")(x)
    return models.Model(x_in, v)
