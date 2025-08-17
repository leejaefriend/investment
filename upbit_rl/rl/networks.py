import tensorflow as tf
from tensorflow.keras import layers, models

def make_actor(input_dim: int):
    x_in = layers.Input(shape=(None, input_dim))
    x = layers.Masking()(x_in)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)  # target weight
    return models.Model(x_in, out)

def make_critic(input_dim: int):
    x_in = layers.Input(shape=(None, input_dim))
    x = layers.Masking()(x_in)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    v = layers.Dense(1, activation="linear")(x)
    return models.Model(x_in, v)