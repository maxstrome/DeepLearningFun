from typing import Tuple
import keras

def basic_cnn_sequential(input_size: Tuple[int, int], num_labels: int) -> keras.Sequential:
    model = keras.Sequential(
        layers=[
            keras.layers.Flatten(input_shape=input_size),
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.Dense(units=num_labels)
        ],
        name="basic_cnn"
    )
    return model