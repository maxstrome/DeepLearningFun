from typing import Tuple
import keras

def basic_cnn_sequential(input_size: Tuple[int, int], num_labels: int) -> keras.Sequential:
    """
    Create a basic CNN using the sequential API

    Returns:
        keras.Sequential: The basic CNN model
    """
    model = keras.Sequential(
        layers=[
            keras.layers.Flatten(input_shape=input_size),
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.Dense(units=num_labels)
        ],
        name="basic_cnn"
    )
    return model

def basic_cnn_functional(input_size: Tuple[int, int], num_labels: int) -> keras.Model:
    """
    Create a basic CNN using the functional API

    Returns:
        keras.Model: The basic CNN model
    """
    inputs = keras.Input(shape=input_size)
    flattened = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(units=128, activation="relu")(flattened)
    outputs = keras.layers.Dense(units=num_labels, activation="linear")(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def actual_cnn(input_size: Tuple[int, int], num_labels: int) -> keras.Model:
    inputs = keras.Input(shape=(input_size[0], input_size[1], 1))
    x = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(inputs)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(.5)(x)
    x = keras.layers.Dense(units=64, activation="relu")(x)
    outputs = keras.layers.Dense(units=num_labels, activation="linear")(x)
    return keras.Model(inputs=inputs, outputs=outputs)

