import keras
import wandb
from dotenv import load_dotenv
import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from wandb.keras import WandbCallback

from DeepLearningFun.definitions import TEMP_DIR

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

if __name__ == '__main__':
    load_dotenv()
    wandb.init(project='TFHub', entity='mstrome', dir=TEMP_DIR)

    # Split the training set into 60% and 40% to end up with 15,000 examples
    # for training, 10,000 examples for validation and 25,000 examples for testing.
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True
    )
    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
    print(train_examples_batch)

    """
    Let's first create a Keras layer that uses a TensorFlow Hub model to embed the sentences, and try it out on a couple of input examples. Note that no matter the length of the input text, the output shape of the embeddings is: (num_examples, embedding_dimension).
    Pretrained embeddings
    """
    embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
    hub_layer = hub.KerasLayer(
        embedding,
        input_shape=[],
        dtype=tf.string,
        trainable=True
    )
    hub_layer(train_examples_batch[:3])
    model = keras.Sequential()
    model.add(hub_layer)
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1))

    model.summary()
    model.compile(
        optimizer='adam',
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_data.shuffle(10000).batch(512),
        epochs=10,
        validation_data=validation_data.batch(512),
        callbacks=[WandbCallback()],
    )
    results = model.evaluate(test_data.batch(512), verbose=2)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))