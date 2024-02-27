import keras
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from DeepLearningFun.definitions import TEMP_DIR
from DeepLearningFun.tensorflow_module.dataset_processing.load_large_movie_review import load_dataset
from DeepLearningFun.tensorflow_module.models.basic_embedding_mlp import basic_embedding_mlp


def train_mlp(train_ds, val_ds, test_ds, max_features: int = 10000, sequence_length: int = 250):

    wandb.init(project='BasicMovieMLP', entity='mstrome', dir=TEMP_DIR)

    model = basic_embedding_mlp(max_features=max_features, sequence_length=sequence_length, embedding_dim=16)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=keras.metrics.BinaryAccuracy(threshold=0.0)
    )
    epochs = 2
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[WandbCallback()],
    )
    return model