from typing import Optional, Tuple

import tensorflow as tf
import keras

from DeepLearningFun.definitions import TEMP_DIR
from DeepLearningFun.tensorflow_module.dataset_processing.load_mnist_fashion import load_and_preprocess_dataset
from DeepLearningFun.tensorflow_module.helper_models import TrainTestDatasetModel
from DeepLearningFun.tensorflow_module.models.basic_cnn import basic_cnn_sequential, basic_cnn_functional
import wandb
from wandb.keras import WandbCallback

def train_basic_cnn(use_gpu: Optional[bool] = True) -> Tuple[keras.Model, TrainTestDatasetModel]:
    if not use_gpu:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'

    wandb.init(project='BasicMNISTCNN', entity='mstrome', dir=TEMP_DIR)

    dataset = load_and_preprocess_dataset()
    input_size = dataset.train_X[0].shape
    num_labels = len(dataset.class_names)
    model = basic_cnn_functional(input_size=input_size, num_labels=num_labels)
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(
        x=dataset.train_X,
        y=dataset.train_Y,
        epochs=5,
        callbacks=[WandbCallback()],
        validation_data=(dataset.test_X, dataset.test_Y),
        batch_size=20
    )
    wandb.log({'model': model})
    return model, dataset