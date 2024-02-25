from typing import Optional

import tensorflow as tf
import keras
from DeepLearningFun.tensorflowfun.dataset_processing.load_mnist_fashion import load_and_preprocess_dataset
from DeepLearningFun.tensorflowfun.models.basic_cnn import basic_cnn_sequential, basic_cnn_functional
import wandb
from wandb.keras import WandbCallback

def train_basic_cnn(use_gpu: Optional[bool] = True):
    if not use_gpu:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'

    wandb.init(project='BasicMNISTCNN', entity='mstrome')

    (x_train, y_train), (x_test, y_test), class_names = load_and_preprocess_dataset()
    input_size = x_train[0].shape
    num_labels = len(class_names)
    model = basic_cnn_functional(input_size=input_size, num_labels=num_labels)
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(
        x=x_train,
        y=y_train,
        epochs=5,
        callbacks=[WandbCallback()],
        validation_data=(x_test, y_test),
        batch_size=20
    )
    wandb.log({'model': model})
    return model, x_test, class_names