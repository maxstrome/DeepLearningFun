from typing import Tuple

import keras
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel

from DeepLearningFun.tensorflow_module.helper_models import TrainTestDatasetModel


def load_mnist_fashion() -> TrainTestDatasetModel:
    """
    Load the mnist fashion dataset as a TrainTestDatasetModel

    Returns:
        TrainTestDatasetModel: The fashion MNIST dataset
    """
    (train_X, train_Y), (test_X, test_Y) = keras.datasets.fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    dataset = TrainTestDatasetModel(
        train_X=train_X,
        train_Y=train_Y,
        test_X=test_X,
        test_Y=test_Y,
        class_names=class_names
    )
    return dataset

def view_image(image: np.ndarray):
    """
    View one of the images
    """
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def preprocess_images(dataset: TrainTestDatasetModel) -> TrainTestDatasetModel:
    """
    Preprocess images by dividing by 255 (max val of RGB pixels), so between 0 and 1

    Args:
        dataset (TrainTestDatasetModel): The dataset to preprocess

    Returns:
        TrainTestDatasetModel: The preprocessed dataset
    """
    dataset.train_X = dataset.train_X / 255.0
    dataset.test_X = dataset.test_X / 255.0
    return dataset

def plot_images(dataset: TrainTestDatasetModel):
    """
    Plot the first 25 images from the dataset

    Args:
        dataset (TrainTestDatasetModel): The dataset to plot
    """
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(dataset.test_X[i], cmap=plt.cm.binary)
        plt.xlabel(dataset.class_names[dataset.test_Y[i]])
    plt.show()

def load_and_preprocess_dataset() -> TrainTestDatasetModel:
    """
    Load the dataset and preprocess it

    Returns:
        TrainTestDatasetModel: The preprocessed dataset
    """
    dataset = load_mnist_fashion()
    dataset = preprocess_images(dataset=dataset)
    return dataset