from typing import Tuple

import keras
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_fashion():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return (x_train, y_train), (x_test, y_test), class_names

def view_image(image: np.array):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def preprocess_images(x_train, x_test):
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, x_test

def plot_images(images, labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

def load_and_preprocess_dataset():
    (x_train, y_train), (x_test, y_test), class_names = load_mnist_fashion()
    x_train, x_test = preprocess_images(x_train, x_test)
    return (x_train, y_train), (x_test, y_test), class_names

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), class_names = load_mnist_fashion()
    x_train, x_test = preprocess_images(x_train, x_test)
    view_image(image=x_train[0])
    plot_images(x_train, y_train, class_names)