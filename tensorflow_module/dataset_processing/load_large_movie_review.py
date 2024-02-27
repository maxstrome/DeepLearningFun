import os
import re
import shutil
import string
from typing import Tuple

import tensorflow as tf
import keras

from DeepLearningFun.definitions import TEMP_DIR


def download_large_movie_reviews():
    """
    Download the large movie review dataset if it doesn't exist, loaded in to the keras cache
    """
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    dataset = keras.utils.get_file(
        "aclImdb_v1",
        url,
        untar=True,
        cache_dir='../../temp',
        cache_subdir=''
    )

    dataset_dir = os.path.dirname(dataset)
    return dataset_dir

def visualize_data(dataset: tf.data.Dataset):
    """
    Print the contents of a file
    """
    for text_batch, label_batch in dataset.take(1):
        for i in range(3):
            print("Review", text_batch.numpy()[i])
            print("Label", label_batch.numpy()[i])

def load_dataset(max_features: int, sequence_lenght: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, keras.layers.TextVectorization]:
    """
    use keras from directory to load the dataset train, val, and test
    """
    batch_size = 32
    seed = 42

    raw_train_ds = keras.utils.text_dataset_from_directory(
        os.path.join(TEMP_DIR, 'aclImdb', 'train'),
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )
    raw_val_ds = keras.utils.text_dataset_from_directory(
        os.path.join(TEMP_DIR, 'aclImdb', 'train'),
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
    )
    raw_test_ds = keras.utils.text_dataset_from_directory(
        os.path.join(TEMP_DIR, 'aclImdb', 'test'),
        batch_size=batch_size
    )

    return preprocess_text_vectorizer(
        train_ds=raw_train_ds,
        val_ds=raw_val_ds,
        test_ds=raw_test_ds,
        max_features=max_features,
        sequence_length=sequence_lenght
    )

def remove_html(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(
      stripped_html,
      '[%s]' % re.escape(string.punctuation),
      ''
  )

def preprocess_text_vectorizer(
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        max_features: int,
        sequence_length: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, keras.layers.TextVectorization]:

    # standardize, tokenize, and vectorize the data
    vectorize_layer = keras.layers.TextVectorization(
        standardize=remove_html,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    vectorize_layer.adapt(train_ds.map(lambda text, label: text))
    def vectorize_text(text, label):
      text = tf.expand_dims(text, -1)
      return vectorize_layer(text), label

    train_ds = train_ds.map(vectorize_text)
    val_ds = val_ds.map(vectorize_text)
    test_ds = test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, vectorize_layer
