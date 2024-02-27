import keras


def basic_embedding_mlp(max_features: int, sequence_length: int, embedding_dim: int) -> keras.Model:
    """
    Create a basic embedding MLP using the functional API

    Returns:
        keras.Model: The basic embedding MLP model
    """
    inputs = keras.Input(shape=(sequence_length))
    x = keras.layers.Embedding(
        input_dim=max_features,
        output_dim=embedding_dim,
        input_length=sequence_length
    )(inputs)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    outputs = keras.layers.Dense(units=1)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model = basic_embedding_mlp(max_features=10000, sequence_length=250, embedding_dim=16)
    model.summary()