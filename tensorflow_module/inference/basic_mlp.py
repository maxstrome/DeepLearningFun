import keras


def basic_mlp_inference(model: keras.Model, vectorize_layer: keras.layers.TextVectorization):
    """
    Run inference on the test set
    """
    export_model = keras.Sequential(
        [
            vectorize_layer,
            model,
            keras.layers.Activation('sigmoid')
        ]
    )

    export_model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )
    return export_model.predict(["This is the worst movie I have ever seen"])