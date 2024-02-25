import keras
from DeepLearningFun.tensorflowfun.dataset_processing.load_mnist_fashion import load_and_preprocess_dataset
from DeepLearningFun.tensorflowfun.models.basic_cnn import basic_cnn_sequential
import wandb
from wandb.keras import WandbCallback

def train_basic_cnn():
    wandb.init(project='BasicMNISTCNN', entity='mstrome')

    (x_train, y_train), (x_test, y_test), class_names = load_and_preprocess_dataset()
    input_size = x_train[0].shape
    num_labels = len(class_names)
    model = basic_cnn_sequential(input_size=input_size, num_labels=num_labels)
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(x=x_train, y=y_train, epochs=10,  callbacks=[WandbCallback()])
    wandb.log({'model': model})
