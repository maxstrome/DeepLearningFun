from DeepLearningFun.tensorflow_module.dataset_processing.load_large_movie_review import load_dataset
from DeepLearningFun.tensorflow_module.inference.basic_mlp import basic_mlp_inference
from DeepLearningFun.tensorflow_module.training.train_mlp import train_mlp
import tensorflow as tf

def execute_mlp(use_gpu: bool = True):
    if not use_gpu:
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    train_ds, val_ds, test_ds, vectorize_layer = load_dataset(max_features=10000, sequence_lenght=250)
    model = train_mlp(train_ds, val_ds, test_ds, max_features=10000, sequence_length=250)
    print(basic_mlp_inference(model, vectorize_layer))