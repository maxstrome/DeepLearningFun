from DeepLearningFun.tensorflowfun.inference.basic_cnn import run_inference_and_visualize
from DeepLearningFun.tensorflowfun.training.train_cnn import train_basic_cnn


def execute_mnist():
    model, x_test, class_names = train_basic_cnn(use_gpu=False)
    run_inference_and_visualize(images=x_test, model=model, class_labels=class_names)