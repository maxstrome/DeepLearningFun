from dotenv import load_dotenv

from DeepLearningFun.tensorflow_module.executor.execute_mlp import execute_mlp
from DeepLearningFun.tensorflow_module.executor.execute_mnist import execute_mnist
from DeepLearningFun.tensorflow_module.inference.basic_cnn import run_inference_and_visualize
from DeepLearningFun.tensorflow_module.training.train_cnn import train_basic_cnn
from DeepLearningFun.tensorflow_module.training.train_mlp import train_mlp

if __name__ == '__main__':
    load_dotenv()
    # execute_mnist()
    execute_mlp()