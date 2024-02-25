from dotenv import load_dotenv

from DeepLearningFun.tensorflowfun.executor.execute_mnist import execute_mnist
from DeepLearningFun.tensorflowfun.inference.basic_cnn import run_inference_and_visualize
from DeepLearningFun.tensorflowfun.training.train_cnn import train_basic_cnn

if __name__ == '__main__':
    load_dotenv()
    execute_mnist()