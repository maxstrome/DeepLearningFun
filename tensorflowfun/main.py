from dotenv import load_dotenv

from DeepLearningFun.tensorflowfun.training.train_cnn import train_basic_cnn

if __name__ == '__main__':
    load_dotenv()
    train_basic_cnn()
