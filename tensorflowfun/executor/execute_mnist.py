from DeepLearningFun.tensorflowfun.inference.basic_cnn import run_inference_and_visualize
from DeepLearningFun.tensorflowfun.training.train_cnn import train_basic_cnn


def execute_mnist():
    """
    Train a basic CNN on the MNIST Fashion dataset and then run inference on the test set
    """
    # Train the model
    model, dataset = train_basic_cnn(use_gpu=False)
    # Run inference and visualize the results
    run_inference_and_visualize(dataset=dataset, model=model)