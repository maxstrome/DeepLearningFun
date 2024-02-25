from typing import List

from pydantic import BaseModel
import numpy as np

class TrainTestDatasetModel(BaseModel):
    """
    BaseModel to hold a train/test dataset along with the class names
    """
    class Config:
        arbitrary_types_allowed = True

    train_X: np.ndarray
    train_Y: np.ndarray
    test_X: np.ndarray
    test_Y: np.ndarray
    class_names: List[str]