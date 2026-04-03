import json 
import os
import logging 

import numpy as np 
from tensorflow import keras 
from sklearn.preprocessing import LabelEncoder
import yaml

with open(os.path.join(os.path.dirname(__file__), "../../models.yml"), "r") as f:
    models = yaml.safe_load(f)["models"]
logger = logging.getLogger(__name__)

class MLProcessor():
    """
    MLProcessor for the model specified in the models.yml file.
    """

    def __init__(self, model_name: str = "mixup_08"):
        """
        Initialize the MLProcessor for the model specified in the models.yml file.
        """
        self.model = keras.models.load_model(models[model_name]["path"])
        self._label_encoder = LabelEncoder()
        with open(models[model_name]["label_encoder"], "r") as f:
            self._label_encoder.classes_ = np.array(json.load(f)["classes"])
        self._input_dim = models[model_name]["input_dim"]
        self._classes = models[model_name]["classes"]
    
    def predict(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Predict the class of the keypoints.
        Args:
            keypoints: Numpy Array of shape (n_frames, 51)
        Returns:
            Predictions as a Numpy Array of shape (n_frames, 4) where 4 is the number of classes
        """
        
        predictions = self.model.predict(keypoints)  
        return np.array(predictions, dtype=np.float32).tolist()
    
    def classify_pose(self, predictions: np.ndarray) -> str:
        """
        Classify the predictions into a class.
        Args:
            predictions: Numpy Array of shape (n_frames, 4) where 4 is the number of classes
        Returns:
            Most common class as a string
        """
        # Use the most frequently predicted class per-frame 
        per_frame_class_idxs = np.argmax(predictions, axis=1)
        most_common_class_idx = np.bincount(per_frame_class_idxs).argmax()
        most_common_class = self._label_encoder.classes_[most_common_class_idx]
        return most_common_class
