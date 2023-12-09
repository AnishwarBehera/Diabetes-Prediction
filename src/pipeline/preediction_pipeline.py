


import os, sys
from src.logger import logging
from src.exception import CustmeException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        preprocessro_path = os.path.join("artifacts/data_transformation", "preprcessor.pkl")
        model_path = os.path.join("artifacts/model_trainer", "model.pkl")

        processor = load_object(preprocessro_path)
        model = load_object(model_path)


        scaled = processor.transform(features)
        pred = model.predict(scaled)

        return pred













