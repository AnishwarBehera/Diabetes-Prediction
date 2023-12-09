import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustmeException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.component.data_transfromation import DataTransformation
from src.component.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts/data_ingestion", "train.csv")
    test_data_path = os.path.join("artifacts/data_ingestion", "test.csv")
    raw_data_path = os.path.join("artifacts/data_ingestion", "raw.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def inititate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            logging.info("Data Reading")
            data = pd.read_csv(os.path.join("notebook/data", "diabetes_prediction_dataset.csv"))
            logging.info("Data Reading completed")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)#Creating raw_data_path
            data.to_csv(self.ingestion_config.raw_data_path, index=False)#saving data to raw_data_path

            train_set, test_set = train_test_split(data, test_size = 0.2)
            logging.info("Data spliteted into train and test")

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)#saving train data
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)#saving test data

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            logging.info("Error occured in data ingestion stage")
            raise CustmeException(e, sys)

if __name__ =="__main__":
    obj = DataIngestion()
    train_data_path , test_data_path = obj.inititate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.inititate_data_transformation(train_data_path , test_data_path)

    modeltrainer = ModelTrainer()
    print(modeltrainer.inititate_model_trainer(train_arr, test_arr))





