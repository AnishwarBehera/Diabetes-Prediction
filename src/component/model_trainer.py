import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustmeException
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting our data into dependent and independent features")
            X_train, y_train, X_test, y_test  = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            model = {
                "Random Forest": RandomForestClassifier(),
                "xgb":xgb.XGBClassifier()
            }

            params = {
                "Random Forest":{
                    'n_estimators': [50,100,150,200],
                    'max_depth': [None,5,8,10],
                    'min_samples_split': [2, 5, 10],
                },
                "xgb":{
                        'n_estimators': [50,100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7]
                   }
            }
                
            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test =y_test,
                                                models = model, params = params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = model[best_model_name]
            
            print(f"Best Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n---------------------------------------------------------------------------\n")
            logging.info(f"Best Model Name is {best_model_name}, accuracy Score: {best_model_score}")


            save_object(file_path=self.model_trainer_config.train_model_file_path,
                        obj = best_model
                        )
        except Exception as e:
            raise CustmeException(e, sys)

