
import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustmeException
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object



@dataclass
class DataTransfromartionConfigs:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation", "preprcessor.pkl") #creating a path for preprocessor.pkl


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransfromartionConfigs()


    def get_data_transformation_obj(self):
        try:

            logging.info(" Data Transformation Started")

            # numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level','blood_glucose_level', 'diabetes']
            numerical_features = ['age','hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level']
            catagorical_features=['gender']
            drop_col=['smoking_history']
            
            steps_cat= Pipeline([('gender_mapping', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))])    

            #Note: It can happen if you're using sklearn.compose.ColumnTransformer You need to make sure the output column is not included in the data you create your Preprocessing pipeline with. 
            # So make sure to separate the input(s) and the output(s) columns first.
            preprocessor = ColumnTransformer([
                ('preprocessing_cat', steps_cat,catagorical_features),
                ('preprocessing_num',StandardScaler(),numerical_features),
                ('drop_col','drop',drop_col)],remainder='passthrough')
            return preprocessor
        
        except Exception as e:
            raise CustmeException(e, sys)
        
    def inititate_data_transformation(self, train_path, test_path):

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            

            preprocess_obj = self.get_data_transformation_obj()

            traget_columns = 'diabetes'
            drop_columns = [traget_columns]

            logging.info("Splitting train data into dependent and independent features")
            input_feature_train_data = train_data.drop(drop_columns, axis = 1)
            traget_feature_train_data = train_data[traget_columns]
            logging.info("Splitting test data into dependent and independent features")
            input_feature_test_data = test_data.drop(drop_columns, axis = 1)
            traget_feature_test_data = test_data[traget_columns]

            input_train_arr= preprocess_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocess_obj.transform(input_feature_test_data)

            logging.info("Resampling the dependent and independent features ")
            sc=SMOTE(sampling_strategy=0.3)
            input_train_arr,traget_feature_train_data_=sc.fit_resample(input_train_arr,np.array(traget_feature_train_data))
            us=RandomUnderSampler(sampling_strategy=0.4)
            input_train_arr,traget_feature_train_data_=us.fit_resample(input_train_arr,traget_feature_train_data_)
                                                                       
            train_array = np.c_[input_train_arr, np.array(traget_feature_train_data_)]
            test_array = np.c_[input_test_arr, np.array(traget_feature_test_data)]



            save_object(file_path=self.data_transformation_config.preprocess_obj_file_path,
                        obj=preprocess_obj)#saving preprocessor object to file path 
            return (train_array,
                    test_array,
                    self.data_transformation_config.preprocess_obj_file_path)
        except Exception as e:
            raise CustmeException(e, sys)
