import sys
from dataclasses import dataclass
import os 

import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import Custom_exception
from src.logger import logging
from src.utils import save_objects

@dataclass
class Data_config:
    preprocessor_obj_path:str = os.path.join("artifacts","preprocessor.pkl") 

class Data_transformation:
    def __init__(self):
        self.data_transformation = Data_config()

    def get_data_transformer(self):
        try:
            numerical_feature = ['duration', 'days_left']
            categorical_feature = [  'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
            num_pipeline = Pipeline(
                steps = [
                        ('imputer',SimpleImputer(strategy='median')),
                        ('scaler',StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                        ('imputer',SimpleImputer(strategy='most_frequent')),
                        ('One_hot',OneHotEncoder())
                ]
            )
            logging.info("Numerical columns has completed")
            logging.info("categorical columns has completed")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_feature),
                    ('cat_pipeline',categorical_pipeline,categorical_feature)
                ]

            )
            return preprocessor
        except Exception as e:
            raise Custom_exception(e,sys)
    def initiate_transform_data(self,train_path,test_path):
        try :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Obtaining preprocessing obj")

            preprocess_obj = self.get_data_transformer()
            target_columns = 'price'

            input_feature_train_df = train_df.drop(columns=[target_columns],axis=1)
            target_feature_train_df = train_df[target_columns]

            input_feature_test_df = test_df.drop(columns=[target_columns],axis=1)
            target_feature_test_df = test_df[target_columns]

            logging.info("apply processing obj on train and test df")
            logging.info(f'{input_feature_train_df.columns}')
            input_feature_train_arr=preprocess_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocess_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr.toarray(),np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.toarray(),np.array(target_feature_test_df)]
            
            logging.info("saved  processing obj")

            save_objects(
                file_path = self.data_transformation.preprocessor_obj_path,
                obj = preprocess_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_path
            )
  
           
        except Exception as e:
            raise Custom_exception(e,sys)