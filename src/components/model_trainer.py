import sys 
import os 
from dataclasses import dataclass

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor

)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.exception import Custom_exception
from src.logger import logging

from src.utils import save_objects,evaluate_model

@dataclass
class Model_trainer_config:
    train_model_path = os.path.join("artifacts","model.pkl")

class Model_trainer:
    def __init__(self):
        self.model_trainer_config = Model_trainer_config()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splite train and test input data")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'AdaBoost': AdaBoostRegressor(),
            'XGBoost': XGBRegressor(),
            'CatBoost': CatBoostRegressor(verbose=1)  # Suppressing verbose output
            }
            params = {
            'Linear Regression': [
                {'fit_intercept': [True, False]}
            ],
            'Decision Tree': [
                {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            ],
            'Random Forest': [
                {
                    'n_estimators': [100, 200, 300],
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            ],
            'Gradient Boosting': [
                {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 1.0]
                }
            ],
            'AdaBoost': [
                {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                }
            ],
            'XGBoost': [
                {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            ],
            'CatBoost': [
                {
                    'iterations': [500, 1000, 1500],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'l2_leaf_reg': [1, 3, 5],
                    'border_count': [32, 64, 128]
                    }
                ]
            }


            
            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = [model_report.keys()][list(model_report.values()).index(best_model_score)]
            logging.info(f"{best_model_name}")
            best_model = models[list(best_model_name)[0]]
            if best_model_score<=0.6:
                raise Custom_exception("Not found best model")
            logging.info(f"Best found model in training and testing data {best_model_name}")

            save_objects(self.model_trainer_config.train_model_path,best_model)

            y_pred = best_model.predict(x_test)
            r2 = r2_score(y_test,y_pred)
            return r2

        except Exception as e :
            raise Custom_exception(e,sys)


