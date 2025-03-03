import os
import sys 

import numpy as np 
import pandas as pd
import dill
from src.exception import Custom_exception
from sklearn.metrics import r2_score
def save_objects(file_path,obj):
    try:
        dir_name= os.path.dirname(file_path)
        
        os.makedirs(dir_name,exist_ok=True)
        
        with open(file_path,'wb') as file:
            dill.dump(obj,file)

    except Exception as e:
        raise Custom_exception(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    report ={}
    for i,value in models.items():
        model = value
        model.fit(x_train,y_train)
        y_train_pred = model.predict(x_train)

        model.fit(x_test,y_test)
        y_test_pred = model.predict(x_test)

        train_model_score = r2_score(y_train,y_train_pred)
        test_model_score = r2_score(y_test,y_test_pred)
        report[i] = test_model_score 
        return report
