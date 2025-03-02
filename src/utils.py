import os
import sys 

import numpy as np 
import pandas as pd
import dill
from src.exception import Custom_exception

def save_objects(file_path,obj):
    try:
        dir_name= os.path.dirname(file_path)
        
        os.makedirs(dir_name,exist_ok=True)
        
        with open(file_path,'wb') as file:
            dill.dump(obj,file)

    except Exception as e:
        raise Custom_exception(e,sys)

