import sys 
import pandas as pd 
from ..exception import Custom_exception
from ..logger import logging 
from src.utils import load_object 

class Predict_pipeline:
    def __init__(self):
        pass
    def predict(self,feature):
        try:           
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaler = preprocessor.transform(feature)
            pred=model.predict(data_scaler)
        except Exception as e:
            raise Custom_exception(e,sys)
        return pred
class CustomData:
    def __init__(self,
                 duration: int,
                 days_left: int,
                 source_city: str,
                 departure_time: str,
                 stops: str,
                 arrival_time: str,
                 destination_city: str,
                 Class: str):
        self.duration = duration
        self.days_left = days_left
        self.source_city = source_city
        self.departure_time = departure_time
        self.stops = stops
        self.arrival_time = arrival_time
        self.destination_city = destination_city
        self.Class = Class
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'duration':[self.duration] , 
                'days_left':[self.days_left] ,
                'source_city':[self.source_city]  ,
                'departure_time':[self.departure_time] ,
                'stops':[self.stops]  ,
                'arrival_time':[self.arrival_time]  ,
                'destination_city':[self.destination_city]  ,
                'class':[self.Class] }
            return pd.DataFrame(custom_data_input_dict)
        except:
            pass
        

        
