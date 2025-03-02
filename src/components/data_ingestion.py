import os 
import sys 
from src.exception import Custom_exception
from src.logger import logging 
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import Data_transformation
@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join("artifacts",'train.csv')
    test_data_path : str=os.path.join("artifacts",'test.csv')
    raw_data_path : str=os.path.join("artifacts",'data.csv')
    
class InitiateIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def Initiate_data_ingestion(self):
        logging.info("Entered the dataIngestion method or components")
        try:
            df= pd.read_csv("notebook/data/Clean_Dataset.csv")
            logging.info("read the dataset from dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("train test split initiate")
            train_set,test_set = train_test_split(df,random_state=42,test_size=0.3)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("ingestion of data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise Custom_exception(e,sys)
if __name__ == "__main__":
    obj = InitiateIngestion()
    train_data,test_data = obj.Initiate_data_ingestion()
    data_transformation =Data_transformation()
    data_transformation.initiate_transform_data(train_data,test_data)