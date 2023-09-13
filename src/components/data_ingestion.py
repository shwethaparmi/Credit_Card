import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df = pd.read_excel(os.path.join('notebooks/data','default of credit card clients.xls'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info(df.shape)
            df =  df.drop_duplicates()
            logging.info(df.shape)

            df['SEX']        = df['SEX'].astype("category")
            df['EDUCATION']  = df['EDUCATION'].astype("category")
            df['MARRIAGE']   = df['MARRIAGE'].astype("category")
            
        
            logging.info('Train test split')
            
            train_sets,test_sets=train_test_split(df,test_size=0.30,random_state=42)
            

            train_sets.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_sets.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data,test_data)


    modeltrainer=ModelTrainer()
    print(modeltrainer.initate_model_training(train_arr,test_arr))
