# data ingestion relates to reading the data

import os
import sys
from mlproject.exception import CustomException
from mlproject.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from mlproject.components.data_transformation import DataTransformation, DataTransformationConfig
from mlproject.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def train_test_split(self):
        logging.info("Entered the train test split component.")
        try:
            df = pd.read_csv(r"notebook\data\std_perf.csv")
            logging.info("Reading the csv file")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            logging.info("Train test initialized")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header = True)
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(sys, e) 
            

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.train_test_split()

    data_preprocessing = DataTransformation()
    train_array, test_array, _ = data_preprocessing.initiate_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_training(train_array, test_array))