import sys, pickle
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from mlproject.exception import CustomException
from mlproject.logger import logging
from mlproject.utils import save_obj
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifact", "preprocessing.pkl")

class DataTransformation:
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_features = ['reading score', 'writing score']
            cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standard scaling done.")
            logging.info("Categorical columns encoding done.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data completed")

            target_col = "math score"

            logging.info("Obtaining preprocessor object")
            processing_obj = self.get_data_transformer_obj()

            input_feature_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info("Applying preprocessing object on train and test dataframes")
            input_feature_train_array = processing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = processing_obj.transform(input_feature_test_df)

            logging.info("Saving preprocessing object into pickle file")
            save_obj(self.datatransformationconfig.preprocessor_obj_file_path, processing_obj)

            train_array = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            return train_array, test_array, self.datatransformationconfig.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
