import os, sys
import pandas as pd
import numpy as np
from mlproject.exception import CustomException
from mlproject.logger import logging
from mlproject.utils import save_obj, load_object, evaluate_models

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict_data(self, features):
        try:
            model_path = "artifact/model.pkl"
            preprocessor_path = "artifact/preprocessing.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)

import pandas as pd

class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            data = {
            "gender": [self.gender],
            "race/ethnicity": [self.race_ethnicity],
            "parental level of education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test preparation course": [self.test_preparation_course],
            "reading score": [self.reading_score],
            "writing score": [self.writing_score]
            }
            return pd.DataFrame(data)
        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)