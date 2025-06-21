import os, sys, numpy as np, pandas as pd, pickle
from sklearn.model_selection import train_test_split
from mlproject.exception import CustomException
from mlproject.logger import logging

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)  # ✅ FIXED

    except Exception as e:
        logging.info(CustomException(e, sys))
        raise CustomException(e, sys)
