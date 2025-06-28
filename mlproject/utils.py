import os, sys, numpy as np, pandas as pd, pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mlproject.exception import CustomException
from mlproject.logger import logging
from sklearn.model_selection import GridSearchCV

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file) 

    except Exception as e:
        logging.info(CustomException(e, sys))
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        logging.info(CustomException(e, sys))
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, model_params):
    report = {}
    try:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(model_params.values())[i]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)
            

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
        
