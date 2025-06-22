import sys, os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error


from mlproject.exception import CustomException
from mlproject.logger import logging
from mlproject.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    modeltrainer_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting data into train and test data")
            X_train, y_train, X_test, y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso Regression": Lasso(),
                "Ridge Regression": Ridge(),
                "ElasticNet Regression": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Support Vector Regressor": SVR(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)
            }
            model_params = {
            "Linear Regression": {},
            "Lasso Regression": {
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "max_iter": [1000, 5000]
            },
            "Ridge Regression": {
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "solver": ["auto", "svd", "cholesky", "lsqr"]
            },
            "ElasticNet Regression": {
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "l1_ratio": [0.1, 0.5, 0.9],
                "max_iter": [1000, 5000]
            },
            "Decision Tree": {
                "criterion": ["squared_error", "friedman_mse"],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "bootstrap": [True, False]
            },
            "AdaBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0]
            },
            "Support Vector Regressor": {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"]
            },
            "KNeighbors Regressor": {
                "n_neighbors": [3, 5, 7, 10],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree"]
            },
            "XGBoost": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0]
            },
            "CatBoost": {
                "iterations": [100, 200],
                "depth": [4, 6, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "l2_leaf_reg": [1, 3, 5],
                "verbose": [0]
            }
        }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, model_params=model_params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info("Best model found")
            save_obj(self.model_trainer_config.modeltrainer_file_path, obj=best_model)
            logging.info("saving the best model into pickle file")
            y_pred = best_model.predict(X_test)
            return r2_score(y_test, y_pred)
        except Exception as e:
            raise CustomException(e, sys)


