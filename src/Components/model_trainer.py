import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "Linear Regression": {},
                "K-Neighbours Regressor": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                },
                "XGB Regressor": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "CatBoosting Regressor": {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'depth': [4, 6]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                }
            }

            model_report: dict = {}

            for i in range(len(models)):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                param_grid = params[model_name]
                
                logging.info(f"Training {model_name} with hyperparameter tuning")
                
                if param_grid:
                    # Perform hyperparameter tuning
                    gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
                    gs.fit(X_train, y_train)
                    
                    # Use the best estimator
                    best_model = gs.best_estimator_
                else:
                    # For models without hyperparameters, train normally
                    model.fit(X_train, y_train)
                    best_model = model
                
                # Make predictions
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                
                # Calculate scores
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                
                model_report[model_name] = test_model_score
                
                logging.info(f"{model_name} - Train Score: {train_model_score}, Test Score: {test_model_score}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            best_model = models[best_model_name]
            best_params = params[best_model_name]
            
            # Retrain the best model with optimal hyperparameters
            if best_params:
                gs = GridSearchCV(best_model, best_params, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                final_best_model = gs.best_estimator_
            else:
                best_model.fit(X_train, y_train)
                final_best_model = best_model

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_best_model
            )
            
            predict = final_best_model.predict(X_test)
            r2_square = r2_score(y_test, predict)
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)