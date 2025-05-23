import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "k-Neighbors":KNeighborsRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor(verbose=0),
                "AdaBoost":AdaBoostRegressor(),
                "Linear Regression":LinearRegression()
            }

            params={
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
                },
                "Linear Regression":{},
                "k-Neighbors":{
                    'n_neighbors':[5,7,9,11]
                },
                "XGBoost":{
                    'learning_rate':[.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,0.01,0.5,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test= X_test, y_test=y_test,models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model saved successfully")

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return best_model_name, r2_square

        except Exception as e:
            raise CustomException(e, sys)
        finally:
            pass
