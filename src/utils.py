import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    '''
    Taking the file path of the pickle file, 
    reading it using read byte mode and loading the object using dill
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_pred)
            report[list(models.keys())[i]] = test_score
        return report
    
    except Exception as e:
        raise CustomException(e, sys)