import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
from src.exception_1 import CustomException
from src.logger_1 import logging

def deg2rad(deg): 
    try:
        return deg * (math.pi/180)
    except Exception as e:
        logging.info('Exception occured while converting points from Degree to radian')
        raise CustomException(e, sys)



def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    try:
        
        R = 6371  #Radius of the earth in km
        dLat = deg2rad(lat2-lat1)  #// deg2rad below
        dLon = deg2rad(lon2-lon1)
     
        a= math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = R * c; # Distance in km
        return d;
    except Exception as e:
        logging.info('Exception occured while calculating the distance')
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)