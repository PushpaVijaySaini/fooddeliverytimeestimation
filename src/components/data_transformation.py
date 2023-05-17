import sys
from dataclasses import dataclass


import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from math import sin, cos, sqrt, atan2, radians
from math import pi
import math

from src.exception_1 import CustomException
from src.logger_1 import logging
import os
from src.utils_1 import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Weather_conditions','Road_traffic_density','Type_of_order','Type_of_vehicle','Festival','City']
            numerical_cols = ['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries','distance_km']
            
            # Define the custom ranking for each ordinal variable
            Road_categories= (['Jam','High','Medium','Low'])
            Weather_categories=(['Fog','Stormy','Sandstorms','Windy','Cloudy','Sunny'])
            Festival_categories=(['No','Yes'])
            City_categories=(['Metropolitian','Urban','Semi-Urban'])
            Order_categories=(['Snack','Meal','Drinks','Buffet'])
            Vehicle_categories=(['motorcycle','scooter','electric_scooter','bicycle'])
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                    ]

                    )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('ordinalencoder',OrdinalEncoder(categories=[Road_categories,Weather_categories,Festival_categories,City_categories,Order_categories,Vehicle_categories],handle_unknown='use_encoded_value',unknown_value=-999)),
    
            ('scaler',StandardScaler())]
                

                )

            preprocessor=ColumnTransformer(
            [('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)])
           
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            # Calculate the distance between each pair of points
            train_df['distance_km'] = np.nan
            test_df['distance_km'] = np.nan
            def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    
  
                R = 6371  #Radius of the earth in km
                dLat = deg2rad(lat2-lat1)  #// deg2rad below
                dLon = deg2rad(lon2-lon1)
     
                a= math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                d = R * c; # Distance in km
                return d;


            def deg2rad(deg): 
                return deg * (math.pi/180)
            for i in range(len(train_df)):
                train_df.loc[i, 'distance_km'] = getDistanceFromLatLonInKm(train_df.loc[i, 'Restaurant_latitude'], 
                                        train_df.loc[i, 'Restaurant_longitude'], 
                                        train_df.loc[i, 'Delivery_location_latitude'], 
                                        train_df.loc[i, 'Delivery_location_longitude'])
            for i in range(len(test_df)):
                test_df.loc[i, 'distance_km'] = getDistanceFromLatLonInKm(test_df.loc[i, 'Restaurant_latitude'], 
                                        test_df.loc[i, 'Restaurant_longitude'], 
                                        test_df.loc[i, 'Delivery_location_latitude'], 
                                        test_df.loc[i, 'Delivery_location_longitude'])
            logging.info('Calculating distance completed')
            
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns=[target_column_name,'ID','Delivery_person_ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Order_Date','Time_Orderd','Time_Order_picked','Type_of_order']


            train_df.drop(columns=drop_columns, axis=1)
            input_feature_train_df = train_df
            target_feature_train_df = train_df[target_column_name]

            # test_df.drop(columns=drop_columns,axis=1)
            input_feature_test_df = test_df
            target_feature_test_df = test_df[target_column_name]

            # Trnasformating using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)


            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)

