import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import save_object
from src.components.data_ingestion import DataIngestion

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer(self):
        '''
        this function is responsible for the data transformation
        '''
        try:
            num_features=['reading_score', 'writing_score']
            cat_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder",OneHotEncoder(drop="first"))
                ]
            )

            preprocessor=ColumnTransformer([
                ("numerical_pipeline",num_pipeline,num_features),
                ("categorical_pipeline",cat_pipeline,cat_features)
            ])
            logging.info("Transformation pipeline created for the dataset.")
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_set=pd.read_csv(train_path)
            test_set=pd.read_csv(test_path)
            logging.info("Training and Test data is fetched")
            target_column_name="math_score"

            preprocessor=self.get_data_transformer()

            logging.info("calling the get_data_transform method")

            input_feature_train=train_set.drop(columns=[target_column_name])
            target_feature_train=train_set[target_column_name]

            input_feature_test=test_set.drop(columns=[target_column_name])
            target_feature_test=test_set[target_column_name]
            logging.info("Applying the preprocessing object on the train and test data")

            input_feature_train_arr=preprocessor.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessor.transform(input_feature_test)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor
            )
            logging.info("saved preprocessing object")

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
    def data_transformation(self):
        try:
            obj=DataIngestion()
            train_path,test_path=obj.initiate_data_ingestion()
            return self.initiate_data_transformation(train_path=train_path,test_path=test_path)
        except Exception as e:
            raise CustomException(e,sys)
        
        