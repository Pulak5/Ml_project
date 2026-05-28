import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Xgboost regressor":XGBRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Nearest Neighbours":KNeighborsRegressor(),
                "Adaboosting Regressor":AdaBoostRegressor(),
                "Gradient Boosting Regressor":GradientBoostingRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(),
                "Support Vector Regressor":SVR()
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "Support Vector Regressor":{},
                "K-Nearest Neighbours":{},
                "Xgboost regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Adaboosting Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                              models=models,param=params)
            
            best_model_name,best_model_score=sorted(model_report.items(),key=lambda x:x[1],reverse=True)[0]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info("Best model found on both training and test data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2=r2_score(y_test,predicted)
            return r2

        except Exception as e:
            raise CustomException(e,sys)
    
    def model_trainer(self):
        obj=DataTransformation()
        train_set,test_set,path=obj.data_transformation()
        return self.initiate_model_trainer(train_array=train_set,test_array=test_set,preprocessor_path=path)


if __name__=="__main__":
    obj=ModelTrainer()
    print(obj.model_trainer())
