import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger= get_logger(__name__)

class DataProcessor:
    
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
    def preprocess_data(self, df):
        try:
            logger.info("Starting data Processing Step")
            
            logger.info("Dropping the columns and duplicates")
            df.drop(columns=["Unnamed: 0","Booking_ID"], inplace=True)
            df.drop_duplicates(inplace=True)
            
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]
            
            logger.info("Applying Lebel Encoding")
            
            label_encoder = LabelEncoder()
            mappings={}
            
            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col]= {label: code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
                
            logger.info("Label mappings are : ")
            for col, mapping in mappings.items():
                logger.info(f"{col} : {mapping}")
                
            logger.info("Perform Skewness Handling")
            
            skew_threshold= self.config["data_processing"]["skewness_threshold"]
            
            skewness=df.skew()
            for col in df.columns:
                if skewness[col] > skew_threshold:
                    df[col] = np.log1p(df[col])
            
            return df
        
        except Exception as e:
            logger.error(f"Error during preprocess step {e}")
            raise CustomException("Error while preprcessing data", e)
        
    def balance_data(self,df):
        try:
            logger.info("Handling Imbalanced data")
            
            X=df.drop(columns="booking_status")
            y=df["booking_status"]
            
            smote= SMOTE(random_state=42)
            X_res, y_res= smote.fit_resample(X,y)
            
            balanced_df= pd.DataFrame(X_res, columns=X.columns)
            balanced_df["booking_status"]= y_res
            
            logger.info("Data balaned successful")
            return balanced_df
        except Exception as e:
            logger.error(f"Error during balanced step {e}")
            raise CustomException("Error while balancing data", e)
        
    def select_features(self,df):
        try:
            logger.info("Starting feature selection process")
            X=df.drop(columns="booking_status")
            y=df["booking_status"]
            model = RandomForestClassifier(random_state=42)
            model.fit(X,y)
            
            feature_importance=model.feature_importances_
            feature_importance_df= pd.DataFrame(
                {
                    "feature" : X.columns,
                    "importance" : feature_importance
                    
                })
            feature_importance_df=feature_importance_df.sort_values(by="importance", ascending=False)
            top_10_features = feature_importance_df["feature"].head(10).values
            logger.info(f"Top 10 selected features are {top_10_features}")
            top_10_df= df[top_10_features.tolist() + ["booking_status"]]
            logger.info("feature selection process completed")
            return top_10_df
            
        except Exception as e:
            logger.error(f"Error during feature selection step {e}")
            raise CustomException("Error while doing feature selection on data", e)
        
    def save_data(self,df,file_path):
        try:
            logger.info("Saving our data in processed folder")
            df.to_csv(file_path, index=False)
            logger.info(f"Data Saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error during saving data step {e}")
            raise CustomException("Error while saving data", e)
        
    def process(self):
        try:
            logger.info("Loading data from raw directory")
            
            train_df= load_data(self.train_path)
            test_df= load_data(self.test_path)
            
            train_df= self.preprocess_data(train_df)
            test_df=self.preprocess_data(test_df)
            
            train_df= self.balance_data(train_df)
            test_df=self.balance_data(test_df)
            
            train_df= self.select_features(train_df)
            test_df= test_df[train_df.columns]
            
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)
            
            logger.info("Data processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline  {e}")
            raise CustomException("Error while preprocessing pipeline ", e)
        
if __name__=="__main__":
    processor= DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()
        
        