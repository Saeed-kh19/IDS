import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def detect_feature_types(dataframe: pd.DataFrame):
    numeric_features = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = dataframe.select_dtypes(include=["object","bool","category"]).columns.tolist()
    
    return numeric_features, categorical_features


def build_preprocessing_pipeline(numric_feature, categorical_features):
    numeric_pipeline = Pipeline(steps=[
        ("numeric_imputer", SimpleImputer(strategy="median")),
        ("numeric_scaler",StandardScaler())
    ])
    
    
    categorical_pipeline = Pipeline (steps=[
        ("categorical_imputer", SimpleImputer(strategy="most_frequent")),
        ("categorical_encoder", OneHotEncoder(handle_unknown="ignoe"))
    ])
    
    preprocessing_pipeline = ColumnTransformer(transformers=[
        ("numerical",numeric_pipeline,numric_feature),
        ("categorical", categorical_pipeline, categorical_features)
    ])
    
    
    return preprocessing_pipeline