# import numpy as np
# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer


# def detect_feature_types(dataframe: pd.DataFrame):
#     numerical_features = dataframe.select_dtypes(include=[np.number]).columns.tolist()
#     categorical_features = dataframe.select_dtypes(
#         include=["object", "bool", "category"]
#     ).columns.tolist()

#     return numerical_features, categorical_features


# def build_preprocessing_pipeline(numerical_features, categorical_features):
#     transformers = []

#     # Numerical features pipeline
#     if numerical_features:
#         numerical_pipeline = Pipeline(steps=[
#             ("numeric_imputer", SimpleImputer(strategy="median")),
#             ("numeric_scaler", StandardScaler())
#         ])

#         transformers.append(
#             ("numerical", numerical_pipeline, numerical_features)
#         )

#     # Categorical features pipeline (safe even if empty)
#     if categorical_features:
#         categorical_pipeline = Pipeline(steps=[
#             ("categorical_imputer", SimpleImputer(strategy="most_frequent")),
#             ("categorical_encoder", OneHotEncoder(handle_unknown="ignore"))
#         ])

#         transformers.append(
#             ("categorical", categorical_pipeline, categorical_features)
#         )

#     preprocessing_pipeline = ColumnTransformer(
#         transformers=transformers
#     )

#     return preprocessing_pipeline


# preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


def detect_feature_types(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessing_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor
