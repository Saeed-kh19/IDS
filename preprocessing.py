# preprocessing.py
"""
Preprocessing utilities for IoTID20 IDS project.
Handles:
- Automatic detection of numeric vs categorical features
- Missing value imputation
- Scaling numeric features
- Encoding categorical features
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def detect_feature_types(X):
    """
    Detect numeric and categorical feature columns.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"[INFO] Detected {len(numeric_features)} numeric features and {len(categorical_features)} categorical features.")
    return numeric_features, categorical_features


def build_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Build preprocessing pipeline:
    - Numeric: impute missing values (median), scale with StandardScaler
    - Categorical: impute missing values (most frequent), encode with OneHotEncoder
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    print("[INFO] Preprocessing pipeline created (imputation + scaling + encoding).")
    return preprocessor
