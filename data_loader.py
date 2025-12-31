import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded with shape: {df.shape}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    cleaned_columns = []
    seen = {}
    for col in df.columns:
        clean_col = re.sub(r"[ /]", "_", col.strip())
        clean_col = re.sub(r"__+", "_", clean_col)
        clean_col = clean_col.lower()

        if clean_col in seen:
            seen[clean_col] += 1
            clean_col = f"{clean_col}_{seen[clean_col]}"
        else:
            seen[clean_col] = 0

        cleaned_columns.append(clean_col)

    df.columns = cleaned_columns
    print(f"[INFO] Columns cleaned and standardized to lowercase.")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
        print("[INFO] Numeric missing values imputed with median.")

    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
        print("[INFO] Categorical missing values imputed with most frequent value.")

    return df


def split_features_and_target(df: pd.DataFrame, target_column: str):
    target_column_clean = target_column.lower().replace(" ", "_")
    df_columns_lower = [c.lower() for c in df.columns]

    if target_column_clean not in df_columns_lower:
        raise ValueError(f"Target column '{target_column}' not found! Available columns:\n{df.columns.tolist()}")

    original_col_name = df.columns[df_columns_lower.index(target_column_clean)]
    df = df.rename(columns={original_col_name: "label"})
    print(f"[INFO] Target column '{original_col_name}' renamed to 'label'")

    X = df.drop(columns=["label"])
    y = df["label"]

    return X, y


def train_test_split_data(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print("[INFO] Train/Test split completed (70% train / 30% test)")
    return X_train, X_test, y_train, y_test
