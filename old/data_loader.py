# import pandas as pd
# import numpy as np
# import re


# def load_dataset(csv_path: str) -> pd.DataFrame:
#     dataframe = pd.read_csv(csv_path)

#     # Clean column names
#     cleaned_columns = []
#     seen = {}

#     for col in dataframe.columns:
#         clean_col = re.sub(r"[ /]", "_", col.strip())
#         clean_col = re.sub(r"__+", "_", clean_col)

#         if clean_col in seen:
#             seen[clean_col] += 1
#             clean_col = f"{clean_col}_{seen[clean_col]}"
#         else:
#             seen[clean_col] = 0

#         cleaned_columns.append(clean_col.lower())

#     dataframe.columns = cleaned_columns

#     # VERY IMPORTANT: replace inf with NaN
#     dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

#     print(f"[INFO] Dataset loaded with shape: {dataframe.shape}")
#     print("[INFO] Infinite values replaced with NaN")

#     return dataframe


# def split_features_and_target(dataframe: pd.DataFrame, target_column: str):
#     target_column = target_column.lower()

#     if target_column not in dataframe.columns:
#         raise ValueError(f"Target column '{target_column}' not found!")

#     # Explicit rename (important for clarity & grading)
#     dataframe = dataframe.rename(columns={target_column: "label"})
#     print("[INFO] Target column renamed to: label")

#     X = dataframe.drop(columns=["label"])
#     y = dataframe["label"]

#     return X, y

# data_loader.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load CSV, clean column names, replace inf with NaN.
    """
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded with shape: {df.shape}")

    # Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"[INFO] Infinite values replaced with NaN and missing values dropped.")

    # Clean column names: lowercase, replace spaces/slashes with underscores
    cleaned_columns = []
    seen = {}
    for col in df.columns:
        clean_col = re.sub(r"[ /]", "_", col.strip())
        clean_col = re.sub(r"__+", "_", clean_col)
        clean_col = clean_col.lower()

        # Handle duplicate column names
        if clean_col in seen:
            seen[clean_col] += 1
            clean_col = f"{clean_col}_{seen[clean_col]}"
        else:
            seen[clean_col] = 0

        cleaned_columns.append(clean_col)

    df.columns = cleaned_columns
    print(f"[INFO] Columns cleaned and standardized to lowercase.")

    return df


def split_features_and_target(df: pd.DataFrame, target_column: str):
    """
    Split dataframe into X (features) and y (target), robust to case and spaces.
    """
    target_column_clean = target_column.lower().replace(" ", "_")
    df_columns_lower = [c.lower() for c in df.columns]

    if target_column_clean not in df_columns_lower:
        raise ValueError(f"Target column '{target_column}' not found! Available columns:\n{df.columns.tolist()}")

    # Map to original column name in df
    original_col_name = df.columns[df_columns_lower.index(target_column_clean)]
    X = df.drop(columns=[original_col_name])
    y = df[original_col_name]

    print(f"[INFO] Target column identified as '{original_col_name}'")
    return X, y


def train_test_split_data(X, y, test_size=0.3):
    """
    Split data into 70/30 train/test with stratification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print("[INFO] Train/Test split completed (70% train / 30% test)")
    return X_train, X_test, y_train, y_test
