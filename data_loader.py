import pandas as pd
import numpy as np
import re


def load_dataset(csv_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)

    # Clean column names
    cleaned_columns = []
    seen = {}

    for col in dataframe.columns:
        clean_col = re.sub(r"[ /]", "_", col.strip())
        clean_col = re.sub(r"__+", "_", clean_col)

        if clean_col in seen:
            seen[clean_col] += 1
            clean_col = f"{clean_col}_{seen[clean_col]}"
        else:
            seen[clean_col] = 0

        cleaned_columns.append(clean_col.lower())

    dataframe.columns = cleaned_columns

    # 🔥 VERY IMPORTANT: replace inf with NaN
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"[INFO] Dataset loaded with shape: {dataframe.shape}")
    print(f"[INFO] Infinite values replaced with NaN")
    print(f"[INFO] Target column renamed to: label")

    return dataframe


def split_features_and_target(dataframe: pd.DataFrame, target_column: str):
    target_column = target_column.lower()

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found!")

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    return X, y
