import pandas as pd

def load_dataset(csv_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    print(f"[info] dataset loaded with shape: {dataframe.shape}")
    return dataframe

def split_featuress_and_target(dataframe: pd.DataFrame, target_column: str):
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found!")
    
    feature_matrix = dataframe.drop(columns=[target_column])
    target_vector = dataframe[target_column]
    
    return feature_matrix, target_vector