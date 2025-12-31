import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def select_top_features_random_forest(X,y,number_of_features:int):
    
    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    random_forest.fit(X,y)
    
    importances = random_forest.feature_importances_
    importance_dataframe = pd.DataFrame({
        "feature_index": range(len(importances)),
        "importance": importances
    })
    
    selected_indices = importance_dataframe.sort_values(
        by="importance",
        ascending=False
    ).head(number_of_features)["feature_index"].values
    
    return selected_indices