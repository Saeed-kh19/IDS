# # feature_selection.py

# import numpy as np
# from sklearn.ensemble import RandomForestClassifier


# def select_top_features_random_forest(X_train, y_train, number_of_features: int):
#     random_forest = RandomForestClassifier(
#         n_estimators=300,
#         class_weight="balanced",
#         random_state=42,
#         n_jobs=-1
#     )

#     random_forest.fit(X_train, y_train)

#     importances = random_forest.feature_importances_
#     selected_indices = np.argsort(importances)[::-1][:number_of_features]

#     return selected_indices



# feature_selection.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def select_top_features(X, y, num_features):
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf.fit(X, y)
    importances = rf.feature_importances_

    selected_indices = np.argsort(importances)[::-1][:num_features]
    return selected_indices
