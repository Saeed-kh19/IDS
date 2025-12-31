import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_features=20, random_state=42, verbose=True):
        self.num_features = num_features
        self.random_state = random_state
        self.selected_indices_ = None
        self.verbose = verbose
        self._printed_once = False

    def fit(self, X, y):
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced"
        )
        rf.fit(X, y)
        importances = rf.feature_importances_

        self.selected_indices_ = np.argsort(importances)[::-1][:self.num_features]

        if self.verbose and not self._printed_once:
            print(f"[INFO] Top {self.num_features} features selected based on RandomForest importance.")
            self._printed_once = True

        return self

    def transform(self, X):
        if self.selected_indices_ is None:
            raise RuntimeError("The transformer has not been fitted yet!")
        return X[:, self.selected_indices_]


def select_top_features(X, y, num_features=20, verbose=True):
    
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(X, y)
    importances = rf.feature_importances_

    selected_indices = np.argsort(importances)[::-1][:num_features]
    if verbose:
        print(f"[INFO] Selected top {num_features} features (standalone mode).")
    return selected_indices
