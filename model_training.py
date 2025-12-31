# model_training.py

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

def evaluate_models(X, y, preprocessing_pipeline):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
    }

    best_model = None
    best_score = -1

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessing", preprocessing_pipeline),
            ("classifier", model)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=3,              #  reduced from 5 to 3
            scoring="f1_macro",
            n_jobs=-1
        )

        mean_score = np.mean(scores)
        print(f"[INFO] {name} F1-score: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = name

    return best_model
