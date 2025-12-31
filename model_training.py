# model_training.py
"""
Model training utilities for IoTID20 IDS project.
Handles:
- Integration of preprocessing + feature selection + classifier
- 5-fold Stratified Cross Validation
- Reporting Accuracy and F1 scores
- Returning the best trained pipeline
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score


def build_model_pipeline(preprocessing_pipeline, classifier, feature_selector=None):
    """
    Build a pipeline with optional feature selection, preprocessing, and classifier.
    """
    steps = []
    if preprocessing_pipeline is not None:
        steps.append(("preprocessing", preprocessing_pipeline))
    if feature_selector is not None:
        steps.append(("feature_selection", feature_selector))
    steps.append(("classifier", classifier))

    pipeline = Pipeline(steps=steps)
    return pipeline


def select_best_model(X_train, y_train, preprocessing_pipeline, feature_selector=None):
    """
    Compare candidate models using 5-fold Stratified CV.
    Evaluate with both Accuracy and F1-macro.
    Return the best trained pipeline.
    """
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_model_name = None
    best_score = -1
    best_pipeline = None

    for name, model in models.items():
        pipeline = build_model_pipeline(preprocessing_pipeline, model, feature_selector)

        # Evaluate with F1-macro
        f1_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        )
        acc_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        mean_f1 = f1_scores.mean()
        mean_acc = acc_scores.mean()

        print(f"[INFO] {name} CV Accuracy: {mean_acc:.4f}, F1-macro: {mean_f1:.4f}")

        if mean_f1 > best_score:  # prioritize F1-macro
            best_score = mean_f1
            best_model_name = name
            best_pipeline = pipeline

    print(f"[INFO] Selected best model: {best_model_name} (F1-macro={best_score:.4f})")

    # Fit best pipeline on full training set
    best_pipeline.fit(X_train, y_train)
    print("[INFO] Best pipeline trained on full training set.")

    return best_pipeline
