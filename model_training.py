from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def build_model_pipeline(preprocessing_pipeline, classifier, feature_selector=None):
    steps = []
    if preprocessing_pipeline is not None:
        steps.append(("preprocessing", preprocessing_pipeline))
    if feature_selector is not None:
        steps.append(("feature_selection", feature_selector))
    steps.append(("classifier", classifier))

    pipeline = Pipeline(steps=steps)
    return pipeline


def _cv_metrics(pipeline, X_train, y_train, cv):
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
    return acc_scores.mean(), f1_scores.mean()


def select_best_model(X_train, y_train, preprocessing_pipeline, feature_selector=None):
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
        ),
}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    print("\n=== CROSS-VALIDATION (5-fold) ===")
    for name, model in models.items():
        pipeline = build_model_pipeline(preprocessing_pipeline, model, feature_selector)
        mean_acc, mean_f1 = _cv_metrics(pipeline, X_train, y_train, cv)
        print(f"[CV] {name:12s} | Accuracy: {mean_acc:.4f} | F1-macro: {mean_f1:.4f}")
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model_name = name
            best_pipeline = pipeline

    print(f"[INFO] Selected best model: {best_model_name} (F1-macro={best_f1:.4f})")

    # Fit best pipeline on full training set
    best_pipeline.fit(X_train, y_train)
    print("[INFO] Best pipeline trained on full training set.")

    return best_pipeline
