import argparse
import os
import pandas as pd

from data_loader import load_dataset, split_features_and_target
from preprocessing import detect_feature_types, build_preprocessing_pipeline
from model_training import evaluate_models
from evaluation import evaluate_final_model

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def main(args):
    os.makedirs("outputs", exist_ok=True)

    dataset = load_dataset(args.data_path)
    X, y = split_features_and_target(dataset, args.target_column)

    numeric_features, categorical_features = detect_feature_types(X)
    preprocessing_pipeline = build_preprocessing_pipeline(
        numeric_features, categorical_features
    )

    best_model_name = evaluate_models(X, y, preprocessing_pipeline)

    if best_model_name == "RandomForest":
        final_model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    elif best_model_name == "ExtraTrees":
        final_model = ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    elif best_model_name == "SVM":
        final_model = SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True
        )
    else:
        final_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

    final_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing_pipeline),
        ("classifier", final_model)
    ])

    evaluate_final_model(final_pipeline, X, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDS Classification - IoTID20")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_column", type=str, default="Label")

    arguments = parser.parse_args()
    main(arguments)
