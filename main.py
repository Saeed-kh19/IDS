# main.py
"""
IoTID20 IDS end-to-end orchestration script.

Pipeline:
1) Load dataset from CSV (Google Drive path or local).
2) Clean column names, replace inf, impute missing values.
3) Split into train/test (stratified).
4) Detect feature types (numeric/categorical).
5) Build preprocessing (imputation + scaling + encoding).
6) (Optional) Feature selection using RandomForest importances.
7) Model selection with 5-fold Stratified CV (Accuracy + F1-macro).
8) Train best pipeline on full training set.
9) Evaluate on held-out test set (Accuracy, F1-macro, F1-weighted).

Notes:
- Adjust DATASET_PATH and TARGET_COLUMN to match your CSV.
- Designed for academic clarity, reproducibility, and modularity.
"""

import os
import argparse
import joblib
from pprint import pprint

# Local modules
from data_loader import load_dataset, split_features_and_target, train_test_split_data
from preprocessing import detect_feature_types, build_preprocessing_pipeline
from feature_selection import TopFeatureSelector
from model_training import select_best_model
from evaluation import evaluate_on_test


def parse_args():
    parser = argparse.ArgumentParser(description="IoTID20 IDS Training & Evaluation")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to IoTID20 CSV dataset (e.g., 'drive/MyDrive/IoTID20.csv').",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="label",
        help="Target column name in the dataset (e.g., 'Label', 'Attack', 'class').",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optional: select top-k features via RandomForest importance. If None, use all features.",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Optional path to save the trained pipeline (e.g., 'artifacts/best_pipeline.joblib').",
    )
    parser.add_argument(
        "--pretty",
        type=str,
        default="tabulate",
        choices=["none", "tabulate", "rich"],
        help="Pretty printing mode for final report: none, tabulate, or rich.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.data
    target_column = args.target
    top_k = args.top_k
    save_model_path = args.save_model
    pretty = args.pretty

    print("=== IoTID20 IDS: Orchestration Start ===")

    # 1) Load dataset
    df = load_dataset(dataset_path)

    # 2) Split features/target
    X, y = split_features_and_target(df, target_column)

    # 3) Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.3)

    # 4) Detect feature types
    numeric_features, categorical_features = detect_feature_types(X_train)

    # 5) Build preprocessing pipeline (impute + scale + encode)
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

    # 6) Optional feature selection
    feature_selector = None
    if top_k is not None and top_k > 0:
        feature_selector = TopFeatureSelector(num_features=top_k, random_state=42, verbose=True)
        print(f"[INFO] Feature selection enabled: Top {top_k} features will be used.")
    else:
        print("[INFO] Feature selection disabled: Using all features.")

    # 7) Model selection with 5-fold CV, return trained best pipeline
    best_pipeline = select_best_model(
        X_train=X_train,
        y_train=y_train,
        preprocessing_pipeline=preprocessor,
        feature_selector=feature_selector
    )

    # 8) Evaluate on held-out test set
    metrics = evaluate_on_test(best_pipeline, X_test, y_test, pretty=pretty)

    print("\n=== SUMMARY METRICS ===")
    pprint({k: v for k, v in metrics.items() if k != "report"})

    # 9) Optionally save the trained pipeline for reproducibility
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        joblib.dump(best_pipeline, save_model_path)
        print(f"[INFO] Trained pipeline saved to: {save_model_path}")

    print("=== IoTID20 IDS: Orchestration Complete ===")


if __name__ == "__main__":
    main()
