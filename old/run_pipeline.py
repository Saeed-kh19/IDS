# # run_pipeline.py

# import argparse
# import os

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression

# from data_loader import load_dataset, split_features_and_target
# from preprocessing import detect_feature_types, build_preprocessing_pipeline
# from model_training import select_best_model
# from evaluation import evaluate_final_model


# def main(args):
#     os.makedirs("outputs", exist_ok=True)

#     # 1️⃣ Load & prepare data
#     dataset = load_dataset(args.data_path)
#     X, y = split_features_and_target(dataset, args.target_column)

#     # 2️⃣ Train / Test split (70 / 30)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.30,
#         random_state=42,
#         stratify=y
#     )

#     print("[INFO] Train/Test split completed (70% train / 30% test)")

#     # 3️⃣ Detect feature types ONLY from training data
#     numeric_features, categorical_features = detect_feature_types(X_train)

#     preprocessing_pipeline = build_preprocessing_pipeline(
#         numeric_features,
#         categorical_features
#     )

#     # 4️⃣ Model selection using CV on TRAINING data only
#     best_model_name = select_best_model(
#         X_train,
#         y_train,
#         preprocessing_pipeline
#     )

#     # 5️⃣ Initialize final model
#     if best_model_name == "RandomForest":
#         final_model = RandomForestClassifier(
#             n_estimators=300,
#             class_weight="balanced",
#             random_state=42,
#             n_jobs=-1
#         )

#     elif best_model_name == "ExtraTrees":
#         final_model = ExtraTreesClassifier(
#             n_estimators=300,
#             class_weight="balanced",
#             random_state=42,
#             n_jobs=-1
#         )

#     elif best_model_name == "SVM":
#         final_model = SVC(
#             kernel="rbf",
#             class_weight="balanced",
#             probability=True
#         )

#     else:
#         final_model = LogisticRegression(
#             max_iter=1000,
#             class_weight="balanced"
#         )

#     # 6️⃣ Final evaluation on UNSEEN TEST DATA
#     evaluate_final_model(
#         model=final_model,
#         preprocessing_pipeline=preprocessing_pipeline,
#         X_train=X_train,
#         y_train=y_train,
#         X_test=X_test,
#         y_test=y_test
#     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="IDS Classification - IoTID20"
#     )
#     parser.add_argument("--data_path", type=str, required=True)
#     parser.add_argument("--target_column", type=str, default="Label")

#     arguments = parser.parse_args()
#     main(arguments)


# run_pipeline.py

import argparse
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from old.data_loader import load_dataset, split_features_and_target, train_test_split_data
from preprocessing import detect_feature_types, build_preprocessing_pipeline
from feature_selection import select_top_features
from model_training import select_best_model
from old.evaluation import evaluate_on_test


def main(args):
    # Load dataset
    df = load_dataset(args.data_path)

    # Split features and target robustly
    X, y = split_features_and_target(df, args.target_column)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Detect feature types and build preprocessing pipeline
    numeric_features, categorical_features = detect_feature_types(X_train)
    preprocessing_pipeline = build_preprocessing_pipeline(
        numeric_features, categorical_features
    )

    # Select best model using CV on training set
    best_model_name = select_best_model(X_train, y_train, preprocessing_pipeline)

    # Create final classifier
    if best_model_name == "RandomForest":
        classifier = RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
        )
    else:
        classifier = ExtraTreesClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
        )

    # Final pipeline
    final_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing_pipeline),
        ("classifier", classifier)
    ])

    # Train on training set
    final_pipeline.fit(X_train, y_train)

    # Evaluate on unseen test set
    evaluate_on_test(final_pipeline, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDS Classification - IoTID20")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_column", type=str, default="label")
    args = parser.parse_args()

    main(args)
