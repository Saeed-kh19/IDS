# # model_training.py

# import numpy as np
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# def select_best_model(X_train, y_train, preprocessing_pipeline):
#     models = {
#         "RandomForest": RandomForestClassifier(
#             n_estimators=200,
#             random_state=42,
#             n_jobs=-1
#         ),
#         "ExtraTrees": ExtraTreesClassifier(
#             n_estimators=200,
#             random_state=42,
#             n_jobs=-1
#         )
#     }

#     best_model_name = None
#     best_f1_score = -1.0

#     for name, model in models.items():
#         pipeline = Pipeline(steps=[
#             ("preprocessing", preprocessing_pipeline),
#             ("classifier", model)
#         ])

#         scores = cross_val_score(
#             pipeline,
#             X_train,
#             y_train,
#             cv=5,
#             scoring="f1_macro",
#             n_jobs=-1
#         )

#         mean_score = np.mean(scores)
#         print(f"[INFO] {name} CV F1-macro: {mean_score:.4f}")

#         if mean_score > best_f1_score:
#             best_f1_score = mean_score
#             best_model_name = name

#     print(f"[INFO] Selected model: {best_model_name}")
#     return best_model_name



# model_training.py

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def select_best_model(X_train, y_train, preprocessing_pipeline):
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

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessing", preprocessing_pipeline),
            ("classifier", model)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        )

        mean_score = scores.mean()
        print(f"[INFO] {name} CV F1-macro: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name

    print(f"[INFO] Selected model: {best_model_name}")
    return best_model_name
