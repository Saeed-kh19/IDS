# # evaluation.py

# from sklearn.metrics import classification_report, f1_score, accuracy_score
# from sklearn.pipeline import Pipeline


# def evaluate_final_model(
#     model,
#     preprocessing_pipeline,
#     X_train,
#     y_train,
#     X_test,
#     y_test
# ):
#     pipeline = Pipeline(steps=[
#         ("preprocessing", preprocessing_pipeline),
#         ("classifier", model)
#     ])

#     pipeline.fit(X_train, y_train)
#     predictions = pipeline.predict(X_test)

#     accuracy = accuracy_score(y_test, predictions)
#     f1_weighted = f1_score(y_test, predictions, average="weighted")
#     f1_macro = f1_score(y_test, predictions, average="macro")

#     print("\nFINAL RESULTS (UNSEEN TEST SET)")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"F1 Weighted: {f1_weighted:.4f}")
#     print(f"F1 Macro: {f1_macro:.4f}")
#     print("\nClassification Report:\n")
#     print(classification_report(y_test, predictions))




# evaluation.py

from sklearn.metrics import classification_report, accuracy_score, f1_score


def evaluate_on_test(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")
    f1_weighted = f1_score(y_test, predictions, average="weighted")

    print("\nFINAL RESULTS (UNSEEN TEST SET)")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))
