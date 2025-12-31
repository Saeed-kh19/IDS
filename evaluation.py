# evaluation.py
"""
Evaluation utilities for IoTID20 IDS project.
Handles:
- Final test set evaluation of trained pipeline
- Reporting Accuracy, F1-macro, F1-weighted
- Printing classification report
"""

from sklearn.metrics import classification_report, accuracy_score, f1_score


def evaluate_on_test(model_pipeline, X_test, y_test):
    """
    Evaluate the trained pipeline on the held-out test set.
    Reports Accuracy, F1-macro, F1-weighted, and classification report.
    Returns metrics as a dictionary.
    """
    predictions = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")
    f1_weighted = f1_score(y_test, predictions, average="weighted")

    print("\n=== FINAL RESULTS (UNSEEN TEST SET) ===")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"F1 Macro   : {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }
    return metrics
