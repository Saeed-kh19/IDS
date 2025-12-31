from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score, accuracy_score


def evaluate_final_model(pipeline, X,y):
    
    cv = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
    predictions = cross_val_predict(pipeline, X,y,cv=cv,n_jobs=-1)
    
    accuracy = accuracy_score(y,predictions)
    f1_weighted = f1_score(y,predictions,average="weighted")
    f1_macro = f1_score(y,predictions,average="macro")
    
    
    print("\n FINAL RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print("\nClassification Report: \n")
    print(classification_report(y,predictions))
    
    
    