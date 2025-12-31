import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def evaluate_models(X,y, preprocessing_pipeline):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        
         "SVM": SVC(
             kernel="rbf",
             class_weight="balanced",
             probability=True
        ),
         
         "LogisticRegression": LogisticRegression(
             max_iter=1000,
             class_weight="balanced"
         )
    }
    
    results = {}
    
    for model_name, model in models.items():
        pipeline= Pipeline(steps=[
            ("preprocessing",preprocessing_pipeline),
            ("classifier",model)
        ])
        
        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        f1_scores = cross_val_score(
            pipeline,X,y,cv=cv,scoring="f1_weighted",n_jobs=-1
        )
        
        results[model_name] = np.mean(f1_scores)
        
        print(f"[INFO] {model_name} F1-score: {results[model_name]:.4f}")
        
        
    best_model_name = max(results,key=results.get)
    print(f"[INFO] Best model selected: {best_model_name}")
    
    
    return best_model_name