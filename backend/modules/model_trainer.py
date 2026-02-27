"""
Module 3: Model Training
Trains multiple classification models and selects the best one.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import os
import json


class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        self.results = {}
        self.best_model_name = None
        self.best_model = None

    def train_all(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            cm = confusion_matrix(y_test, y_pred).tolist()
            fpr, tpr, _ = roc_curve(y_test, y_prob)

            self.results[name] = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
                "confusion_matrix": cm,
                "roc_fpr": fpr.tolist(),
                "roc_tpr": tpr.tolist(),
            }
            print(f"  ROC-AUC: {self.results[name]['roc_auc']}")

        # Select best model by ROC-AUC
        self.best_model_name = max(self.results, key=lambda k: self.results[k]["roc_auc"])
        self.best_model = self.models[self.best_model_name]
        print(f"\nBest Model: {self.best_model_name}")
        return self.results

    def get_feature_importance(self, feature_names):
        model = self.best_model
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return {}

        fi = dict(zip(feature_names, importances.tolist()))
        fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        return fi

    def predict_risk(self, X) -> dict:
        prob = self.best_model.predict_proba(X)[0][1]
        label = self.best_model.predict(X)[0]
        risk_level = "Low" if prob < 0.33 else "Medium" if prob < 0.66 else "High"
        return {
            "attrition_probability": round(float(prob), 4),
            "risk_score": round(float(prob) * 100, 1),
            "risk_level": risk_level,
            "prediction": "Yes" if label == 1 else "No",
        }

    def save(self, path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if path is None:
            path = os.path.join(base_dir, "models", "best_model.pkl")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        
        # Save results as JSON
        results_copy = {}
        for name, r in self.results.items():
            results_copy[name] = {k: v for k, v in r.items() if k not in ("roc_fpr", "roc_tpr")}
        
        results_path = os.path.join(os.path.dirname(path), "results.json")
        with open(results_path, "w") as f:
            json.dump(results_copy, f, indent=2)
        print(f"Model saved to {path}")
        print(f"Results saved to {results_path}")

    @staticmethod
    def load(path: str = None):
        if path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, "models", "best_model.pkl")
        return joblib.load(path)


if __name__ == "__main__":
    import sys
    import os
    # Add project root to path for module imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.data_generator import generate_dataset
    from modules.preprocessor import DataPreprocessor

    df = generate_dataset()
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    trainer = ModelTrainer()
    results = trainer.train_all(X_train, X_test, y_train, y_test)

    fi = trainer.get_feature_importance(preprocessor.feature_columns)
    print("\nTop Features:", list(fi.items())[:5])

    preprocessor.save()
    trainer.save()
