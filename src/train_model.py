"""
train_model.py
------------------------
Trains a classifier (XGBoost, falls back to RandomForest if xgboost
isn't installed) on the engineered features to predict is_fake,
evaluates it, and saves the trained model + feature list to disk.

Run:
    python src/train_model.py
"""

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)

from feature_engineering import build_features, FEATURE_COLUMNS

try:
    from xgboost import XGBClassifier
    MODEL_NAME = "XGBoost"
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as XGBClassifier
    MODEL_NAME = "RandomForest (xgboost not installed)"


def main():
    print("Loading data...")
    df = pd.read_csv("data/reviews.csv")

    print("Engineering features...")
    df = build_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["is_fake"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Training {MODEL_NAME}...")
    if MODEL_NAME.startswith("XGBoost"):
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42
        )
    else:
        model = XGBClassifier(n_estimators=200, max_depth=6, random_state=42)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    report = classification_report(y_test, preds, target_names=["Genuine", "Fake"])
    print(report)

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

    auc = roc_auc_score(y_test, probs)
    print(f"\nROC-AUC: {auc:.4f}")

    # Feature importance
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
        importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
        print("\n=== Feature Importances ===")
        for feat, score in importances.items():
            print(f"  {feat:35s} {score:.4f}")

    # Save model + metadata
    joblib.dump(model, "models/fake_review_model.joblib")
    with open("models/metadata.json", "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "features": FEATURE_COLUMNS,
            "roc_auc": auc,
            "feature_importances": importances,
        }, f, indent=2)

    print("\nSaved model -> models/fake_review_model.joblib")
    print("Saved metadata -> models/metadata.json")


if __name__ == "__main__":
    main()
