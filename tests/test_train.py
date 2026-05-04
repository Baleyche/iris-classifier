import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris


# -----------------------------
# Split Data
# -----------------------------
def split_data(df):
    X = df.drop("species", axis=1)
    y = df["species"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# Train Models
# -----------------------------
def train_models(X_train, y_train):
    models = {
        "decision_tree": DecisionTreeClassifier(),
        "logistic_regression": LogisticRegression(max_iter=200),
        "svm": SVC(probability=True)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


# -----------------------------
# Evaluate Models
# -----------------------------
def evaluate(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": metrics.accuracy_score(y_test, y_pred),
            "precision": metrics.precision_score(y_test, y_pred, average='weighted'),
            "recall": metrics.recall_score(y_test, y_pred, average='weighted'),
            "f1_score": metrics.f1_score(y_test, y_pred, average='weighted')
        }

    return results


# -----------------------------
# Save Outputs
# -----------------------------
def save_outputs(models, results):
    os.makedirs("outputs", exist_ok=True)

    # Save best model (Decision Tree for now)
    joblib.dump(models["decision_tree"], "outputs/model.pkl")

    # Save metrics
    with open("outputs/metrics.json", "w") as f:
        json.dump(results, f, indent=4)


# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    print("🚀 Training Iris Model...")

    df, iris = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    models = train_models(X_train, y_train)
    results = evaluate(models, X_test, y_test)

    save_outputs(models, results)

    print("✅ Training Complete")
    print("📊 Results:", results)


if __name__ == "__main__":
    main()
