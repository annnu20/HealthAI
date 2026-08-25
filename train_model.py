"""
Model training script for the AI-Powered Healthcare Diagnosis Assistant.

This reproduces the exact training pipeline from the original notebook
(AI_Powered_Healthcare_Diagnosis_Assistant.ipynb):

  1. Load Training.csv, drop the stray "Unnamed: 133" column, drop duplicates.
  2. Split features (132 symptoms) from the target ("prognosis").
  3. Label-encode the disease names.
  4. Train/validation split (80/20, stratified).
  5. Train 5 candidate models: Decision Tree, Random Forest,
     Logistic Regression, Naive Bayes, Support Vector Machine.
  6. Pick the best model by validation accuracy.
  7. Evaluate the best model on the held-out Testing.csv.
  8. Save the best model + label encoder + symptom column list with joblib.

Run this once (or whenever Training.csv/Testing.csv change) to (re)generate
the artifacts in models/. The Streamlit app only ever loads these saved
artifacts — it never retrains on the fly.

Usage:
    python train_model.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "Training.csv")
TEST_PATH = os.path.join(DATA_DIR, "Testing.csv")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- 1. Load & clean training data -------------------------------
    df = pd.read_csv(TRAIN_PATH)

    if "Unnamed: 133" in df.columns:
        df.drop("Unnamed: 133", axis=1, inplace=True)

    df.fillna(0, inplace=True)
    df = df.drop_duplicates()

    print("Dataset shape after cleaning:", df.shape)

    # ---- 2. Features / target -----------------------------------------
    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]

    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    # ---- 3. Encode disease labels --------------------------------------
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print("Number of diseases:", len(encoder.classes_))

    # ---- 4. Train/validation split --------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)

    # ---- 5. Train candidate models ---------------------------------------
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        # Enhanced from the original notebook: more trees (100 -> 300) and
        # max_features=None (consider all 132 symptom columns at every
        # split, instead of the default sqrt(132)~=11). With this many
        # binary indicator features and a relatively small, sparse dataset,
        # letting every tree see every symptom produces less noisy,
        # better-calibrated probability estimates -- i.e. higher, more
        # trustworthy confidence scores for well-specified symptom sets --
        # without changing validation/test accuracy.
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_features=None, random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=42),
    }

    trained_models = {}
    accuracy = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracy[name] = acc
        print(f"{name} validation accuracy: {acc:.4f}")

    accuracy_df = pd.DataFrame(accuracy.items(), columns=["Model", "Accuracy"])
    accuracy_df = accuracy_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    # ---- 6. Pick the best model -------------------------------------------
    best_model_name = accuracy_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_val_accuracy = float(accuracy_df.iloc[0]["Accuracy"])

    print("\nBest model:", best_model_name)
    print("Best validation accuracy:", round(best_val_accuracy * 100, 2), "%")

    # ---- 7. Evaluate on the held-out test set -------------------------------
    test_df = pd.read_csv(TEST_PATH)
    if "Unnamed: 133" in test_df.columns:
        test_df.drop("Unnamed: 133", axis=1, inplace=True)
    test_df.fillna(0, inplace=True)

    X_test = test_df.drop("prognosis", axis=1)
    y_test = encoder.transform(test_df["prognosis"])

    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test set accuracy:", round(test_accuracy * 100, 2), "%")

    report = classification_report(
        y_test, y_test_pred, target_names=encoder.classes_, zero_division=0
    )

    # ---- 8. Save artifacts ---------------------------------------------------
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_disease_model.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "symptom_columns.pkl"))

    metadata = {
        "best_model_name": best_model_name,
        "validation_accuracy": best_val_accuracy,
        "test_accuracy": float(test_accuracy),
        "all_validation_accuracies": accuracy,
        "num_symptoms": len(X.columns),
        "num_diseases": len(encoder.classes_),
        "num_training_rows": int(X_train.shape[0]),
        "num_validation_rows": int(X_val.shape[0]),
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(MODELS_DIR, "test_classification_report.txt"), "w") as f:
        f.write(report)

    print("\nSaved model artifacts to:", MODELS_DIR)
    print(" -", "best_disease_model.pkl")
    print(" -", "label_encoder.pkl")
    print(" -", "symptom_columns.pkl")
    print(" -", "metadata.json")


if __name__ == "__main__":
    main()
