# model_trainer.py
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import precision_score , accuracy_score ,confusion_matrix , f1_score , recall_score

import numpy as np

def train_model(X_train , y_train):
    print("Training model...")

    model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Training Complete")
    return model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='FAKE')
    recall = recall_score(y_test, y_pred, pos_label='FAKE')
    f1 = f1_score(y_test, y_pred, pos_label='FAKE')

    cm = confusion_matrix(y_test, y_pred, labels=['REAL', 'FAKE'])

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Fake): {precision:.4f}")
    print(f"Recall (Fake): {recall:.4f}")
    print(f"F1-Score (Fake): {f1:.4f}")
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               REAL   FAKE")
    print(f"Actual REAL:   {cm[0, 0]:<5}  {cm[0, 1]:<5}")
    print(f"Actual FAKE:   {cm[1, 0]:<5}  {cm[1, 1]:<5}")
    print("------------------------------")
    print(f"True Negatives (REAL correctly identified): {cm[0, 0]}")
    print(f"False Positives (REAL incorrectly identified as FAKE): {cm[0, 1]}")
    print(f"False Negatives (FAKE incorrectly identified as REAL): {cm[1, 0]}")
    print(f"True Positives (FAKE correctly identified): {cm[1, 1]}")
    print("Evaluation finished.")

