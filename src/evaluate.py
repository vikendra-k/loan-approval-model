from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on test data.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, cm, report

import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    """
    Evaluates model using a custom probability threshold.
    """
    probs = model.predict_proba(X_test)[:, 1]  # probability of class 1
    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    return cm
