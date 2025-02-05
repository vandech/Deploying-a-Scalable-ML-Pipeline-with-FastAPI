import pytest

# TODO: add necessary import
import numpy as np
import os
import logging
import pickle
import multiprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.model import train_model, save_model, load_model, compute_model_metrics, inference  # Imports functions


# TODO: implement the first test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # add description for the first test
    Check if it returns the expected model type
    """
    # Your code here
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == 1.0  # Or 0.5 depending on how you want to handle this case.
    assert recall == 0.5
    assert fbeta == 0.6666666666666666y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == 1.0  # Or 0.5 depending on how you want to handle this case.
    assert recall == 0.5
    assert fbeta == 0.6666666666666666



# TODO: implement the second test. Change the function name and input as needed
def save_model(tmp_path):
    """
    # add description for the second test
     check whether the file was created
    """
    # Your code here

    model = GradientBoostingClassifier()  # or any model object
    path = tmp_path / "test_model.pkl"  # creates a Path object for a temporary file
    save_model(model, str(path))  # save model

    assert path.is_file()  # check if file was created



# TODO: implement the third test. Change the function name and input as needed
def test_inference():
    """
    # add description for the third test

    Check whether the inference function is working.
    """
    # Your code here

    # Create a simple mock model (replace with a real trained model for more thorough testing)
    class MockModel:
        def predict(self, X):
            return np.array([1] * len(X))  # Always predict 1

    model = MockModel()
    X = np.array([[1, 2], [3, 4], [5, 6]])

    preds = inference(model, X)

    assert np.array_equal(preds, np.array([1, 1, 1]))




