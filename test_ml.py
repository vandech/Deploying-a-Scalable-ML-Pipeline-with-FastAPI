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




# Mock logging (important for tests)
logging.basicConfig(level=logging.INFO)  # Setting the default level
test_logger = logging.getLogger(__name__)  # Creating a logger for the test file
@pytest.fixture
def mock_logger(monkeypatch):
    def mock_info(msg, *args, **kwargs):
        test_logger.info(msg, *args, **kwargs)
        monkeypatch.setattr(logging, "info", mock_info)
        return test_logger


# Mock multiprocessing (to avoid actual parallel processing in tests)
@pytest.fixture
def mock_cpu_count(monkeypatch):
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 2)  # simulate 2 cores

# TODO: implement the first test. Change the function name and input as needed
def test_train_model(mock_cpu_count, mock_logger): # Injects mocks
    """
    # add description for the first test
    Check if it returns the expected model type
    """
    # Your code here
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, GridSearchCV)  # Check if it's a GridSearchCV object
    assert hasattr(model, 'best_estimator_')  # Check that best_estimator_ is present after training
    assert isinstance(model.best_estimator_, GradientBoostingClassifier)  # Check if best_estimator_ is a GradientBoostingClassifier
    assert model.csv == 3



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




