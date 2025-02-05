import pytest
import os
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from ml.model import inference, compute_model_metrics
from ml.data import process_data


# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_features(data, features):
    """
    # add description for the first test
    Check that categorical features are in dataset
    """
    # Your code here
    try:
        assert sorted(set(data.columns).intersection(features)) == sorted(features)
    except AssertionError as err:
        logging.error(
            "Testing dataset: Features are missing in the data columns")
        raise err



# TODO: implement the second test. Change the function name and input as needed
def test_inference(data):
    """
    # add description for the second test
    Check inference function
    """
    # Your code here

    X_train, y_train = data

    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        model = pickle.load(open(savepath, 'rb'))

        try:
            preds = inference(model, X_train)
        except Exception as err:
            logging.error(
            "Inference cannot be performed on saved model and train data")
            raise err
    else:
        pass


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics(data):
    """
    # add description for the third test

    Check calculation of performance metrics function
    """
    # Your code here

    X_train, y_train = data

    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        model = pickle.load(open(savepath, 'rb'))
        preds = inference(model, X_train)

        try:
            precision, recall, fbeta = compute_model_metrics(y_train, preds)
        except Exception as err:
            logging.error(
            "Performance metrics cannot be calculated on train data")
            raise err
    else:
        pass
