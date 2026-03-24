"""
Unit tests for the ML model and data processing.
"""
import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model
)
import os
import tempfile


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = pd.DataFrame({
        'age': [39, 50, 38, 53, 28],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private',
                      'Private', 'Private'],
        'fnlgt': [77516, 83311, 215646, 234721, 338409],
        'education': ['Bachelors', 'Bachelors', 'HS-grad',
                      'Some-college', 'Bachelors'],
        'education-num': [13, 13, 9, 7, 13],
        'marital-status': ['Never-married', 'Married-civ-spouse',
                           'Divorced', 'Married-civ-spouse', 'Married-civ-spouse'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
                       'Handlers-cleaners', 'Prof-specialty'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family',
                         'Husband', 'Wife'],
        'race': ['White', 'White', 'White', 'Black', 'Black'],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Female'],
        'capital-gain': [2174, 0, 0, 0, 0],
        'capital-loss': [0, 0, 0, 0, 0],
        'hours-per-week': [40, 13, 40, 40, 40],
        'native-country': ['United-States'] * 5,
        'salary': ['<=50K', '>50K', '<=50K', '<=50K', '>50K']
    })
    return data


@pytest.fixture
def categorical_features():
    """Define categorical features for testing."""
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_process_data_training(sample_data, categorical_features):
    """Test data processing in training mode."""
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    # Check that X and y have correct shapes
    assert X.shape[0] == sample_data.shape[0]
    assert y.shape[0] == sample_data.shape[0]

    # Check that encoder and label binarizer are fitted
    assert encoder is not None
    assert lb is not None

    # Check that y is binary
    assert set(y).issubset({0, 1})


def test_process_data_inference(sample_data, categorical_features):
    """Test data processing in inference mode."""
    # First process in training mode to get encoder and lb
    _, _, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    # Now test inference mode
    X_inf, y_inf, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Check shapes
    assert X_inf.shape[0] == sample_data.shape[0]
    assert y_inf.shape[0] == sample_data.shape[0]


def test_train_model(sample_data, categorical_features):
    """Test model training."""
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    model = train_model(X, y)

    # Check that model is trained
    assert model is not None
    assert hasattr(model, 'predict')


def test_compute_model_metrics():
    """Test metrics computation."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Check that metrics are in valid range
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_inference(sample_data, categorical_features):
    """Test model inference."""
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    model = train_model(X, y)
    preds = inference(model, X)

    # Check predictions shape
    assert preds.shape[0] == X.shape[0]

    # Check predictions are binary
    assert set(preds).issubset({0, 1})


def test_save_load_model(sample_data, categorical_features):
    """Test model saving and loading."""
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    model = train_model(X, y)

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        tmp_path = tmp.name

    try:
        # Save model
        save_model(model, tmp_path)

        # Check file exists
        assert os.path.exists(tmp_path)

        # Load model
        loaded_model = load_model(tmp_path)

        # Check loaded model works
        preds_original = inference(model, X)
        preds_loaded = inference(loaded_model, X)

        # Predictions should be identical
        assert np.array_equal(preds_original, preds_loaded)
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_data_shape_consistency(sample_data, categorical_features):
    """Test that processed data maintains consistent shapes."""
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    # All samples should be processed
    assert X_train.shape[0] == len(sample_data)
    assert y_train.shape[0] == len(sample_data)

    # Features should be consistent
    assert X_train.shape[1] > len(categorical_features)  # One-hot encoding expands


def test_model_predictions_valid(sample_data, categorical_features):
    """Test that model predictions are valid binary outputs."""
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    model = train_model(X, y)
    preds = inference(model, X)

    # All predictions should be 0 or 1
    assert all(p in [0, 1] for p in preds)

    # Should have same length as input
    assert len(preds) == len(X)
