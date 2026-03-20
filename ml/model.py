"""
Machine learning model functions for training, inference, and evaluation.
"""
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Train a machine learning model.

    Parameters
    ----------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validate the trained machine learning model using precision, recall,
    and F1.

    Parameters
    ----------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Parameters
    ----------
    model : sklearn model
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : sklearn model
        Trained machine learning model.
    path : str
        Path to save the model.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    """
    Load a trained model from disk.

    Parameters
    ----------
    path : str
        Path to the saved model.

    Returns
    -------
    model : sklearn model
        Loaded machine learning model.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def performance_on_categorical_slice(
    df,
    feature,
    y,
    preds
):
    """
    Compute performance metrics on slices of the data based on categorical
    feature values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the feature.
    feature : str
        Name of the categorical feature to slice on.
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    results : dict
        Dictionary with feature values as keys and metrics as values.
    """
    results = {}

    for value in df[feature].unique():
        mask = df[feature] == value
        y_slice = y[mask]
        preds_slice = preds[mask]

        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)

        results[value] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta,
            'count': mask.sum()
        }

    return results
