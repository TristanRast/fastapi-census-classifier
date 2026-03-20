"""
Data processing functions for the ML model.
"""
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=[],
    label=None,
    training=True,
    encoder=None,
    lb=None
):
    """
    Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features
    and a label binarizer for the labels. This can be used in either
    training or inference/validation.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features` will be one-hot encoded.
    categorical_features : list[str]
        List containing the names of the categorical features (default=[])
    label : str, optional
        Name of the label column in `X`. If None, then an empty array will be
        returned for y (default=None)
    training : bool, optional
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed in.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = pd.Series([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            pass

    X = pd.concat(
        [pd.DataFrame(X_continuous.values), pd.DataFrame(X_categorical)],
        axis=1
    )
    return X, y, encoder, lb
