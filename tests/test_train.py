import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from src.train import train_model


def build_sample_data():
    X_train = pd.DataFrame(
        {
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [2.0, 4.0, 6.0, 8.0, 10.0],
            'feature3': [1.5, 3.0, 4.5, 6.0, 7.5]
        }
        )
    y_train = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    return X_train, y_train


def build_preprocessor():
    """Create a basic ColumnTransformer for preprocessing."""
    preprocessor = make_column_transformer(
        (StandardScaler(), ['feature1', 'feature2', 'feature3']),
        remainder='passthrough'
    )
    return preprocessor


def test_train_model_regression_success():
    """Test successful regression model training."""
    X_train, y_train = build_sample_data()
    preprocessor = build_preprocessor()
    model = train_model(X_train, y_train, preprocessor,
                        problem_type="regression")
    assert model is not None
    assert hasattr(model, 'predict')


def test_train_model_classification_fail():
    """Test successful classification model training."""
    X_train, y_train = build_sample_data()
    preprocessor = build_preprocessor()
    y_train_binary = pd.Series([0, 1, 0, 1, 0])
    with pytest.raises(ValueError,
                       match="Training failed: problem_type not supported"):
        train_model(X_train, y_train_binary, preprocessor,
                    problem_type='classification')


def test_train_model_empty_X_train():
    """Test that training fails with empty X_train."""
    _, y_train = build_sample_data()
    X_train_empty = pd.DataFrame()
    preprocessor = build_preprocessor()
    with pytest.raises(ValueError, match="Training failed: X_train is empty."):
        train_model(X_train_empty, y_train, preprocessor)


def test_train_model_empty_y_train():
    """Test that training fails with empty y_train."""
    X_train, _ = build_sample_data()
    y_train_empty = pd.Series(dtype=float)
    preprocessor = build_preprocessor()
    with pytest.raises(ValueError, match="Training failed: y_train is empty."):
        train_model(X_train, y_train_empty, preprocessor)


def test_train_model_none_X_train():
    """Test that training fails with None X_train."""
    _, y_train = build_sample_data()
    preprocessor = build_preprocessor()
    with pytest.raises((ValueError, TypeError)):
        train_model(None, y_train, preprocessor)


def test_train_model_none_y_train():
    """Test that training fails with None y_train."""
    X_train, _ = build_sample_data()
    preprocessor = build_preprocessor()
    with pytest.raises((ValueError, TypeError)):
        train_model(X_train, None, preprocessor)


def test_train_model_invalid_problem_type():
    """Test that invalid problem_type raises an error."""
    X_train, y_train = build_sample_data()
    preprocessor = build_preprocessor()
    with pytest.raises(ValueError,
                       match="Training failed: problem_type not supported"):
        train_model(X_train, y_train, preprocessor, problem_type="clustering")


def test_train_model_default_problem_type():
    """Test that default problem_type is 'regression'."""
    X_train, y_train = build_sample_data()
    preprocessor = build_preprocessor()
    model = train_model(X_train, y_train, preprocessor)
    assert model is not None
    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train)


def test_train_model_mismatched_X_y_lengths():
    """Test that training fails when X_train and y_train have different
    lengths."""
    X_train = pd.DataFrame({'feature1': [1.0, 2.0, 3.0]})
    y_train = pd.Series([10.0, 20.0])
    preprocessor = build_preprocessor()
    with pytest.raises(ValueError):
        train_model(X_train, y_train, preprocessor)


def test_train_model_predictions_shape():
    """Test that model predictions have correct shape."""
    X_train, y_train = build_sample_data()
    preprocessor = build_preprocessor()
    model = train_model(X_train, y_train, preprocessor,
                        problem_type="regression")
    predictions = model.predict(X_train)
    assert predictions.shape[0] == X_train.shape[0]
