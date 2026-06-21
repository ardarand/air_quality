import pandas as pd
import numpy as np

import preprocessing

from utils import *

def test_join_categories():
    """Unit test for join_categories() in preprocessing.py"""

    config = load_config()
    
    # Arrange
    mock_data = {
        'category': ['BAIK', 'TIDAK SEHAT', 'SEDANG', 'TIDAK SEHAT', 'BAIK', 'TIDAK SEHAT', 'SEDANG', 'BAIK']
    }
    mock_data = pd.DataFrame(mock_data)

    expected_data = {
        'category': ['BAIK', 'TIDAK BAIK', 'TIDAK BAIK', 'TIDAK BAIK', 'BAIK', 'TIDAK BAIK', 'TIDAK BAIK', 'BAIK']
    }
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = preprocessing.join_categories(mock_data, config)

    # Assert
    assert processed_data.equals(expected_data)

def test_nan_replace():
    """Unit test for nan_replace() in preprocessing.py"""

    # Arrange
    mock_data = {
        'data': [2, 3, -1, 4, 0, -1]
    }
    mock_data = pd.DataFrame(mock_data)

    expected_data = {
        'data': [2, 3, np.nan, 4, 0, np.nan]
    }
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = preprocessing.nan_replace(mock_data)

    # Assert
    assert processed_data.equals(expected_data)

def test_transform_ohe():
    """Unit test for transform_ohe_encoder() in preprocessing.py"""

    config = load_config()
    path_ohe = config['path_encoder_stasiun']
    ohe_encoder = deserialize_data(path_ohe)

    # Arrange
    mock_data = {
        'stasiun': ['DKI1 (Bunderan HI)', 'DKI2 (Kelapa Gading)', 'DKI3 (Jagakarsa)', 'DKI4 (Lubang Buaya)', 'DKI5 (Kebon Jeruk) Jakarta Barat']
    }
    mock_data = pd.DataFrame(mock_data)

    expected_data = {
        'DKI1 (Bunderan HI)': [1, 0, 0, 0, 0],
        'DKI2 (Kelapa Gading)': [0, 1, 0, 0, 0],
        'DKI3 (Jagakarsa)': [0, 0, 1, 0, 0],
        'DKI4 (Lubang Buaya)': [0, 0, 0, 1, 0],
        'DKI5 (Kebon Jeruk) Jakarta Barat': [0, 0, 0, 0, 1]
    }
    expected_data = pd.DataFrame(expected_data)
    expected_data = expected_data.astype(float)

    # Act
    processed_data = preprocessing.transform_ohe_encoder(mock_data, ohe_encoder)

    # Assert
    assert processed_data.equals(expected_data)

def test_transform_label():
    """Unit test for transform_label_encoder() in preprocessing.py"""

    config = load_config()
    path_le = config['path_encoder_label']
    label_encoder = deserialize_data(path_le)

    # Arrange
    mock_data = {
        'category': ['BAIK', 'TIDAK BAIK', 'BAIK', 'BAIK', 'TIDAK BAIK']
    }
    mock_data = pd.DataFrame(mock_data)

    expected_data = np.array([0, 1, 0, 0, 1])
    expected_data = pd.Series(expected_data)

    # Act
    processed_data = preprocessing.transform_label_encoder(mock_data['category'], label_encoder)

    # Assert
    assert processed_data.equals(expected_data)