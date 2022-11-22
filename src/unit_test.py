import data_pipeline
import util as utils
import pandas as pd
import numpy as np

def test_convert_datetime():
    # Arrange
    config = utils.load_config()

    mock_data = {
            "UTC" : [1669133247]}
    mock_data = pd.DataFrame(mock_data)

    expected_data = {
            "UTC" : pd.date_range("2022-11-22 16:07:27", periods = 1)}
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = data_pipeline.convert_datetime(mock_data, config)

    # Assert
    assert processed_data.equals(expected_data)
