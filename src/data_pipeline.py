import pandas as pd
import util as utils
import copy
from sklearn.model_selection import train_test_split

def read_raw_data(config: dict) -> pd.DataFrame:
    # Return raw dataset
    return pd.read_csv(config["dataset_path"])

def convert_datetime(input_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    input_data = input_data.copy()

    # Convert to datetime
    input_data[config["datetime_columns"][0]] = pd.to_datetime(
            input_data[config["datetime_columns"][0]],
            unit = "s"
    )

    return input_data

def check_data(input_data: pd.DataFrame, config: dict, api: bool = False):
    input_data = copy.deepcopy(input_data)
    config = copy.deepcopy(config)

    if not api:
        # Check column data types
        assert input_data.select_dtypes("datetime").columns.to_list() == \
            config["datetime_columns"], "an error occurs in datetime column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            config["int_columns"], "an error occurs in int column(s)."
        assert input_data.select_dtypes("float").columns.to_list() == \
            config["float_columns"], "an error occurs in float column(s)."

        # Check range of UTC
        min_datetime_definition = pd.to_datetime(config["utc"][0], dayfirst = True)
        min_data_datetime = input_data[config["datetime_columns"][0]].sort_values().iloc[0]
            
        max_datetime_definition = pd.to_datetime(config["utc"][1], dayfirst = True)
        max_data_datetime = input_data[config["datetime_columns"][0]].sort_values().iloc[-1]

        assert ((min_datetime_definition < min_data_datetime) & (max_datetime_definition > max_data_datetime))

        # Check range of pm2.5
        assert input_data[config["float_columns"][4]].between(
                config["range_pm25"][0],
                config["range_pm25"][1]
                ).sum() == len(input_data), "an error occurs in range_pm25."
        
        # Check range of nc0.5
        assert input_data[config["float_columns"][5]].between(
                config["range_nc05"][0],
                config["range_nc05"][1]
                ).sum() == len(input_data), "an error occurs in range_nc05."
        
        # Check range of nc1.0
        assert input_data[config["float_columns"][6]].between(
                config["range_nc1"][0],
                config["range_nc1"][1]
                ).sum() == len(input_data), "an error occurs in range_nc1."
        
        # Check range of nc2.5
        assert input_data[config["float_columns"][7]].between(
                config["range_nc25"][0],
                config["range_nc25"][1]
                ).sum() == len(input_data), "an error occurs in range_nc25."

        # Check range of fire alarm
        assert input_data[config["int_columns"][5]].between(
                config["range_fire_alarm"][0],
                config["range_fire_alarm"][1]
                ).sum() == len(input_data), "an error occurs in range_fire_alarm."

    else:
        # In case checking data from api
        # Last 2 column names in list of int columns are not used as predictor (CNT and Fire Alarm)
        int_columns = config["int_columns"]
        del int_columns[-2:]

        # Last 4 column names in list of int columns are not used as predictor (NC2.5, NC1.0, NC0.5, and PM2.5)
        float_columns = config["float_columns"]
        del float_columns[-4:]

        # Check column data types
        assert input_data.select_dtypes("int64").columns.to_list() == \
            int_columns, "an error occurs in int column(s)."
        assert input_data.select_dtypes("float64").columns.to_list() == \
            float_columns, "an error occurs in float column(s)."
    
    # Check range of temperature
    assert input_data[config["float_columns"][0]].between(
            config["range_temperature"][0],
            config["range_temperature"][1]
            ).sum() == len(input_data), "an error occurs in range_temperature."
    
    # Check range of humidity
    assert input_data[config["float_columns"][1]].between(
            config["range_humidity"][0],
            config["range_humidity"][1]
            ).sum() == len(input_data), "an error occurs in range_humidity."
    
    # Check range of pressure
    assert input_data[config["float_columns"][2]].between(
            config["range_pressure"][0],
            config["range_pressure"][1]
            ).sum() == len(input_data), "an error occurs in range_pressure."
    
    # Check range of pm1.0
    assert input_data[config["float_columns"][3]].between(
            config["range_pm1"][0],
            config["range_pm1"][1]
            ).sum() == len(input_data), "an error occurs in range_pm1."
 
    # Check range of tvoc
    assert input_data[config["int_columns"][0]].between(
            config["range_tvoc"][0],
            config["range_tvoc"][1]
            ).sum() == len(input_data), "an error occurs in range_tvoc."
    
    # Check range of eco2
    assert input_data[config["int_columns"][1]].between(
            config["range_eco2"][0],
            config["range_eco2"][1]
            ).sum() == len(input_data), "an error occurs in range_eco2."
    
    # Check range of raw h2
    assert input_data[config["int_columns"][2]].between(
            config["range_raw_h2"][0],
            config["range_raw_h2"][1]
            ).sum() == len(input_data), "an error occurs in range_raw_h2."
    
    # Check range of raw ethanol
    assert input_data[config["int_columns"][3]].between(
            config["range_raw_ethanol"][0],
            config["range_raw_ethanol"][1]
            ).sum() == len(input_data), "an error occurs in range_raw_ethanol."

def split_data(input_data: pd.DataFrame, config: dict):
    # Split predictor and label
    x = input_data[config["predictors"]].copy()
    y = input_data[config["label"]].copy()

    # 1st split train and test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = config["test_size"],
        random_state = 42,
        stratify = y
    )

    # 2nd split test and valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = config["valid_size"],
        random_state = 42,
        stratify = y_test
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config)

    # 3. Convert to datetime
    raw_dataset = convert_datetime(raw_dataset, config)

    # 4. Data defense for non API data
    check_data(raw_dataset, config)

    # 5. Splitting train, valid, and test set
    x_train, x_valid, x_test, \
        y_train, y_valid, y_test = split_data(raw_dataset, config)

    # 6. Save train, valid and test set
    utils.pickle_dump(x_train, config["train_set_path"][0])
    utils.pickle_dump(y_train, config["train_set_path"][1])

    utils.pickle_dump(x_valid, config["valid_set_path"][0])
    utils.pickle_dump(y_valid, config["valid_set_path"][1])

    utils.pickle_dump(x_test, config["test_set_path"][0])
    utils.pickle_dump(y_test, config["test_set_path"][1])

    utils.pickle_dump(raw_dataset, config["dataset_cleaned_path"])