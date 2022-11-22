import pandas as pd
import util as utils
from imblearn.under_sampling import RandomUnderSampler

def load_dataset(config_data: dict):
    x_train = utils.pickle_load(config_data["train_set_path"][0])
    y_train = utils.pickle_load(config_data["train_set_path"][1])

    x_valid = utils.pickle_load(config_data["valid_set_path"][0])
    y_valid = utils.pickle_load(config_data["valid_set_path"][1])

    x_test = utils.pickle_load(config_data["test_set_path"][0])
    y_test = utils.pickle_load(config_data["test_set_path"][1])

    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    return train_set, valid_set, test_set

def rus_fit_resample(set_data, config):
    x_rus, y_rus = RandomUnderSampler(random_state = 42).fit_resample(  # type: ignore
        set_data.drop(columns = config["label"]),
        set_data[config["label"]]
    )
    train_set_bal = pd.concat([x_rus, y_rus], axis = 1)

    return train_set_bal

def remove_outliers(set_data):
    set_data = set_data.copy()
    list_of_set_data = list()

    for col_name in set_data.columns[:-1]:
        q1 = set_data[col_name].quantile(0.25)
        q3 = set_data[col_name].quantile(0.75)
        iqr = q3 - q1
        set_data_cleaned = set_data[~((set_data[col_name] < (q1 - 1.5 * iqr)) | (set_data[col_name] > (q3 + 1.5 * iqr)))].copy()
        list_of_set_data.append(set_data_cleaned.copy())
    
    set_data_cleaned = pd.concat(list_of_set_data)
    count_duplicated_index = set_data_cleaned.index.value_counts()
    used_index_data = count_duplicated_index[count_duplicated_index == (set_data.shape[1]-1)].index
    set_data_cleaned = set_data_cleaned.loc[used_index_data].drop_duplicates()

    return set_data_cleaned

if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config)

    # 3. Undersampling dataset
    train_set_bal = rus_fit_resample(train_set, config)

    # 4. Removing outliers
    train_set_bal_cleaned = remove_outliers(train_set_bal)

    # 5. Dump set data
    utils.pickle_dump(
            train_set_bal_cleaned[config["predictors"]],
            config["train_feng_set_path"][0]
    )
    utils.pickle_dump(
            train_set_bal_cleaned[config["label"]],
            config["train_feng_set_path"][1]
    )


    utils.pickle_dump(
            valid_set[config["predictors"]],
            config["valid_feng_set_path"][0]
    )
    utils.pickle_dump(
            valid_set[config["label"]],
            config["valid_feng_set_path"][1]
    )


    utils.pickle_dump(
            test_set[config["predictors"]],
            config["test_feng_set_path"][0]
    )
    utils.pickle_dump(
            test_set[config["label"]],
            config["test_feng_set_path"][1]
    )

    