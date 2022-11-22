import util as utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def load_train_feng(params: dict):
    # Load train set
    x_train = utils.pickle_load(params["train_feng_set_path"][0])
    y_train = utils.pickle_load(params["train_feng_set_path"][1])

    return x_train, y_train

def load_valid(params: dict):
    # Load valid set
    x_valid = utils.pickle_load(params["valid_feng_set_path"][0])
    y_valid = utils.pickle_load(params["valid_feng_set_path"][1])

    return x_valid, y_valid

def load_test(params: dict):
    # Load tets set
    x_test = utils.pickle_load(params["test_feng_set_path"][0])
    y_test = utils.pickle_load(params["test_feng_set_path"][1])

    return x_test, y_test

def train_model(x_train, y_train, x_valid, y_valid):
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)

    y_pred = dtc.predict(x_valid)
    print(classification_report(y_valid, y_pred))

    return dtc

if __name__ == "__main__" :
    # 1. Load config file
    config = utils.load_config()

    # 2. Load set data
    x_train, y_train = load_train_feng(config)
    x_valid, y_valid = load_valid(config)
    x_test, y_test = load_test(config)

    # 3. Train model
    dtc = train_model(x_train, y_train, x_valid, y_valid)

    # 4. Dump model
    utils.pickle_dump(dtc, config["production_model_path"])