import pandas as pd
import numpy as np
import yaml
import os
import tqdm
import util

from sklearn.model_selection import train_test_split

def read_raw_data(config: dict) -> pd.DataFrame:
    raw_data_path = config['raw_dataset_path']
    selected_columns = config['int64_columns'] + config['object_columns']
    df = pd.read_csv(raw_data_path)[selected_columns]
    return df

def check_data(input_data, params, api=False):
    if api == False:
        """
        If the data input not coming from API, then we check all required columns.
        """
        # Data type checking
        assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "Error occurs in object columns"
        assert input_data.select_dtypes("int64").columns.to_list() == params["int64_columns"], "Error occurs in object columns"

        # Range data checking for obj columns
        for col in config_data["object_columns"]:
            assert set(input_data[f"{col}"]).issubset(set(params[f"{col}_range"])), f"Error occurs in {col} column range"

        # Range data checking for int64 columns
        for col in config_data["int64_columns"]:
            assert input_data[f"{col}"].between(params[f"{col}_range"]["min"], params[f"{col}_range"]["max"]).sum() == len(input_data), f"Error occurs in {col} column range"
    else:
        """
        If the data input coming from API, then we only check columns that selected as predictor
        """
        expected_object_columns = set(list((set(params["object_columns"]) & set(params["predictor_columns"]))))
        input_object_columns = set(input_data.select_dtypes("object").columns.to_list())

        expected_int64_columns = set(list((set(params["int64_columns"]) & set(params["predictor_columns"]))))
        input_int64_columns = set(input_data.select_dtypes("int64").columns.to_list())

        assert input_object_columns == expected_object_columns, f"Error occurs in object columns, {expected_object_columns - input_object_columns}"
        assert input_int64_columns == expected_int64_columns, f"Error occurs in int64 columns, {expected_int64_columns - input_int64_columns}"

        # Range data checking for obj columns
        for col in (set(params["object_columns"]) & set(params["predictor_columns"])):
            assert set(input_data[f"{col}"]).issubset(set(params[f"{col}_range"])), f"Error occurs in {col} column range"

        # Range data checking for int64 columns
        for col in (set(params["int64_columns"]) & set(params["predictor_columns"])):
            assert input_data[f"{col}"].between(params[f"{col}_range"]["min"], params[f"{col}_range"]["max"]).sum() == len(input_data), f"Error occurs in {col} column range"

if __name__ == "__main__":
    config_data = util.load_config()

    raw_dataset = read_raw_data(config_data)
    util.pickle_dump(raw_dataset, config_data["raw_dataset_collected_path"])

    check_data(raw_dataset, config_data, api=False)

    x = raw_dataset[config_data["predictor_columns"]].copy() # note that this is different with notebook where x_train still contain non-predictor features
    y = raw_dataset[config_data["target_column"]].copy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42, stratify = y)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 42, stratify = y_test)

    util.pickle_dump(x_train, config_data["train_set_path"]["x"])
    util.pickle_dump(y_train, config_data["train_set_path"]["y"])

    util.pickle_dump(x_valid, config_data["valid_set_path"]["x"])
    util.pickle_dump(y_valid, config_data["valid_set_path"]["y"])

    util.pickle_dump(x_test, config_data["test_set_path"]["x"])
    util.pickle_dump(y_test, config_data["test_set_path"]["y"])

