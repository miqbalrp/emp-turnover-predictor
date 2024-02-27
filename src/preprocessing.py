import pandas as pd
import numpy as np
import seaborn as sns
import util as util
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"]['x'])
    y_train = util.pickle_load(config_data["train_set_path"]['y'])

    x_valid = util.pickle_load(config_data["valid_set_path"]['x'])
    y_valid = util.pickle_load(config_data["valid_set_path"]['y'])

    x_test = util.pickle_load(config_data["test_set_path"]['x'])
    y_test = util.pickle_load(config_data["test_set_path"]['y'])

    # Concatenate x and y each set
    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    # Return 3 set of data
    return train_set, valid_set, test_set

def ohe_fit(categorical_attribute, ohe_model_path):
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(np.array(categorical_attribute).reshape(-1, 1))
    util.pickle_dump(ohe, ohe_model_path)
    return ohe

def ohe_transform(set_data: pd.DataFrame, transformed_column: str, ohe_path: str) -> pd.DataFrame:
    set_data = set_data.copy()
    ohe = util.pickle_load(ohe_path)

    features = ohe.transform(np.array(set_data[transformed_column].to_list()).reshape(-1, 1))
    features = pd.DataFrame(features.tolist(), columns=ohe.get_feature_names_out([transformed_column]))
    
    features.set_index(set_data.index, inplace=True)
    set_data = pd.concat([features, set_data], axis=1)
    set_data.drop(columns=transformed_column, inplace=True)

    new_col = [str(col_name) for col_name in set_data.columns.to_list()]
    set_data.columns = new_col

    return set_data

def rus_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    set_data = set_data.copy()
    rus = RandomUnderSampler(random_state = 42)
    x_rus, y_rus = rus.fit_resample(set_data.drop("Attrition", axis = 1), set_data['Attrition'])
    set_data_rus = pd.concat([x_rus, y_rus], axis = 1)
    return set_data_rus

def ros_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    set_data = set_data.copy()
    ros = RandomOverSampler(random_state = 11)
    x_ros, y_ros = ros.fit_resample(set_data.drop("Attrition", axis = 1), set_data['Attrition'])
    set_data_ros = pd.concat([x_ros, y_ros], axis = 1)
    return set_data_ros

def sm_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    set_data = set_data.copy()
    sm = SMOTE(random_state = 112)
    x_sm, y_sm = sm.fit_resample(set_data.drop("Attrition", axis = 1), set_data['Attrition'])
    set_data_sm = pd.concat([x_sm, y_sm], axis = 1)
    return set_data_sm

def le_fit(data_tobe_fitted: dict, le_path: str) -> LabelEncoder:
    le_encoder = LabelEncoder()
    le_encoder.fit(data_tobe_fitted)
    util.pickle_dump(le_encoder, le_path)
    return le_encoder

def le_transform(label_data: pd.Series, config_data: dict) -> pd.Series:
    label_data = label_data.copy()
    le_encoder = util.pickle_load(config_data["le_encoder_path"])

    # If categories both label data and trained le matched
    if len(set(label_data.unique()) - set(le_encoder.classes_) | set(le_encoder.classes_) - set(label_data.unique())) == 0:
        label_data = le_encoder.transform(label_data)
    else:
        raise RuntimeError("Check category in label data and label encoder.")

    return label_data

if __name__ == "__main__":
    config_data = util.load_config()
    train_set, valid_set, test_set = load_dataset(config_data)

    impute_values = config_data['missing_value_handling']
    train_set.fillna(value=impute_values, inplace=True)
    valid_set.fillna(value=impute_values, inplace=True)
    test_set.fillna(value=impute_values, inplace=True)

    ohe_Department = ohe_fit(config_data['Department_range'], config_data['ohe_Department_path'])
    ohe_JobRole = ohe_fit(config_data['JobRole_range'], config_data['ohe_JobRole_path'])
    ohe_Gender = ohe_fit(config_data['Gender_range'], config_data['ohe_Gender_path'])
    ohe_OverTime = ohe_fit(config_data['OverTime_range'], config_data['ohe_OverTime_path'])

    for col in ['Department', 'JobRole', 'Gender', 'OverTime']:
        train_set = ohe_transform(train_set, col, config_data[f"ohe_{col}_path"])
        valid_set = ohe_transform(valid_set, col, config_data[f"ohe_{col}_path"])
        test_set = ohe_transform(test_set, col, config_data[f"ohe_{col}_path"])

    train_set_rus = rus_fit_resample(train_set)
    train_set_ros = ros_fit_resample(train_set)
    train_set_sm = sm_fit_resample(train_set)

    le_encoder = le_fit(config_data["Attrition_range"], config_data["le_encoder_path"])

    train_set['Attrition'] = le_transform(train_set['Attrition'], config_data)

    train_set_rus['Attrition'] = le_transform(train_set_rus['Attrition'], config_data)
    train_set_ros['Attrition'] = le_transform(train_set_ros['Attrition'], config_data)
    train_set_sm['Attrition'] = le_transform(train_set_sm['Attrition'], config_data)
    valid_set['Attrition'] = le_transform(valid_set['Attrition'], config_data)
    test_set['Attrition'] = le_transform(test_set['Attrition'], config_data)

    train_set.Attrition.value_counts()

    x_train = {
        "No sampling" : train_set.drop(columns = "Attrition"),
        "Undersampling" : train_set_rus.drop(columns = "Attrition"),
        "Oversampling" : train_set_ros.drop(columns = "Attrition"),
        "SMOTE" : train_set_sm.drop(columns = "Attrition")
    }

    y_train = {
        "No sampling" : train_set.Attrition,
        "Undersampling" : train_set_rus.Attrition,
        "Oversampling" : train_set_ros.Attrition,
        "SMOTE" : train_set_sm.Attrition
    }

    util.pickle_dump(x_train, config_data['train_set_feng_path']['x'])
    util.pickle_dump(y_train, config_data['train_set_feng_path']['y'])

    util.pickle_dump(valid_set.drop(columns = "Attrition"), config_data['valid_set_feng_path']['x'])
    util.pickle_dump(valid_set.Attrition, config_data['valid_set_feng_path']['y'])

    util.pickle_dump(test_set.drop(columns = "Attrition"), config_data['test_set_feng_path']['x'])
    util.pickle_dump(test_set.Attrition, config_data['test_set_feng_path']['y'])