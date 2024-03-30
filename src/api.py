from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import util as util
import data_pipeline as data_pipeline
import preprocessing as preprocessing 

import shap

config_data = util.load_config()

ohe_Department = config_data["ohe_Department_path"]
ohe_JobRole = config_data["ohe_JobRole_path"]
ohe_OverTime = config_data["ohe_OverTime_path"]

le_encoder = util.pickle_load(config_data["le_encoder_path"])
model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    JobLevel : int
    Age : int
    DistanceFromHome : int
    YearsAtCompany : int
    YearsInCurrentRole : int
    YearsWithCurrManager : int
    MonthlyIncome : int
    EnvironmentSatisfaction : int
    JobSatisfaction : int
    WorkLifeBalance : int
    Department : str
    JobRole : str
    OverTime : str

app = FastAPI()

@app.get("/")
def home():
    return "Employer-Turnover-Predictor API is up!"

@app.post("/predict/")
def predict(data: api_data):
    print(data)
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)
    print(data)
    data_columns = data.columns.to_list()

    # data type
    data = pd.concat(
        [
            data[list(set(config_data["predictor_columns"]) & set(config_data["object_columns"]))].astype(object),
            data[list(set(config_data["predictor_columns"]) & set(config_data["int64_columns"]))].astype(int)
        ],
        axis = 1
    )
    data = data[data_columns]

    # check data
    try:
        data_pipeline.check_data(data, config_data, api=True)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # encoding
    data = preprocessing.ohe_transform(data, "Department", ohe_Department)
    data = preprocessing.ohe_transform(data, "JobRole", ohe_JobRole)
    data = preprocessing.ohe_transform(data, "OverTime", ohe_OverTime)

    # predict
    prediction = model_data["model_data"]["model_object"].predict_proba(data)[:, 1][0]
    # result = f'The probability of employee turnover is: {prediction:.0%}'

    explainer = shap.Explainer(model_data["model_data"]["model_object"])
    shap_base_values = explainer(data).base_values
    shap_values = explainer(data).values
    shap_features = explainer(data).data
    shap_feature_name = data.columns

    print()

    return {"res" : prediction, 
            "shap_values" : shap_values.tolist(), 
            "shap_base_values" : shap_base_values.tolist(), 
            "shap_feature" : shap_features.tolist(), 
            "shap_feature_name" : shap_feature_name.tolist(), 
            "error_msg" : ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080)