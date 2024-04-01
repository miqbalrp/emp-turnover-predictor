import streamlit as st
import requests
from PIL import Image
import util as util
import time

import shap as shap
import streamlit_shap as st_shap

import numpy as np
import pandas as pd

config_data = util.load_config()

job_levels_mapping = {
    1: "Level I - Entry",
    2: "Level II - Intermediate",
    3: "Level III - Advanced",
    4: "Level IV - Senior",
    5: "Level V - Leadership",
}

st.title("Employer Turnover Prediction")
st.write("""
        Welcome to ETP Module! \n
        Please enter the employee information on the below section to make prediction!
        """)

with st.sidebar:
    with st.form(key = "emp_features"):
        
        st.subheader('1. Employee basic information')
        col1, col2 = st.columns(2)

        with col1:
            employee_name = st.text_input(
                label='Employee name', 
                placeholder='e.g., Dwight Schrute',
                key="employee_name"
            )
            department = st.selectbox(
                label="Department",
                options=config_data['Department_range']
            )
            selected_job_level = st.selectbox(
                "Job level", 
                list(job_levels_mapping.keys()), 
                format_func=lambda x: job_levels_mapping[x]
            )
        
        with col2:    
            age = st.number_input(
                label="Age",
                min_value=config_data['Age_range']['min'],
                max_value=config_data['Age_range']['max']
            )
            job_role = st.selectbox(
                label="Job role",
                options=config_data['JobRole_range']
            )

        st.subheader('2. Employee work details')
        col1, col2 = st.columns(2)

        with col1:
            years_at_company = st.number_input(
                label="Years at company",
                min_value=config_data['YearsAtCompany_range']['min'],
                max_value=config_data['YearsAtCompany_range']['max']
            )
            years_with_current_manager = st.number_input(
                label="Years with current manager",
                min_value=config_data['YearsWithCurrManager_range']['min'],
                max_value=config_data['YearsWithCurrManager_range']['max']
            )
            monthly_income = st.number_input(
                label="Current monthly income (USD)",
                min_value=config_data['MonthlyIncome_range']['min'],
                max_value=config_data['MonthlyIncome_range']['max']
            )

        with col2:
            years_in_current_role = st.number_input(
                label="Years in current role",
                min_value=config_data['YearsInCurrentRole_range']['min'],
                max_value=config_data['YearsInCurrentRole_range']['max']
            )
            overtime = st.selectbox(
                label="Over time",
                options=config_data['OverTime_range']
            )
            distance_from_home = st.number_input(
                label="Distance from home",
                min_value=config_data['DistanceFromHome_range']['min'],
                max_value=config_data['DistanceFromHome_range']['max']
            )

        st.subheader('3. Job satisfaction')
        job_satisfaction = st.slider(
            label='Job satisfaction',
            min_value=config_data['JobSatisfaction_range']['min'],
            max_value=config_data['JobSatisfaction_range']['max'] 
        )
        environment_satisfaction = st.slider(
            label='Environment satisfaction',
            min_value=config_data['EnvironmentSatisfaction_range']['min'],
            max_value=config_data['EnvironmentSatisfaction_range']['max'] 
        )
        work_life_balance = st.slider(
            label='Work life balance',
            min_value=config_data['WorkLifeBalance_range']['min'],
            max_value=config_data['WorkLifeBalance_range']['max'] 
        )
        

        submitted = st.form_submit_button("Predict!")

if submitted:
    st.sidebar.write("Employee information has been submitted")

    raw_data = {
        "JobLevel": selected_job_level,
        "Age": age,
        "DistanceFromHome": distance_from_home,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsWithCurrManager": years_with_current_manager,
        "MonthlyIncome": monthly_income,
        "EnvironmentSatisfaction": environment_satisfaction,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "Department": department,
        "JobRole": job_role, 
        "OverTime": overtime
    }

    with st.spinner("Sending data to prediction server ..."):
        time.sleep(1)
        res = requests.post("http://0.0.0.0:8080/predict", json = raw_data).json()
    
    st.header('Result', divider='rainbow')
    if res["error_msg"] != "":
        st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
    else:
        st.write(f'The likelihood that {employee_name} will turnover is:')
        st.metric('Probability', f'{res["res"]:.1%}', label_visibility='collapsed')

        base_value = np.array(res["shap_base_values"])[0][1]
        shap_values = np.array(res["shap_values"])[:,:,1]
        feature = np.array(res["shap_feature"])
        feature_names = res["shap_feature_name"]

        p = shap.force_plot(base_value, shap_values, feature, feature_names)

        def st_shap(plot, height=None):
            import streamlit.components.v1 as components
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)
        
        st.header('Factor Analyses', divider='rainbow')
        st_shap(p, height=None)

        factors = pd.DataFrame({'feature_name' : feature_names,
                                'feature_value' : feature[0],
                                'shap_value' : shap_values[0]
                                }, index=feature_names)
        positive_factors = factors[factors['shap_value'] > 0].sort_values(by='shap_value', ascending=False).head(3)
        negative_factors = factors[factors['shap_value'] <= 0].sort_values(by='shap_value', ascending=True).head(3)

        st.subheader("Factors that most contribute to the employee being likely to turnover:")
        for i in range(len(positive_factors)): 
            st.write(f"{i+1}. {positive_factors['feature_name'][i]} = {positive_factors['feature_value'][i]} ({positive_factors['shap_value'][i]:.0%})")

        st.subheader("Factors that most significantly reduces the likelihood of employee turnover")
        for i in range(len(negative_factors)): 
            st.write(f"{i+1}. {negative_factors['feature_name'][i]} = {negative_factors['feature_value'][i]} ({negative_factors['shap_value'][i]:.0%})")



        
        