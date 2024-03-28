import streamlit as st
import requests
from PIL import Image
import util as util
import time

config_data = util.load_config()

job_levels_mapping = {
    1: "Level I - Entry",
    2: "Level II - Intermediate",
    3: "Level III - Advanced",
    4: "Level IV - Senior",
    5: "Level V - Leadership",
}

st.title("Employer Turnover Prediction")
st.info("""
        Welcome to ETP Module! \n
        Please enter the employee information on the below section to make prediction!
        """)

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
        st.write("Employee information has been submitted")

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
            time.sleep(3)
            res = requests.post("http://0.0.0.0:8080/predict", json = raw_data).json()
        
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            st.write(res["res"])