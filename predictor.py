import joblib
import pandas as pd
import sklearn
import streamlit as st

sklearn.set_config(transform_output="pandas")

model = joblib.load("voting.pkl")
pipeline = joblib.load("pipeline.pkl")


# Set page title and description
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
st.title("Heart Disease Prediction")
st.write("Fill out the following information to predict your heart disease status.")

# Sidebar title
st.sidebar.title("Input Data")

# Sidebar inputs
with st.sidebar:
    st.subheader("Personal Information")
    sex_options = {"Male": "M", "Female": "F"}
    sex = st.radio("Sex:", options=list(sex_options.keys()))

    age = st.number_input("Age", min_value=0, step=1, value=None)

    st.subheader("Health Metrics")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", step=0.1, value=None)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", step=1, value=None)

    max_hr = st.number_input("Maximum Heart Rate Achieved", step=1, value=None)
    oldpeak = st.number_input(
        "ST Depression induced by exercise relative to rest", step=0.1, value=None
    )

    st.subheader("Medical History")
    chest_pain_options = {
        "Asymptomatic": "ASY",
        "Non-Anginal Pain": "NAP",
        "Typical Angina": "TA",
        "Atypical Angina": "ATA",
    }
    chest_pain = st.selectbox(
        "Chest Pain Type:", options=list(chest_pain_options.keys())
    )
    fasting_bs_options = {"Yes": 1, "No": 0}
    fasting_bs = st.radio(
        "Fasting Blood Sugar > 120 mg/dl?", options=list(fasting_bs_options.keys())
    )
    resting_ecg_options = {
        "Normal": "Normal",
        "Having ST segment or T wave deviations": "ST",
        "ECG signs of left ventricular hypertrophy": "LVH",
    }
    resting_ecg = st.selectbox(
        "Resting ECG Type:", options=list(resting_ecg_options.keys())
    )
    angina_options = {"Yes": "Y", "No": "N"}
    angina = st.radio("Exercise-Induced Angina?", options=list(angina_options.keys()))
    st_slope_options = {"Downsloping": "Down", "Upsloping": "Up", "No": "Flat"}
    st_slope = st.selectbox(
        "ST Segment Deviations:", options=list(st_slope_options.keys())
    )

# Transform input data
data = pd.DataFrame(
    {
        "Age": [age],
        "Sex": [sex_options[sex]],
        "ChestPainType": [chest_pain_options[chest_pain]],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs_options[fasting_bs]],
        "RestingECG": [resting_ecg_options[resting_ecg]],
        "MaxHR": [max_hr],
        "ExerciseAngina": [angina_options[angina]],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope_options[st_slope]],
    }
)

# Predict button
button = st.button("Predict")

# Prediction logic
if button:
    try:
        data_transformed = pipeline.transform(data)
        prediction = model.predict(data_transformed)

        if prediction == 1:
            st.error("You have a heart disease 💔")
            st.image(
                "deadge.png",
            )
        else:
            st.success("You don't have a heart disease ❤️")
            st.image("healthy.png")
    except ValueError:
        st.write("Fill all fields on a side bar first.")
