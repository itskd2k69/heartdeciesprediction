import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Stroke Predictor", page_icon="❤️", layout="centered")

# Load model and scalers
try:
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
except Exception as e:
    st.error("🚨 Failed to load model or files. Please ensure all `.pkl` files are in the correct directory.")
    st.stop()

# Title & Intro
st.markdown("<h1 style='text-align: center; color: crimson;'>❤️ Heart Stroke Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill the form below to check your risk level for heart disease.</p>", unsafe_allow_html=True)
st.markdown("---")

# Inputs in columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🧓 Age", 18, 100, 40)
    sex = st.selectbox("⚧️ Sex", ['M', 'F'])
    chest_pain = st.selectbox("💔 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("🍔 Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("🧪 Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("📉 Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("🏃‍♂️ Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("💥 Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("📉 Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("📈 ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")

# Predict Button
if st.button("🔍 Predict Risk", use_container_width=True):
    # Build input
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create DataFrame
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1] * 100

    # Result
    st.markdown("---")
    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease Detected!")
    else:
        st.success("✅ Low Risk of Heart Disease.")

    st.progress(int(prob))
    st.markdown(f"**🧠 Probability of Heart Disease:** `{prob:.2f}%`")

    st.markdown("---")
    st.caption("🔒 Your data is not stored or shared. This tool is for educational use only.")
