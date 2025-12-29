import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("eye_state_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="EEG Eye State Detection",
    layout="wide"
)

# Title
st.title("🧠 EEG Eye State Detection App")
st.write("Predict whether the eyes are **OPEN** or **CLOSED** using EEG signals.")

# EEG Feature Names (REAL)
FEATURE_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

st.sidebar.header("🎛 EEG Signal Inputs")

inputs = []

for feature in FEATURE_NAMES:
    value = st.sidebar.number_input(
        label=f"{feature}",
        min_value=-10000.0,     # allow wide EEG range
        max_value=10000.0,
        value=0.0,
        step=1.0,               # 👈 FIX: no more 0.0001 lock
        format="%.2f"
    )
    inputs.append(value)

# Predict button
if st.sidebar.button("🔍 Predict Eye State"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    confidence = np.max(model.predict_proba(input_scaled)) * 100

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("👀 Eyes are **OPEN**")
    else:
        st.error("😴 Eyes are **CLOSED**")

    st.write(f"**Confidence:** {confidence:.2f}%")
