import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("eye_state_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(page_title="EEG Eye State Detection", layout="centered")

# App title
st.title("🧠 EEG Eye State Detection App")
st.write("This app predicts whether the **eyes are OPEN or CLOSED** using EEG signals.")

# Sidebar for input
st.sidebar.header("Enter EEG Feature Values")

# Number of EEG features (dataset dependent)
NUM_FEATURES = 14   # update if required

input_features = []
for i in range(NUM_FEATURES):
    value = st.sidebar.number_input(
        f"EEG Feature {i+1}",
        value=0.0,
        format="%.4f"
    )
    input_features.append(value)

# Prediction
if st.sidebar.button("Predict Eye State"):
    input_array = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("👀 Eyes are **OPEN**")
    else:
        st.error("😴 Eyes are **CLOSED**")

    st.write(f"Confidence: **{np.max(probability) * 100:.2f}%**")
