import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# Load Model & Scaler
# ===============================
model = joblib.load("eye_state_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="EEG Eye State Detection",
    page_icon="🧠",
    layout="wide"
)

# ===============================
# EEG Feature Names (REAL)
# ===============================
FEATURE_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

# ===============================
# Header
# ===============================
st.markdown(
    """
    <h1 style='text-align:center;'>🧠 EEG Eye State Detection App</h1>
    <p style='text-align:cen
