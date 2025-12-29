# ===============================
# Imports
# ===============================
import streamlit as st
import numpy as np
import joblib
import os

# ===============================
# Page Config (must be first Streamlit call)
# ===============================
st.set_page_config(
    page_title="EEG Eye State Detection",
    page_icon="🧠",
    layout="wide"
)

# ===============================
# Safe Model & Scaler Loading
# ===============================
if not os.path.exists("eye_state_model.pkl"):
    st.error("❌ eye_state_model.pkl not found")
    st.stop()

if not os.path.exists("scaler.pkl"):
    st.error("❌ scaler.pkl not found")
    st.stop()

try:
    model = joblib.load("eye_state_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model/scaler: {e}")
    st.stop()

# ===============================
# EEG Feature Names
# ===============================
FEATURE_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

# ===============================
# Header
# ===============================
st.markdown(
    "<h1 style='text-align:center;'>🧠 EEG Eye State Detection App</h1>"
    "<p style='text-align:center; font-size:18px;'>"
    "Predict whether the eyes are <b>OPEN</b> or <b>CLOSED</b> using EEG signals"
    "</p>",
    unsafe_allow_html=True
)

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("🎛 EEG Signal Inputs (-5000 to 5000)")
st.sidebar.caption("Values represent raw EEG signal amplitudes")

inputs = []
for feature in FEATURE_NAMES:
    value = st.sidebar.number_input(
        label=feature,
        min_value=-5000.0,
        max_value=5000.0,
        value=0.0,
        step=50.0,
        format="%.1f"
    )
    inputs.append(value)

# ===============================
# Prediction Section
# ===============================
if st.sidebar.button("🔍 Predict Eye State"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    probs = model.predict_proba(input_scaled)[0]
    prediction = np.argmax(probs)
    confidence = probs[prediction] * 100

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("👀 Eyes are **OPEN**")
        else:
            st.error("😴 Eyes are **CLOSED**")

    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    # Show both probabilities for clarity
    st.write(
        f"😴 Closed: **{probs[0]*100:.2f}%** | "
        f"👀 Open: **{probs[1]*100:.2f}%**"
    )

    if abs(probs[0] - probs[1]) < 0.10:
        st.warning("⚠ Low confidence prediction")

# ===============================
# Footer
# ===============================
st.markdown(
    "<hr>"
    "<p style='text-align:center;'>"
    "Built with ❤️ using <b>Random Forest & Streamlit</b>"
    "</p>",
    unsafe_allow_html=True
)
