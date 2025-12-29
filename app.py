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
    "<h1 style='text-align:center;'>🧠 EEG Eye State Detection App</h1>"
    "<p style='text-align:center; font-size:18px;'>"
    "Predict whether the eyes are <b>OPEN</b> or <b>CLOSED</b> using EEG signals"
    "</p>",
    unsafe_allow_html=True
)


# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("🎛 EEG Signal Inputs")

inputs = []
for feature in FEATURE_NAMES:
    value = st.sidebar.number_input(
        feature,
        min_value=-10000.0,
        max_value=10000.0,
        value=0.0,
        step=1.0,
        format="%.2f"
    )
    inputs.append(value)

# ===============================
# Prediction Section
# ===============================
if st.sidebar.button("🔍 Predict Eye State"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    confidence = np.max(model.predict_proba(input_scaled)) * 100

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("👀 Eyes are **OPEN**")
        else:
            st.error("😴 Eyes are **CLOSED**")

    with col2:
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

# ===============================
# Feature Importance Plot
# ===============================
st.subheader("📈 Feature Importance (Random Forest)")

importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.invert_yaxis()
ax.set_xlabel("Importance Score")
ax.set_title("EEG Feature Importance")

st.pyplot(fig)

# ===============================
# CSV Upload Section
# ===============================
st.subheader("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload EEG CSV file (must contain all EEG feature columns)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Validate columns
    missing_cols = set(FEATURE_NAMES) - set(data.columns)
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
    else:
        scaled_data = scaler.transform(data[FEATURE_NAMES])
        predictions = model.predict(scaled_data)

        data["Eye_State_Prediction"] = predictions.map(
            {0: "Closed", 1: "Open"}
        )

        st.success("✅ Prediction Completed")
        st.dataframe(data)

        st.download_button(
            "⬇ Download Results",
            data.to_csv(index=False),
            file_name="EEG_Eye_State_Predictions.csv",
            mime="text/csv"
        )

# ===============================
# Footer
# ===============================
st.markdown(
    """
    <hr>
    <p style='text-align:center;'>
    Built with ❤️ using <b>Random Forest & Streamlit</b>
    </p>
    """,
    unsafe_allow_html=True
)
