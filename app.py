# ============================================================
# STREAMLIT EYE OPEN / EYE CLOSED DETECTION APP
# ============================================================

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Eye Open / Closed Detection",
    layout="centered"
)

st.title("👁️ Eye Open / Closed Detection")
st.markdown("Real-time CNN-based eye state detection using OpenCV")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_cnn_model():
    return load_model("eye_open_close_cnn_model.h5")

model = load_cnn_model()

IMG_SIZE = 64

# -------------------- LOAD HAAR CASCADE --------------------
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# -------------------- START / STOP BUTTON --------------------
run = st.checkbox("▶ Start Camera")

FRAME_WINDOW = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in eyes:
            eye = gray[y:y+h, x:x+w]
            eye = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
            eye = eye / 255.0
            eye = eye.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            prediction = model.predict(eye, verbose=0)[0][0]

            if prediction > 0.5:
                label = "EYES OPEN 👀"
                color = (0, 255, 0)
            else:
                label = "EYES CLOSED 👁️"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

else:
    st.info("Click 'Start Camera' to begin eye detection")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("🧠 **CNN + OpenCV | Real-Time Eye State Detection**")
