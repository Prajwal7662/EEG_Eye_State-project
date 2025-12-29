# EEG_Eye_State-project
# 🧠 EEG Eye State Detection App  
### Random Forest | Streamlit | Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

An end-to-end **Machine Learning web application** that predicts whether a person’s **eyes are OPEN or CLOSED** using **EEG signal features**, deployed using **Streamlit**.

Streamlit App Link = https://egjpbeie7cnz7v2ctyijtb.streamlit.app/
---

## ✨ Highlights

✅ End-to-end ML pipeline  
✅ Random Forest classifier  
✅ Real-time & batch predictions  
✅ Clean Streamlit UI  
✅ Joblib model serialization  
✅ Interview & resume ready  

---

## 📌 Project Description

Electroencephalography (EEG) signals reflect brain activity and are widely used in **Brain–Computer Interface (BCI)** systems.  
This project leverages **EEG signal features** to classify eye states using a **Random Forest Classifier**, known for handling non-linear and noisy data effectively.

The trained model and scaler are saved as `.pkl` files and deployed via a **Streamlit web application** for real-time inference.

---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| ML Model | Random Forest (Scikit-learn) |
| Web App | Streamlit |
| Data Handling | Pandas, NumPy |
| Model Saving | Joblib |

---

## 📂 Project Structure

EEG_Eye_State_Streamlit_App/

│

├── app.py # Streamlit application

├── eye_state_model.pkl # Trained Random Forest model

├── scaler.pkl # Feature scaler

├── requirements.txt # Dependencies

└── README.md # Documentation

---

## 📊 Dataset Details

- **Dataset:** EEG Eye State Classification  
- **Input:** EEG signal features  
- **Target Column:** `eyeDetection`  
  - `0` → Eyes Closed 😴  
  - `1` → Eyes Open 👀  

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone <your-repository-link>
cd EEG_Eye_State_Streamlit_App
2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py


🌐 The app will open automatically in your browser.
🧪 How the App Works

User enters EEG feature values or uploads a CSV file

Features are scaled using the saved scaler

Random Forest model predicts eye state

Result + confidence score are displayed

📈 Machine Learning Model

Algorithm: Random Forest Classifier

Why Random Forest?

Handles non-linear EEG patterns

Robust to noise

Reduces overfitting via ensemble learning

🎯 Sample Output

👀 Eyes Open

😴 Eyes Closed

📊 Confidence Score (%)

🌍 Deployment Options

Streamlit Cloud

Hugging Face Spaces

Render

Railway

🚀 Use Cases

Brain–Computer Interfaces (BCI)

Cognitive & attention monitoring

Neuro-signal analysis

Research & education


