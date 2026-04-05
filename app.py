import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
import requests
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="PragyanAI Pro", layout="wide")

# -------------------------------
# CSS (PRO UI)
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.block-container {
    padding-top: 1rem;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.title("🌾 PragyanAI - Smart Crop Intelligence System")
st.caption("AI-powered Disease Prediction | Weather Analytics | Smart Advisory")

# -------------------------------
# MODEL
# -------------------------------
MODEL_FILE = "model.pkl"

def train_model():
    data = pd.read_csv("data.csv")
    X = data[["temperature", "humidity", "rainfall"]]
    y = data["disease"]

    model = RandomForestClassifier()
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model

def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_model()
    return pickle.load(open(MODEL_FILE, "rb"))

model = load_model()

# -------------------------------
# WEATHER (REAL + FALLBACK)
# -------------------------------
API_KEY = "YOUR_API_KEY"

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()

        if "main" not in res:
            raise Exception("API failed")

        temp = res["main"]["temp"]
        humidity = res["main"]["humidity"]
        rainfall = res.get("rain", {}).get("1h", 0)

        return temp, humidity, rainfall, "Live Data"

    except:
        # fallback demo
        np.random.seed(len(city))
        return (
            np.random.randint(20, 40),
            np.random.randint(50, 95),
            np.random.randint(0, 20),
            "Demo Data"
        )

# -------------------------------
# SIDEBAR CONTROL PANEL
# -------------------------------
st.sidebar.title("⚙️ Control Panel")

city = st.sidebar.text_input("📍 Location", "Delhi")
crop = st.sidebar.selectbox("🌾 Crop Type", ["Rice", "Wheat", "Corn"])
stage = st.sidebar.selectbox("🌱 Growth Stage", ["Seedling", "Vegetative", "Flowering", "Harvest"])

advanced = st.sidebar.checkbox("⚡ Advanced Mode")

if advanced:
    custom_temp = st.sidebar.slider("Temperature Override", 10, 50, 30)
    custom_humidity = st.sidebar.slider("Humidity Override", 10, 100, 70)
    custom_rain = st.sidebar.slider("Rainfall Override", 0, 50, 10)

# -------------------------------
# TABS (PRO FEATURE)
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📸 Image Analysis", "📈 Analytics"])

# ===============================
# TAB 1 - PREDICTION
# ===============================
with tab1:

    if st.button("🚀 Run Analysis"):

        with st.spinner("Analyzing data..."):

            temp, humidity, rainfall, source = get_weather(city)

            if advanced:
                temp, humidity, rainfall = custom_temp, custom_humidity, custom_rain

            col1, col2, col3 = st.columns(3)

            col1.metric("🌡 Temp", f"{temp} °C")
            col2.metric("💧 Humidity", f"{humidity}%")
            col3.metric("🌧 Rainfall", f"{rainfall} mm")

            st.caption(f"Data Source: {source}")

            # DFI
            dfi = (humidity * 0.5) + (rainfall * 0.3) + (temp * 0.2)

            st.subheader("🧠 Disease Favorability Index")
            st.progress(int(min(dfi, 100)))

            # Prediction
            prob = model.predict_proba([[temp, humidity, rainfall]])[0][1]

            st.subheader("⚠️ Risk Score")
            st.progress(int(prob * 100))

            # Risk
            if prob < 0.3:
                st.success("🟢 Low Risk")
            elif prob < 0.7:
                st.warning("🟡 Medium Risk")
            else:
                st.error("🔴 High Risk")
                st.info("💊 Spray recommended within 2–3 days")

            # Explainability
            st.subheader("📌 Why this risk?")
            st.write(f"High humidity ({humidity}%) and rainfall ({rainfall}mm) increase fungal growth chances.")

            # What-if
            st.subheader("🔮 Scenario Simulation")
            new_prob = model.predict_proba([[temp, humidity, rainfall + 10]])[0][1]
            st.write(f"If rainfall increases → Risk: {round(new_prob,2)}")

# ===============================
# TAB 2 - IMAGE
# ===============================
with tab2:

    file = st.file_uploader("Upload Leaf Image")

    if file:
        img = Image.open(file)
        st.image(img, width=300)

        avg = np.array(img).mean()

        if avg < 100:
            st.error("Disease Detected")
        else:
            st.success("Healthy Leaf")

# ===============================
# TAB 3 - ANALYTICS
# ===============================
with tab3:

    data = pd.read_csv("data.csv")

    st.subheader("📊 Weather Trends")
    st.line_chart(data[["temperature", "humidity", "rainfall"]])

    st.subheader("🌾 Disease Distribution")
    st.bar_chart(data["disease"].value_counts())

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("🚀 Built with AI for Smart Farming | PragyanAI")
