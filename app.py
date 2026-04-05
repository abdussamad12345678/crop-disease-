# ================================
# IMPORTS
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# ================================
# CONFIG
# ================================
st.set_page_config(
    page_title="PragyanAI Pro",
    layout="wide",
    page_icon="🌾"
)

MODEL_FILE = "model.pkl"
DATA_FILE = "data.csv"
API_KEY = "YOUR_API_KEY"  # Replace with actual key

# ================================
# STYLING
# ================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.title("🌾 PragyanAI - Smart Crop Intelligence System")
st.caption("AI-powered Disease Prediction | Weather Analytics | Smart Advisory")

# ================================
# MODEL FUNCTIONS
# ================================
@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_FILE)
    X = data[["temperature", "humidity", "rainfall"]]
    y = data["disease"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_model()

    try:
        return pickle.load(open(MODEL_FILE, "rb"))
    except:
        return train_model()


model = load_model()

# ================================
# WEATHER FUNCTION
# ================================
def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()

        if "main" not in data:
            raise ValueError("Invalid API response")

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0)

        return temp, humidity, rainfall, "Live Data"

    except Exception:
        # Fallback demo data
        np.random.seed(len(city))
        return (
            np.random.randint(20, 40),
            np.random.randint(50, 90),
            np.random.randint(0, 20),
            "Demo Data"
        )

# ================================
# SIDEBAR
# ================================
st.sidebar.header("⚙️ Control Panel")

city = st.sidebar.text_input("📍 Location", "Bangalore")
crop = st.sidebar.selectbox("🌾 Crop Type", ["Rice", "Wheat", "Corn"])
stage = st.sidebar.selectbox("🌱 Growth Stage",
                             ["Seedling", "Vegetative", "Flowering", "Harvest"])

advanced = st.sidebar.checkbox("⚡ Advanced Mode")

if advanced:
    custom_temp = st.sidebar.slider("Temperature (°C)", 10, 50, 30)
    custom_humidity = st.sidebar.slider("Humidity (%)", 10, 100, 70)
    custom_rain = st.sidebar.slider("Rainfall (mm)", 0, 50, 10)

# ================================
# TABS
# ================================
tab1, tab2, tab3 = st.tabs([
    "📊 Prediction",
    "📸 Image Analysis",
    "📈 Analytics"
])

# ================================
# TAB 1 - PREDICTION
# ================================
with tab1:

    st.subheader("🌦 Weather-Based Disease Prediction")

    if st.button("🚀 Run Analysis", use_container_width=True):

        with st.spinner("Analyzing conditions..."):

            temp, humidity, rainfall, source = get_weather(city)

            if advanced:
                temp, humidity, rainfall = custom_temp, custom_humidity, custom_rain

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡 Temperature", f"{temp} °C")
            col2.metric("💧 Humidity", f"{humidity}%")
            col3.metric("🌧 Rainfall", f"{rainfall} mm")

            st.caption(f"📡 Data Source: {source}")

            # Disease Favorability Index
            dfi = (humidity * 0.5) + (rainfall * 0.3) + (temp * 0.2)
            st.subheader("🧠 Disease Favorability Index")
            st.progress(min(int(dfi), 100))

            # Model Prediction
            prob = model.predict_proba([[temp, humidity, rainfall]])[0][1]

            st.subheader("⚠️ Disease Risk Score")
            st.progress(int(prob * 100))

            # Risk Levels
            if prob < 0.3:
                st.success("🟢 Low Risk - Conditions are safe")
            elif prob < 0.7:
                st.warning("🟡 Moderate Risk - Monitor closely")
            else:
                st.error("🔴 High Risk - Immediate action required")
                st.info("💊 Recommended: Apply preventive fungicide within 48 hours")

            # Explainability
            st.subheader("📌 Explanation")
            st.write(
                f"High humidity ({humidity}%) and rainfall ({rainfall} mm) "
                f"create favorable conditions for disease development."
            )

            # Scenario Simulation
            st.subheader("🔮 What-if Simulation")
            future_prob = model.predict_proba([[temp, humidity, rainfall + 10]])[0][1]
            st.write(f"If rainfall increases by 10 mm → Risk becomes: **{round(future_prob, 2)}**")

# ================================
# TAB 2 - IMAGE ANALYSIS
# ================================
with tab2:

    st.subheader("📸 Leaf Health Detection")

    file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", width=300)

        avg_pixel = np.array(img).mean()

        if avg_pixel < 100:
            st.error("⚠️ Disease Detected")
        else:
            st.success("✅ Healthy Leaf")

# ================================
# TAB 3 - ANALYTICS
# ================================
with tab3:

    st.subheader("📈 Historical Data Insights")

    try:
        data = pd.read_csv(DATA_FILE)

        st.write("### 🌦 Weather Trends")
        st.line_chart(data[["temperature", "humidity", "rainfall"]])

        st.write("### 🌾 Disease Distribution")
        st.bar_chart(data["disease"].value_counts())

    except Exception:
        st.error("Dataset not found or invalid.")

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("🚀 Built for Smart Farming | PragyanAI Pro")
