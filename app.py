import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import folium
from streamlit_folium import st_folium
import requests

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="PragyanAI Pro", layout="wide")

# -------------------------------
# CUSTOM UI
# -------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}
.metric-card {
    background: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("🌾 PragyanAI – Smart Crop Intelligence Platform")
st.caption("AI-based Crop Disease Prediction & Advisory System")

# -------------------------------
# MODEL
# -------------------------------
@st.cache_resource
def load_model():
    data = pd.read_csv("data.csv")
    X = data[["temperature", "humidity", "rainfall"]]
    y = data["disease"]

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = load_model()

# -------------------------------
# WEATHER
# -------------------------------
API_KEY = "YOUR_API_KEY"

def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()

        if "main" not in res:
            raise Exception()

        temp = res["main"]["temp"]
        humidity = res["main"]["humidity"]
        rainfall = res.get("rain", {}).get("1h", 0)

        return temp, humidity, rainfall, "LIVE"

    except:
        np.random.seed(len(city))
        return np.random.randint(20,35), np.random.randint(60,90), np.random.randint(0,15), "DEMO"

def get_coordinates(city):
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        res = requests.get(url).json()
        return res[0]["lat"], res[0]["lon"]
    except:
        return 28.61, 77.23

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Control Panel")

city = st.sidebar.text_input("📍 Location", "Delhi")
crop = st.sidebar.selectbox("🌾 Crop", ["Rice","Wheat","Corn"])
stage = st.sidebar.selectbox("🌱 Growth Stage", ["Seedling","Vegetative","Flowering","Harvest"])

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📸 Image", "📈 Analytics"])

# ===============================
# TAB 1 – PREDICTION
# ===============================
with tab1:

    if st.button("🚀 Run Analysis"):

        temp, humidity, rainfall, source = get_weather(city)

        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 Temp", f"{temp}°C")
        col2.metric("💧 Humidity", f"{humidity}%")
        col3.metric("🌧 Rainfall", f"{rainfall} mm")

        st.caption(f"Data Source: {source}")

        prob = model.predict_proba([[temp, humidity, rainfall]])[0][1]

        st.subheader("⚠️ Risk Score")
        st.progress(int(prob*100))

        # Risk color
        if prob < 0.3:
            st.success("🟢 Low Risk")
        elif prob < 0.7:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

        # Recommendation
        st.subheader("💊 Recommendation")

        if prob > 0.7:
            st.write("Apply fungicide within 48 hours")
        elif prob > 0.4:
            st.write("Monitor crop regularly")
        else:
            st.write("No action needed")

        # Map
        st.subheader("🌍 Risk Map")

        lat, lon = get_coordinates(city)
        m = folium.Map(location=[lat, lon], zoom_start=6)

        color = "green" if prob < 0.3 else "orange" if prob < 0.7 else "red"

        folium.Marker(
            [lat, lon],
            popup=f"Risk: {round(prob,2)}",
            icon=folium.Icon(color=color)
        ).add_to(m)

        st_folium(m, width=700)

# ===============================
# TAB 2 – IMAGE
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
# TAB 3 – ANALYTICS
# ===============================
with tab3:

    data = pd.read_csv("data.csv")

    st.subheader("📊 Weather Trends")
    st.line_chart(data[["temperature","humidity","rainfall"]])

    st.subheader("🌾 Disease Distribution")
    st.bar_chart(data["disease"].value_counts())

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.write("🚀 Built with AI for Smart Farming")
