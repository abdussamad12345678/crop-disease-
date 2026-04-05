import requests

API_KEY = "9f244592efe26bbd55cf0f9ddaeb63d6"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    res = requests.get(url)
    data = res.json()

    if "main" not in data:
        raise Exception(data.get("message", "Weather API Error"))

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    rainfall = data.get("rain", {}).get("1h", 0)

    return temp, humidity, rainfall
