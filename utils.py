import requests

API_KEY = "9f244592efe26bbd55cf0f9ddaeb63d6"

def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            raise Exception("API not responding")

        data = response.json()

        if "main" not in data:
            raise Exception(data.get("message", "Invalid API"))

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        rainfall = 0
        if "rain" in data:
            rainfall = data["rain"].get("1h", 0)

        return temp, humidity, rainfall

    except Exception as e:
        print("API ERROR:", e)

        # 🔥 fallback demo data (important)
        return 30, 75, 10
