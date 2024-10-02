import streamlit as st
import requests

# Assuming your API key is stored in st.secrets like this:
# st.secrets["openweathermap_api_key"]
API_KEY = st.secrets["openweathermap_api_key"]

def get_current_weather(location):
    if "," in location:
        location = location.split(",")[0].strip()
    url_base = "https://api.openweathermap.org/data/2.5/weather"
    url = f"{url_base}?q={location}&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()

    # Extract temperatures and convert from Kelvin to Celsius
    temp = data['main']['temp'] - 273.15
    feels_like = data['main']['feels_like'] - 273.15
    temp_min = data['main']['temp_min'] - 273.15
    temp_max = data['main']['temp_max'] - 273.15
    humidity = data['main']['humidity']

    return {
        "location": location,
        "temperature": round(temp, 2),
        "feels_like": round(feels_like, 2),
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),
        "humidity": round(humidity, 2)
    }

# Testing the function
st.write(get_current_weather('Syracuse, NY'))
st.write(get_current_weather('London, England'))

def suggest_clothing(weather_data):
    # Placeholder for clothing suggestion logic
    temp = weather_data['temperature']
    if temp > 20:
        return "It's warm! Wear light clothing."
    elif temp > 10:
        return "Moderate temperature. Consider a light jacket."
    else:
        return "It's cold! Dress warmly."

# Streamlit user interface
st.title('Weather and Clothing Suggestion Bot')
location = st.text_input('Enter a city:', 'Syracuse, NY')
weather_data = get_current_weather(location)
st.write('Weather Data:', weather_data)
suggestion = suggest_clothing(weather_data)
st.write('Clothing Suggestion:', suggestion)
