import os
import requests
from dotenv import load_dotenv
import sys

load_dotenv(dotenv_path='weather_skill/.env')
API_KEY = os.getenv("WEATHER_API_KEY")

def get_current_weather(location):
    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": API_KEY,
        "q": location
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        weather_data = response.json()
        
        if "error" in weather_data:
            print(f"Error: {weather_data['error']['message']}")
            return

        current = weather_data["current"]
        location_name = weather_data["location"]["name"]
        region = weather_data["location"]["region"]
        country = weather_data["location"]["country"]
        temp_c = current["temp_c"]
        condition = current["condition"]["text"]
        
        print(f"Current weather in {location_name}, {region}, {country}:")
        print(f"Temperature: {temp_c}°C")
        print(f"Condition: {condition}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        LOCATION = " ".join(sys.argv[1:])
    else:
        LOCATION = "Toronto, Ontario" # Default location

    get_current_weather(LOCATION)
