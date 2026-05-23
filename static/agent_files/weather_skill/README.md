# Weather Skill

This directory contains a Python script to fetch current weather information.

## `get_weather.py`

This script retrieves the current weather for a specified location using the WeatherAPI.com service.

### Usage:

The script can fetch the weather for a specified city. If no city is provided, it defaults to "Toronto, Ontario".

To get the weather for a specific city, execute the script with the city name as an argument:
`python get_weather.py "City Name"`
Example: `python get_weather.py "London, UK"`

### Setup:

1.  **API Key:** Ensure you have a `.env` file in the `weather_skill` directory with your WeatherAPI.com key:
    ```
    WEATHER_API_KEY=YOUR_API_KEY
    ```
    Replace `YOUR_API_KEY` with your actual key.

### Dependencies:

*   `python-dotenv`
*   `requests`

These libraries are not currently installed in the sandbox environment. If this were a local environment, you would install them using pip:
`pip install python-dotenv requests`
