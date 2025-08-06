from django.http import HttpResponse
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os
from django.shortcuts import render

API_KEY = '5c4f49700f913b44be32fc286f7762a9'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

def get_current_weather(city):
    try:
        url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()  # Raises exception for 4XX/5XX errors
        data = response.json()
        
        # Safely extract data with .get() to avoid KeyError
        return {
            'city': data.get('name', ''),
            'current_temp': round(data.get('main', {}).get('temp', 0)),
            'feels_like': round(data.get('main', {}).get('feels_like', 0)),
            'temp_min': round(data.get('main', {}).get('temp_min', 0)),
            'temp_max': round(data.get('main', {}).get('temp_max', 0)),
            'humidity': round(data.get('main', {}).get('humidity', 0)),
            'description': data.get('weather', [{}])[0].get('description', 'N/A'),
            'country': data.get('sys', {}).get('country', ''),
            'wind_gust_dir': data.get('wind', {}).get('deg', 0),
            'pressure': data.get('main', {}).get('pressure', 0),
            'Wind_Gust_Speed': data.get('wind', {}).get('speed', 0),
            'clouds': data.get('clouds', {}).get('all', 0),
            'visibility': data.get('visibility', 0),
        }
    except Exception as e:
        print(f"Weather API Error: {str(e)}")
        return None

def read_historical_data(filename):
    try:
        df = pd.read_csv(filename)
        df = df.dropna()
        df = df.drop_duplicates()
        return df
    except Exception as e:
        print(f"Error reading historical data: {str(e)}")
        return None

def prepare_data(data):
    try:
        le = LabelEncoder()
        data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
        data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
        X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
        y = data['RainTomorrow']
        return X, y, le
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return None, None, None

def train_rain_model(x, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Mean Squared Error for Rain Model: {mean_squared_error(y_test, y_pred)}")
        return model
    except Exception as e:
        print(f"Error training rain model: {str(e)}")
        return None

def prepare_regression_data(data, feature):
    try:
        X, y = [], []
        for i in range(len(data) - 1):
            X.append(data[feature].iloc[i])
            y.append(data[feature].iloc[i+1])
        return np.array(X).reshape(-1, 1), np.array(y)
    except Exception as e:
        print(f"Error preparing regression data: {str(e)}")
        return None, None

def train_regression_model(x, y):
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x, y)
        return model
    except Exception as e:
        print(f"Error training regression model: {str(e)}")
        return None

def predict_future(model, current_value):
    try:
        predictions = [current_value]
        for _ in range(5):
            next_value = model.predict(np.array([[predictions[-1]]]))
            predictions.append(next_value[0])
        return predictions[1:]
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return [current_value] * 5  # Return current value as fallback

def weather_view(request):
    # Default context with safe fallback values
    context = {
        'error': None,
        'location': '',
        'current_temp': 0,
        'MinTemp': 0,
        'MaxTemp': 0,
        'FeelsLike': 0,
        'humidity': 0,
        'clouds': 0,
        'description': 'N/A',
        'city': '',
        'country': '',
        'time': datetime.now(),
        'date': datetime.now().strftime("%B %d, %Y"),
        'wind': 0,
        'pressure': 0,
        'visibility': 0,
        'time1': '--:--', 'time2': '--:--', 'time3': '--:--', 'time4': '--:--', 'time5': '--:--',
        'temp1': '0', 'temp2': '0', 'temp3': '0', 'temp4': '0', 'temp5': '0',
        'hum1': '0', 'hum2': '0', 'hum3': '0', 'hum4': '0', 'hum5': '0'
    }

    if request.method == 'POST':
        city = request.POST.get('city', '').strip()
        if not city:
            context['error'] = 'Please enter a city name'
            return render(request, 'weather.html', context)

        current_weather = get_current_weather(city)
        if not current_weather:
            context['error'] = 'Could not fetch weather data for this location'
            return render(request, 'weather.html', context)

        # Load historical data
        csv_path = os.path.join('E:\\ProjectTY\\weather.csv')
        historical_data = read_historical_data(csv_path)
        if historical_data is None:
            context['error'] = 'Could not load historical weather data'
            return render(request, 'weather.html', context)

        # Prepare and train models
        X, y, le = prepare_data(historical_data)
        if X is None or le is None:
            context['error'] = 'Error preparing weather data'
            return render(request, 'weather.html', context)

        rain_model = train_rain_model(X, y)
        if rain_model is None:
            context['error'] = 'Error training weather model'
            return render(request, 'weather.html', context)

        # Prepare current data for prediction
        wind_deg = current_weather.get('wind_gust_dir', 0) % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]
        compass_direction = next((point for point, start, end in compass_points if start <= wind_deg < end), "N")
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else 0

        current_data = {
            'MinTemp': current_weather.get('temp_min', 0),
            'MaxTemp': current_weather.get('temp_max', 0),
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather.get('Wind_Gust_Speed', 0),
            'Humidity': current_weather.get('humidity', 0),
            'Pressure': current_weather.get('pressure', 0),
            'Temp': current_weather.get('current_temp', 0),
        }

        # Train regression models
        x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
        temp_model = train_regression_model(x_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)

        # Make predictions
        future_temp = predict_future(temp_model, current_data['Temp'])
        future_humidity = predict_future(hum_model, current_data['Humidity'])

        # Prepare time slots
        timezone = pytz.timezone('Asia/Karachi')
        now = datetime.now(timezone)
        future_times = [(now + timedelta(hours=i+1)).strftime("%H:00") for i in range(5)]

        # Update context
        context.update({
            'location': city,
            'current_temp': current_weather.get('current_temp', 0),
            'MinTemp': current_weather.get('temp_min', 0),
            'MaxTemp': current_weather.get('temp_max', 0),
            'FeelsLike': current_weather.get('feels_like', 0),
            'humidity': current_weather.get('humidity', 0),
            'clouds': current_weather.get('clouds', 0),
            'description': current_weather.get('description', 'N/A'),
            'city': current_weather.get('city', ''),
            'country': current_weather.get('country', ''),
            'wind': current_weather.get('Wind_Gust_Speed', 0),
            'pressure': current_weather.get('pressure', 0),
            'visibility': current_weather.get('visibility', 0),
            'time1': future_times[0], 'time2': future_times[1], 'time3': future_times[2],
            'time4': future_times[3], 'time5': future_times[4],
            'temp1': f"{round(future_temp[0], 1)}", 'temp2': f"{round(future_temp[1], 1)}",
            'temp3': f"{round(future_temp[2], 1)}", 'temp4': f"{round(future_temp[3], 1)}",
            'temp5': f"{round(future_temp[4], 1)}",
            'hum1': f"{round(future_humidity[0], 1)}", 'hum2': f"{round(future_humidity[1], 1)}",
            'hum3': f"{round(future_humidity[2], 1)}", 'hum4': f"{round(future_humidity[3], 1)}",
            'hum5': f"{round(future_humidity[4], 1)}"
        })

    return render(request, 'weather.html', context)

def map_view(request):
    return render(request, 'map.html')

# forecast/views.py
def index_view(request):
    return render(request, 'index.html')