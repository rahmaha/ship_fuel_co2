import requests


sample = {
    'ship_type' : 'Oil Service Boat',
    'route_id'  : 'Lagos-Apapa',
    'month'     : 'December',
    'distance'  : 134,
    'fuel_type' : 'HFO',
    'weather_condition' : "Calm",
    'engine_efficiency' : 90
}


url = "http://localhost:9696/predict"
response = requests.post(url, json=sample)
print(response.json())