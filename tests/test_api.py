from fastapi.testclient import TestClient
from deployment.main import app

client = TestClient(app)


def test_predict_endpoint():
    sample = {
        "ship_type": "Oil Service Boat",
        "route_id": "Lagos-Apapa",
        "month": "December",
        "distance": 134,
        "fuel_type": "HFO",
        "weather_condition": "Calm",
        "engine_efficiency": 90,
    }

    response = client.post("/predict", json=sample)

    # testing
    assert response.status_code == 200

    data = response.json()
    assert "fuel_consumption" in data
    assert "CO2_emissions" in data
    assert isinstance(data["fuel_consumption"], float)
    assert isinstance(data["CO2_emissions"], float)
