import pickle


def test_pipeline_prediction():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/dv.pkl", "rb") as f:
        dv = pickle.load(f)

    sample = {
        "ship_type": "Oil Service Boat",
        "route_id": "Lagos-Apapa",
        "month": "December",
        "distance": 134,
        "fuel_type": "HFO",
        "weather_condition": "Calm",
        "engine_efficiency": 90,
    }

    X = dv.transform([sample])
    pred = model.predict(X)
    assert pred.shape == (1, 2)
