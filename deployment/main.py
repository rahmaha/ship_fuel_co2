from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import pickle
import os
import numpy as np
from dotenv import load_dotenv

app = FastAPI()

# load from .env
load_dotenv()


def load_model_and_dv(model_path: str, dv_path: str) -> tuple:
    """Load the trained model and the DictVectorizer from local or S3."""
    s3 = boto3.client(
        service_name="s3",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        endpoint_url="http://localhost:4566",
    )

    os.makedirs("models", exist_ok=True)

    # Only download if not already present
    if not os.path.exists(model_path):
        s3.download_file("ship-model-artifacts", "model.pkl", model_path)

    if not os.path.exists(dv_path):
        s3.download_file("ship-model-artifacts", "dv.pkl", dv_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(dv_path, "rb") as f:
        dv = pickle.load(f)

    return model, dv


model_path = os.getenv("MODEL_PATH")
dv_path = os.getenv("DV_PATH")
model, dv = load_model_and_dv(model_path, dv_path)


# Input schema
class ShipInput(BaseModel):
    ship_type: str
    route_id: str
    month: str
    distance: float
    fuel_type: str
    weather_condition: str
    engine_efficiency: float


@app.post("/predict")
def predict(input_data: ShipInput) -> dict:
    try:
        data_dict = input_data.model_dump()
        X = dv.transform([data_dict])
        y_pred = model.predict(X)
        y_pred_ori = np.expm1(y_pred)

        return {
            "fuel_consumption": float(y_pred_ori[0][0]),
            "CO2_emissions": float(y_pred_ori[0][1]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model_ready")
def model_ready():
    return {"model_loaded": model is not None, "dv_loaded": dv is not None}
