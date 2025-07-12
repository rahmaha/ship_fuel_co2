from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import mlflow
import pickle
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

app = FastAPI()

# load from .env
load_dotenv()

def load_model_and_dv(model_path: str, dv_path: str) -> tuple:
    """Load the trained model and the DictVectorizer from local files."""

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    # Load DictVectorizer
    if not os.path.exists(dv_path):
        raise FileNotFoundError(f"DictVectorizer file not found at path: {dv_path}")
    
    try:
        with open(dv_path, 'rb') as f:
            dv = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load DictVectorizer: {str(e)}")

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


@app.post('/predict')
def predict(input_data: ShipInput) -> dict:
    try:
        data_dict = input_data.model_dump()
        X = dv.transform([data_dict])
        y_pred = model.predict(X)
        y_pred_ori = np.expm1(y_pred)

        return {
            'fuel_consumption': float(y_pred_ori[0][0]),
            'CO2_emissions': float(y_pred_ori[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get('/health')
def health_check():
    return {'status': 'ok'}
