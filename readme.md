1. Problem Description (Make it Worth 2)

In your README.md, describe:

    The business context (e.g. “Predict fuel consumption and CO₂ emissions of ships for better environmental compliance”)

    Why it matters

    What your ML model does

    The outcome (MLflow tracking, local API, etc.)

    XGBRegressor,"{'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1}",23d42f65f3c54b1ca2423d0204ca989e,0.14975189304885136,5.565986394882202


command on root level:

prefect deploy
prefect worker start --pool ship_pool
prefect deployment run main-flow/ship_training

uvicorn deployment.main:app --reload --port 9696
python deployment\test.py

docker build -f docker/Dockerfile.prefect -t prefect-ship-fuel .

docker build -f docker/Dockerfile -t ship_fuel_co2 .
docker run -it --rm -p 9696:9696 ship_fuel_co2
docker run --rm --env-file .env -p 9696:9696 ship_fuel_co2

prefect deploy 
and etc

docker-compose up --build

