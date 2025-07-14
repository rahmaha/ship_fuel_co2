# 🚢 Ship Fuel Consumption and CO₂ Emissions Prediction 

This Project builds a full MLOps pipeline for **predicting fuel consumption and CO₂ emissions (multioutput regression)**. This Project also as final project for MLOps Zoomcamp 2025.

## Problem Description
Ship is one of the most used transportation either people or goods across waterway. They serve different purpose as they are very important like global trade, defense, fishing and more. Because of that they also contribute to the most CO₂ emissions, making them consume a lot of fuel that make it concers for environtmel and mairtime industry. This project aims to **predict both fuel consumption and CO₂ emissions** of ships based on various operational parameters (e.g., route, weather, engine efficiency). The goal is to help optimize fuel usage and reduce environmental impact by enabling **data-driven decisions**.

The problem is framed as a **multi-output regression** task. The model takes ship features as input and outputs two continuous targets:  
- `fuel_consumption`  
- `CO2_emissions`



## Project Overview
- **Type:** Multi-output Regression
- **Targets:** `fuel_consumption` & `CO2_emissions`
- **Model:** XGBoost (wrapped with `MultiOutputRegressor`)
- **Monitoring:** Prefect + Evidently
- **Storage:** Model artifacts stored in **LocalStack S3**
- **Tracking:** MLflow
- **Deployment:** FastAPI

---

## ⚙️ Tools & Frameworks

| Area        | Tool/Library           |
|-------------|------------------------|
| Modeling    | XGBoost, scikit-learn  |
| Workflow    | Prefect                |
| Tracking    | MLflow                 |
| Monitoring  | Evidently              |
| Cloud Mock  | LocalStack (S3)        |
| API         | FastAPI                |
| CI/CD       | GitHub Actions         |
| Packaging   | Docker, pipenv         |
| Automating  | Makefile               |
| task        |                        |

---

## Reproducibility
This project doesn't use real cloud. So, don't need real account.

### Prerequisites
- Docker
- Pipenv
- Prefect
- Makefile 

(please try to meet the prerequisites on your own machine). Also for makefile it doesn't include in pipfile or any on the requirements because it is not package-library.  It’s a system-level tool that runs on your operating system. 

- Windows: install via [Chocolatey](https://chocolatey.org/)
- Linux/macOS: usually pre-installed

Then run tasks like:
```sh
make test```


### Dataset
The dataset that we used for this project is from kaggle ![kaggle]https://www.kaggle.com/datasets/jeleeladekunlefijabi/ship-fuel-consumption-and-co2-emissions-analysis. It's really nice dataset. This dataset saved in `data/` folder.

### Step-by-Step
- first you can using command `make install` or pipenv `pipenv install --dev` to set up your virtual environment and install dependencies
- run `prefect server start on your terminal`
- open separate terminal and you can do command make build up
- open another terminal again and do make deploy and prefect deployment 

1. Problem Description (Make it Worth 2)

In your README.md, describe:

    The business context (e.g. “Predict fuel consumption and CO₂ emissions of ships for better environmental compliance”)

    Why it matters

    What your ML model does

    The outcome (MLflow tracking, local API, etc.)

    XGBRegressor,"{'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1}",23d42f65f3c54b1ca2423d0204ca989e,0.14975189304885136,5.565986394882202


command on root level:

docker run --rm -it -p 4566:4566 -p 4571:4571 localstack/localstack

prefect deploy
prefect worker start --pool ship_pool
prefect deployment run main-flow/ship_training

uvicorn deployment.main:app --reload --port 9696
python deployment\test.py

docker build -f docker/Dockerfile -t ship_fuel_co2 .
docker run -it --rm -p 9696:9696 ship_fuel_co2
docker run --rm --env-file .env -p 9696:9696 ship_fuel_co2
python deployment\test.py

prefect deploy 
and etc

docker-compose up --build

Unittest: 
-test_main.py
-test_train.py
Integration test:
- test_api.py
$env:PYTHONPATH="."; pytest (for testing) -> using powershell on windows
or do this set PYTHONPATH=.
pytest (on cmd)

Using black and ruff as linter and code formatter. 
    - black -> is a code formatter.
    - Ruff -> is a fast, high-performance linter and formatter.

    ruff check .
    black .

pre-commit install

make Makefile install

# Start LocalStack
docker-compose up -d


docker run -d -p 4566:4566 -p 4571:4571 localstack/localstack


aws --endpoint-url=http://localhost:4566 s3 ls \
  --region us-east-1 \
  --no-sign-request \
  --access-key test \
  --secret-key test
