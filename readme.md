# 🚢 Ship Fuel Consumption and CO₂ Emissions Prediction

This project builds a complete MLOps pipeline to **predict ship fuel consumption and CO₂ emissions** using multi-output regression. It serves as the final project for the **MLOps Zoomcamp 2025**.

---

## 📌 Problem Description

Ships are widely used for transporting goods and people across waterways. From global trade and defense to fishing and logistics, their role is critical. However, they also contribute significantly to CO₂ emissions and fuel usage, raising environmental concerns.

This project aims to address this issue by predicting **fuel consumption** and **CO₂ emissions** based on operational parameters like route, weather, fuel type, and engine efficiency. The model helps ship operators and engineers make data-driven decisions to **optimize fuel usage and reduce environmental impact**.

This is formulated as a **multi-output regression** task where the model predicts two targets:
- `fuel_consumption`
- `CO2_emissions`

---

## 🛠 Project Overview

| Component       | Description                                |
|------------------|--------------------------------------------|
| **Problem Type** | Multi-output Regression                    |
| **Targets**      | `fuel_consumption`, `CO2_emissions`        |
| **Model**        | XGBoost + MultiOutputRegressor             |
| **Tracking**     | MLflow                                     |
| **Workflow**     | Prefect                                    |
| **Monitoring**   | Evidently                                  |
| **Storage**      | LocalStack (S3 emulation)                  |
| **Deployment**   | FastAPI                                    |
| **Automation**   | Makefile                                   |
| **CI/CD**        | GitHub Actions                             |

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
| Automation  | Makefile               |

---

## 🧪 Reproducibility

This project is fully reproducible **without any real cloud account**. Everything runs locally using Docker and LocalStack.

### 🔧 Prerequisites

- Docker
- Python 3.10
- Pipenv
- Prefect
- Make (for automation)

> ⚠️ `Make` is a system-level tool and not included in `Pipfile`.  
> - **Windows:** Install via [Chocolatey](https://chocolatey.org/)  
> - **macOS/Linux:** Usually pre-installed

---

## 📂 Dataset

The dataset comes from [Kaggle](https://www.kaggle.com/datasets/jeleeladekunlefijabi/ship-fuel-consumption-and-co2-emissions-analysis) and is included in the `data/` directory.

---

## 🚀 How to Run

### Step-by-Step

1. **Install dependencies:**
   ```bash
   make install
   # or
   pipenv install --dev
   ```

2. **Start Prefect server:**
   ```bash
   prefect server start or you can use make run-ui
   ```
3. **Prefect Worker Pool Setup**

   Before running the project, make sure you have a Prefect worker pool named `ship_pool`.

   You can create it via the CLI:

   ```bash
   prefect worker pool create ship_pool --type process


4. **In a new terminal, run Docker services:**
   ```bash
   make build
   make up
   ```

5. **In another terminal, deploy flows and start the workers:**
   ```bash
   make deploy start-worker 
   ```
   Note: when prefect asked `Your Prefect workers will need access to this flow's code in order to run it. Would you like your workers to pull your flow code from a remote storage location when running this flow? [y/n] (y)`: n
   I recommend just always input `n`

6. **Run in other command prefect deployment run**
   So, if you just want to see the monitoring only you can choose `prefect deployment run monitoring-flow/monitoring-flow` but if you do edit the test.py like maybe change the weather from calm to the stormy then you need to do this command first `prefect deployment run main-flow/ship_training`

6. **Testing The API:**
   you can test using Script-based – Edit the input inside deployment/test.py


---

## 📌 Notes

- This project uses **multi-output regression**, meaning it predicts two continuous variables.
- Some processes  need to be started in separate terminals.
- Need a lot of further improvement (especially wrap prefect on docker-compose, using IaC and cloud and project structure)