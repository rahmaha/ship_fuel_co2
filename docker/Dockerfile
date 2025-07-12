FROM python:3.10-slim

# System update & pipenv
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /app

# Copy everything needed for the app
COPY ../Pipfile ../Pipfile.lock ./  
RUN pipenv install --system --deploy

COPY ../.env .env
COPY ../deployment/main.py ./deployment/main.py
COPY ../models/model.pkl ./models/model.pkl
COPY ../models/dv.pkl ./models/dv.pkl

EXPOSE 9696

# Serve FastAPI app using uvicorn, referencing the module inside 'deployment' folder
ENTRYPOINT ["uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "9696"]
