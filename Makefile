.PHONY: install lint format test unit-test integration-test \
        build up down run-ui deploy start-worker train monitor clean

# === Setup ===
install:
	pipenv install --dev

# === Code Quality ===
lint:
	pipenv run ruff . --fix

format:
	pipenv run black .

# === Testing ===
test: unit-test integration-test

unit-test:
	PYTHONPATH=. pipenv run pytest tests/test_train.py
	PYTHONPATH=. pipenv run pytest tests/test_main.py

integration-test:
	PYTHONPATH=. pipenv run pytest tests/test_api.py

# === Prefect Flows ===
train:
	pipenv run python pipeline.py

monitor:
	pipenv run python monitoring/monitoring.py

run-ui:
	pipenv run prefect server start
deploy:
# 	pipenv run prefect deploy --all
	prefect deploy --all
start-worker:
	pipenv run prefect worker start --pool ship_pool

# === Docker Compose ===
build:
	docker-compose build

up:
	docker-compose up

down:
	docker-compose down

# === Clean Up (optional) ===
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf models/*.pkl mlruns mlflow.db

