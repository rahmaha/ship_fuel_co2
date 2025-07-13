.PHONY: install lint format test unit-test integration-test monitoring docker-build docker-up deploy pre-commit lint-check format-check style-check

install:
	pipenv install --dev

test: unit-test integration-test
	PYTHONPATH=. pipenv run pytest

unit-test:
	pytest tests/test_train.py
	pytest tests/test_main.py

integration-test:
	pytest tests/test_api.py

lint:
	ruff . --fix

lint-check:
	ruff .

format:
	black .

format-check:
	black --check .

style-check: lint-check format-check

monitoring:
	pipenv run python monitoring/monitoring.py

docker-build:
	docker build -t ship-fuel-co2 -f docker/Dockerfile .

docker-up:
	docker-compose up --build

deploy:
	prefect deploy --all

pre-commit:
	pipenv run pre-commit run --all-files
