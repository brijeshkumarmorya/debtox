.PHONY: help install setup test build-frontend train analyze run-api run-ui run docker-build docker-up docker-down clean

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
UVICORN ?= .venv/bin/uvicorn
PYTEST ?= .venv/bin/pytest

help:
	@echo "DebtOx Platform Management"
	@echo "------------------------------------------------------"
	@echo "make install         - Install backend and frontend dependencies"
	@echo "make test            - Run all unit and integration tests"
	@echo "make build-frontend  - Build production frontend bundle"
	@echo "make train           - Train models and run research experiments"
	@echo "make analyze         - Analyze sample repository with DebtOx CLI"
	@echo "make run-api         - Run FastAPI backend development server"
	@echo "make run-ui          - Run Vite frontend development server"
	@echo "make run             - Run production unified server (API + UI on :8000)"
	@echo "make docker-build    - Build Docker containers"
	@echo "make docker-up       - Start Docker Compose stack"
	@echo "make docker-down     - Stop Docker Compose stack"
	@echo "make clean           - Remove caches, temporary files and builds"

install:
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

setup: install build-frontend
	@echo "DebtOx environment setup complete."

test:
	$(PYTHON) -m pytest backend/tests/ -v

build-frontend:
	@echo "Building frontend bundle..."
	cd frontend && npm run build

train:
	@echo "Running empirical training and experiment benchmarks..."
	$(PYTHON) -m ml.experiments.runner

analyze:
	@echo "Analyzing sample Java repository..."
	./debt-ox analyze data/sample_repo

run-api:
	$(UVICORN) backend.app.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	cd frontend && npm run dev

run: build-frontend
	$(UVICORN) backend.app.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf frontend/dist
