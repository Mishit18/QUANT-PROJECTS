# Makefile for Kalman Filter + HMM project

.PHONY: help install test run clean lint format

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run unit tests"
	@echo "  make run        - Run full pipeline"
	@echo "  make clean      - Clean generated files"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code"
	@echo "  make notebook   - Start Jupyter notebook"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

run:
	python run_pipeline.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

lint:
	flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503

format:
	black src/ tests/ --line-length=120

notebook:
	jupyter notebook notebooks/
