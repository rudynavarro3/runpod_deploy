# Hugging Face Model Fine-tuning Makefile

.PHONY: help setup build run clean test deploy

help:
	@echo "Available commands:"
	@echo "  setup    - Set up the project environment"
	@echo "  build    - Build the Docker image"
	@echo "  run      - Run the training locally"
	@echo "  clean    - Clean up generated files"
	@echo "  test     - Run tests"
	@echo "  deploy   - Deploy to RunPod"

setup:
	pip install -r requirements.txt
	python setup.py

build:
	docker build -t runpod_deploy .

run:
	docker-compose up

clean:
	docker-compose down -v
	docker system prune -f
	rm -rf results/ fine_tuned_model/ models/.cache/

test:
	python -m pytest tests/ -v

deploy:
	python deploy_runpod.py --help

# Development commands
dev-install:
	pip install -r requirements.txt
	pip install jupyter notebook ipython

dev-notebook:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Docker commands
docker-shell:
	docker run -it --rm --gpus all -v $(PWD):/app runpod_deploy bash

docker-logs:
	docker-compose logs -f model-trainer
