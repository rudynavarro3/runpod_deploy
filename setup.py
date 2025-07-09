#!/usr/bin/env python3
"""
Project Setup Script
Creates the necessary directory structure and files for the HF fine-tuning project
"""

import os
import json
import argparse
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure"""
    directories = [
        "data",
        "config",
        "results",
        "models",
        "scripts",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def create_sample_config():
    """Create sample configuration files"""
    
    # Training configuration
    train_config = {
        "model_name": "microsoft/DialoGPT-small",
        "output_model_name": "your-username/my-finetuned-model",
        "training_args": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "logging_steps": 100,
            "save_steps": 500,
            "eval_steps": 500,
            "warmup_steps": 100,
            "max_steps": -1,
            "fp16": True,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False
        },
        "dataset": {
            "name": "squad",
            "config": "plain_text",
            "split": "train[:1000]"
        }
    }
    
    with open("config/train_config.json", "w") as f:
        json.dump(train_config, f, indent=2)
    
    print("Created config/train_config.json")

def create_sample_data():
    """Create sample training data"""
    sample_data = [
        "What is machine learning?",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "How do neural networks work?",
        "Neural networks process information through layers of interconnected nodes, mimicking the human brain.",
        "What is fine-tuning?",
        "Fine-tuning is the process of adapting a pre-trained model to a specific task or domain.",
        "Why use transformers?",
        "Transformers are powerful architectures that excel at processing sequential data like text.",
        "What is GPU acceleration?",
        "GPU acceleration uses graphics cards to speed up parallel computations in machine learning."
    ]
    
    with open("data/sample_data.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("Created data/sample_data.json")

def create_env_template():
    """Create environment template file"""
    env_template = """# Hugging Face Model Fine-tuning Environment Variables

# Hugging Face Hub token (required)
HF_TOKEN=your_huggingface_token_here

# Your Hugging Face username (required for model uploads)
HF_USERNAME=your_username

# RunPod API key (for deployment)
RUNPOD_API_KEY=your_runpod_api_key

# Docker registry settings
DOCKER_REGISTRY=docker.io
DOCKER_USERNAME=your_docker_username

# Model settings
BASE_MODEL_NAME=microsoft/DialoGPT-small
OUTPUT_MODEL_NAME=${HF_USERNAME}/my-finetuned-model

# Training settings
EPOCHS=3
BATCH_SIZE=2
LEARNING_RATE=5e-5

# Cache directories
TRANSFORMERS_CACHE=./models/.cache
HF_HOME=./models/.cache
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
    
    print("Created .env.template")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Model files and cache
models/
*.bin
*.safetensors
results/
fine_tuned_model/
.cache/

# Jupyter
.ipynb_checkpoints

# Docker
.dockerignore

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
wandb/
tensorboard/

# RunPod
runpod_config.json
runpod_setup.sh
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("Created .gitignore")

def create_makefile():
    """Create Makefile for common tasks"""
    makefile_content = """# Hugging Face Model Fine-tuning Makefile

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
	docker build -t hf-model-trainer .

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
	docker run -it --rm --gpus all -v $(PWD):/app hf-model-trainer bash

docker-logs:
	docker-compose logs -f model-trainer
"""
    
    with open("Makefile", "w") as f:
        f.write(makefile_content)
    
    print("Created Makefile")

def main():
    parser = argparse.ArgumentParser(description="Set up HF fine-tuning project")
    parser.add_argument("--all", action="store_true", help="Create all files and directories")
    
    args = parser.parse_args()
    
    print("Setting up Hugging Face Model Fine-tuning Project...")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Create configuration files
    create_sample_config()
    create_sample_data()
    create_env_template()
    create_gitignore()
    create_makefile()
    
    print("\n" + "=" * 50)
    print("Project setup complete!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and fill in your credentials")
    print("2. Modify config/train_config.json with your model settings")
    print("3. Add your training data to data/")
    print("4. Run 'make build' to build the Docker image")
    print("5. Run 'make run' to start training locally")
    print("6. Use 'make deploy' for RunPod deployment")
    
    print("\nProject structure:")
    print("├── train.py              # Main training script")
    print("├── Dockerfile            # Docker configuration")
    print("├── docker-compose.yml    # Local development setup")
    print("├── requirements.txt      # Python dependencies")
    print("├── setup.py             # This setup script")
    print("├── deploy_runpod.py     # RunPod deployment script")
    print("├── Makefile             # Common tasks")
    print("├── .env.template        # Environment variables template")
    print("├── config/              # Configuration files")
    print("├── data/                # Training data")
    print("├── results/             # Training outputs")
    print("├── models/              # Model cache")
    print("└── scripts/             # Additional scripts")

if __name__ == "__main__":
    main()