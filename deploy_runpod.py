#!/usr/bin/env python3
"""
RunPod Deployment Script for Model Fine-tuning
"""

import os
import subprocess
import json
import argparse
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunPodDeployer:
    def __init__(self, api_key: str=None):
        self.api_key = os.getenv("RUNPOD_API_KEY", api_key) 
        self.base_url = "https://api.runpod.io/graphql"
        
    def build_and_push_image(self, image_name: str, dockerfile_path: str = "."):
        """Build and push Docker image to registry"""
        logger.info(f"Building Docker image: {image_name}")
        
        # Build image
        build_cmd = f"docker build -t {image_name} {dockerfile_path}"
        result = subprocess.run(build_cmd.split(), capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Docker build failed: {result.stderr}")
            raise Exception("Docker build failed")
        
        # Push to registry
        logger.info(f"Pushing image to registry: {image_name}")
        push_cmd = f"docker push {image_name}"
        result = subprocess.run(push_cmd.split(), capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Docker push failed: {result.stderr}")
            raise Exception("Docker push failed")
        
        logger.info("Image built and pushed successfully")
        
    def create_runpod_config(self, 
                           image_name: str,
                           model_name: str,
                           output_model_name: str,
                           hf_token: str,
                           gpu_type: str = "NVIDIA RTX A4000",
                           container_disk_size: int = 50) -> Dict[str, Any]:
        """Create RunPod configuration"""
        
        config = {
            "name": "runpod_deploy",
            "imageName": image_name,
            "gpuTypeId": gpu_type,
            "containerDiskInGb": container_disk_size,
            "volumeInGb": 100,
            "volumeMountPath": "/workspace",
            "env": [
                {"key": "HF_TOKEN", "value": hf_token},
                {"key": "TRANSFORMERS_CACHE", "value": "/workspace/.cache"},
                {"key": "HF_HOME", "value": "/workspace/.cache"}
            ],
            "dockerArgs": f"python train.py --model_name {model_name} --output_model_name {output_model_name} --epochs 3 --batch_size 4",
            "ports": "8888/http",
            "volumeKey": "hf-training-volume"
        }
        
        return config
    
    def create_runpod_template(self, config: Dict[str, Any]) -> str:
        """Create RunPod template"""
        template_query = """
        mutation {
            saveTemplate(input: {
                name: "%s"
                imageName: "%s"
                containerDiskInGb: %d
                volumeInGb: %d
                volumeMountPath: "%s"
                env: [%s]
                dockerArgs: "%s"
                ports: "%s"
                readme: "Hugging Face Model Fine-tuning Template"
                isPublic: false
            }) {
                id
                name
            }
        }
        """ % (
            config["name"],
            config["imageName"],
            config["containerDiskInGb"],
            config["volumeInGb"],
            config["volumeMountPath"],
            ", ".join([f'{{key: "{env["key"]}", value: "{env["value"]}"}}' for env in config["env"]]),
            config["dockerArgs"],
            config["ports"]
        )
        
        # This is a simplified version - in practice, you'd use the RunPod SDK
        # or make HTTP requests to their GraphQL API
        logger.info("Template configuration created")
        return template_query
    
    def generate_runpod_script(self, config: Dict[str, Any]) -> str:
        """Generate a bash script for RunPod deployment"""
        script = f"""#!/bin/bash
# RunPod Setup Script for Model Fine-tuning

# Update system
apt-get update

# Install additional dependencies if needed
pip install --upgrade pip

# Set up environment
export HF_TOKEN="{config['env'][0]['value']}"
export TRANSFORMERS_CACHE="/workspace/.cache"
export HF_HOME="/workspace/.cache"

# Create cache directory
mkdir -p /workspace/.cache

# Navigate to app directory
cd /app

# Run the training script
{config['dockerArgs']}

# Keep container running for debugging if needed
echo "Training completed. Container will remain running for inspection."
tail -f /dev/null
"""
        return script

def main():
    parser = argparse.ArgumentParser(description="Deploy to RunPod")
    parser.add_argument("--api_key", required=False, help="RunPod API key")
    parser.add_argument("--image_name", required=True, help="Docker image name")
    parser.add_argument("--model_name", required=True, help="Base model name")
    parser.add_argument("--output_model_name", required=True, help="Output model name")
    parser.add_argument("--hf_token", default=os.getenv("HUGGINGFACE_API_KEY"), required=False, help="Hugging Face token")
    parser.add_argument("--gpu_type", default="NVIDIA RTX A4000", help="GPU type")
    parser.add_argument("--build_image", action="store_true", help="Build and push image")
    
    args = parser.parse_args()
    
    deployer = RunPodDeployer(args.api_key)
    
    # Build and push image if requested
    if args.build_image:
        deployer.build_and_push_image(args.image_name)
    
    # Create configuration
    config = deployer.create_runpod_config(
        image_name=args.image_name,
        model_name=args.model_name,
        output_model_name=args.output_model_name,
        hf_token=args.hf_token,
        gpu_type=args.gpu_type
    )
    
    # Generate RunPod script
    script = deployer.generate_runpod_script(config)
    
    # Save script to file
    with open("runpod_setup.sh", "w") as f:
        f.write(script)
    
    # Save config to file
    with open("runpod_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("RunPod configuration and setup script generated!")
    logger.info("Files created:")
    logger.info("- runpod_setup.sh: Script to run in RunPod container")
    logger.info("- runpod_config.json: Configuration for RunPod deployment")

if __name__ == "__main__":
    main()