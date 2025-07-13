#!/usr/bin/env python3
"""
Hugging Face Model Fine-tuning Script
Loads a model, fine-tunes it, and pushes back to Hugging Face Hub
"""

import os
import argparse
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelFineTuner:
    def __init__(self, model_name: str, output_model_name: str, hf_token: str):
        self.model_name = model_name
        self.output_model_name = output_model_name
        self.hf_token = hf_token
        self.tokenizer = None
        self.model = None
        
    def authenticate(self):
        """Authenticate with Hugging Face Hub"""
        try:
            login(token=self.hf_token)
            logger.info("Successfully authenticated with Hugging Face Hub")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def prepare_dataset(self, dataset_name: str = None, custom_data: list = None):
        """Prepare dataset for training"""
        if custom_data:
            # Use custom data
            dataset = Dataset.from_dict({"text": custom_data})
        elif dataset_name:
            # Load from Hugging Face datasets
            dataset = load_dataset(dataset_name, split="train")
        else:
            # Default example dataset
            example_texts = [
                "Hello, how are you today?",
                "I'm doing great, thanks for asking!",
                "What's the weather like?",
                "It's sunny and warm outside.",
                "Can you help me with a Python question?",
                "Of course! I'd be happy to help with Python."
            ]
            dataset = Dataset.from_dict({"text": example_texts})
        
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def fine_tune(self, dataset, training_args_dict: dict = None):
        """Fine-tune the model"""
        logger.info("Starting fine-tuning process...")
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Default training arguments
        default_args = {
            "output_dir": "./results",
            "overwrite_output_dir": True,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "save_steps": 500,
            "save_total_limit": 2,
            "prediction_loss_only": True,
            "logging_steps": 100,
            "logging_first_step": True,
            "fp16": torch.cuda.is_available(),
            "dataloader_pin_memory": False,
            "gradient_checkpointing": True,
            "remove_unused_columns": False,
        }
        
        # Update with user-provided arguments
        if training_args_dict:
            default_args.update(training_args_dict)
        
        training_args = TrainingArguments(**default_args)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model locally
        trainer.save_model("./fine_tuned_model")
        self.tokenizer.save_pretrained("./fine_tuned_model")
        
        logger.info("Fine-tuning completed successfully")
    
    def push_to_hub(self):
        """Push the fine-tuned model to Hugging Face Hub"""
        logger.info(f"Pushing model to Hub: {self.output_model_name}")
        
        try:
            # Load the saved model
            model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
            tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
            
            # Push to hub
            model.push_to_hub(self.output_model_name)
            tokenizer.push_to_hub(self.output_model_name)
            
            logger.info(f"Model successfully pushed to: https://huggingface.co/{self.output_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to push model to hub: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model")
    parser.add_argument("--model_name", required=True, help="Base model name from Hugging Face")
    parser.add_argument("--output_model_name", required=True, help="Output model name for Hub")
    parser.add_argument("--hf_token", help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--dataset_name", help="Dataset name from Hugging Face datasets")
    parser.add_argument("--custom_data_file", help="Path to JSON file with custom training data")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    
    args = parser.parse_args()
    
    # Get HF token from argument or environment
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token must be provided via --hf_token or HF_TOKEN env var")
    
    # Initialize fine-tuner
    fine_tuner = ModelFineTuner(
        model_name=args.model_name,
        output_model_name=args.output_model_name,
        hf_token=hf_token
    )
    
    # Authenticate
    fine_tuner.authenticate()
    
    # Load model and tokenizer
    fine_tuner.load_model_and_tokenizer()
    
    # Prepare dataset
    custom_data = None
    if args.custom_data_file:
        with open(args.custom_data_file, 'r') as f:
            custom_data = json.load(f)
    
    dataset = fine_tuner.prepare_dataset(
        dataset_name=args.dataset_name,
        custom_data=custom_data
    )
    
    # Training arguments
    training_args = {
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
    }
    
    # Fine-tune
    fine_tuner.fine_tune(dataset, training_args)
    
    # Push to hub
    fine_tuner.push_to_hub()
    
    logger.info("Fine-tuning pipeline completed successfully!")

if __name__ == "__main__":
    main()